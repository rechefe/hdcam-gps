"""Calibrating an HdCam classifier against an operating point.

The threshold and the vote rule of OneBitHdCamClassifier can be derived from a
chance model, and that model is wrong in a way that matters: it treats every non
matching row as a coin flip, when a strong satellite cross correlates with the
other PRNs' codes by a fixed amount that repeats on every look. The measured
false alarm rate is then orders of magnitude worse than the model predicts.

So this module measures instead of assuming. The caller states an operating
point - a C/N0 to work at, a missed detection rate and a false alarm rate to stay
under - and calibrate searches the vote rule and threshold for the setting that
meets it, scoring every candidate on the same recorded scenarios.

Two things make that affordable. Records are scored through distance_table, so
the expensive sweep over the search space happens once per record rather than
once per candidate. And the false alarm trials include a satellite, since the
cross correlation floor only exists when one is present, and are counted per
acquisition on the PRNs that are not there.
"""

from dataclasses import dataclass
from math import comb

import numpy as np
from tqdm.auto import tqdm

from hdcam_gps.acq_base import AcqConfig, PrnResult
from hdcam_gps.hdcam_acq import OneBitHdCamClassifier
from hdcam_gps.signal_gen import SatelliteTruth, generate_synthetic

DEFAULT_VOTE_CHOICES: tuple[float, ...] = (0.2, 1 / 3, 0.5, 0.7)
DEFAULT_SIGMA_GRID: tuple[float, ...] = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)


@dataclass(frozen=True)
class OperatingPoint:
    """The specification a calibrated classifier has to meet."""

    cn0_dbhz: float = 40.0  # The C/N0 it has to work at
    max_pmd: float = 0.10  # Missed detections tolerated, as a fraction
    max_pfa: float = 1e-4  # False alarms tolerated, per acquisition
    code_phase_tolerance: int = 1  # Samples of code phase error to forgive


@dataclass(frozen=True)
class Candidate:
    """One setting of the two knobs, and how it did."""

    hd_threshold: int
    min_votes: int
    n_trials: int
    n_detected: int  # Trials where the satellite was found correctly
    n_false: int  # Trials reporting a PRN that was not there

    @property
    def pmd(self) -> float:
        """Measured missed detection rate."""
        return 1.0 - self.n_detected / self.n_trials if self.n_trials else 1.0

    @property
    def pfa(self) -> float:
        """Measured false alarm rate, per acquisition."""
        return self.n_false / self.n_trials if self.n_trials else 1.0

    @property
    def pfa_upper(self) -> float:
        """Upper confidence bound on the false alarm rate.

        A rate of zero over a hundred trials is not a rate of zero. This is what
        the trials actually support, at 95 percent.
        """
        return binomial_upper_bound(self.n_false, self.n_trials)

    def meets(self, target: OperatingPoint) -> bool:
        """Whether this candidate satisfies an operating point.

        The false alarm side is judged on the confidence bound rather than the
        point estimate, so a target cannot be met simply by running few trials.

        Args:
            target (OperatingPoint): The specification.

        Returns:
            bool: True when both rates are within the target.
        """
        return self.pmd <= target.max_pmd and self.pfa_upper <= target.max_pfa


@dataclass(frozen=True)
class CalibrationResult:
    """What the search found."""

    target: OperatingPoint
    candidates: tuple[Candidate, ...]  # Every setting tried, best first
    n_trials: int

    @property
    def best(self) -> Candidate | None:
        """The lowest missed detection rate among the settings that qualify."""
        passing = [c for c in self.candidates if c.meets(self.target)]
        return min(passing, key=lambda c: c.pmd) if passing else None

    @property
    def trials_needed(self) -> int:
        """Trials required before the false alarm target could be confirmed.

        With no false alarms observed, the 95 percent bound is about three over
        the number of trials, so this is how many are needed for that bound to
        reach the target at all.
        """
        return int(np.ceil(3.0 / self.target.max_pfa))

    def table(self) -> str:
        """Renders every candidate as a fixed width table.

        Returns:
            str: One row per setting tried.
        """
        header = (
            f"{'thr':>5} {'votes':>6} {'Pmd':>7} {'Pfa':>8} {'Pfa 95%':>9} {'':>5}"
        )
        rows = [header, "-" * len(header)]
        for candidate in self.candidates:
            mark = "OK" if candidate.meets(self.target) else ""
            rows.append(
                f"{candidate.hd_threshold:5d} {candidate.min_votes:6d} "
                f"{candidate.pmd:7.3f} {candidate.pfa:8.4f} "
                f"{candidate.pfa_upper:9.4f} {mark:>5}"
            )
        return "\n".join(rows)


def binomial_upper_bound(successes: int, trials: int, confidence: float = 0.95) -> float:
    """The Clopper-Pearson upper bound on a rate.

    Args:
        successes (int): Events observed.
        trials (int): Trials run.
        confidence (float): The confidence level.

    Returns:
        float: The largest rate consistent with the observation.
    """
    if trials == 0:
        return 1.0
    if successes >= trials:
        return 1.0
    alpha = 1.0 - confidence
    low, high = successes / trials, 1.0
    for _ in range(100):  # bisection on the binomial lower tail
        middle = (low + high) / 2
        tail = sum(
            comb(trials, k) * middle**k * (1 - middle) ** (trials - k)
            for k in range(successes + 1)
        )
        if tail > alpha:
            low = middle
        else:
            high = middle
    return high


def score_table(
    classifier: OneBitHdCamClassifier,
    table: np.ndarray,
    hd_threshold: int,
    min_votes: int,
) -> list[PrnResult]:
    """Replays the decision over a precomputed distance table.

    This mirrors what the classifier does, without touching the CAM, so a whole
    grid of settings can be scored from one sweep of a record.

    Args:
        classifier (OneBitHdCamClassifier): Supplies the codebook layout.
        table (np.ndarray): Distances of shape (n_starts, n_rows).
        hd_threshold (int): The per look threshold to apply.
        min_votes (int): How many looks have to match.

    Returns:
        list[PrnResult]: What the classifier would have returned.
    """
    samples_per_code = classifier.config.samples_per_code
    n_starts = table.shape[0]
    matched = table <= hd_threshold

    phases = np.arange(n_starts) % samples_per_code
    votes = np.zeros((samples_per_code, classifier.n_rows), dtype=np.int32)
    best = np.full((samples_per_code, classifier.n_rows), table.max() + 1, np.int32)
    np.add.at(votes, phases, matched)
    np.minimum.at(best, phases, table)

    rows, columns = np.nonzero(votes >= min_votes)
    distances = {
        (int(row), int(phase)): int(best[phase, row])
        for phase, row in zip(rows, columns)
    }
    return classifier._best_per_prn(distances)


def calibrate(
    config: AcqConfig,
    target: OperatingPoint = OperatingPoint(),
    n_trials: int = 100,
    sigma_grid: tuple[float, ...] = DEFAULT_SIGMA_GRID,
    vote_fractions: tuple[float, ...] = DEFAULT_VOTE_CHOICES,
    seed: int = 0,
    progress: bool = True,
) -> CalibrationResult:
    """Searches the threshold and vote rule for a setting that meets a target.

    Every trial places one satellite at the target C/N0, at a random PRN, Doppler
    bin and code phase drawn from the configuration's own search space. A trial
    counts as a detection when that satellite comes back with the right Doppler
    and code phase, and as a false alarm when any other PRN is reported. The
    satellite is present in every trial on purpose: the cross correlation floor
    that the chance model misses only exists when one is.

    Args:
        config (AcqConfig): The acquisition configuration to calibrate for.
        target (OperatingPoint): The C/N0, Pmd and Pfa to meet.
        n_trials (int): Records to measure each candidate on.
        sigma_grid (tuple[float, ...]): Thresholds to try, in standard deviations
            below the chance floor.
        vote_fractions (tuple[float, ...]): Vote rules to try, as a fraction of
            the looks available.
        seed (int): Seed of the scenario draw.
        progress (bool): Show a progress bar.

    Returns:
        CalibrationResult: Every setting tried and the best that qualifies.
    """
    assert n_trials >= 1, "At least one trial is needed."
    classifier = OneBitHdCamClassifier(config)
    chance = classifier.n_columns / 2
    deviation = np.sqrt(classifier.n_columns) / 2

    settings = sorted(
        {
            (int(round(chance - sigma * deviation)), votes)
            for sigma in sigma_grid
            for votes in {
                max(1, min(classifier.n_looks, int(round(f * classifier.n_looks))))
                for f in vote_fractions
            }
        }
    )
    detected = {setting: 0 for setting in settings}
    false = {setting: 0 for setting in settings}

    rng = np.random.default_rng(seed)
    for trial in tqdm(
        range(n_trials), disable=not progress, desc="calibrating", unit="trial"
    ):
        satellite = SatelliteTruth(
            prn=int(rng.choice(np.array(config.prn_list))),
            doppler_hz=float(rng.choice(config.doppler_grid_hz)),
            code_phase=int(rng.integers(config.samples_per_code)),
            cn0_dbhz=target.cn0_dbhz,
            carrier_phase_rad=float(rng.uniform(0, 2 * np.pi)),
        )
        scenario = generate_synthetic(
            config, [satellite], seed=int(rng.integers(2**31))
        )
        table = classifier.distance_table(scenario.samples)
        wanted = scenario.expected_results()[0]

        for hd_threshold, min_votes in settings:
            results = score_table(classifier, table, hd_threshold, min_votes)
            hit = next((r for r in results if r.prn == wanted.prn), None)
            if hit is not None and hit.doppler_hz == wanted.doppler_hz:
                error = abs(hit.code_phase - wanted.code_phase)
                error = min(error, config.samples_per_code - error)
                if error <= target.code_phase_tolerance:
                    detected[(hd_threshold, min_votes)] += 1
            if any(r.prn != wanted.prn for r in results):
                false[(hd_threshold, min_votes)] += 1

    candidates = tuple(
        sorted(
            (
                Candidate(
                    hd_threshold=hd_threshold,
                    min_votes=min_votes,
                    n_trials=n_trials,
                    n_detected=detected[(hd_threshold, min_votes)],
                    n_false=false[(hd_threshold, min_votes)],
                )
                for hd_threshold, min_votes in settings
            ),
            key=lambda c: (c.pmd, c.pfa),
        )
    )
    return CalibrationResult(target=target, candidates=candidates, n_trials=n_trials)
