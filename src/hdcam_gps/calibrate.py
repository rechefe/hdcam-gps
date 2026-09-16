"""Calibrating a classifier against an operating point, by measurement.

The threshold and the vote rule of a CAM classifier can be derived from a chance
model, and that model is wrong in a way that matters: it treats every non
matching row as a coin flip, when a strong satellite cross correlates with the
other PRNs' codes by a fixed amount that repeats on every look. The measured
false alarm rate is then orders of magnitude worse than the model predicts.

So this module measures instead of assuming. The caller states an operating
point - a C/N0 to work at, a missed detection rate and a false alarm rate to stay
under - and calibrate searches the vote rule and threshold for the setting that
meets it, scoring every candidate on the same recorded scenarios.

Three things make that affordable and fair.

Records are scored through distance_table and replayed through decide, so the
expensive sweep over the search space happens once per record rather than once
per candidate, and the replay runs the classifier's own decision rather than a
copy of it. The false alarm trials include a satellite, since the cross
correlation floor only exists when one is present, and are counted per
acquisition on the PRNs that are not there.

The FFT reference goes through the identical protocol over a grid of peak
ratios, replayed from cached correlation surfaces. Comparing a calibrated CAM
against a reference left at its default threshold is not a comparison, and
PeakRatioCalibrator is what stops that happening.

The honest target is 1e-2 false alarms per acquisition, not the 1e-4 a receiver
wants. Confirming 1e-4 by the rule of three needs 30 000 records per family. So
pfa_upper is printed in every table and 1e-4 is not demonstrated anywhere.
"""

from dataclasses import dataclass
from math import comb
from typing import ClassVar

import numpy as np
from tqdm.auto import tqdm

from hdcam_gps.acq_base import AcqConfig, PrnResult
from hdcam_gps.cam_acq import CamAcqClassifier
from hdcam_gps.evaluate import AcquisitionEvents, score_events
from hdcam_gps.fft_acq import FftAcqClassifier
from hdcam_gps.hdcam_acq import OneBitHdCamClassifier
from hdcam_gps.signal_gen import SatelliteTruth, Scenario, generate_synthetic

DEFAULT_VOTE_CHOICES: tuple[float, ...] = (0.2, 1 / 3, 0.5, 0.7)
DEFAULT_SIGMA_GRID: tuple[float, ...] = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)
DEFAULT_RATIO_GRID: tuple[float, ...] = (1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0)
DEFAULT_TARGET_PFA: float = 1e-2  # Per acquisition, and all the records support


@dataclass(frozen=True)
class OperatingPoint:
    """The specification a calibrated classifier has to meet."""

    cn0_dbhz: float = 40.0  # The C/N0 it has to work at
    max_pmd: float = 0.10  # Missed detections tolerated, as a fraction
    max_pfa: float = 1e-4  # False alarms tolerated, per acquisition
    code_phase_tolerance: int = 1  # Samples of code phase error to forgive
    doppler_bins: float = 1.0  # Bins of Doppler error to forgive
    band_dbhz: tuple[float, float] = (38.0, 42.0)  # Where detection is compared


class RateCounts:
    """The rates a candidate reports, from counters its dataclass holds."""

    @property
    def pmd(self) -> float:
        """Measured missed detection rate, per acquisition."""
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

    @property
    def pfa_per_prn(self) -> float:
        """False reports per (acquisition, absent PRN) opportunity."""
        return (
            self.n_prn_false / self.n_prn_trials if self.n_prn_trials else float("nan")
        )

    @property
    def pfa_per_prn_upper(self) -> float:
        """Upper confidence bound on the per absent PRN rate.

        A supporting number only: the absent PRNs of one record face the same
        satellites, so they are not independent trials.
        """
        return binomial_upper_bound(self.n_prn_false, self.n_prn_trials)

    @property
    def pd(self) -> float:
        """Pooled per satellite detection rate, over every record."""
        return self.n_found / self.n_satellites if self.n_satellites else 0.0

    @property
    def pd_in_band(self) -> float:
        """Per satellite detection rate inside the operating point's C/N0 band."""
        return (
            self.n_band_found / self.n_band_satellites if self.n_band_satellites else 0.0
        )

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
class Candidate(RateCounts):
    """One setting of the two CAM knobs, and how it did."""

    hd_threshold: int
    min_votes: int
    n_trials: int
    n_detected: int  # Trials where every present satellite was found correctly
    n_false: int  # Trials reporting a false alarm of either kind
    n_wrong_fix: int = 0  # Satellites reported at the wrong Doppler or phase
    n_satellites: int = 0  # Satellites observed, pooled over the trials
    n_found: int = 0  # Of those, found within both tolerances
    n_prn_trials: int = 0  # (trial, absent PRN) opportunities
    n_prn_false: int = 0  # Of those, reported
    n_band_satellites: int = 0  # Satellites inside the target's C/N0 band
    n_band_found: int = 0  # Of those, found

    SETTING_HEADER: ClassVar[str] = f"{'thr':>5} {'votes':>6}"

    @property
    def setting(self) -> str:
        """The two knobs, formatted for a table row."""
        return f"{self.hd_threshold:5d} {self.min_votes:6d}"


@dataclass(frozen=True)
class RatioCandidate(RateCounts):
    """One peak ratio of the FFT reference, and how it did."""

    peak_ratio: float
    n_trials: int
    n_detected: int
    n_false: int
    n_wrong_fix: int = 0
    n_satellites: int = 0
    n_found: int = 0
    n_prn_trials: int = 0
    n_prn_false: int = 0
    n_band_satellites: int = 0
    n_band_found: int = 0

    SETTING_HEADER: ClassVar[str] = f"{'ratio':>12}"

    @property
    def setting(self) -> str:
        """The one knob, formatted for a table row."""
        return f"{self.peak_ratio:12.2f}"


@dataclass(frozen=True)
class CalibrationResult:
    """What the search found."""

    target: OperatingPoint
    candidates: tuple  # Every setting tried, best first
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
        setting_header = (
            self.candidates[0].SETTING_HEADER
            if self.candidates
            else Candidate.SETTING_HEADER
        )
        header = (
            f"{setting_header} {'Pmd':>7} {'Pfa':>8} {'Pfa 95%':>9} {'':>5}"
        )
        rows = [header, "-" * len(header)]
        for candidate in self.candidates:
            mark = "OK" if candidate.meets(self.target) else ""
            rows.append(
                f"{candidate.setting} "
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


def binomial_lower_bound(successes: int, trials: int, confidence: float = 0.95) -> float:
    """The Clopper-Pearson lower bound on a rate.

    Args:
        successes (int): Events observed.
        trials (int): Trials run.
        confidence (float): The confidence level.

    Returns:
        float: The smallest rate consistent with the observation.
    """
    if trials == 0:
        return 0.0
    if successes <= 0:
        return 0.0
    alpha = 1.0 - confidence
    low, high = 0.0, successes / trials
    for _ in range(100):  # bisection on the binomial upper tail
        middle = (low + high) / 2
        tail = sum(
            comb(trials, k) * middle**k * (1 - middle) ** (trials - k)
            for k in range(successes, trials + 1)
        )
        if tail > alpha:
            high = middle
        else:
            low = middle
    return low


def binomial_interval(
    successes: int, trials: int, confidence: float = 0.95
) -> tuple[float, float]:
    """The two sided Clopper-Pearson interval on a rate.

    Each tail carries half the remaining probability, so a 95 percent interval is
    the 97.5 percent one sided bounds. This is what a detection rate is reported
    with, and a bin holding under a hundred observations cannot carry a
    sensitivity claim however good its point estimate looks.

    Args:
        successes (int): Events observed.
        trials (int): Trials run.
        confidence (float): The coverage of the interval.

    Returns:
        tuple[float, float]: The lower and upper bound.
    """
    one_sided = 1.0 - (1.0 - confidence) / 2.0
    return (
        binomial_lower_bound(successes, trials, one_sided),
        binomial_upper_bound(successes, trials, one_sided),
    )


def score_table(
    classifier: CamAcqClassifier,
    table: np.ndarray,
    hd_threshold: int,
    min_votes: int,
) -> list[PrnResult]:
    """Replays the decision over a precomputed distance table.

    The replay is the classifier's own decide, not a second implementation of the
    same rule, so there is nothing for the two to disagree about.

    Args:
        classifier (CamAcqClassifier): The classifier whose rule to replay.
        table (np.ndarray): Distances of shape (n_queries, n_rows).
        hd_threshold (int): The per look threshold to apply.
        min_votes (int): How many looks have to match.

    Returns:
        list[PrnResult]: What the classifier would have returned.
    """
    return classifier.decide(table, hd_threshold, min_votes)


class CamCalibrator:
    """Replays a grid of (threshold, vote rule) settings from one sweep per record."""

    def __init__(
        self,
        classifier: CamAcqClassifier,
        sigma_grid: tuple[float, ...] = DEFAULT_SIGMA_GRID,
        vote_fractions: tuple[float, ...] = DEFAULT_VOTE_CHOICES,
        floor: tuple[float, float] | None = None,
    ):
        """Fixes the grid of settings against the measured chance floor.

        Args:
            classifier (CamAcqClassifier): The family to calibrate.
            sigma_grid (tuple[float, ...]): Thresholds to try, in standard
                deviations below the measured chance floor.
            vote_fractions (tuple[float, ...]): Vote rules to try, as a fraction
                of the looks available.
            floor (tuple[float, float] | None): The chance floor as (mean, sd).
                None measures it, which is the point: the binomial model is
                right for a 1 bit codebook and wrong for a thermometer one.
        """
        self.classifier = classifier
        chance, deviation = floor if floor is not None else classifier.chance_floor()
        self.chance_mean, self.chance_sd = chance, deviation
        votes = {
            max(1, min(classifier.n_looks, int(round(f * classifier.n_looks))))
            for f in vote_fractions
        }
        self.settings = sorted(
            {
                (int(round(chance - sigma * deviation)), vote)
                for sigma in sigma_grid
                for vote in votes
            }
        )

    def prepare(self, scenario: Scenario):
        """The one sweep per record every setting is then scored from.

        Args:
            scenario (Scenario): The record to sweep.

        Returns:
            np.ndarray: The distance table.
        """
        return self.classifier.distance_table(scenario.samples)

    def replay(self, prepared, setting) -> list[PrnResult]:
        """Scores one setting off the prepared sweep.

        Args:
            prepared: The distance table.
            setting: A (hd_threshold, min_votes) pair.

        Returns:
            list[PrnResult]: What the classifier would have returned.
        """
        return self.classifier.decide(prepared, setting[0], setting[1])

    def candidate(self, setting, counts: dict) -> Candidate:
        """Packs the counters of one setting into a Candidate.

        Args:
            setting: A (hd_threshold, min_votes) pair.
            counts (dict): The accumulated event counters.

        Returns:
            Candidate: The setting and how it did.
        """
        return Candidate(hd_threshold=setting[0], min_votes=setting[1], **counts)


class PeakRatioCalibrator:
    """Replays a grid of peak ratios from one set of correlation surfaces.

    The reference has to be calibrated by the same protocol as the families it is
    the reference for, or the comparison is against whichever default threshold
    happened to be in the file.
    """

    def __init__(
        self,
        classifier: FftAcqClassifier,
        ratio_grid: tuple[float, ...] = DEFAULT_RATIO_GRID,
    ):
        """Fixes the grid of peak ratios to try.

        Args:
            classifier (FftAcqClassifier): The reference to calibrate.
            ratio_grid (tuple[float, ...]): Peak to second peak ratios to try.
        """
        self.classifier = classifier
        self.settings = sorted(float(ratio) for ratio in ratio_grid)

    def prepare(self, scenario: Scenario):
        """The correlation surfaces every ratio is then scored from.

        Args:
            scenario (Scenario): The record to correlate.

        Returns:
            dict[int, np.ndarray]: One surface per PRN.
        """
        return self.classifier.surfaces(scenario.samples)

    def replay(self, prepared, setting) -> list[PrnResult]:
        """Scores one peak ratio off the prepared surfaces.

        Args:
            prepared: The correlation surfaces.
            setting: The peak ratio.

        Returns:
            list[PrnResult]: What the reference would have returned.
        """
        return self.classifier.decide(prepared, setting)

    def candidate(self, setting, counts: dict) -> RatioCandidate:
        """Packs the counters of one ratio into a RatioCandidate.

        Args:
            setting: The peak ratio.
            counts (dict): The accumulated event counters.

        Returns:
            RatioCandidate: The setting and how it did.
        """
        return RatioCandidate(peak_ratio=setting, **counts)


def synthetic_trials(
    config: AcqConfig, cn0_dbhz: float, n_trials: int, seed: int
) -> list[Scenario]:
    """One satellite per record, drawn from the configuration's own search space.

    The satellite is present in every trial on purpose: the cross correlation
    floor that the chance model misses only exists when one is.

    Args:
        config (AcqConfig): The configuration to build for.
        cn0_dbhz (float): The C/N0 to place the satellite at.
        n_trials (int): How many records to draw.
        seed (int): Seed of the draw.

    Returns:
        list[Scenario]: The records and their truth.
    """
    rng = np.random.default_rng(seed)
    scenarios = []
    for _ in range(n_trials):
        satellite = SatelliteTruth(
            prn=int(rng.choice(np.array(config.prn_list))),
            doppler_hz=float(rng.choice(config.doppler_grid_hz)),
            code_phase=int(rng.integers(config.samples_per_code)),
            cn0_dbhz=cn0_dbhz,
            carrier_phase_rad=float(rng.uniform(0, 2 * np.pi)),
        )
        scenarios.append(
            generate_synthetic(config, [satellite], seed=int(rng.integers(2**31)))
        )
    return scenarios


def _empty_counts(n_trials: int) -> dict:
    """The counters one setting accumulates over a run of records.

    Args:
        n_trials (int): How many records the run holds.

    Returns:
        dict: Every counter at zero, ready to pass to a Candidate.
    """
    return {
        "n_trials": n_trials,
        "n_detected": 0,
        "n_false": 0,
        "n_wrong_fix": 0,
        "n_satellites": 0,
        "n_found": 0,
        "n_prn_trials": 0,
        "n_prn_false": 0,
        "n_band_satellites": 0,
        "n_band_found": 0,
    }


def _accumulate(counts: dict, events: AcquisitionEvents, band: tuple[float, float]):
    """Folds one acquisition's events into a setting's counters.

    Args:
        counts (dict): The counters, modified in place.
        events (AcquisitionEvents): What the acquisition got right and wrong.
        band (tuple[float, float]): The C/N0 band detection is compared in.
    """
    low, high = band
    counts["n_satellites"] += events.n_satellites
    counts["n_found"] += events.n_found
    counts["n_wrong_fix"] += events.n_wrong_fix
    counts["n_prn_trials"] += events.n_absent_prns
    counts["n_prn_false"] += events.n_false_prns
    counts["n_detected"] += int(
        events.n_satellites > 0 and events.n_found == events.n_satellites
    )
    counts["n_false"] += int(events.n_false_alarms > 0)
    in_band_found = sum(low <= cn0 <= high for cn0 in events.found_cn0_dbhz)
    in_band_missed = sum(low <= cn0 <= high for cn0 in events.missed_cn0_dbhz)
    counts["n_band_found"] += in_band_found
    counts["n_band_satellites"] += in_band_found + in_band_missed


def run_calibration(
    replayer,
    scenarios: list[Scenario],
    target: OperatingPoint,
    progress: bool = True,
) -> CalibrationResult:
    """Scores every setting of a replayer on the same records.

    Args:
        replayer: A CamCalibrator or PeakRatioCalibrator.
        scenarios (list[Scenario]): The records to score on.
        target (OperatingPoint): The specification, and the tolerances to score
            against.
        progress (bool): Show a progress bar.

    Returns:
        CalibrationResult: Every setting tried and the best that qualifies.
    """
    assert scenarios, "At least one record is needed."
    counters = {
        setting: _empty_counts(len(scenarios)) for setting in replayer.settings
    }
    for scenario in tqdm(
        scenarios, disable=not progress, desc="calibrating", unit="record"
    ):
        prepared = replayer.prepare(scenario)
        for setting in replayer.settings:
            events = score_events(
                scenario,
                replayer.replay(prepared, setting),
                code_phase_tolerance=target.code_phase_tolerance,
                doppler_bins=target.doppler_bins,
            )
            _accumulate(counters[setting], events, target.band_dbhz)

    candidates = tuple(
        sorted(
            (
                replayer.candidate(setting, counts)
                for setting, counts in counters.items()
            ),
            key=lambda c: (c.pmd, c.pfa),
        )
    )
    return CalibrationResult(
        target=target, candidates=candidates, n_trials=len(scenarios)
    )


def match_false_alarm(
    result: CalibrationResult, target_pfa: float = DEFAULT_TARGET_PFA
):
    """The setting the study compares at, picked at a matched false alarm rate.

    For every vote rule, the loosest threshold whose false alarm bound clears the
    target; among those, the one detecting most inside the operating point's C/N0
    band. Comparing families at their own best false alarm rates would reward
    whichever happened to be most conservative, so the rate is fixed first and
    sensitivity is read off afterwards.

    Args:
        result (CalibrationResult): Every setting tried.
        target_pfa (float): The per acquisition rate every family is held to.

    Returns:
        Candidate | RatioCandidate | None: The pick, or None when no setting
            clears the target.
    """
    passing = [c for c in result.candidates if c.pfa_upper <= target_pfa]
    if not passing:
        return None
    loosest: dict = {}
    for candidate in passing:
        knob = getattr(candidate, "min_votes", None)
        current = loosest.get(knob)
        if current is None or _looseness(candidate) > _looseness(current):
            loosest[knob] = candidate
    return max(
        loosest.values(), key=lambda c: (c.pd_in_band, c.pd, _looseness(c))
    )


def _looseness(candidate) -> float:
    """How permissive a setting is, whichever knob it turns.

    Args:
        candidate: A Candidate or a RatioCandidate.

    Returns:
        float: Larger means more detections and more false alarms.
    """
    if isinstance(candidate, Candidate):
        return float(candidate.hd_threshold)
    return -float(candidate.peak_ratio)


def calibrate(
    config: AcqConfig,
    target: OperatingPoint = OperatingPoint(),
    n_trials: int = 100,
    sigma_grid: tuple[float, ...] = DEFAULT_SIGMA_GRID,
    vote_fractions: tuple[float, ...] = DEFAULT_VOTE_CHOICES,
    seed: int = 0,
    progress: bool = True,
    classifier: CamAcqClassifier | None = None,
    scenarios=None,
) -> CalibrationResult:
    """Searches the threshold and vote rule for a setting that meets a target.

    With no scenarios, every trial places one satellite at the target C/N0, at a
    random PRN, Doppler bin and code phase drawn from the configuration's own
    search space. A satellite counts as found when it comes back within one
    Doppler bin and code_phase_tolerance samples, and a record counts as a false
    alarm when it reports an absent PRN or fixes a present one wrongly.

    With a ScenarioBank, the records are real skies of ten satellites and the
    trial count is whatever the bank holds. A threshold tuned on one satellite
    does not transfer to a ten satellite sky, which is why the study calibrates
    on a bank and evaluates on a disjoint one.

    Args:
        config (AcqConfig): The acquisition configuration to calibrate for.
        target (OperatingPoint): The C/N0, Pmd and Pfa to meet.
        n_trials (int): Records to measure each candidate on, scenarios aside.
        sigma_grid (tuple[float, ...]): Thresholds to try, in standard deviations
            below the measured chance floor.
        vote_fractions (tuple[float, ...]): Vote rules to try, as a fraction of
            the looks available.
        seed (int): Seed of the scenario draw.
        progress (bool): Show a progress bar.
        classifier (CamAcqClassifier | None): The family to calibrate. None
            builds the 1 bit baseline for the configuration.
        scenarios (ScenarioBank | None): Records to calibrate on. None draws
            single satellite synthetic trials.

    Returns:
        CalibrationResult: Every setting tried and the best that qualifies.
    """
    assert n_trials >= 1, "At least one trial is needed."
    if classifier is None:
        classifier = OneBitHdCamClassifier(config)
    assert classifier.config == config, (
        "The classifier was built for a different configuration."
    )
    records = (
        bank_records(scenarios)
        if scenarios is not None
        else synthetic_trials(config, target.cn0_dbhz, n_trials, seed)
    )
    replayer = CamCalibrator(classifier, sigma_grid, vote_fractions)
    return run_calibration(replayer, records, target, progress)


def calibrate_peak_ratio(
    classifier: FftAcqClassifier,
    target: OperatingPoint = OperatingPoint(),
    n_trials: int = 100,
    ratio_grid: tuple[float, ...] = DEFAULT_RATIO_GRID,
    seed: int = 0,
    progress: bool = True,
    scenarios=None,
) -> CalibrationResult:
    """Runs the FFT reference through the protocol calibrate runs a family through.

    Args:
        classifier (FftAcqClassifier): The reference to calibrate.
        target (OperatingPoint): The C/N0, Pmd and Pfa to meet.
        n_trials (int): Records to measure each ratio on, scenarios aside.
        ratio_grid (tuple[float, ...]): Peak to second peak ratios to try.
        seed (int): Seed of the scenario draw.
        progress (bool): Show a progress bar.
        scenarios (ScenarioBank | None): Records to calibrate on. None draws
            single satellite synthetic trials.

    Returns:
        CalibrationResult: Every ratio tried and the best that qualifies.
    """
    assert n_trials >= 1, "At least one trial is needed."
    records = (
        bank_records(scenarios)
        if scenarios is not None
        else synthetic_trials(
            classifier.config, target.cn0_dbhz, n_trials, seed
        )
    )
    return run_calibration(
        PeakRatioCalibrator(classifier, ratio_grid), records, target, progress
    )


def bank_records(scenarios) -> list[Scenario]:
    """Every record a ScenarioBank holds, in a fixed order.

    Args:
        scenarios (ScenarioBank): The bank to flatten.

    Returns:
        list[Scenario]: One entry per (scaling, sky).
    """
    return [
        scenarios.get(cn0_dbhz, index)
        for cn0_dbhz in scenarios.cn0_dbhz
        for index in scenarios.indices
    ]
