"""How far the right answer sits from the wrong ones, before any threshold.

This is the cheap screen of docs/CAM_FAMILY_STUDY.md section 4.6, and it is what
PROPOSAL risk 1 asks for. A CAM fires on a Hamming distance below a tolerance, so
a design is buildable only if the tolerance it needs is one silicon can offer:
the JSSC 2025 macro demonstrates 8 bits in a 64 bit word, 12.5 %, and a coin flip
puts the ceiling near 50 %. Nothing in this repo measured that fraction.

Two distributions answer it, and neither needs a decision rule, so the same code
screens every family:

* D_true, the distance from a query at the satellite's own code phase to the row
  that stands for it, kept per look so the single look distribution exists. The
  tolerance has to be at least this, or the right row never fires.
* D_wrong, the distance to every row of a PRN the record does not contain, at
  every start. The tolerance has to be below this, or everything fires. Its mean
  and standard deviation are the chance floor measured **with satellites
  present**, which is the floor that matters: a strong satellite cross
  correlates with the other PRNs' codes by a fixed amount that a pure noise
  measurement never sees.

From them, per C/N0 bin: d' = (mean D_wrong - mean D_true) / sd D_wrong, and the
required tolerance fraction T = the 90th percentile of single look D_true over
n_columns. A family whose T is above 50 % cannot work at all; one between 12.5 %
and 50 % works only on a CAM nobody has built.

Running this costs one distance sweep per record, read in slices so the
segmented family's 21504 rows never sit in memory at once.
"""

from dataclasses import dataclass, field

import numpy as np
from tqdm.auto import tqdm

from hdcam_gps.cam_acq import CamAcqClassifier
from hdcam_gps.signal_gen import Scenario

SILICON_TOLERANCE_FRACTION: float = 8 / 64  # JSSC 60(8):3009, 8 bits of 64
CHANCE_TOLERANCE_FRACTION: float = 0.5  # Where a wrong row already sits
DEFAULT_BIN_WIDTH_DB: float = 2.0  # C/N0 bin width of the screen
DEFAULT_PERCENTILE: float = 90.0  # Of single look D_true, per section 4.6
KILL_BIN_DBHZ: float = 45.0  # The bin section 8's kill rule reads d' in
DEFAULT_CODE_PHASE_TOLERANCE: int = 1  # Samples, as section 4.1 scores a detection


@dataclass(frozen=True)
class ScreenBin:
    """The screen's numbers for one C/N0 bin."""

    cn0_dbhz: float  # Bin centre
    n_looks: int  # Single look observations pooled into it
    true_mean: float  # Mean D_true
    true_percentile: float  # The percentile of D_true the tolerance must reach
    d_prime: float  # (mean D_wrong - mean D_true) / sd D_wrong
    tolerance_fraction: float  # true_percentile / n_columns
    resolution_fraction: float  # (mean D_wrong - true_percentile) / n_columns
    resolution_sigma: float  # The same gap in units of sd D_wrong

    @property
    def buildable(self) -> bool:
        """Whether the tolerance this bin needs is one silicon has demonstrated."""
        return self.tolerance_fraction <= SILICON_TOLERANCE_FRACTION


@dataclass(frozen=True)
class Separation:
    """What one family's distance tables say, before any threshold is chosen."""

    family: str
    n_rows: int
    n_columns: int
    n_records: int
    true_distance: np.ndarray = field(repr=False)  # One per (record, satellite, look)
    true_cn0_dbhz: np.ndarray = field(repr=False)  # The satellite's C/N0, matching
    wrong_mean: float  # Chance floor with satellites present
    wrong_sd: float
    n_wrong: int  # Distances it was measured over
    noise_mean: float  # Chance floor on noise alone, for reference
    noise_sd: float

    def bin_edges(self, width: float = DEFAULT_BIN_WIDTH_DB) -> np.ndarray:
        """The C/N0 bin edges covering every satellite observed.

        Args:
            width (float): Bin width in dB.

        Returns:
            np.ndarray: Edges, so there is at least one bin.
        """
        if self.true_cn0_dbhz.size == 0:
            return np.array([0.0, width])
        low = np.floor(self.true_cn0_dbhz.min() / width) * width
        high = np.ceil(self.true_cn0_dbhz.max() / width) * width
        return np.arange(low, max(high, low + width) + width / 2, width)

    def bins(
        self,
        width: float = DEFAULT_BIN_WIDTH_DB,
        percentile: float = DEFAULT_PERCENTILE,
        min_looks: int = 20,
    ) -> list[ScreenBin]:
        """The screen's numbers, one entry per populated C/N0 bin.

        Args:
            width (float): Bin width in dB.
            percentile (float): Of single look D_true, the one the tolerance has
                to reach.
            min_looks (int): Bins holding fewer observations are dropped, since
                a percentile of ten samples is not a percentile.

        Returns:
            list[ScreenBin]: Ordered by C/N0.
        """
        edges = self.bin_edges(width)
        result = []
        for low, high in zip(edges[:-1], edges[1:]):
            inside = (self.true_cn0_dbhz >= low) & (self.true_cn0_dbhz < high)
            if inside.sum() < min_looks:
                continue
            result.append(
                self._bin((low + high) / 2, self.true_distance[inside], percentile)
            )
        return result

    def _bin(
        self, centre: float, distances: np.ndarray, percentile: float
    ) -> ScreenBin:
        """Builds one bin's entry from the D_true values that fell in it.

        Args:
            centre (float): The bin centre in dB-Hz.
            distances (np.ndarray): Single look D_true values inside the bin.
            percentile (float): The percentile the tolerance has to reach.

        Returns:
            ScreenBin: The bin's numbers.
        """
        reach = float(np.percentile(distances, percentile))
        spread = self.wrong_sd if self.wrong_sd > 0 else float("inf")
        return ScreenBin(
            cn0_dbhz=centre,
            n_looks=int(distances.size),
            true_mean=float(distances.mean()),
            true_percentile=reach,
            d_prime=(self.wrong_mean - float(distances.mean())) / spread,
            tolerance_fraction=reach / self.n_columns,
            resolution_fraction=(self.wrong_mean - reach) / self.n_columns,
            resolution_sigma=(self.wrong_mean - reach) / spread,
        )

    def at(
        self, cn0_dbhz: float, width: float = DEFAULT_BIN_WIDTH_DB, **kwargs
    ) -> ScreenBin | None:
        """The bin holding a C/N0, or None when nothing landed in it.

        Args:
            cn0_dbhz (float): The C/N0 to look up.
            width (float): Bin width in dB.
            **kwargs: Passed to bins.

        Returns:
            ScreenBin | None: The bin containing that C/N0.
        """
        for entry in self.bins(width=width, **kwargs):
            if abs(entry.cn0_dbhz - cn0_dbhz) <= width / 2:
                return entry
        return None

    def table(self, width: float = DEFAULT_BIN_WIDTH_DB, **kwargs) -> str:
        """Renders the screen as a fixed width table.

        Args:
            width (float): Bin width in dB.
            **kwargs: Passed to bins.

        Returns:
            str: A header naming the family, then one row per C/N0 bin.
        """
        header = (
            f"{'C/N0':>6} {'looks':>7} {'D_true':>8} {'p90':>8} {'d prime':>8} "
            f"{'tol':>7} {'res':>7} {'res sd':>7} {'build':>6}"
        )
        rows = [
            f"{self.family}: {self.n_rows} x {self.n_columns} bits, "
            f"{self.n_records} records, "
            f"D_wrong {self.wrong_mean:.1f} +- {self.wrong_sd:.1f} "
            f"(noise alone {self.noise_mean:.1f} +- {self.noise_sd:.1f})",
            header,
            "-" * len(header),
        ]
        for entry in self.bins(width=width, **kwargs):
            rows.append(
                f"{entry.cn0_dbhz:6.1f} {entry.n_looks:7d} {entry.true_mean:8.1f} "
                f"{entry.true_percentile:8.1f} {entry.d_prime:8.2f} "
                f"{entry.tolerance_fraction:7.3f} {entry.resolution_fraction:7.3f} "
                f"{entry.resolution_sigma:7.2f} "
                f"{'yes' if entry.buildable else 'no':>6}"
            )
        return "\n".join(rows)


def true_targets(
    classifier: CamAcqClassifier,
    satellite,
    code_phase_tolerance: int = DEFAULT_CODE_PHASE_TOLERANCE,
) -> list[tuple]:
    """The (queries, rows, looks) a satellite's own hypothesis occupies.

    A hit is the right answer when its row stands for the satellite and its
    query sits at the satellite's code phase, which is cam_acq's mapping run
    backwards. Rows starting at different points of the code period read
    different code phases out of one window, so they are grouped by that offset.

    The code phase is taken within a tolerance rather than exactly, for the same
    reason section 4.1 scores a detection that way: gps_sdr_sim.labels rounds a
    fractional code phase to the nearest sample, and at 1.023 MHz one sample is
    one chip, so the labelled sample is often not the one the signal is at.
    Measured on a real sky, seven of nine satellites peaked one sample after
    their label and read the chance floor at the label itself.

    Args:
        classifier (CamAcqClassifier): The family being screened.
        satellite (SatelliteTruth): The satellite to locate.
        code_phase_tolerance (int): Samples either side of the label to accept.

    Returns:
        list[tuple]: One (query indices, row indices, look of each query) group
            per distinct segment offset.
    """
    rows = classifier.rows()
    queries = classifier.queries()
    samples_per_code = classifier.config.samples_per_code
    wanted = classifier.true_rows(satellite.prn, satellite.doppler_hz)

    groups = []
    for offset in np.unique(rows.segment_offset[wanted]):
        of_offset = wanted[rows.segment_offset[wanted] == offset]
        error = (
            queries.start - int(offset) - satellite.code_phase
        ) % samples_per_code
        error = np.minimum(error, samples_per_code - error)  # the phase wraps
        at_phase = error <= code_phase_tolerance
        if classifier.doppler_is_in_the_query:
            bin_index = classifier.nearest_doppler_bin(satellite.doppler_hz)
            at_phase &= queries.doppler_bin == bin_index
        found = np.flatnonzero(at_phase)
        if found.size:
            groups.append((found, of_offset, queries.start[found] // samples_per_code))
    return groups


def screen_record(
    classifier: CamAcqClassifier,
    scenario: Scenario,
    code_phase_tolerance: int = DEFAULT_CODE_PHASE_TOLERANCE,
) -> tuple[list[tuple[float, np.ndarray]], tuple[float, float, int]]:
    """Reads D_true and D_wrong off one record's distance sweep.

    Args:
        classifier (CamAcqClassifier): The family being screened.
        scenario (Scenario): A labelled record.
        code_phase_tolerance (int): Samples of code phase error to accept.

    Returns:
        tuple: Per satellite (C/N0, per look D_true), and the sum, sum of
            squares and count of D_wrong.
    """
    config = classifier.config
    present = [s for s in scenario.truth if s.prn in config.prn_list]
    absent = np.flatnonzero(~np.isin(classifier.rows().prn, [s.prn for s in present]))

    targets = [
        (s, true_targets(classifier, s, code_phase_tolerance)) for s in present
    ]
    best: list[dict[int, int]] = [{} for _ in targets]
    total = squares = 0.0
    count = 0

    for low, block in classifier.distance_batches(scenario.samples):
        high = low + len(block)
        wrong = block[:, absent].astype(np.int64)
        total += float(wrong.sum())
        squares += float((wrong**2).sum())
        count += wrong.size
        for index, (_, groups) in enumerate(targets):
            for found, of_offset, looks in groups:
                inside = (found >= low) & (found < high)
                if not inside.any():
                    continue
                local = block[np.ix_(found[inside] - low, of_offset)].min(axis=1)
                for look, distance in zip(looks[inside], local):
                    look = int(look)
                    if distance < best[index].get(look, classifier.n_columns + 1):
                        best[index][look] = int(distance)

    per_satellite = [
        (satellite.cn0_dbhz, np.array(sorted(looks.values()), dtype=np.int32))
        for (satellite, _), looks in zip(targets, best)
    ]
    return per_satellite, (total, squares, count)


def screen(
    classifier: CamAcqClassifier,
    records: list[Scenario],
    family: str | None = None,
    progress: bool = True,
    noise_draws: int = 256,
    seed: int = 0,
    code_phase_tolerance: int = DEFAULT_CODE_PHASE_TOLERANCE,
) -> Separation:
    """Measures D_true and D_wrong for one family over a set of records.

    Args:
        classifier (CamAcqClassifier): The family to screen. Its threshold is
            never read, because the screen makes no decision.
        records (list[Scenario]): The records to measure on, normally the
            calibration half of a ScenarioBank.
        family (str | None): The name to print. None takes the class name.
        progress (bool): Show a progress bar.
        noise_draws (int): Query windows the pure noise floor is measured over.
        seed (int): Seed of that noise record.
        code_phase_tolerance (int): Samples of code phase error to accept when
            locating a satellite's own hypothesis.

    Returns:
        Separation: The two distributions and everything derived from them.
    """
    assert records, "The screen needs at least one record."
    distances: list[np.ndarray] = []
    cn0_values: list[np.ndarray] = []
    total = squares = 0.0
    count = 0

    for scenario in tqdm(
        records, disable=not progress, desc="screening", unit="record"
    ):
        per_satellite, wrong = screen_record(
            classifier, scenario, code_phase_tolerance
        )
        for cn0_dbhz, looks in per_satellite:
            if looks.size:
                distances.append(looks)
                cn0_values.append(np.full(looks.size, cn0_dbhz))
        total += wrong[0]
        squares += wrong[1]
        count += wrong[2]

    mean = total / count if count else 0.0
    variance = squares / count - mean**2 if count else 0.0
    noise_mean, noise_sd = classifier.chance_floor(n_draws=noise_draws, seed=seed)
    return Separation(
        family=family or type(classifier).__name__,
        n_rows=classifier.n_rows,
        n_columns=classifier.n_columns,
        n_records=len(records),
        true_distance=(
            np.concatenate(distances) if distances else np.array([], dtype=np.int32)
        ),
        true_cn0_dbhz=(
            np.concatenate(cn0_values) if cn0_values else np.array([], dtype=float)
        ),
        wrong_mean=float(mean),
        wrong_sd=float(np.sqrt(max(variance, 0.0))),
        n_wrong=int(count),
        noise_mean=noise_mean,
        noise_sd=noise_sd,
    )


def kill_verdict(
    separation: Separation,
    baseline: Separation | None = None,
    cn0_dbhz: float = KILL_BIN_DBHZ,
    **kwargs,
) -> str:
    """Applies test (i) of the study's kill rule to one family.

    A family is dead when d' in the 45 dB-Hz bin is below 1.0, or below half the
    baseline's. The rule was fixed before the runs so it cannot be argued with
    afterwards.

    Args:
        separation (Separation): The family's screen.
        baseline (Separation | None): The baseline's screen, for the relative
            half of the test. None applies the absolute half only.
        cn0_dbhz (float): The bin to read d' in.
        **kwargs: Passed to bins.

    Returns:
        str: One line naming the verdict and the number behind it.
    """
    entry = separation.at(cn0_dbhz, **kwargs)
    if entry is None:
        return f"{separation.family}: no {cn0_dbhz:.0f} dB-Hz bin to judge on"
    if entry.d_prime < 1.0:
        return (
            f"{separation.family}: dead, d' = {entry.d_prime:.2f} at "
            f"{cn0_dbhz:.0f} dB-Hz, below 1.0"
        )
    if baseline is not None:
        reference = baseline.at(cn0_dbhz, **kwargs)
        if reference is not None and entry.d_prime < reference.d_prime / 2:
            return (
                f"{separation.family}: dead, d' = {entry.d_prime:.2f} against the "
                f"baseline's {reference.d_prime:.2f} at {cn0_dbhz:.0f} dB-Hz"
            )
    return (
        f"{separation.family}: survives, d' = {entry.d_prime:.2f} at "
        f"{cn0_dbhz:.0f} dB-Hz, tolerance {entry.tolerance_fraction:.1%}"
    )
