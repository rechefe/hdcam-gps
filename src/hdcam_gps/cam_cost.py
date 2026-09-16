"""What one acquisition costs the CAM, counted rather than modelled.

The comparison the study exists for is sensitivity against CAM cost, and the two
numbers that decide whether a family is buildable are not sensitivity at all:

* tolerance_fraction, the calibrated threshold as a fraction of the row width.
  The JSSC 2025 silicon demonstrates 8 bits in a 64-bit word, 12.5 %. The
  baseline design asks for 44-47 %. PROPOSAL risk 1 is that no CAM does that.
* resolution_fraction, the gap between the chance floor and the threshold, in the
  same units. It is what the family has left to discriminate with.

Everything else follows from counting. n_searches and n_threshold_writes come
from running a real acquisition through a CountingHdCam, so the bisections
tightest_match issues are included - about eleven per surviving look, which a
closed form of the first pass alone would miss (PROPOSAL risk 5).

The chance floor is measured, not assumed. A 1 bit codebook really does sit at
n_columns / 2, but a thermometer codebook does not: its unary words make the bits
of one sample dependent, so the binomial with p = 0.5 is the wrong distribution
and its standard deviation is the wrong scale to quote a threshold in.

Energy and latency are the JSSC figures applied to the counted searches, so they
are that paper's silicon running this codebook, not a circuit model of our own.
"""

from dataclasses import dataclass

import numpy as np

from hdcam_gps.cam_acq import CamAcqClassifier
from hdcam_gps.hdcam_packed import PackedHdCam

ENERGY_PER_BIT_FJ: float = 0.19  # JSSC 60(8):3009, 65 nm, per bit per search
LATENCY_PER_SEARCH_NS: float = 8.0  # The same macro at 125 MHz


@dataclass(frozen=True)
class CamCost:
    """The area, energy and latency of one acquisition on one family."""

    n_rows: int  # Codebook rows
    n_columns: int  # Bits per row
    n_searches: int  # search_cam calls, counted over a whole acquisition
    n_threshold_writes: int  # set_hd_threshold calls, counted the same way
    n_first_pass_searches: int  # Of n_searches, the ones the shortlist issued
    hits_mean: float  # Rows returned per first pass search
    hits_max: int  # The worst one, which is what a readout has to survive
    hd_threshold: int  # The threshold the acquisition ran at
    chance_mean: float  # Measured distance of an unrelated row
    chance_sd: float  # Its measured standard deviation

    @property
    def total_bits(self) -> int:
        """Bits of CAM the codebook occupies."""
        return self.n_rows * self.n_columns

    @property
    def bit_comparisons(self) -> int:
        """Bit comparisons one acquisition makes, the energy proxy."""
        return self.n_searches * self.total_bits

    @property
    def energy_uj(self) -> float:
        """Search energy of one acquisition in microjoules."""
        return self.bit_comparisons * ENERGY_PER_BIT_FJ * 1e-15 * 1e6

    @property
    def latency_us(self) -> float:
        """Search latency of one acquisition in microseconds, searches in series."""
        return self.n_searches * LATENCY_PER_SEARCH_NS * 1e-3

    @property
    def tolerance_fraction(self) -> float:
        """The threshold as a fraction of the row width.

        This is the number PROPOSAL risk 1 turns on. Silicon demonstrates 0.125.
        """
        return self.hd_threshold / self.n_columns

    @property
    def resolution_fraction(self) -> float:
        """The gap from the chance floor down to the threshold, in row widths."""
        return (self.chance_mean - self.hd_threshold) / self.n_columns

    @property
    def resolution_sigma(self) -> float:
        """The same gap, in standard deviations of the measured chance floor."""
        if self.chance_sd == 0:
            return float("inf")
        return (self.chance_mean - self.hd_threshold) / self.chance_sd

    def table(self) -> str:
        """Renders the cost as a fixed width block.

        Returns:
            str: One line per quantity.
        """
        lines = [
            f"{'rows x columns':<22} {self.n_rows} x {self.n_columns}",
            f"{'total bits':<22} {self.total_bits:,}",
            f"{'searches':<22} {self.n_searches:,} "
            f"({self.n_first_pass_searches:,} in the first pass)",
            f"{'threshold writes':<22} {self.n_threshold_writes:,}",
            f"{'bit comparisons':<22} {self.bit_comparisons:.3e}",
            f"{'energy':<22} {self.energy_uj:.1f} uJ",
            f"{'latency':<22} {self.latency_us:.1f} us",
            f"{'hits per search':<22} {self.hits_mean:.2f} mean, {self.hits_max} max",
            f"{'chance floor':<22} {self.chance_mean:.1f} +- {self.chance_sd:.1f}",
            f"{'tolerance fraction':<22} {self.tolerance_fraction:.3f}",
            f"{'resolution fraction':<22} {self.resolution_fraction:.3f} "
            f"({self.resolution_sigma:.1f} sigma)",
        ]
        return "\n".join(lines)


class CountingHdCam(PackedHdCam):
    """A PackedHdCam that records what was asked of it.

    Every counter is public and additive, so a caller can snapshot one between
    the two passes of an acquisition.
    """

    def __init__(self, n_rows: int, n_columns: int, hd_threshold: int):
        """Builds the CAM with every counter at zero.

        Args:
            n_rows (int): Amount of rows in the grid.
            n_columns (int): Amount of columns in the grid.
            hd_threshold (int): The hamming distance threshold for the HdCam.
        """
        super().__init__(n_rows, n_columns, hd_threshold)
        self.n_searches = 0
        self.n_threshold_writes = 0
        self.hits: list[int] = []

    def search_cam(self, query):
        """Answers the search and counts it, with how many rows it returned.

        Args:
            query (list[int]): The query to search for.

        Returns:
            np.ndarray: The indices of the matching rows.
        """
        matched = super().search_cam(query)
        self.n_searches += 1
        self.hits.append(len(matched))
        return matched

    def set_hd_threshold(self, hd_threshold: int):
        """Retunes the CAM and counts the write.

        Args:
            hd_threshold (int): The new Hamming distance threshold.
        """
        super().set_hd_threshold(hd_threshold)
        self.n_threshold_writes += 1


def noise_record(classifier: CamAcqClassifier, seed: int = 0) -> np.ndarray:
    """A unit power noise record of the length the classifier acquires.

    Args:
        classifier (CamAcqClassifier): Supplies the record length.
        seed (int): Seed of the noise.

    Returns:
        np.ndarray: Complex samples, samples_per_acquisition long.
    """
    rng = np.random.default_rng(seed)
    n_samples = classifier.config.samples_per_acquisition
    return rng.normal(scale=np.sqrt(0.5), size=n_samples) + 1j * rng.normal(
        scale=np.sqrt(0.5), size=n_samples
    )


def measure_cost(
    classifier: CamAcqClassifier,
    samples: np.ndarray | None = None,
    seed: int = 0,
    n_draws: int = 256,
) -> CamCost:
    """Runs one acquisition through a counting CAM and reports what it cost.

    The classifier's own CAM is swapped out and put back, so the measurement
    leaves it exactly as it was found. The two passes are composed here from the
    same methods _acquire uses, so the counts describe the real decision rather
    than a copy of it.

    Args:
        classifier (CamAcqClassifier): The family to measure.
        samples (np.ndarray | None): The record to acquire. None uses noise,
            which shortlists nothing and so gives the first pass on its own.
        seed (int): Seed of that noise record, and of the chance floor draw.
        n_draws (int): Query windows the chance floor is measured over.

    Returns:
        CamCost: The counted cost of one acquisition.
    """
    if samples is None:
        samples = noise_record(classifier, seed)
    original = classifier.cam
    counting = CountingHdCam(
        original.n_rows, original.n_columns, original.hd_threshold
    )
    counting.write_array(original.grid)
    classifier.cam = counting
    try:
        cells = classifier.shortlist_cells(
            classifier.matched_table(samples), classifier.min_votes
        )
        first_pass = counting.n_searches
        hits = np.array(counting.hits[:first_pass])
        classifier.rank_shortlist(samples, cells)
    finally:
        classifier.cam = original
    chance_mean, chance_sd = classifier.chance_floor(n_draws=n_draws, seed=seed)
    return CamCost(
        n_rows=classifier.n_rows,
        n_columns=classifier.n_columns,
        n_searches=counting.n_searches,
        n_threshold_writes=counting.n_threshold_writes,
        n_first_pass_searches=first_pass,
        hits_mean=float(hits.mean()) if hits.size else 0.0,
        hits_max=int(hits.max()) if hits.size else 0,
        hd_threshold=original.hd_threshold,
        chance_mean=chance_mean,
        chance_sd=chance_sd,
    )
