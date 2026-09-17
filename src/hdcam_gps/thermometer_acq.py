"""Thermometer rows: a Hamming distance that measures amplitude, not just sign.

A 1 bit front end throws away 1.96 dB, and about 1.4 dB of that comes back at 2
bits. The obstacle is that a CAM compares words by Hamming distance, which knows
nothing about the value a multi bit number stands for - 01 and 10 are two bits
apart while standing one level apart. Unary coding fixes exactly that. Writing
level L as L ones followed by zeros makes the Hamming distance between two
levels their difference, so the distance between two words is the L1 distance
between the sample runs they encode, and a CAM computes a soft correlation
without changing.

The price is area: four levels need three bits per component, so a 2046 bit row
becomes 6138 and the codebook grows from 2.75 to 8.25 Mbit. Three times the area
and three times the energy for 1.4 dB.

**The gain is not guaranteed and the study exists to find out.** The stored row
is a noiseless replica whose amplitude is constant, while the query at 40 dB-Hz
is noise with a bias in it. If the extra bits only describe the noise, the
magnitude term adds a constant to every distance and discriminates nothing. The
screen measures that rather than assuming it.

Two things follow from the unary code and are worth stating before reading the
quantiser. Negating a level is complementing and reversing its word, so a
quarter turn of the carrier is still a bit operation on the query. And the
chance floor is no longer binomial with p = 0.5: the bits of one sample are not
independent, so cam_cost and calibrate both measure that floor instead of
computing it.

The gain of the quantiser comes from the record, never from Scenario.noise_sigma
- a receiver has an AGC, not the generator's parameters.
"""

from statistics import NormalDist

import numpy as np

from hdcam_gps.acq_base import AcqConfig
from hdcam_gps.ca_code import sampled_ca_code
from hdcam_gps.cam_acq import (
    QUERY_ROTATIONS,
    CamAcqClassifier,
    QueryIndex,
    RowIndex,
    per_look_false_alarm,
)
from hdcam_gps.hdcam_packed import PackedHdCam

DEFAULT_LEVELS: int = 4  # Per component, so three bits of unary
DEFAULT_CODEBOOK_PHASES: int = 2  # Stored phases, as the baseline carries
# The optimal uniform step for a Gaussian at two bits, in standard deviations.
DEFAULT_LEVEL_STEP: float = 0.9957


def level_boundaries(n_levels: int, step: float) -> np.ndarray:
    """The thresholds a uniform quantiser of n_levels puts on one component.

    Args:
        n_levels (int): How many levels to quantise to.
        step (float): The spacing between them, in the same units as the input.

    Returns:
        np.ndarray: n_levels - 1 thresholds, symmetric about zero.
    """
    return (np.arange(1, n_levels) - n_levels / 2) * step


def thermometer(levels: np.ndarray, n_levels: int) -> np.ndarray:
    """Unary codes a run of levels, so Hamming distance is level difference.

    Args:
        levels (np.ndarray): Values in 0..n_levels-1.
        n_levels (int): How many levels the code spans.

    Returns:
        np.ndarray: A boolean array of (n_levels - 1) bits per level, level
            major.
    """
    weights = np.arange(n_levels - 1)
    return (levels[:, None] > weights[None, :]).reshape(-1)


def levels_of(bits: np.ndarray, n_levels: int) -> np.ndarray:
    """Reads the levels back out of a unary coded run.

    Args:
        bits (np.ndarray): A unary coded run, as thermometer produces.
        n_levels (int): How many levels the code spans.

    Returns:
        np.ndarray: One level per word.
    """
    return np.asarray(bits, dtype=bool).reshape(-1, n_levels - 1).sum(axis=1)


def negate_thermometer(bits: np.ndarray, n_levels: int) -> np.ndarray:
    """Turns every level into its reflection about the middle of the scale.

    Level L becomes n_levels - 1 - L, which on a unary word is the complement
    reversed. A quarter turn of the carrier needs this and nothing else, so the
    query side stays a bit operation.

    Args:
        bits (np.ndarray): A unary coded run.
        n_levels (int): How many levels the code spans.

    Returns:
        np.ndarray: The reflected run, of the same width.
    """
    words = np.asarray(bits, dtype=bool).reshape(-1, n_levels - 1)
    return (~words[:, ::-1]).reshape(-1)


def rotate_thermometer(bits: np.ndarray, rotations: int, n_levels: int) -> np.ndarray:
    """Turns a thermometer query by whole quadrants, using bit operations only.

    Multiplying by j maps (I, Q) to (-Q, I), which here is a swap of the two
    blocks with the new leading one reflected.

    Args:
        bits (np.ndarray): A thermometer query, the I block then the Q block.
        rotations (int): How many quarter turns to apply.
        n_levels (int): How many levels the code spans.

    Returns:
        np.ndarray: The rotated query, of the same width.
    """
    for _ in range(rotations % QUERY_ROTATIONS):
        in_phase, quadrature = np.split(bits, 2)
        bits = np.concatenate((negate_thermometer(quadrature, n_levels), in_phase))
    return bits


def quantize_thermometer(
    samples: np.ndarray, n_levels: int, scale: float, step: float = DEFAULT_LEVEL_STEP
) -> np.ndarray:
    """Quantizes complex samples to unary coded levels, I block then Q block.

    Args:
        samples (np.ndarray): The complex samples to quantize.
        n_levels (int): How many levels per component.
        scale (float): The per component standard deviation the quantiser is
            sized against, which is what an AGC supplies.
        step (float): Level spacing in units of scale.

    Returns:
        np.ndarray: A boolean array of 2 * (n_levels - 1) bits per sample.
    """
    boundaries = level_boundaries(n_levels, step * max(scale, 1e-12))
    in_phase = np.searchsorted(boundaries, np.real(samples))
    quadrature = np.searchsorted(boundaries, np.imag(samples))
    return np.concatenate(
        (thermometer(in_phase, n_levels), thermometer(quadrature, n_levels))
    )


def component_scale(samples: np.ndarray) -> float:
    """The per component standard deviation of a record, which is the AGC.

    Args:
        samples (np.ndarray): Complex samples.

    Returns:
        float: The root mean square of one component.
    """
    return float(np.sqrt(np.mean(np.abs(samples) ** 2) / 2))


class ThermometerHdCamClassifier(CamAcqClassifier):
    """Acquisition by a unary coded multi level codebook lookup in an HdCam."""

    def __init__(
        self,
        config: AcqConfig,
        hd_threshold: int | None = None,
        n_levels: int = DEFAULT_LEVELS,
        n_codebook_phases: int = DEFAULT_CODEBOOK_PHASES,
        min_votes: int | None = None,
        level_step: float = DEFAULT_LEVEL_STEP,
        false_alarm_rate: float = 1e-2,
        search_mode: str = "cam",
        cam_factory=PackedHdCam,
    ):
        """Builds the thermometer codebook and writes it into a fresh HdCam.

        Args:
            config (AcqConfig): The acquisition configuration. Its PRN list and
                Doppler grid are the two hypothesis dimensions of the codebook.
            hd_threshold (int | None): The per look Hamming distance threshold.
                None derives it from the vote rule and false_alarm_rate.
            n_levels (int): Levels per component. Two is the 1 bit baseline in
                this coding, and four is the design the study measures.
            n_codebook_phases (int): Carrier phase offsets stored per (PRN, CFO).
            min_votes (int | None): How many of its looks a hypothesis has to
                match on. None takes a third of them.
            level_step (float): Quantiser spacing in standard deviations.
            false_alarm_rate (float): False detections tolerated per acquisition,
                used only when hd_threshold is None.
            search_mode (str): "cam" or "table", as CamAcqClassifier defines.
            cam_factory: The HdCam class to build.
        """
        assert n_levels >= 2, "A thermometer code needs at least two levels."
        assert n_codebook_phases >= 1, "At least one codebook phase is needed."
        assert config.n_codes >= 2, (
            "At least two code periods are needed, so that a full contiguous "
            "window is available at every code phase."
        )
        self.n_levels = n_levels
        self.level_step = level_step
        self.n_codebook_phases = n_codebook_phases
        self.n_columns = 2 * (n_levels - 1) * config.samples_per_code
        self._agc_cache: tuple[np.ndarray, float] | None = None
        super().__init__(
            config,
            hd_threshold=hd_threshold,
            min_votes=min_votes,
            false_alarm_rate=false_alarm_rate,
            search_mode=search_mode,
            cam_factory=cam_factory,
        )

    def default_hd_threshold(self, false_alarm_rate: float) -> int:
        """The per look threshold, from the floor this family actually has.

        The 1 bit answer would put it above the floor rather than below: the
        bits of one unary word are not independent, so an unrelated row sits
        near 39 % of the width rather than the 50 % a binomial with p = 0.5
        predicts, and a threshold derived from that model lets every PRN
        through. The floor is measured instead.

        Args:
            false_alarm_rate (float): False detections tolerated per acquisition.

        Returns:
            int: The Hamming distance threshold, within [0, n_columns).
        """
        per_look = per_look_false_alarm(
            self.n_looks, self.min_votes, self.n_cells, false_alarm_rate
        )
        mean, deviation = self.chance_floor()
        reach = NormalDist().inv_cdf(per_look)  # negative, the floor's lower tail
        return int(np.clip(round(mean + reach * deviation), 0, self.n_columns - 1))

    @property
    def n_phases(self) -> int:
        """Returns the carrier phase hypotheses covering the circle."""
        return QUERY_ROTATIONS * self.n_codebook_phases

    @property
    def n_rows(self) -> int:
        """Returns the number of codebook rows, one per PRN, CFO and phase."""
        return len(self.config.prn_list) * self.n_doppler_bins * self.n_codebook_phases

    def row_of(self, prn: int, doppler_bin: int, codebook_phase: int = 0) -> int:
        """Returns the codebook row of one hypothesis.

        Args:
            prn (int): A PRN number from the configuration's PRN list.
            doppler_bin (int): An index into the configuration's Doppler grid.
            codebook_phase (int): Which stored carrier phase offset.

        Returns:
            int: The row index in the CAM.
        """
        prn_position = self.config.prn_list.index(prn)
        hypothesis = prn_position * self.n_doppler_bins + doppler_bin
        return hypothesis * self.n_codebook_phases + codebook_phase

    def build_codebook(self) -> np.ndarray:
        """Builds the unary coded replica of every stored hypothesis.

        The replica is quantised against its own component scale, the same rule
        the query is quantised by, so a distance between them is an L1 distance
        between two runs measured on one ruler.

        Returns:
            np.ndarray: A boolean array of shape (n_rows, n_columns).
        """
        config = self.config
        time_s = np.arange(config.samples_per_code) / config.fs_hz
        codebook = np.empty((self.n_rows, self.n_columns), dtype=bool)
        for prn in config.prn_list:
            code = sampled_ca_code(prn, config.fs_hz)
            for doppler_bin, doppler_hz in enumerate(config.doppler_grid_hz):
                carrier = np.exp(2j * np.pi * doppler_hz * time_s)
                for phase in range(self.n_codebook_phases):
                    offset = np.exp(2j * np.pi * (phase + 0.5) / self.n_phases)
                    replica = code * carrier * offset
                    codebook[self.row_of(prn, doppler_bin, phase)] = (
                        quantize_thermometer(
                            replica,
                            self.n_levels,
                            component_scale(replica),
                            self.level_step,
                        )
                    )
        return codebook

    def row_index(self) -> RowIndex:
        """Returns the PRN and Doppler bin of every row.

        Returns:
            RowIndex: Parallel arrays of n_rows entries.
        """
        hypotheses = np.arange(self.n_rows) // self.n_codebook_phases
        positions = hypotheses // self.n_doppler_bins
        return RowIndex(
            prn=np.array(self.config.prn_list)[positions],
            doppler_bin=hypotheses % self.n_doppler_bins,
            segment_offset=np.zeros(self.n_rows, dtype=int),
        )

    def query_index(self, n_samples: int | None = None) -> QueryIndex:
        """Returns the start of every window the record is searched at.

        Args:
            n_samples (int | None): Length of the record. None takes the
                configuration's own acquisition length.

        Returns:
            QueryIndex: One entry per contiguous window that fits.
        """
        if n_samples is None:
            n_samples = self.config.samples_per_acquisition
        n_starts = n_samples - self.config.samples_per_code + 1
        return QueryIndex(
            start=np.arange(n_starts),
            doppler_bin=np.full(n_starts, -1, dtype=int),
        )

    def gain(self, samples: np.ndarray) -> float:
        """The quantiser scale, read off the record the way an AGC would.

        Measured over the whole record rather than per window, because a window
        is one millisecond and an AGC is slower than that. Scenario.noise_sigma
        is never consulted: it is the generator's parameter, not a receiver's.

        Args:
            samples (np.ndarray): The full input record.

        Returns:
            float: The per component standard deviation to quantise against.
        """
        cached = self._agc_cache
        if cached is not None and cached[0] is samples:
            return cached[1]
        scale = component_scale(samples)
        # The record itself, not its id: a freed array's id gets reused, and the
        # chance floor's noise record is freed right before a real one arrives.
        self._agc_cache = (samples, scale)
        return scale

    def query_window(self, samples: np.ndarray, start: int) -> np.ndarray:
        """Takes one contiguous code period of samples out of the record.

        Args:
            samples (np.ndarray): The full input record.
            start (int): The first sample of the window.

        Returns:
            np.ndarray: A view of samples_per_code complex samples.
        """
        end = start + self.config.samples_per_code
        assert end <= len(samples), "The query window must fit inside the record."
        return samples[start:end]

    def query_variants(self, samples: np.ndarray, query: int) -> list[np.ndarray]:
        """The four quarter turns of one window, quantised once.

        Args:
            samples (np.ndarray): The full input record.
            query (int): The window start, which is also the query index.

        Returns:
            list[np.ndarray]: Four boolean arrays, each n_columns wide.
        """
        bits = quantize_thermometer(
            self.query_window(samples, query),
            self.n_levels,
            self.gain(samples),
            self.level_step,
        )
        return [
            rotate_thermometer(bits, rotation, self.n_levels)
            for rotation in range(QUERY_ROTATIONS)
        ]
