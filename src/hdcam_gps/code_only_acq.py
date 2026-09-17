"""Code-only rows: the CFO moves from the codebook into the query.

The baseline spends a row on every (PRN, CFO, stored phase), so a 32 PRN search
over +-5 kHz needs 1344 rows of 2046 bits. Only one dimension of that is really
in the codebook: a Doppler hypothesis is a carrier the receiver can wipe off the
query instead of storing. Doing so leaves 32 PRNs times 2 stored phases, 64
rows, and 21 times less CAM. This is the reduced-row design PROPOSAL step 4
names.

It is not free, and the trade is the point of measuring it. The search now asks
one question per (start, CFO) rather than per start, so it issues 21 times as
many searches against 21 times fewer rows: the same bit comparisons, the same
energy, 21 times the latency. Nothing else in the study moves area without
moving energy.

**The mixer is where the 1 bit front end is won or lost.**

* mixer="quadrant" rotates the quantized stream by a 2 bit phase ramp, one
  mod-4 add per sample. The samples stay 1 bit wide, so this is a design a 1 bit
  front end can drive. Rounding each rotation to a quarter turn leaves up to 45
  degrees of phase error per sample, which is the price.
* mixer="exact" multiplies the full precision samples by an NCO before
  quantizing. That is no longer a 1 bit front end, and it is reported as an
  upper bound only. Any headline number for this family is the quadrant one.

The quadrant stream is what makes the cheap mixer possible. A 1 bit (I, Q) pair
is a phase rounded to a quadrant, so it is a 2 bit number, and multiplying by j
is adding one to it. A carrier of f Hz is then a ramp of 4*f*t quarter turns,
rounded, subtracted mod 4 - no multiplier anywhere.
"""

import numpy as np

from hdcam_gps.acq_base import AcqConfig
from hdcam_gps.ca_code import sampled_ca_code
from hdcam_gps.cam_acq import (
    QUERY_ROTATIONS,
    CamAcqClassifier,
    QueryIndex,
    RowIndex,
    quantize_iq,
    rotate_quarter_turns,
)
from hdcam_gps.hdcam_packed import PackedHdCam

DEFAULT_CODEBOOK_PHASES: int = 2  # Stored phases, as the baseline carries
MIXERS: tuple[str, ...] = ("quadrant", "exact")


def quadrants(bits: np.ndarray) -> np.ndarray:
    """The 2 bit phase of every sample of a quantized query.

    Args:
        bits (np.ndarray): A quantized query, as produced by quantize_iq.

    Returns:
        np.ndarray: One value in 0..3 per sample, counting quarter turns
            anticlockwise from the first quadrant.
    """
    in_phase, quadrature = np.split(np.asarray(bits, dtype=bool), 2)
    return ((in_phase ^ quadrature) + 2 * ~quadrature).astype(np.int8)


def bits_from_quadrants(quadrant: np.ndarray) -> np.ndarray:
    """The quantized query a run of 2 bit phases stands for.

    Args:
        quadrant (np.ndarray): One value in 0..3 per sample.

    Returns:
        np.ndarray: A boolean array of twice the length, the inverse of
            quadrants.
    """
    quadrature = quadrant < 2
    in_phase = (quadrant + 1) % 4 < 2
    return np.concatenate((in_phase, quadrature))


def rotate_quadrants(bits: np.ndarray, turns: np.ndarray) -> np.ndarray:
    """Turns every sample of a query by its own number of quarter turns.

    rotate_quarter_turns is this with one turn count for the whole word. A
    carrier is the case where the count is a ramp, which is what makes wiping a
    CFO off a 1 bit stream an addition rather than a multiplication.

    Args:
        bits (np.ndarray): A quantized query, as produced by quantize_iq.
        turns (np.ndarray): Quarter turns to add, one per sample.

    Returns:
        np.ndarray: The rotated query, of the same width.
    """
    return bits_from_quadrants((quadrants(bits) + turns) % QUERY_ROTATIONS)


class CodeOnlyHdCamClassifier(CamAcqClassifier):
    """Acquisition against zero-Doppler rows, with the CFO wiped off the query."""

    def __init__(
        self,
        config: AcqConfig,
        hd_threshold: int | None = None,
        n_codebook_phases: int = DEFAULT_CODEBOOK_PHASES,
        min_votes: int | None = None,
        mixer: str = "quadrant",
        false_alarm_rate: float = 1e-2,
        search_mode: str = "cam",
        cam_factory=PackedHdCam,
    ):
        """Builds the zero-Doppler codebook and writes it into a fresh HdCam.

        Args:
            config (AcqConfig): The acquisition configuration. Only its PRN list
                is a codebook dimension here; the Doppler grid sizes the queries.
            hd_threshold (int | None): The per look Hamming distance threshold.
                None derives it from the vote rule and false_alarm_rate.
            n_codebook_phases (int): Carrier phase offsets stored per PRN, on top
                of the four quarter turns of the query.
            min_votes (int | None): How many of its looks a hypothesis has to
                match on. None takes a third of them.
            mixer (str): "quadrant" to wipe the CFO off the 1 bit stream,
                "exact" to wipe it off the full precision samples. Only the
                first is a 1 bit front end.
            false_alarm_rate (float): False detections tolerated per acquisition,
                used only when hd_threshold is None.
            search_mode (str): "cam" or "table", as CamAcqClassifier defines.
            cam_factory: The HdCam class to build.
        """
        assert mixer in MIXERS, f"mixer is one of {MIXERS}, not {mixer!r}."
        assert n_codebook_phases >= 1, "At least one codebook phase is needed."
        assert config.n_codes >= 2, (
            "At least two code periods are needed, so that a full contiguous "
            "window is available at every code phase."
        )
        self.mixer = mixer
        self.n_codebook_phases = n_codebook_phases
        self.n_columns = 2 * config.samples_per_code
        self._ramps = self._build_ramps(config)
        self._window_cache: tuple[np.ndarray, int, np.ndarray] | None = None
        super().__init__(
            config,
            hd_threshold=hd_threshold,
            min_votes=min_votes,
            false_alarm_rate=false_alarm_rate,
            search_mode=search_mode,
            cam_factory=cam_factory,
        )

    @staticmethod
    def _build_ramps(config: AcqConfig) -> np.ndarray:
        """The quarter turns a code period of each CFO hypothesis sweeps through.

        Args:
            config (AcqConfig): Supplies the Doppler grid and the sampling rate.

        Returns:
            np.ndarray: Shape (n_doppler_bins, samples_per_code), in quarter
                turns and not yet rounded, since the start offset shifts them.
        """
        time_s = np.arange(config.samples_per_code) / config.fs_hz
        return -QUERY_ROTATIONS * np.outer(config.doppler_grid_hz, time_s)

    @property
    def n_phases(self) -> int:
        """Returns the carrier phase hypotheses covering the circle."""
        return QUERY_ROTATIONS * self.n_codebook_phases

    @property
    def n_rows(self) -> int:
        """Returns the number of codebook rows, one per PRN and stored phase."""
        return len(self.config.prn_list) * self.n_codebook_phases

    def row_of(self, prn: int, codebook_phase: int = 0) -> int:
        """Returns the codebook row of one hypothesis.

        Args:
            prn (int): A PRN number from the configuration's PRN list.
            codebook_phase (int): Which stored carrier phase offset.

        Returns:
            int: The row index in the CAM.
        """
        return self.config.prn_list.index(prn) * self.n_codebook_phases + (
            codebook_phase
        )

    def build_codebook(self) -> np.ndarray:
        """Builds the quantized zero-Doppler replica of every PRN.

        The stored phases subdivide a quadrant, exactly as the baseline's do, so
        that no replica lands on an axis with an empty quadrature plane.

        Returns:
            np.ndarray: A boolean array of shape (n_rows, 2 * samples_per_code).
        """
        codebook = np.empty((self.n_rows, self.n_columns), dtype=bool)
        for prn in self.config.prn_list:
            code = sampled_ca_code(prn, self.config.fs_hz)
            for phase in range(self.n_codebook_phases):
                offset = np.exp(2j * np.pi * (phase + 0.5) / self.n_phases)
                codebook[self.row_of(prn, phase)] = quantize_iq(code * offset)
        return codebook

    def row_index(self) -> RowIndex:
        """Returns the PRN of every row, and no Doppler bin.

        Returns:
            RowIndex: Parallel arrays of n_rows entries, with doppler_bin at -1
                because the query carries it.
        """
        positions = np.arange(self.n_rows) // self.n_codebook_phases
        return RowIndex(
            prn=np.array(self.config.prn_list)[positions],
            doppler_bin=np.full(self.n_rows, -1, dtype=int),
            segment_offset=np.zeros(self.n_rows, dtype=int),
        )

    def query_index(self, n_samples: int | None = None) -> QueryIndex:
        """Returns one query per window start and CFO hypothesis.

        The bins of one start are kept together, so a batch of queries reuses
        the same quantized window.

        Args:
            n_samples (int | None): Length of the record. None takes the
                configuration's own acquisition length.

        Returns:
            QueryIndex: n_starts * n_doppler_bins entries.
        """
        if n_samples is None:
            n_samples = self.config.samples_per_acquisition
        starts = np.arange(n_samples - self.config.samples_per_code + 1)
        bins = np.arange(self.n_doppler_bins)
        return QueryIndex(
            start=np.repeat(starts, len(bins)),
            doppler_bin=np.tile(bins, len(starts)),
        )

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

    def _quantized_window(self, samples: np.ndarray, start: int) -> np.ndarray:
        """The quantized window of a start, kept for the other CFO bins of it.

        Args:
            samples (np.ndarray): The full input record.
            start (int): The first sample of the window.

        Returns:
            np.ndarray: The quantized window, n_columns bits wide.
        """
        cached = self._window_cache
        if cached is not None and cached[0] is samples and cached[1] == start:
            return cached[2]
        bits = quantize_iq(self.query_window(samples, start))
        # The record itself, not its id: a freed array's id gets reused.
        self._window_cache = (samples, start, bits)
        return bits

    def wipe_doppler(
        self, samples: np.ndarray, start: int, doppler_bin: int
    ) -> np.ndarray:
        """Removes one CFO hypothesis from a window, by whichever mixer is set.

        Args:
            samples (np.ndarray): The full input record.
            start (int): The first sample of the window.
            doppler_bin (int): An index into the configuration's Doppler grid.

        Returns:
            np.ndarray: The quantized, Doppler corrected query.
        """
        if self.mixer == "exact":
            doppler_hz = float(self.config.doppler_grid_hz[doppler_bin])
            window = self.query_window(samples, start)
            time_s = (start + np.arange(len(window))) / self.config.fs_hz
            return quantize_iq(window * np.exp(-2j * np.pi * doppler_hz * time_s))

        # The whole-turn part of the start offset is a constant rotation, which
        # the four query rotations already cover, so only the ramp is rounded.
        doppler_hz = float(self.config.doppler_grid_hz[doppler_bin])
        offset = -QUERY_ROTATIONS * doppler_hz * start / self.config.fs_hz
        turns = np.rint(self._ramps[doppler_bin] + offset).astype(np.int8)
        return rotate_quadrants(self._quantized_window(samples, start), turns)

    def query_variants(self, samples: np.ndarray, query: int) -> list[np.ndarray]:
        """The four quarter turns of one window at one CFO hypothesis.

        Args:
            samples (np.ndarray): The full input record.
            query (int): An index into the query index.

        Returns:
            list[np.ndarray]: Four boolean arrays, each n_columns wide.
        """
        index = self.queries(len(samples))
        bits = self.wipe_doppler(
            samples, int(index.start[query]), int(index.doppler_bin[query])
        )
        return [
            rotate_quarter_turns(bits, rotation) for rotation in range(QUERY_ROTATIONS)
        ]
