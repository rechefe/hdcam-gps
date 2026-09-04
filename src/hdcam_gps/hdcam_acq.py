"""1 bit HdCam based GPS L1 C/A acquisition.

The codebook written into the HdCam holds the replicas the acquisition is
searching for. A row is one C/A code period of a (PRN, CFO) hypothesis,
quantized to one bit per component: the sign of I followed by the sign of Q,
which is the carrier phase of every sample rounded to a quadrant.

The two dimensions the codebook cannot span on its own are handled around the
search:

* Code phase, by sliding the query window over the record. The row index of a
  hit gives the PRN and the CFO, the window offset gives the code phase. The
  windows are contiguous, never wrapped: only the code repeats every period, the
  Doppler carrier does not, so the record must hold at least two code periods
  for every code phase to be reachable.

* Carrier phase, which a Hamming distance cannot ignore. A codeword and the same
  codeword received in antiphase sit at opposite ends of the distance scale, and
  the CAM only fires on small distances, so the circle has to be covered
  explicitly. This is split across the two sides of the comparison:

    - The query contributes the four quarter turns, which are all a one bit
      sample can express. They are bit operations, not arithmetic: a quarter
      turn maps (I, Q) to (-Q, I), so it is a swap of the two bit planes with
      one of them complemented.
    - The codebook contributes the phases in between, because it is built
      offline in full precision and can be quantized at any starting phase.

  Four query rotations alone are not enough. Their worst case residual phase of
  45 degrees costs n_columns / 4 in Hamming distance, which is the same order as
  a wrong CFO bin of the same PRN, so the two become indistinguishable. Two
  codebook phases, for eight effective hypotheses, restore the separation.
"""

from statistics import NormalDist

import numpy as np

from hdcam_gps.acq_base import AcqConfig, GpsL1AcqClassifier, PrnResult
from hdcam_gps.ca_code import sampled_ca_code
from hdcam_gps.hdcam import HdCam

QUERY_ROTATIONS: int = 4  # Quarter turns, all a one bit sample can express
DEFAULT_CODEBOOK_PHASES: int = 2  # Extra phases carried by the codebook
DEFAULT_FALSE_ALARM_RATE: float = 1e-6  # Per single row, per single search
DEFAULT_MIN_VOTES: int = 1  # Hits needed across the sliding windows


def quantize_iq(samples: np.ndarray) -> np.ndarray:
    """Quantizes complex samples to one bit per component.

    The two bit planes are concatenated rather than interleaved, so that a
    quarter turn of the carrier is a swap of the two halves.

    Args:
        samples (np.ndarray): The complex samples to quantize.

    Returns:
        np.ndarray: A boolean array of twice the input length, the sign bits of
            I followed by the sign bits of Q.
    """
    return np.concatenate((np.real(samples) > 0, np.imag(samples) > 0))


def rotate_quarter_turns(bits: np.ndarray, rotations: int) -> np.ndarray:
    """Turns a quantized query by whole quadrants, using bit operations only.

    Multiplying by j maps (I, Q) to (-Q, I), which on sign bits is a swap of the
    two planes with the new leading one complemented. Nothing here needs the
    samples the bits came from, so a one bit front end supports it.

    Args:
        bits (np.ndarray): A quantized query, as produced by quantize_iq.
        rotations (int): How many quarter turns to apply.

    Returns:
        np.ndarray: The rotated query, of the same width.
    """
    for _ in range(rotations % QUERY_ROTATIONS):
        in_phase, quadrature = np.split(bits, 2)
        bits = np.concatenate((~quadrature, in_phase))
    return bits


def hd_threshold_for_false_alarm(n_columns: int, false_alarm_rate: float) -> int:
    """The loosest threshold that still rejects an unrelated codebook row.

    Against a row of a different PRN the sign bits agree at chance, so the
    Hamming distance is binomial with n_columns trials and probability one half.
    The threshold is the normal approximation of that distribution's lower tail.

    Args:
        n_columns (int): The width of a codebook row in bits.
        false_alarm_rate (float): The wanted probability that an unrelated row
            falls inside the threshold.

    Returns:
        int: The Hamming distance threshold, within [0, n_columns).
    """
    assert 0 < false_alarm_rate < 0.5, "False alarm rate must be in (0, 0.5)."
    z = NormalDist().inv_cdf(false_alarm_rate)  # negative, chance is n_columns / 2
    threshold = int(np.floor(n_columns / 2 + z * np.sqrt(n_columns) / 2))
    return int(np.clip(threshold, 0, n_columns - 1))


def hd_threshold_for_phase_quantization(n_columns: int, n_phases: int) -> int:
    """The tightest threshold that still admits a true match.

    A quadrant comparison disagrees on a fraction phi / pi of the bits for a
    residual carrier phase phi, so with n_phases hypotheses covering the circle
    the worst placed true match sits at n_columns / n_phases.

    This is the binding bound in practice. Rows of one PRN at neighbouring CFOs
    share their code, so they sit far below the unrelated row floor of
    n_columns / 2, and only a threshold of this order tells the CFO bins apart.

    Args:
        n_columns (int): The width of a codebook row in bits.
        n_phases (int): The carrier phase hypotheses covering the circle.

    Returns:
        int: The Hamming distance threshold, within [0, n_columns).
    """
    return int(np.clip(n_columns // n_phases, 0, n_columns - 1))


class OneBitHdCamClassifier(GpsL1AcqClassifier):
    """Acquisition by a single bit codebook lookup in an HdCam."""

    def __init__(
        self,
        config: AcqConfig,
        hd_threshold: int | None = None,
        n_codebook_phases: int = DEFAULT_CODEBOOK_PHASES,
        min_votes: int = DEFAULT_MIN_VOTES,
        false_alarm_rate: float = DEFAULT_FALSE_ALARM_RATE,
    ):
        """Builds the codebook and writes it into a fresh HdCam.

        Args:
            config (AcqConfig): The acquisition configuration. Its PRN list and
                Doppler grid are the two hypothesis dimensions of the codebook.
            hd_threshold (int | None): The Hamming distance threshold. When left
                as None it is the tighter of the two bounds above: tight enough
                to separate the CFO bins of one PRN, loose enough to admit a true
                match limited only by the phase quantization.
            n_codebook_phases (int): Carrier phase offsets stored per (PRN, CFO),
                on top of the four quarter turns of the query. One means quarter
                turns only, which does not separate neighbouring CFO bins.
            min_votes (int): How many hits a hypothesis needs, across the sliding
                windows and the rotations, to be reported.
            false_alarm_rate (float): Target false alarm rate of a single row in
                a single search, used only when hd_threshold is None.
        """
        super().__init__(config)
        assert n_codebook_phases >= 1, "At least one codebook phase is needed."
        assert min_votes >= 1, "At least one hit is needed to report a hypothesis."
        assert config.n_codes >= 2, (
            "At least two code periods are needed, so that a full contiguous "
            "window is available at every code phase."
        )
        self.n_codebook_phases = n_codebook_phases
        self.min_votes = min_votes

        self.n_columns = 2 * config.samples_per_code
        if hd_threshold is None:
            hd_threshold = min(
                hd_threshold_for_phase_quantization(self.n_columns, self.n_phases),
                hd_threshold_for_false_alarm(self.n_columns, false_alarm_rate),
            )
        self.cam = HdCam(self.n_rows, self.n_columns, hd_threshold)
        self.cam.write_array(self.build_codebook())

    @property
    def n_phases(self) -> int:
        """Returns the carrier phase hypotheses covering the circle."""
        return QUERY_ROTATIONS * self.n_codebook_phases

    @property
    def n_doppler_bins(self) -> int:
        """Returns the number of CFO bins of the configuration."""
        return len(self.config.doppler_grid_hz)

    @property
    def n_rows(self) -> int:
        """Returns the number of codebook rows, one per PRN, CFO and phase."""
        return len(self.config.prn_list) * self.n_doppler_bins * self.n_codebook_phases

    @property
    def hd_threshold(self) -> int:
        """Returns the Hamming distance threshold currently used by the CAM."""
        return self.cam.hd_threshold

    def set_hd_threshold(self, hd_threshold: int):
        """Retunes the CAM without rebuilding the codebook.

        Args:
            hd_threshold (int): The new Hamming distance threshold.
        """
        self.cam.set_hd_threshold(hd_threshold)

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

    def hypothesis_of(self, row: int) -> tuple[int, float]:
        """Returns the (PRN, Doppler) hypothesis a codebook row stands for.

        Args:
            row (int): The row index in the CAM.

        Returns:
            tuple[int, float]: The PRN number and the Doppler frequency in Hz.
        """
        hypothesis = row // self.n_codebook_phases
        prn = self.config.prn_list[hypothesis // self.n_doppler_bins]
        doppler_bin = hypothesis % self.n_doppler_bins
        return prn, float(self.config.doppler_grid_hz[doppler_bin])

    def build_codebook(self) -> np.ndarray:
        """Builds the quantized replica of every stored hypothesis.

        The stored phase offsets subdivide a quadrant, since the query already
        supplies the quarter turns.

        Returns:
            np.ndarray: A boolean array of shape (n_rows, 2 * samples_per_code).
        """
        config = self.config
        time_s = np.arange(config.samples_per_code) / config.fs_hz
        codebook = np.empty((self.n_rows, self.n_columns), dtype=bool)
        for prn_position, prn in enumerate(config.prn_list):
            code = sampled_ca_code(prn, config.fs_hz)
            for doppler_bin, doppler_hz in enumerate(config.doppler_grid_hz):
                carrier = np.exp(2j * np.pi * doppler_hz * time_s)
                for phase in range(self.n_codebook_phases):
                    offset = np.exp(2j * np.pi * phase / self.n_phases)
                    row = self.row_of(prn, doppler_bin, phase)
                    codebook[row] = quantize_iq(code * carrier * offset)
        return codebook

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

    def _acquire(self, samples: np.ndarray) -> list[PrnResult]:
        """Searches every code phase of the record and votes on the hits.

        Args:
            samples (np.ndarray): Input samples for acquisition.

        Returns:
            list[PrnResult]: One result per acquired PRN, in configuration order.
        """
        config = self.config
        votes = np.zeros((self.n_rows, config.samples_per_code), dtype=int)
        n_starts = config.samples_per_acquisition - config.samples_per_code + 1
        for start in range(n_starts):
            query = quantize_iq(self.query_window(samples, start))
            code_phase = start % config.samples_per_code
            for rotation in range(QUERY_ROTATIONS):
                rotated = rotate_quarter_turns(query, rotation)
                votes[self.cam.search_cam(rotated), code_phase] += 1
        return self._results_from_votes(votes)

    def _results_from_votes(self, votes: np.ndarray) -> list[PrnResult]:
        """Reduces the vote table to at most one result per PRN.

        Args:
            votes (np.ndarray): Hits per codebook row and code phase.

        Returns:
            list[PrnResult]: The winning hypothesis of every PRN that reached
                min_votes, in configuration order.
        """
        rows_per_prn = self.n_doppler_bins * self.n_codebook_phases
        results = []
        for prn_position, prn in enumerate(self.config.prn_list):
            block = votes[
                prn_position * rows_per_prn : (prn_position + 1) * rows_per_prn
            ]
            if block.max() < self.min_votes:
                continue
            row_in_block, code_phase = np.unravel_index(np.argmax(block), block.shape)
            doppler_bin = row_in_block // self.n_codebook_phases
            results.append(
                PrnResult(
                    prn=prn,
                    doppler_hz=float(self.config.doppler_grid_hz[doppler_bin]),
                    code_phase=int(code_phase),
                )
            )
        return results
