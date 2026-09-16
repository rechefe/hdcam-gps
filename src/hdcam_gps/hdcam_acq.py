"""1 bit HdCam based GPS L1 C/A acquisition.

The codebook written into the HdCam holds the replicas the acquisition is
searching for. A row is one C/A code period of a (PRN, CFO) hypothesis,
quantized to one bit per component: the sign of I followed by the sign of Q,
which is the carrier phase of every sample rounded to a quadrant. This is the
baseline family of the study in docs/CAM_FAMILY_STUDY.md, and the one the other
four are measured against.

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

The vote-then-rank decision this drives is not written here. It is shared with
every other family and lives in cam_acq.CamAcqClassifier, which this class
supplies four hooks to: the codebook, what its rows mean, what its queries mean,
and the four rotations of each query.
"""

import numpy as np

from hdcam_gps.acq_base import AcqConfig
from hdcam_gps.ca_code import sampled_ca_code
from hdcam_gps.cam_acq import (  # noqa: F401  - the study's public spelling
    DEFAULT_FALSE_ALARM_RATE,
    DEFAULT_VOTE_FRACTION,
    QUERY_ROTATIONS,
    CamAcqClassifier,
    Cell,
    QueryIndex,
    RowIndex,
    binomial_tail,
    hd_threshold_for_false_alarm,
    per_look_false_alarm,
    quantize_iq,
    rotate_quarter_turns,
)
from hdcam_gps.hdcam_packed import PackedHdCam

DEFAULT_CODEBOOK_PHASES: int = 2  # Extra phases carried by the codebook


class OneBitHdCamClassifier(CamAcqClassifier):
    """Acquisition by a single bit codebook lookup in an HdCam."""

    def __init__(
        self,
        config: AcqConfig,
        hd_threshold: int | None = None,
        n_codebook_phases: int = DEFAULT_CODEBOOK_PHASES,
        min_votes: int | None = None,
        false_alarm_rate: float = DEFAULT_FALSE_ALARM_RATE,
        search_mode: str = "cam",
        cam_factory=PackedHdCam,
    ):
        """Builds the codebook and writes it into a fresh HdCam.

        Args:
            config (AcqConfig): The acquisition configuration. Its PRN list and
                Doppler grid are the two hypothesis dimensions of the codebook.
            hd_threshold (int | None): The per look Hamming distance threshold.
                When left as None it follows from the vote rule: the per look
                rate that turns min_votes out of n_looks into false_alarm_rate,
                converted into a distance through the chance floor.
            n_codebook_phases (int): Carrier phase offsets stored per (PRN, CFO),
                on top of the four quarter turns of the query. One means quarter
                turns only, which does not separate neighbouring CFO bins.
            min_votes (int | None): How many of its looks a hypothesis has to
                match on. None takes a third of them, which is what lets the per
                look threshold sit close to chance.
            false_alarm_rate (float): False detections tolerated per acquisition,
                used only when hd_threshold is None. Lower it to trade
                sensitivity for a cleaner answer.
            search_mode (str): "cam" to issue one search_cam per query, "table"
                to read the same hits off a precomputed distance table.
            cam_factory: The HdCam class to build.
        """
        assert n_codebook_phases >= 1, "At least one codebook phase is needed."
        assert config.n_codes >= 2, (
            "At least two code periods are needed, so that a full contiguous "
            "window is available at every code phase."
        )
        self.n_codebook_phases = n_codebook_phases
        self.n_columns = 2 * config.samples_per_code
        super().__init__(
            config,
            hd_threshold=hd_threshold,
            min_votes=min_votes,
            false_alarm_rate=false_alarm_rate,
            search_mode=search_mode,
            cam_factory=cam_factory,
        )

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

    def row_index(self) -> RowIndex:
        """Returns the PRN and Doppler bin of every row.

        Returns:
            RowIndex: Parallel arrays of n_rows entries. The Doppler bin is in
                the row here, so every query is Doppler blind.
        """
        hypotheses = np.arange(self.n_rows) // self.n_codebook_phases
        prn_positions = hypotheses // self.n_doppler_bins
        return RowIndex(
            prn=np.array(self.config.prn_list)[prn_positions],
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

    def build_codebook(self) -> np.ndarray:
        """Builds the quantized replica of every stored hypothesis.

        The stored phase offsets subdivide a quadrant, since the query already
        supplies the quarter turns. They sit at the middle of their share rather
        than at its edge, so no replica ever lands on an axis: a replica at
        exactly zero phase and zero Doppler would be purely real, and its whole
        quadrature bit plane would be the sign of zero, carrying nothing and
        matching anything.

        Returns:
            np.ndarray: A boolean array of shape (n_rows, 2 * samples_per_code).
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

    def query_variants(self, samples: np.ndarray, query: int) -> list[np.ndarray]:
        """The four quarter turns of one window.

        A look counts once if any of them matched, so carrier phase is free
        inside a look.

        Args:
            samples (np.ndarray): The full input record.
            query (int): The window start, which is also the query index.

        Returns:
            list[np.ndarray]: Four boolean arrays, each n_columns wide.
        """
        bits = quantize_iq(self.query_window(samples, query))
        return [
            rotate_quarter_turns(bits, rotation) for rotation in range(QUERY_ROTATIONS)
        ]

    def shortlist(self, samples: np.ndarray) -> dict[tuple[int, int], list[np.ndarray]]:
        """Collects every hypothesis that falls inside the threshold anywhere.

        A hypothesis is kept when it matched on at least min_votes of the looks
        its code phase gets, which is the non-coherent accumulation. The queries
        returned for it are every rotation of every look that matched, not only
        the rotations that did: a rotation that missed is further away than the
        threshold, so it can never win the minimum the ranking takes.

        Args:
            samples (np.ndarray): Input samples for acquisition.

        Returns:
            dict[tuple[int, int], list[np.ndarray]]: The queries to measure,
                keyed by the codebook row and the code phase it matched at.
        """
        cells = self.shortlist_cells(self.matched_table(samples), self.min_votes)
        return {
            (cell.row, cell.code_phase): [
                bits
                for query in queries
                for bits in self.query_variants(samples, query)
            ]
            for cell, queries in cells.items()
        }
