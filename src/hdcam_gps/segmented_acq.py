"""Segmented rows: the match line narrowed to a width silicon demonstrates.

The baseline asks a CAM to compare 2046 bits at once. The JSSC 2025 macro
compares 64, and match line design is what limits that: every bit of a row
discharges the same wire, so a wider word means a smaller signal per bit and a
longer settle. Cutting a code period into K sub-rows of segment_bits each puts
the comparison back inside that limit at almost no cost in stored bits - 1344
rows of 2046 become 20160 of 128, 2.58 Mbit against 2.75. The 6 % it does save is
not a saving: it is the tail of the code period that no sub-row covers, and the
paragraph on K below explains it.

**The non-obvious part is that the searches do not follow the rows.** Segment k
of a hypothesis at code phase p is segment 0 of the same hypothesis at code
phase p + k*segment_samples, so one short query sliding over the record already
visits every segment at every alignment. A hit on (hypothesis, segment k) at
start s votes for code phase s - k*segment_samples, which is exactly the
segment_offset field of cam_acq's mapping. The searches rise by 10 % against the
baseline, because a 64 sample window starts in more places than a 1023 sample
one, and not by the fifteen the rows rose by.

What is paid for instead is two things.

* One coherent sum of 2046 bits becomes m-of-K binary integration over 128 bit
  pieces, worth 1 to 2 dB. The tolerance **fraction** does not improve, because
  the signal to noise ratio sets it; only the absolute count falls, and the
  relative spread of the chance floor worsens from 1.1 % of the width to 4.4 %.
* **The CFO stops being resolvable at all, and no decision rule wins it back.**
  A 64 sample window is 62 microseconds, whose frequency resolution is 1/T =
  16 kHz. The entire +-5 kHz search fits inside one resolution cell, so a
  segment of the right PRN at the right code phase matches every Doppler bin
  equally - measured at distance zero for all of them, at every threshold. The
  m-of-K rule counts segments and they all match, so counting cannot separate
  bins either. Like the differential family, this one is a PRN and code phase
  detector, and a receiver would need a second stage to finish the job. Section
  3 expected m-of-K to buy back 1 to 2 dB of integration loss; the loss it does
  not touch is this one.
* The readout floods. At a 47 % tolerance on 128 bits about a quarter of the
  rows fall inside every search, so an acquisition produces hits by the hundred
  million. That is a digital readout problem rather than a CAM problem, and it
  is why this family is scored through the table path.

A code period of 1023 samples does not divide into 64 sample segments, so K is
the floor and the last 63 samples of the period are in no sub-row. The
hypothesis is tested on 94 % of its energy, and every code phase stays reachable
because the queries slide over all of them.
"""

from math import ceil

import numpy as np

from hdcam_gps.acq_base import AcqConfig
from hdcam_gps.ca_code import sampled_ca_code
from hdcam_gps.cam_acq import (
    DEFAULT_VOTE_FRACTION,
    QUERY_ROTATIONS,
    CamAcqClassifier,
    QueryIndex,
    RowIndex,
    quantize_iq,
    rotate_quarter_turns,
)
from hdcam_gps.hdcam_packed import PackedHdCam

DEFAULT_SEGMENT_BITS: int = 128  # Two per sample, so 64 samples of the period
DEFAULT_CODEBOOK_PHASES: int = 2  # Stored phases, as the baseline carries


class SegmentedHdCamClassifier(CamAcqClassifier):
    """Acquisition against sub-rows of one code period, searched in one pass."""

    def __init__(
        self,
        config: AcqConfig,
        hd_threshold: int | None = None,
        segment_bits: int = DEFAULT_SEGMENT_BITS,
        n_codebook_phases: int = DEFAULT_CODEBOOK_PHASES,
        min_votes: int | None = None,
        min_segments: int | None = None,
        false_alarm_rate: float = 1e-2,
        search_mode: str = "table",
        cam_factory=PackedHdCam,
    ):
        """Builds the segmented codebook and writes it into a fresh HdCam.

        Args:
            config (AcqConfig): The acquisition configuration. Its PRN list and
                Doppler grid are the two hypothesis dimensions of the codebook.
            hd_threshold (int | None): The per look Hamming distance threshold.
                None derives it from the vote rule and false_alarm_rate.
            segment_bits (int): Width of a sub-row, two bits per sample.
            n_codebook_phases (int): Carrier phase offsets stored per (PRN, CFO).
            min_votes (int | None): How many of its looks a hypothesis has to
                match on. None takes a third of them.
            min_segments (int | None): How many of the K sub-rows have to match
                inside a look before it votes. None takes a third of them, which
                is a starting point for calibration rather than an answer.
            false_alarm_rate (float): False detections tolerated per acquisition,
                used only when hd_threshold is None.
            search_mode (str): "cam" or "table". The default is the table,
                because the CAM path of this family returns hits by the hundred
                million and that is the thing being measured, not simulated.
            cam_factory: The HdCam class to build.
        """
        assert segment_bits >= 2 and segment_bits % 2 == 0, (
            "A segment holds two bits per sample, so its width is even."
        )
        assert n_codebook_phases >= 1, "At least one codebook phase is needed."
        assert config.n_codes >= 2, (
            "At least two code periods are needed, so that a full contiguous "
            "window is available at every code phase."
        )
        self.segment_samples = segment_bits // 2
        assert self.segment_samples <= config.samples_per_code, (
            "A segment cannot be longer than the code period it cuts up."
        )
        self.n_segments = config.samples_per_code // self.segment_samples
        self.n_codebook_phases = n_codebook_phases
        self.n_columns = segment_bits
        if min_segments is None:
            min_segments = max(1, ceil(DEFAULT_VOTE_FRACTION * self.n_segments))
        assert 1 <= min_segments <= self.n_segments, (
            f"min_segments must be between 1 and the {self.n_segments} sub-rows "
            "a hypothesis has."
        )
        super().__init__(
            config,
            hd_threshold=hd_threshold,
            min_votes=min_votes,
            false_alarm_rate=false_alarm_rate,
            search_mode=search_mode,
            cam_factory=cam_factory,
            min_segments=min_segments,
        )

    @property
    def n_phases(self) -> int:
        """Returns the carrier phase hypotheses covering the circle."""
        return QUERY_ROTATIONS * self.n_codebook_phases

    @property
    def n_hypotheses(self) -> int:
        """Returns the (PRN, CFO, stored phase) triples, before segmenting."""
        return len(self.config.prn_list) * self.n_doppler_bins * self.n_codebook_phases

    @property
    def n_rows(self) -> int:
        """Returns the number of sub-rows, one per hypothesis and segment."""
        return self.n_hypotheses * self.n_segments

    def row_of(
        self, prn: int, doppler_bin: int, segment: int = 0, codebook_phase: int = 0
    ) -> int:
        """Returns the sub-row of one hypothesis and segment.

        Args:
            prn (int): A PRN number from the configuration's PRN list.
            doppler_bin (int): An index into the configuration's Doppler grid.
            segment (int): Which sub-row of the code period.
            codebook_phase (int): Which stored carrier phase offset.

        Returns:
            int: The row index in the CAM.
        """
        prn_position = self.config.prn_list.index(prn)
        hypothesis = prn_position * self.n_doppler_bins + doppler_bin
        hypothesis = hypothesis * self.n_codebook_phases + codebook_phase
        return hypothesis * self.n_segments + segment

    def build_codebook(self) -> np.ndarray:
        """Cuts every stored replica into its sub-rows.

        A sub-row holds the I bits then the Q bits of its own segment_samples,
        so a query of the same window is comparable to it directly.

        Returns:
            np.ndarray: A boolean array of shape (n_rows, segment_bits).
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
                    for segment in range(self.n_segments):
                        first = segment * self.segment_samples
                        row = self.row_of(prn, doppler_bin, segment, phase)
                        codebook[row] = quantize_iq(
                            replica[first : first + self.segment_samples]
                        )
        return codebook

    def row_index(self) -> RowIndex:
        """Returns the PRN, Doppler bin and segment offset of every sub-row.

        Returns:
            RowIndex: Parallel arrays of n_rows entries. The segment offset is
                what turns a window start into a code phase, and the hypothesis
                is what groups the K sub-rows the m-of-K rule votes on together.
        """
        rows = np.arange(self.n_rows)
        hypothesis = rows // self.n_segments
        triples = hypothesis // self.n_codebook_phases
        positions = triples // self.n_doppler_bins
        return RowIndex(
            prn=np.array(self.config.prn_list)[positions],
            doppler_bin=triples % self.n_doppler_bins,
            segment_offset=(rows % self.n_segments) * self.segment_samples,
            hypothesis=hypothesis,
        )

    def query_index(self, n_samples: int | None = None) -> QueryIndex:
        """Returns the start of every short window the record is searched at.

        Args:
            n_samples (int | None): Length of the record. None takes the
                configuration's own acquisition length.

        Returns:
            QueryIndex: One entry per window of segment_samples that fits.
        """
        if n_samples is None:
            n_samples = self.config.samples_per_acquisition
        n_starts = n_samples - self.segment_samples + 1
        return QueryIndex(
            start=np.arange(n_starts),
            doppler_bin=np.full(n_starts, -1, dtype=int),
        )

    def query_variants(self, samples: np.ndarray, query: int) -> list[np.ndarray]:
        """The four quarter turns of one short window.

        Args:
            samples (np.ndarray): The full input record.
            query (int): The window start, which is also the query index.

        Returns:
            list[np.ndarray]: Four boolean arrays, each segment_bits wide.
        """
        end = query + self.segment_samples
        assert end <= len(samples), "The query window must fit inside the record."
        bits = quantize_iq(samples[query:end])
        return [
            rotate_quarter_turns(bits, rotation) for rotation in range(QUERY_ROTATIONS)
        ]
