"""Differential rows: multiply the record by its own delay and the carrier goes.

Every other family in the study spends most of its CAM on carrier phase. The
baseline stores two phases per hypothesis and turns the query four ways, eight
hypotheses for something the receiver does not want to know. Multiplying the
record by a delayed conjugate copy of itself deletes that dimension outright:
x[n]*conj(x[n-L]) carries the code times its own delay, and whatever carrier
phase the record arrived with cancels. No rotations, no stored phases, 128 rows
against the baseline's 1344, and 9208 searches against 36832.

**Two problems, both found during design and both worth stating before the
code.**

1. The CFO does not survive either. After the delay multiply a Doppler f is a
   constant phase 2*pi*f*L/fs, and one bit of I and Q resolves that into four
   classes. Resolving 500 Hz would need L near 256; staying unambiguous over
   +-5 kHz needs L below 102. Both cannot hold, so this is structurally a PRN
   and code phase detector with an ambiguous CFO, and a receiver would need a
   second stage to finish the job. The 42 times energy saving is what deleting
   hypotheses buys, not a free lunch.
2. Per sample signal to noise at 40 dB-Hz is -20 dB, and a delay multiply
   detector goes as the square of it at low signal to noise, so the noise times
   noise penalty is of the order of 20 dB.

The prediction is that this family dies in the phase 1 screen. The screen is
built to find that out cheaply rather than to be argued with, so the classifier
here is the minimum that produces a distance table, and PrnResult.doppler_hz is
the grid bin nearest its class centre rather than a measurement.
"""

import numpy as np

from hdcam_gps.acq_base import AcqConfig
from hdcam_gps.ca_code import sampled_ca_code
from hdcam_gps.cam_acq import (
    CamAcqClassifier,
    QueryIndex,
    RowIndex,
    quantize_iq,
)
from hdcam_gps.hdcam_packed import PackedHdCam

DOPPLER_CLASSES: int = 4  # All a one bit I and Q pair resolves a constant phase into


def unambiguous_lag(config: AcqConfig) -> int:
    """The longest delay whose phase still stays inside one turn.

    The delay multiply turns a Doppler f into a phase 2*pi*f*L/fs. At
    fs / (2 * f_max) the two ends of the search range land half a turn either
    side of zero and wrap onto each other, so the answer is one sample short of
    that, and never longer than a code period.

    Args:
        config (AcqConfig): Supplies the sampling rate and the Doppler range.

    Returns:
        int: The lag in samples, at least one.
    """
    longest = config.samples_per_code - 1
    reach = max(abs(config.doppler_min_hz), abs(config.doppler_max_hz))
    if reach <= 0:
        return max(1, longest)
    return max(1, min(longest, int(np.ceil(config.fs_hz / (2 * reach))) - 1))


def class_doppler_hz(class_index: int, lag_samples: int, fs_hz: float) -> float:
    """The Doppler at the centre of one class.

    Args:
        class_index (int): A class in 0..DOPPLER_CLASSES-1.
        lag_samples (int): The delay the record is multiplied against.
        fs_hz (float): The sampling frequency in Hz.

    Returns:
        float: The Doppler in Hz that lands in the middle of that class.
    """
    phase = (class_index - (DOPPLER_CLASSES - 1) / 2) * 2 * np.pi / DOPPLER_CLASSES
    return float(phase * fs_hz / (2 * np.pi * lag_samples))


def doppler_class(doppler_hz: float, lag_samples: int, fs_hz: float) -> int:
    """The class a Doppler falls in, once the delay multiply has flattened it.

    Args:
        doppler_hz (float): A Doppler in Hz.
        lag_samples (int): The delay the record is multiplied against.
        fs_hz (float): The sampling frequency in Hz.

    Returns:
        int: A class in 0..DOPPLER_CLASSES-1.
    """
    phase = 2 * np.pi * doppler_hz * lag_samples / fs_hz
    turns = (phase / (2 * np.pi) + 0.5) % 1.0  # class 0 starts half a turn back
    return int(turns * DOPPLER_CLASSES) % DOPPLER_CLASSES


def differential(samples: np.ndarray, lag_samples: int) -> np.ndarray:
    """Multiplies a run of samples by its own delayed conjugate.

    Args:
        samples (np.ndarray): Complex samples, longer than the lag.
        lag_samples (int): The delay in samples.

    Returns:
        np.ndarray: lag_samples fewer entries than the input.
    """
    return samples[lag_samples:] * np.conj(samples[:-lag_samples])


class DifferentialHdCamClassifier(CamAcqClassifier):
    """Acquisition against delay multiplied rows, with no carrier phase at all."""

    def __init__(
        self,
        config: AcqConfig,
        hd_threshold: int | None = None,
        lag_samples: int | None = None,
        min_votes: int | None = None,
        false_alarm_rate: float = 1e-2,
        search_mode: str = "cam",
        cam_factory=PackedHdCam,
    ):
        """Builds the delay multiplied codebook and writes it into a fresh HdCam.

        Args:
            config (AcqConfig): The acquisition configuration. Its PRN list is a
                codebook dimension; its Doppler range only sizes the lag.
            hd_threshold (int | None): The per look Hamming distance threshold.
                None derives it from the vote rule and false_alarm_rate.
            lag_samples (int | None): The delay to multiply against. None takes
                the longest one that keeps the Doppler range unambiguous.
            min_votes (int | None): How many of its looks a hypothesis has to
                match on. None takes a third of them.
            false_alarm_rate (float): False detections tolerated per acquisition,
                used only when hd_threshold is None.
            search_mode (str): "cam" or "table", as CamAcqClassifier defines.
            cam_factory: The HdCam class to build.
        """
        assert config.n_codes >= 2, (
            "At least two code periods are needed, so that a full contiguous "
            "window is available at every code phase."
        )
        self.lag_samples = (
            unambiguous_lag(config) if lag_samples is None else int(lag_samples)
        )
        assert 1 <= self.lag_samples < config.samples_per_code, (
            "The lag has to be shorter than a code period and at least one."
        )
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
    def n_rows(self) -> int:
        """Returns the number of rows, one per PRN and Doppler class."""
        return len(self.config.prn_list) * DOPPLER_CLASSES

    def row_of(self, prn: int, class_index: int = 0) -> int:
        """Returns the codebook row of one hypothesis.

        Args:
            prn (int): A PRN number from the configuration's PRN list.
            class_index (int): Which Doppler class.

        Returns:
            int: The row index in the CAM.
        """
        return self.config.prn_list.index(prn) * DOPPLER_CLASSES + class_index

    def code_differential(self, prn: int) -> np.ndarray:
        """One code period multiplied by its own delayed copy.

        The code is periodic, so the delay wraps inside the period and the
        result is periodic too.

        Args:
            prn (int): The PRN number.

        Returns:
            np.ndarray: samples_per_code values of plus or minus one.
        """
        code = sampled_ca_code(prn, self.config.fs_hz)
        return code * np.roll(code, self.lag_samples)

    def build_codebook(self) -> np.ndarray:
        """Builds the quantized delay multiplied replica of every hypothesis.

        A row is the code differential turned by its class's constant phase.
        Nothing else is stored, because nothing else survived the delay
        multiply.

        Returns:
            np.ndarray: A boolean array of shape (n_rows, 2 * samples_per_code).
        """
        codebook = np.empty((self.n_rows, self.n_columns), dtype=bool)
        for prn in self.config.prn_list:
            product = self.code_differential(prn).astype(complex)
            for class_index in range(DOPPLER_CLASSES):
                doppler_hz = class_doppler_hz(
                    class_index, self.lag_samples, self.config.fs_hz
                )
                phase = 2 * np.pi * doppler_hz * self.lag_samples / self.config.fs_hz
                row = self.row_of(prn, class_index)
                codebook[row] = quantize_iq(product * np.exp(1j * phase))
        return codebook

    def row_index(self) -> RowIndex:
        """Returns the PRN and the nearest grid bin of every row.

        The Doppler bin here is the grid bin closest to the class centre, not a
        measurement. This family resolves a class, and every table it appears in
        is flagged Doppler blind.

        Returns:
            RowIndex: Parallel arrays of n_rows entries. The segment offset
                carries the lag, so that a window start maps to a code phase.
        """
        rows = np.arange(self.n_rows)
        positions = rows // DOPPLER_CLASSES
        grid = self.config.doppler_grid_hz
        centres = [
            class_doppler_hz(index, self.lag_samples, self.config.fs_hz)
            for index in rows % DOPPLER_CLASSES
        ]
        nearest = [int(np.argmin(np.abs(grid - centre))) for centre in centres]
        return RowIndex(
            prn=np.array(self.config.prn_list)[positions],
            doppler_bin=np.array(nearest),
            segment_offset=np.full(
                self.n_rows, (-self.lag_samples) % self.config.samples_per_code
            ),
        )

    def true_rows(self, prn: int, doppler_hz: float) -> np.ndarray:
        """The row a satellite lands on, which is its class and not its bin.

        Args:
            prn (int): The PRN of the satellite.
            doppler_hz (float): Its true Doppler in Hz.

        Returns:
            np.ndarray: The one row index of that hypothesis.
        """
        class_index = doppler_class(doppler_hz, self.lag_samples, self.config.fs_hz)
        return np.array([self.row_of(prn, class_index)])

    def query_index(self, n_samples: int | None = None) -> QueryIndex:
        """Returns the start of every window the record is searched at.

        A window is a code period plus the lag, so the differential it produces
        is a whole code period long.

        Args:
            n_samples (int | None): Length of the record. None takes the
                configuration's own acquisition length.

        Returns:
            QueryIndex: One entry per window that fits.
        """
        if n_samples is None:
            n_samples = self.config.samples_per_acquisition
        span = self.config.samples_per_code + self.lag_samples
        return QueryIndex(
            start=np.arange(max(0, n_samples - span + 1)),
            doppler_bin=np.full(max(0, n_samples - span + 1), -1, dtype=int),
        )

    def query_variants(self, samples: np.ndarray, query: int) -> list[np.ndarray]:
        """The one query a window produces, since there is no phase to turn.

        Args:
            samples (np.ndarray): The full input record.
            query (int): The window start, which is also the query index.

        Returns:
            list[np.ndarray]: A single boolean array, n_columns wide.
        """
        span = self.config.samples_per_code + self.lag_samples
        end = query + span
        assert end <= len(samples), "The query window must fit inside the record."
        return [quantize_iq(differential(samples[query:end], self.lag_samples))]
