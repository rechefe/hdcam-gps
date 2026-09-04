"""Baseline FFT based GPS L1 C/A acquisition.

This is the textbook parallel code phase search, kept deliberately simple so it
can serve as the reference the HdCam based classifiers are compared against.

For every PRN and every Doppler bin the input is corrected by an NCO at the
candidate carrier, correlated against the local code for all code phases at
once (an FFT, a multiply and an inverse FFT), and the resulting power is
summed non-coherently over the code periods of the observation. A PRN is declared
acquired when the peak of that grid stands far enough above the surrounding
noise floor.
"""

import numpy as np

from hdcam_gps.acq_base import (
    CHIPS_PER_CODE,
    AcqConfig,
    GpsL1AcqClassifier,
    PrnResult,
)
from hdcam_gps.ca_code import sampled_ca_code

DEFAULT_PEAK_RATIO: float = 2.5  # Peak to second peak ratio needed to declare a PRN


class FftAcqClassifier(GpsL1AcqClassifier):
    """Baseline acquisition by parallel code phase search."""

    def __init__(self, config: AcqConfig, peak_ratio: float = DEFAULT_PEAK_RATIO):
        """Initializes the classifier.

        Args:
            config (AcqConfig): The acquisition configuration.
            peak_ratio (float): The peak to second peak ratio above which a PRN
                is considered acquired.
        """
        super().__init__(config)
        assert peak_ratio > 0, "Peak ratio must be positive."
        self.peak_ratio = peak_ratio
        self._code_ffts: dict[int, np.ndarray] = {}

    def correlate(self, samples: np.ndarray, prn: int) -> np.ndarray:
        """Builds the Doppler by code phase power grid of a single PRN.

        Args:
            samples (np.ndarray): Input samples, samples_per_acquisition long.
            prn (int): The PRN number to correlate against.

        Returns:
            np.ndarray: The non-coherently accumulated correlation power, of
                shape (n_doppler_bins, samples_per_code).
        """
        config = self.config
        n_samples = config.samples_per_code
        doppler_grid = config.doppler_grid_hz
        code_fft = self._code_fft(prn)

        time_s = np.arange(config.samples_per_acquisition) / config.fs_hz
        grid = np.empty((len(doppler_grid), n_samples))
        for bin_index, doppler_hz in enumerate(doppler_grid):
            nco_corrected = samples * np.exp(-2j * np.pi * doppler_hz * time_s)
            periods = nco_corrected.reshape(config.n_codes, n_samples)
            correlation = np.fft.ifft(np.fft.fft(periods, axis=1) * code_fft, axis=1)
            grid[bin_index] = np.sum(np.abs(correlation) ** 2, axis=0)
        return grid

    def _acquire(self, samples: np.ndarray) -> list[PrnResult]:
        """Acquires every configured PRN that clears the peak ratio.

        Args:
            samples (np.ndarray): Input samples for acquisition.

        Returns:
            list[PrnResult]: One result per acquired PRN.
        """
        results = []
        for prn in self.config.prn_list:
            grid = self.correlate(samples, prn)
            doppler_bin, code_phase = np.unravel_index(np.argmax(grid), grid.shape)
            if self._peak_ratio(grid[doppler_bin], code_phase) >= self.peak_ratio:
                results.append(
                    PrnResult(
                        prn=prn,
                        doppler_hz=float(self.config.doppler_grid_hz[doppler_bin]),
                        code_phase=int(code_phase),
                    )
                )
        return results

    def _peak_ratio(self, doppler_row: np.ndarray, code_phase: int) -> float:
        """Ratio between the peak and the largest sample more than a chip away.

        Args:
            doppler_row (np.ndarray): The correlation power of the winning Doppler bin.
            code_phase (int): The code phase of the peak.

        Returns:
            float: The peak to second peak ratio. Zero for a silent row, so that
                an all zero input acquires nothing, and infinity when the peak
                stands over an exactly zero floor.
        """
        peak = doppler_row[code_phase]
        if peak == 0:
            return 0.0

        samples_per_chip = self.config.samples_per_code / CHIPS_PER_CODE
        exclusion = int(np.ceil(samples_per_chip))
        offsets = np.arange(len(doppler_row)) - code_phase
        # Wrap the offsets so the exclusion zone is circular, like the correlation.
        offsets = np.minimum(offsets % len(doppler_row), -offsets % len(doppler_row))
        outside_peak = doppler_row[offsets > exclusion]
        if outside_peak.size == 0 or outside_peak.max() == 0:
            return float("inf")
        return float(peak / outside_peak.max())

    def _code_fft(self, prn: int) -> np.ndarray:
        """Returns the conjugated FFT of a PRN's sampled code, generated once per PRN.

        Args:
            prn (int): The PRN number.

        Returns:
            np.ndarray: The conjugated FFT of the local code replica.
        """
        if prn not in self._code_ffts:
            code = sampled_ca_code(prn, self.config.fs_hz)
            self._code_ffts[prn] = np.conj(np.fft.fft(code))
        return self._code_ffts[prn]
