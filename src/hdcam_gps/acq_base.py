from dataclasses import dataclass

import numpy as np

CHIPS_PER_CODE: int = 1023  # Number of chips in a GPS C/A code
CODE_PERIOD_S: float = 1e-3  # Period of a GPS C/A code in seconds
L1_CARRIER_HZ: float = 1575.42e6  # L1 carrier frequency in Hz


@dataclass(frozen=True)
class AcqConfig:
    """Acquisition configuration parameters - for general GPS acquisition."""

    fs_hz: float  # Sampling frequency in Hz
    prn_list: tuple[int, ...]  # List of PRN numbers to acquire
    doppler_min_hz: float  # Minimum Doppler frequency in Hz
    doppler_max_hz: float  # Maximum Doppler frequency in Hz
    doppler_step_hz: float  # Doppler frequency step in Hz
    n_codes: int  # Number of C/A code periods to use for acquisition

    @property
    def samples_per_code(self) -> int:
        """Returns the number of samples per C/A code period."""
        return int(self.fs_hz * CODE_PERIOD_S)  # Number of samples per C/A code period

    @property
    def observation_time_s(self) -> float:
        """Returns the total observation time in seconds."""
        return self.n_codes * CODE_PERIOD_S  # Total observation time in seconds

    @property
    def samples_per_acquisition(self) -> int:
        """Returns the total number of samples for the entire acquisition."""
        return (
            self.samples_per_code * self.n_codes
        )  # Total samples for the entire acquisition

    @property
    def doppler_grid_hz(self) -> np.ndarray:
        """Returns the Doppler frequency grid based on the configuration."""
        return np.arange(
            self.doppler_min_hz,
            self.doppler_max_hz + self.doppler_step_hz,
            self.doppler_step_hz,
        )


@dataclass(frozen=True)
class PrnResult:
    """Result of a PRN acquisition."""

    prn: int  # PRN number
    doppler_hz: float  # Detected Doppler frequency in Hz
    code_phase: int  # Detected code phase in samples


class GpsL1AcqClassifier:
    """Base class for classifiers
    Extracts PrnResult(s) from a set of input samples and given acquisition configuration.
    """

    def __init__(self, config: AcqConfig):
        self.config = config

    def acquire(self, samples: np.ndarray) -> list[PrnResult]:
        """Acquires PRN(s) from the given samples based on the acquisition configuration.

        Args:
            samples (np.ndarray): Input samples for acquisition.

        Returns:
            list[PrnResult]: List of acquired PRN results.
        """
        assert len(samples) == self.config.samples_per_acquisition, (
            f"Input samples length must be {self.config.samples_per_acquisition}, "
            f"but got {len(samples)}."
        )
        return self._acquire(samples)

    def _acquire(self, samples: np.ndarray) -> list[PrnResult]:
        """Internal method to be implemented by subclasses for specific acquisition logic.

        Args:
            samples (np.ndarray): Input samples for acquisition.
        """
        raise NotImplementedError("Subclasses must implement the _acquire method.")
