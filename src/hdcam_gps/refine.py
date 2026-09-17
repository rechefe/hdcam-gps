"""The second stage two of the families need, and what it costs them.

Two designs in this study delete the carrier from the codebook and get their area
back for it. Both then discover they have deleted the CFO as well:

* the differential family turns a Doppler into one constant phase, which a 1 bit
  I and Q pair resolves into four classes and no finer;
* the segmented family compares 64 sample windows, whose frequency resolution is
  1/T = 16 kHz, so the whole +-5 kHz search sits inside one resolution cell.

Neither is a tuning problem and no decision rule fixes either. What fixes both is
a second stage: once the CAM has named a PRN and a code phase, sweep the Doppler
grid over the record at that code phase and take the strongest bin. Section 4.1
is explicit that their sensitivity is not comparable with the rest until this is
costed in, so DopplerRefiner counts what it does.

**The second stage is a correlator, which is the thing the CAM was supposed to
replace.** What makes it affordable is that it runs on the handful of (PRN, code
phase) pairs the CAM already found rather than on the whole search space: about
ten satellites times 21 bins times one record, against 32 PRNs times 21 bins
times every code phase. It stays a 1 bit correlator - the quadrant mixer wipes
the carrier off the quantized stream with one mod-4 add per sample, exactly as
code_only_acq does - so the front end the study is arguing for is unchanged.

RefinedClassifier composes the two, so a Doppler blind family and its second
stage measure as one classifier without either knowing about the other.
"""

from dataclasses import dataclass

import numpy as np

from hdcam_gps.acq_base import AcqConfig, GpsL1AcqClassifier, PrnResult
from hdcam_gps.ca_code import sampled_ca_code
from hdcam_gps.cam_acq import QUERY_ROTATIONS, quantize_iq
from hdcam_gps.code_only_acq import MIXERS, bits_from_quadrants, quadrants

# One complex multiply-accumulate per sample per bin per satellite, on operands
# a 1 bit front end produces. Counted, not modelled.
OPERATIONS_PER_SAMPLE: int = 1


@dataclass(frozen=True)
class RefinerCost:
    """What resolving the Doppler of a set of detections costs."""

    n_results: int  # Detections the second stage was asked about
    n_bins: int  # Doppler hypotheses swept per detection
    n_samples: int  # Samples of record swept per hypothesis
    mixer: str  # Which mixer produced the wiped stream

    @property
    def n_operations(self) -> int:
        """Multiply-accumulates the second stage performs."""
        return self.n_results * self.n_bins * self.n_samples * OPERATIONS_PER_SAMPLE

    def against_a_full_search(self, n_prn: int, samples_per_code: int) -> float:
        """How much smaller this is than correlating the whole search space.

        Args:
            n_prn (int): PRNs a full search would cover.
            samples_per_code (int): Code phases it would cover, per PRN.

        Returns:
            float: The factor the second stage saves, or infinity when it was
                asked about nothing.
        """
        full = n_prn * self.n_bins * self.n_samples * samples_per_code
        return full / self.n_operations if self.n_operations else float("inf")


class DopplerRefiner:
    """Picks the Doppler bin of a detection the CAM could not resolve."""

    def __init__(self, config: AcqConfig, mixer: str = "quadrant"):
        """Precomputes the phase ramp of every Doppler hypothesis.

        Args:
            config (AcqConfig): The acquisition configuration. Its Doppler grid
                is what the sweep covers.
            mixer (str): "quadrant" to wipe the carrier off the 1 bit stream,
                "exact" to wipe it off the full precision samples. Only the
                first is a 1 bit front end.
        """
        assert mixer in MIXERS, f"mixer is one of {MIXERS}, not {mixer!r}."
        self.config = config
        self.mixer = mixer
        self.n_results = 0
        time_s = np.arange(config.samples_per_acquisition) / config.fs_hz
        self._ramps = -QUERY_ROTATIONS * np.outer(config.doppler_grid_hz, time_s)
        self._carriers = np.exp(-2j * np.pi * np.outer(config.doppler_grid_hz, time_s))
        self._stream_cache: tuple[np.ndarray, np.ndarray] | None = None

    def quadrant_stream(self, samples: np.ndarray) -> np.ndarray:
        """The record as one 2 bit phase per sample, built once and kept.

        Args:
            samples (np.ndarray): The full input record.

        Returns:
            np.ndarray: One value in 0..3 per sample.
        """
        cached = self._stream_cache
        if cached is not None and cached[0] is samples:
            return cached[1]
        stream = quadrants(quantize_iq(samples))
        self._stream_cache = (samples, stream)
        return stream

    def wipe_doppler(self, samples: np.ndarray, doppler_bin: int) -> np.ndarray:
        """Removes one Doppler hypothesis from the whole record.

        Args:
            samples (np.ndarray): The full input record.
            doppler_bin (int): An index into the configuration's Doppler grid.

        Returns:
            np.ndarray: Complex samples, unit magnitude under the quadrant mixer.
        """
        if self.mixer == "exact":
            return samples * self._carriers[doppler_bin]
        turns = np.rint(self._ramps[doppler_bin]).astype(np.int8)
        stream = (self.quadrant_stream(samples) + turns) % QUERY_ROTATIONS
        bits = bits_from_quadrants(stream)
        in_phase, quadrature = np.split(bits, 2)
        return (2.0 * in_phase - 1.0) + 1j * (2.0 * quadrature - 1.0)

    def scores(
        self, samples: np.ndarray, prn: int, code_phase: int
    ) -> np.ndarray:
        """The correlation power of every Doppler bin at one code phase.

        The sum is coherent inside a code period and non-coherent across them,
        so a navigation bit transition costs one period rather than the answer.

        Args:
            samples (np.ndarray): The full input record.
            prn (int): The PRN the CAM named.
            code_phase (int): The code phase it named, in samples.

        Returns:
            np.ndarray: One power per bin of the configuration's Doppler grid.
        """
        config = self.config
        samples_per_code = config.samples_per_code
        n_codes = len(samples) // samples_per_code
        n_used = n_codes * samples_per_code
        code = np.roll(
            np.tile(sampled_ca_code(prn, config.fs_hz), n_codes), code_phase
        )
        powers = np.empty(len(config.doppler_grid_hz))
        for doppler_bin in range(len(config.doppler_grid_hz)):
            wiped = self.wipe_doppler(samples, doppler_bin)[:n_used]
            periods = (wiped * code).reshape(n_codes, samples_per_code)
            powers[doppler_bin] = np.sum(np.abs(periods.sum(axis=1)) ** 2)
        return powers

    def resolve(self, samples: np.ndarray, prn: int, code_phase: int) -> float:
        """The Doppler of one detection, in Hz on the configuration's grid.

        Args:
            samples (np.ndarray): The full input record.
            prn (int): The PRN the CAM named.
            code_phase (int): The code phase it named, in samples.

        Returns:
            float: The Doppler of the strongest bin.
        """
        powers = self.scores(samples, prn, code_phase)
        return float(self.config.doppler_grid_hz[int(np.argmax(powers))])

    def refine(
        self, samples: np.ndarray, results: list[PrnResult]
    ) -> list[PrnResult]:
        """Replaces the Doppler of every detection with a resolved one.

        Args:
            samples (np.ndarray): The full input record.
            results (list[PrnResult]): What the CAM returned. Their PRN and code
                phase are kept; only the Doppler is decided here.

        Returns:
            list[PrnResult]: The same detections, in the same order.
        """
        self.n_results += len(results)
        return [
            PrnResult(
                prn=result.prn,
                doppler_hz=self.resolve(samples, result.prn, result.code_phase),
                code_phase=result.code_phase,
            )
            for result in results
        ]

    def cost(self, n_results: int | None = None) -> RefinerCost:
        """What the second stage cost, counted over the detections it saw.

        Args:
            n_results (int | None): Detections to cost. None takes every one
                this refiner has been asked about.

        Returns:
            RefinerCost: The sweep it performed.
        """
        return RefinerCost(
            n_results=self.n_results if n_results is None else n_results,
            n_bins=len(self.config.doppler_grid_hz),
            n_samples=self.config.samples_per_acquisition,
            mixer=self.mixer,
        )


class RefinedClassifier(GpsL1AcqClassifier):
    """A Doppler blind family and the second stage that finishes its answer."""

    def __init__(self, inner: GpsL1AcqClassifier, mixer: str = "quadrant"):
        """Composes a classifier with a refiner built for the same configuration.

        Args:
            inner (GpsL1AcqClassifier): The family that names a PRN and a code
                phase. Its own Doppler output is discarded.
            mixer (str): Passed to DopplerRefiner.
        """
        super().__init__(inner.config)
        self.inner = inner
        self.refiner = DopplerRefiner(inner.config, mixer=mixer)

    def _acquire(self, samples: np.ndarray) -> list[PrnResult]:
        """Acquires with the inner family, then resolves each Doppler.

        Args:
            samples (np.ndarray): Input samples for acquisition.

        Returns:
            list[PrnResult]: One result per acquired PRN, Doppler resolved.
        """
        return self.refiner.refine(samples, self.inner.acquire(samples))
