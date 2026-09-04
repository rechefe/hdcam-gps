"""Directed tests for the acquisition base module.

Covers the derived-number properties of AcqConfig, the PrnResult container and
the length assertion / dispatch behavior of GpsL1AcqClassifier.acquire.
"""

import dataclasses

import numpy as np
import pytest

from hdcam_gps.acq_base import (
    CHIPS_PER_CODE,
    CODE_PERIOD_S,
    L1_CARRIER_HZ,
    AcqConfig,
    GpsL1AcqClassifier,
    PrnResult,
)


def make_config(**overrides) -> AcqConfig:
    """A small, round-numbered config: 4 MHz sampling, 4000 samples per code."""
    kwargs = {
        "fs_hz": 4e6,
        "prn_list": (1, 2, 3),
        "doppler_min_hz": -5000.0,
        "doppler_max_hz": 5000.0,
        "doppler_step_hz": 500.0,
        "n_codes": 2,
    }
    kwargs.update(overrides)
    return AcqConfig(**kwargs)


# --------------------------------------------------------------------------
# module constants
# --------------------------------------------------------------------------


def test_gps_constants():
    assert CHIPS_PER_CODE == 1023
    assert CODE_PERIOD_S == 1e-3
    assert L1_CARRIER_HZ == 1575.42e6


# --------------------------------------------------------------------------
# AcqConfig - derived numbers
# --------------------------------------------------------------------------


def test_samples_per_code():
    # 4 MHz * 1 ms = 4000 samples in one C/A code period
    assert make_config().samples_per_code == 4000


@pytest.mark.parametrize(
    ("fs_hz", "expected"),
    [
        (1e6, 1000),
        (2.046e6, 2046),  # two samples per chip
        (4e6, 4000),
        (16.368e6, 16368),
    ],
)
def test_samples_per_code_for_common_sampling_rates(fs_hz, expected):
    assert make_config(fs_hz=fs_hz).samples_per_code == expected


def test_samples_per_code_truncates_a_fractional_result():
    # 1.5 kHz * 1 ms = 1.5 samples -> truncated down to 1
    config = make_config(fs_hz=1500.0)
    assert config.samples_per_code == 1
    assert isinstance(config.samples_per_code, int)


@pytest.mark.parametrize("n_codes", [1, 2, 5, 10])
def test_observation_time_scales_with_the_code_count(n_codes):
    config = make_config(n_codes=n_codes)
    assert config.observation_time_s == pytest.approx(n_codes * 1e-3)


@pytest.mark.parametrize("n_codes", [1, 2, 5, 10])
def test_samples_per_acquisition_is_samples_per_code_times_n_codes(n_codes):
    config = make_config(n_codes=n_codes)
    assert config.samples_per_acquisition == 4000 * n_codes
    assert config.samples_per_acquisition == config.samples_per_code * n_codes


def test_samples_per_acquisition_matches_rate_times_observation_time():
    config = make_config(n_codes=3)
    assert config.samples_per_acquisition == pytest.approx(
        config.fs_hz * config.observation_time_s
    )


# --------------------------------------------------------------------------
# AcqConfig - doppler grid
# --------------------------------------------------------------------------


def test_doppler_grid_covers_both_endpoints():
    grid = make_config().doppler_grid_hz
    assert grid[0] == pytest.approx(-5000.0)
    assert grid[-1] == pytest.approx(5000.0)


def test_doppler_grid_values_and_length():
    grid = make_config().doppler_grid_hz
    # -5000 .. 5000 in 500 Hz steps -> 21 bins
    assert len(grid) == 21
    assert np.allclose(grid, np.arange(-5000.0, 5000.0 + 500.0, 500.0))


def test_doppler_grid_is_evenly_spaced_by_the_step():
    grid = make_config(doppler_step_hz=250.0).doppler_grid_hz
    assert len(grid) == 41
    assert np.allclose(np.diff(grid), 250.0)


def test_doppler_grid_with_a_single_bin():
    config = make_config(doppler_min_hz=0.0, doppler_max_hz=0.0, doppler_step_hz=500.0)
    grid = config.doppler_grid_hz
    assert len(grid) == 1
    assert grid[0] == pytest.approx(0.0)


def test_doppler_grid_is_a_numpy_array():
    assert isinstance(make_config().doppler_grid_hz, np.ndarray)


# --------------------------------------------------------------------------
# dataclass behavior
# --------------------------------------------------------------------------


def test_acq_config_is_frozen():
    config = make_config()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.fs_hz = 8e6


def test_prn_result_holds_its_fields():
    result = PrnResult(prn=7, doppler_hz=1500.0, code_phase=1234)
    assert result.prn == 7
    assert result.doppler_hz == 1500.0
    assert result.code_phase == 1234


def test_prn_result_is_frozen_and_compares_by_value():
    result = PrnResult(prn=7, doppler_hz=1500.0, code_phase=1234)
    assert result == PrnResult(prn=7, doppler_hz=1500.0, code_phase=1234)
    assert result != PrnResult(prn=8, doppler_hz=1500.0, code_phase=1234)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.prn = 8


# --------------------------------------------------------------------------
# GpsL1AcqClassifier
# --------------------------------------------------------------------------


class RecordingClassifier(GpsL1AcqClassifier):
    """Minimal subclass that records what _acquire was handed."""

    def __init__(self, config: AcqConfig):
        super().__init__(config)
        self.seen: list[np.ndarray] = []

    def _acquire(self, samples: np.ndarray) -> list[PrnResult]:
        self.seen.append(samples)
        return [
            PrnResult(prn=prn, doppler_hz=0.0, code_phase=0)
            for prn in self.config.prn_list
        ]


def test_classifier_keeps_its_config():
    config = make_config()
    assert GpsL1AcqClassifier(config).config is config


def test_acquire_passes_correctly_sized_samples_through_to__acquire():
    config = make_config()
    classifier = RecordingClassifier(config)
    samples = np.zeros(config.samples_per_acquisition)

    results = classifier.acquire(samples)

    assert len(classifier.seen) == 1
    assert classifier.seen[0] is samples
    assert results == [
        PrnResult(prn=1, doppler_hz=0.0, code_phase=0),
        PrnResult(prn=2, doppler_hz=0.0, code_phase=0),
        PrnResult(prn=3, doppler_hz=0.0, code_phase=0),
    ]


@pytest.mark.parametrize("delta", [-1, 1, -4000, 4000])
def test_acquire_asserts_on_a_wrong_number_of_samples(delta):
    config = make_config()
    classifier = RecordingClassifier(config)
    samples = np.zeros(config.samples_per_acquisition + delta)

    with pytest.raises(AssertionError):
        classifier.acquire(samples)

    assert classifier.seen == []  # _acquire was never reached


def test_acquire_assertion_message_names_both_lengths():
    config = make_config()
    classifier = RecordingClassifier(config)

    with pytest.raises(AssertionError) as excinfo:
        classifier.acquire(np.zeros(10))

    message = str(excinfo.value)
    assert str(config.samples_per_acquisition) in message
    assert "but got 10" in message


def test_acquire_asserts_on_an_empty_input():
    classifier = RecordingClassifier(make_config())
    with pytest.raises(AssertionError):
        classifier.acquire(np.zeros(0))


def test_acquire_length_requirement_follows_the_config():
    config = make_config(n_codes=5)  # 5 * 4000 = 20000 samples
    classifier = RecordingClassifier(config)

    classifier.acquire(np.zeros(20000))
    assert len(classifier.seen) == 1

    with pytest.raises(AssertionError):
        classifier.acquire(np.zeros(4000))


def test_base_acquire_raises_not_implemented():
    config = make_config()
    classifier = GpsL1AcqClassifier(config)
    with pytest.raises(NotImplementedError):
        classifier.acquire(np.zeros(config.samples_per_acquisition))


def test_base_class_checks_the_length_before_reaching__acquire():
    # the assertion fires first, so the caller sees AssertionError, not NotImplementedError
    config = make_config()
    classifier = GpsL1AcqClassifier(config)
    with pytest.raises(AssertionError):
        classifier.acquire(np.zeros(1))
