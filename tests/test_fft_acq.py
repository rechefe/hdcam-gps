"""Directed tests for the baseline FFT acquisition.

The signals here are synthetic and deliberately trivial - a clean code replica
at a Doppler that sits exactly on the search grid, occasionally with a little
noise. They exercise the wiring of the classifier (shapes, dispatch, caching,
thresholds, the reported numbers), not the sensitivity of the algorithm.
"""

import numpy as np
import pytest

from hdcam_gps.acq_base import AcqConfig, PrnResult
from hdcam_gps.ca_code import sampled_ca_code
from hdcam_gps.fft_acq import DEFAULT_PEAK_RATIO, FftAcqClassifier

# 1.023 MHz means one sample per chip, so a code phase is also a chip index.
FS_HZ = 1.023e6
SAMPLES_PER_CODE = 1023


def make_config(**overrides) -> AcqConfig:
    kwargs = {
        "fs_hz": FS_HZ,
        "prn_list": (1, 2, 3),
        "doppler_min_hz": -1000.0,
        "doppler_max_hz": 1000.0,
        "doppler_step_hz": 500.0,
        "n_codes": 2,
    }
    kwargs.update(overrides)
    return AcqConfig(**kwargs)


def make_signal(
    config: AcqConfig,
    prn_idx: int,
    doppler_hz: float = 0.0,
    code_phase: int = 0,
    amplitude: float = 1.0,
) -> np.ndarray:
    """A noise free replica of one satellite at a known Doppler and code phase."""
    code = np.tile(sampled_ca_code(prn_idx, config.fs_hz), config.n_codes)
    code = np.roll(code, code_phase).astype(complex)
    time_s = np.arange(config.samples_per_acquisition) / config.fs_hz
    return amplitude * code * np.exp(2j * np.pi * doppler_hz * time_s)


def noise(config: AcqConfig, sigma: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = config.samples_per_acquisition
    return sigma * (rng.normal(size=n) + 1j * rng.normal(size=n))


# --------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------


def test_classifier_keeps_config_and_default_threshold():
    config = make_config()
    classifier = FftAcqClassifier(config)
    assert classifier.config is config
    assert classifier.peak_ratio == DEFAULT_PEAK_RATIO


def test_classifier_accepts_a_custom_threshold():
    assert FftAcqClassifier(make_config(), peak_ratio=10.0).peak_ratio == 10.0


@pytest.mark.parametrize("bad_ratio", [0.0, -1.0])
def test_classifier_rejects_a_non_positive_threshold(bad_ratio):
    with pytest.raises(AssertionError):
        FftAcqClassifier(make_config(), peak_ratio=bad_ratio)


def test_classifier_starts_with_an_empty_code_cache():
    assert FftAcqClassifier(make_config())._code_ffts == {}


# --------------------------------------------------------------------------
# correlate - the Doppler by code phase grid
# --------------------------------------------------------------------------


def test_correlate_returns_a_real_non_negative_grid_of_the_right_shape():
    config = make_config()
    classifier = FftAcqClassifier(config)
    grid = classifier.correlate(make_signal(config, 1), prn=1)

    assert grid.shape == (len(config.doppler_grid_hz), SAMPLES_PER_CODE)
    assert grid.shape == (5, 1023)
    assert np.isrealobj(grid)
    assert (grid >= 0).all()


@pytest.mark.parametrize("code_phase", [0, 1, 300, 1022])
def test_correlate_peaks_at_the_injected_code_phase(code_phase):
    config = make_config()
    classifier = FftAcqClassifier(config)
    signal = make_signal(config, 1, doppler_hz=0.0, code_phase=code_phase)

    grid = classifier.correlate(signal, prn=1)
    _, peak_phase = np.unravel_index(np.argmax(grid), grid.shape)
    assert peak_phase == code_phase


@pytest.mark.parametrize("doppler_hz", [-1000.0, -500.0, 0.0, 500.0, 1000.0])
def test_correlate_peaks_in_the_injected_doppler_bin(doppler_hz):
    config = make_config()
    classifier = FftAcqClassifier(config)
    signal = make_signal(config, 1, doppler_hz=doppler_hz, code_phase=100)

    grid = classifier.correlate(signal, prn=1)
    peak_bin, _ = np.unravel_index(np.argmax(grid), grid.shape)
    assert config.doppler_grid_hz[peak_bin] == doppler_hz


def test_correlate_against_the_wrong_prn_gives_no_strong_peak():
    config = make_config()
    classifier = FftAcqClassifier(config)
    signal = make_signal(config, 1, code_phase=300)

    matched = classifier.correlate(signal, prn=1).max()
    mismatched = classifier.correlate(signal, prn=2).max()
    assert mismatched < matched / 10


def test_correlate_does_not_modify_its_input():
    config = make_config()
    signal = make_signal(config, 1, doppler_hz=500.0, code_phase=42)
    original = signal.copy()

    FftAcqClassifier(config).correlate(signal, prn=1)
    assert np.array_equal(signal, original)


def test_correlate_scales_with_signal_power():
    config = make_config()
    classifier = FftAcqClassifier(config)
    weak = classifier.correlate(make_signal(config, 1, amplitude=1.0), prn=1).max()
    strong = classifier.correlate(make_signal(config, 1, amplitude=2.0), prn=1).max()
    assert strong == pytest.approx(4.0 * weak)  # power, so amplitude squared


def test_correlate_grid_of_zeros_input_is_all_zeros():
    config = make_config()
    grid = FftAcqClassifier(config).correlate(
        np.zeros(config.samples_per_acquisition), prn=1
    )
    assert not grid.any()


# --------------------------------------------------------------------------
# the local code cache
# --------------------------------------------------------------------------


def test_code_fft_is_generated_once_per_prn_and_reused():
    config = make_config()
    classifier = FftAcqClassifier(config)
    signal = make_signal(config, 1)

    first = classifier._code_fft(1)
    classifier.correlate(signal, prn=1)
    assert classifier._code_fft(1) is first
    assert list(classifier._code_ffts) == [1]

    classifier.correlate(signal, prn=2)
    assert sorted(classifier._code_ffts) == [1, 2]


def test_code_fft_has_one_code_period_of_bins():
    classifier = FftAcqClassifier(make_config())
    assert classifier._code_fft(5).shape == (SAMPLES_PER_CODE,)


# --------------------------------------------------------------------------
# _acquire / acquire
# --------------------------------------------------------------------------


def test_acquire_finds_a_single_clean_satellite():
    config = make_config()
    signal = make_signal(config, 2, doppler_hz=500.0, code_phase=300)

    assert FftAcqClassifier(config).acquire(signal) == [
        PrnResult(prn=2, doppler_hz=500.0, code_phase=300)
    ]


def test_acquire_result_field_types():
    config = make_config()
    signal = make_signal(config, 1, doppler_hz=-500.0, code_phase=7)

    (result,) = FftAcqClassifier(config).acquire(signal)
    assert isinstance(result.prn, int)
    assert isinstance(result.doppler_hz, float)
    assert isinstance(result.code_phase, int)


def test_acquire_finds_several_satellites_in_config_order():
    config = make_config()
    signal = (
        make_signal(config, 3, doppler_hz=1000.0, code_phase=10)
        + make_signal(config, 1, doppler_hz=-500.0, code_phase=800)
    )

    assert FftAcqClassifier(config).acquire(signal) == [
        PrnResult(prn=1, doppler_hz=-500.0, code_phase=800),
        PrnResult(prn=3, doppler_hz=1000.0, code_phase=10),
    ]


def test_acquire_skips_satellites_that_are_not_present():
    config = make_config()
    signal = make_signal(config, 1, code_phase=55)

    results = FftAcqClassifier(config).acquire(signal)
    assert [result.prn for result in results] == [1]


def test_acquire_survives_a_little_noise():
    config = make_config()
    signal = make_signal(config, 2, doppler_hz=500.0, code_phase=300) + noise(
        config, sigma=0.5
    )

    assert FftAcqClassifier(config).acquire(signal) == [
        PrnResult(prn=2, doppler_hz=500.0, code_phase=300)
    ]


def test_acquire_returns_nothing_for_a_silent_input():
    config = make_config()
    assert FftAcqClassifier(config).acquire(np.zeros(config.samples_per_acquisition)) == []


def test_acquire_returns_nothing_for_noise_only():
    config = make_config()
    assert FftAcqClassifier(config).acquire(noise(config, sigma=1.0)) == []


# --------------------------------------------------------------------------
# the peak ratio threshold
# --------------------------------------------------------------------------


def test_a_very_high_threshold_rejects_everything():
    config = make_config()
    signal = make_signal(config, 1, code_phase=300)
    assert FftAcqClassifier(config, peak_ratio=1e9).acquire(signal) == []


def test_a_very_low_threshold_accepts_every_configured_prn():
    config = make_config()
    signal = make_signal(config, 1, code_phase=300)

    results = FftAcqClassifier(config, peak_ratio=1e-9).acquire(signal)
    assert [result.prn for result in results] == list(config.prn_list)


def test_threshold_decides_between_a_matched_and_mismatched_prn():
    config = make_config()
    classifier = FftAcqClassifier(config)
    signal = make_signal(config, 1, code_phase=300)

    matched_grid = classifier.correlate(signal, prn=1)
    mismatched_grid = classifier.correlate(signal, prn=2)

    def ratio_of(grid):
        peak_bin, peak_phase = np.unravel_index(np.argmax(grid), grid.shape)
        return classifier._peak_ratio(grid[peak_bin], peak_phase)

    assert ratio_of(matched_grid) > DEFAULT_PEAK_RATIO
    assert ratio_of(mismatched_grid) < DEFAULT_PEAK_RATIO


def test_peak_ratio_of_a_silent_row_is_zero():
    classifier = FftAcqClassifier(make_config())
    assert classifier._peak_ratio(np.zeros(SAMPLES_PER_CODE), 0) == 0.0


def test_peak_ratio_of_an_isolated_peak_is_infinite():
    classifier = FftAcqClassifier(make_config())
    row = np.zeros(SAMPLES_PER_CODE)
    row[100] = 1.0
    assert classifier._peak_ratio(row, 100) == float("inf")


def test_peak_ratio_ignores_the_chip_around_the_peak():
    classifier = FftAcqClassifier(make_config())
    row = np.zeros(SAMPLES_PER_CODE)
    row[100] = 10.0
    row[101] = 9.0  # inside the one chip exclusion zone, must not count
    row[500] = 2.0  # the real second peak
    assert classifier._peak_ratio(row, 100) == pytest.approx(5.0)


def test_peak_ratio_exclusion_zone_wraps_around():
    classifier = FftAcqClassifier(make_config())
    row = np.zeros(SAMPLES_PER_CODE)
    row[0] = 10.0
    row[-1] = 9.0  # one sample before the peak, circularly
    row[500] = 2.0
    assert classifier._peak_ratio(row, 0) == pytest.approx(5.0)


# --------------------------------------------------------------------------
# inherited behavior
# --------------------------------------------------------------------------


def test_acquire_still_asserts_on_the_sample_count():
    config = make_config()
    classifier = FftAcqClassifier(config)
    with pytest.raises(AssertionError):
        classifier.acquire(np.zeros(config.samples_per_acquisition + 1))


def test_sample_count_requirement_follows_n_codes():
    config = make_config(n_codes=4)
    classifier = FftAcqClassifier(config)
    signal = make_signal(config, 1, doppler_hz=500.0, code_phase=17)

    assert classifier.acquire(signal) == [
        PrnResult(prn=1, doppler_hz=500.0, code_phase=17)
    ]
    with pytest.raises(AssertionError):
        classifier.acquire(signal[:SAMPLES_PER_CODE])


def test_single_code_period_acquisition():
    config = make_config(n_codes=1)
    signal = make_signal(config, 3, doppler_hz=0.0, code_phase=512)

    assert FftAcqClassifier(config).acquire(signal) == [
        PrnResult(prn=3, doppler_hz=0.0, code_phase=512)
    ]
