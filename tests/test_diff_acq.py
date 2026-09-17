"""Directed tests for the differential family.

The claim this design rests on is that the carrier phase cancels, so most of
these are about what survives the delay multiply and what does not: the code
does, the initial phase does not, and the CFO comes back only as one of four
classes. The sampling rate is deliberately low, so a whole acquisition run is
cheap.

The acquisition tests are noise free, and that is the point: docs
/CAM_FAMILY_STUDY.md section 3 predicts this family dies on a noise times noise
penalty of about 20 dB, which a noise free test cannot see and the phase 1
screen can. What these check is that the wiring is right, so that the screen's
verdict is about the design rather than about a bug.
"""

import numpy as np
import pytest

from hdcam_gps.acq_base import AcqConfig
from hdcam_gps.ca_code import sampled_ca_code
from hdcam_gps.cam_acq import quantize_iq
from hdcam_gps.diff_acq import (
    DOPPLER_CLASSES,
    DifferentialHdCamClassifier,
    class_doppler_hz,
    differential,
    doppler_class,
    unambiguous_lag,
)
from hdcam_gps.hdcam_acq import OneBitHdCamClassifier

FS_HZ = 204.6e3  # 204 samples per code period, enough to stay correct and cheap
SAMPLES_PER_CODE = 204


def make_config(**overrides) -> AcqConfig:
    kwargs = {
        "fs_hz": FS_HZ,
        "prn_list": (1, 2, 3),
        "doppler_min_hz": -500.0,
        "doppler_max_hz": 500.0,
        "doppler_step_hz": 500.0,
        "n_codes": 4,
    }
    kwargs.update(overrides)
    return AcqConfig(**kwargs)


def make_signal(
    config: AcqConfig,
    prn: int,
    doppler_hz: float = 0.0,
    code_phase: int = 0,
    carrier_phase: float = 0.7,
) -> np.ndarray:
    """A noise free replica at a known Doppler, code phase and carrier phase."""
    code = np.tile(sampled_ca_code(prn, config.fs_hz), config.n_codes)
    code = np.roll(code, code_phase).astype(complex)
    time_s = np.arange(config.samples_per_acquisition) / config.fs_hz
    return code * np.exp(2j * np.pi * doppler_hz * time_s + 1j * carrier_phase)


# --------------------------------------------------------------------------
# what the delay multiply keeps and what it throws away
# --------------------------------------------------------------------------


def test_the_delay_multiply_shortens_the_run_by_the_lag():
    samples = np.arange(10) + 0j
    assert len(differential(samples, 3)) == 7


def test_the_carrier_phase_cancels():
    # The whole argument for the family: the record's own phase leaves no trace,
    # so no rotation and no stored phase has to cover it.
    config = make_config()
    quiet = make_signal(config, 1, doppler_hz=500.0, carrier_phase=0.0)
    turned = make_signal(config, 1, doppler_hz=500.0, carrier_phase=2.1)
    assert np.allclose(differential(quiet, 50), differential(turned, 50))


def test_a_doppler_becomes_one_constant_phase():
    config = make_config()
    lag = 50
    product = differential(make_signal(config, 1, doppler_hz=300.0), lag)
    expected = 2 * np.pi * 300.0 * lag / config.fs_hz
    # The code flips the sign, so the phase is the same modulo half a turn.
    assert np.allclose(np.angle(product) % np.pi, expected % np.pi, atol=1e-6)


def test_the_code_differential_is_periodic_and_real():
    classifier = DifferentialHdCamClassifier(make_config())
    product = classifier.code_differential(2)
    assert len(product) == SAMPLES_PER_CODE
    assert set(np.unique(product).tolist()) <= {-1.0, 1.0}


# --------------------------------------------------------------------------
# the lag, and the ambiguity it cannot escape
# --------------------------------------------------------------------------


def test_the_default_lag_keeps_the_doppler_range_inside_one_turn():
    config = make_config()
    lag = unambiguous_lag(config)
    for doppler_hz in (config.doppler_min_hz, config.doppler_max_hz):
        assert abs(2 * np.pi * doppler_hz * lag / config.fs_hz) < np.pi


def test_a_wider_doppler_range_forces_a_shorter_lag():
    narrow = unambiguous_lag(make_config(doppler_min_hz=-500.0, doppler_max_hz=500.0))
    wide = unambiguous_lag(make_config(doppler_min_hz=-5000.0, doppler_max_hz=5000.0))
    assert wide < narrow


def test_the_lag_cannot_resolve_a_bin_and_stay_unambiguous_at_once():
    # Section 3's arithmetic, as a test: resolving one 500 Hz bin into its own
    # class needs a lag four times longer than the range allows.
    config = make_config(
        doppler_min_hz=-5000.0, doppler_max_hz=5000.0, fs_hz=1.023e6
    )
    unambiguous = unambiguous_lag(config)
    resolving = config.fs_hz / (DOPPLER_CLASSES * config.doppler_step_hz)
    assert resolving > unambiguous


@pytest.mark.parametrize("class_index", range(DOPPLER_CLASSES))
def test_a_class_centre_maps_back_to_its_own_class(class_index):
    lag = unambiguous_lag(make_config())
    centre = class_doppler_hz(class_index, lag, FS_HZ)
    assert doppler_class(centre, lag, FS_HZ) == class_index


def test_the_classes_cover_the_search_range_in_order():
    config = make_config()
    lag = unambiguous_lag(config)
    classes = [
        doppler_class(float(doppler_hz), lag, config.fs_hz)
        for doppler_hz in np.linspace(config.doppler_min_hz, config.doppler_max_hz, 9)
    ]
    assert classes == sorted(classes)
    assert set(classes) == set(range(DOPPLER_CLASSES))


def test_a_lag_as_long_as_the_code_period_is_rejected():
    config = make_config()
    with pytest.raises(AssertionError):
        DifferentialHdCamClassifier(config, lag_samples=config.samples_per_code)


# --------------------------------------------------------------------------
# the shape the cancelled phase buys
# --------------------------------------------------------------------------


def test_there_is_one_row_per_prn_and_class():
    config = make_config()
    classifier = DifferentialHdCamClassifier(config)
    assert classifier.n_rows == len(config.prn_list) * DOPPLER_CLASSES
    assert classifier.n_columns == 2 * SAMPLES_PER_CODE


def test_the_rows_fall_against_the_baseline():
    # 128 rows against 1344 at the study's grid, because carrier phase and the
    # CFO both stopped being codebook dimensions.
    config = make_config(doppler_min_hz=-5000.0, doppler_max_hz=5000.0)
    assert DifferentialHdCamClassifier(config).n_rows < (
        OneBitHdCamClassifier(config).n_rows // 5
    )


def test_a_query_needs_no_rotation():
    config = make_config()
    classifier = DifferentialHdCamClassifier(config)
    assert len(classifier.query_variants(make_signal(config, 1), 0)) == 1


def test_a_window_is_a_code_period_plus_the_lag():
    config = make_config()
    classifier = DifferentialHdCamClassifier(config)
    span = config.samples_per_code + classifier.lag_samples
    assert classifier.n_queries == config.samples_per_acquisition - span + 1


def test_a_window_past_the_end_of_the_record_is_rejected():
    config = make_config()
    classifier = DifferentialHdCamClassifier(config)
    samples = make_signal(config, 1)
    classifier.query_variants(samples, classifier.n_queries - 1)
    with pytest.raises(AssertionError):
        classifier.query_variants(samples, classifier.n_queries)


def test_the_lag_is_carried_in_the_segment_offset():
    # The window starts a lag before the code phase it stands for, and cam_acq's
    # mapping subtracts the segment offset, so that is where the lag lives.
    config = make_config()
    classifier = DifferentialHdCamClassifier(config)
    rows = classifier.row_index()
    offset = rows.segment_offset[0]
    start = 40
    assert (start - offset) % config.samples_per_code == (
        (start + classifier.lag_samples) % config.samples_per_code
    )


def test_a_codebook_row_is_the_turned_code_differential():
    config = make_config()
    classifier = DifferentialHdCamClassifier(config)
    prn, class_index = 2, 3
    doppler_hz = class_doppler_hz(class_index, classifier.lag_samples, config.fs_hz)
    phase = 2 * np.pi * doppler_hz * classifier.lag_samples / config.fs_hz
    expected = quantize_iq(
        classifier.code_differential(prn).astype(complex) * np.exp(1j * phase)
    )
    assert np.array_equal(
        classifier.build_codebook()[classifier.row_of(prn, class_index)], expected
    )


def test_the_true_row_is_the_class_and_not_the_nearest_bin():
    config = make_config()
    classifier = DifferentialHdCamClassifier(config)
    for doppler_hz in config.doppler_grid_hz:
        (row,) = classifier.true_rows(2, float(doppler_hz))
        expected = doppler_class(
            float(doppler_hz), classifier.lag_samples, config.fs_hz
        )
        assert row == classifier.row_of(2, expected)


# --------------------------------------------------------------------------
# acquisition, noise free
# --------------------------------------------------------------------------


@pytest.mark.parametrize("code_phase", [0, 55, 150])
def test_a_planted_satellite_comes_back_at_its_own_code_phase(code_phase):
    config = make_config()
    classifier = DifferentialHdCamClassifier(config)
    signal = make_signal(config, 2, doppler_hz=500.0, code_phase=code_phase)
    reported = [r for r in classifier.acquire(signal) if r.prn == 2]
    assert [r.code_phase for r in reported] == [code_phase]


@pytest.mark.parametrize("carrier_phase", [0.0, 1.0, 2.5, 4.0])
def test_the_answer_does_not_move_with_the_carrier_phase(carrier_phase):
    config = make_config()
    classifier = DifferentialHdCamClassifier(config)
    signal = make_signal(config, 2, doppler_hz=0.0, code_phase=31, carrier_phase=carrier_phase)
    reported = [r for r in classifier.acquire(signal) if r.prn == 2]
    assert [r.code_phase for r in reported] == [31]


def test_a_silent_record_acquires_nothing():
    config = make_config()
    classifier = DifferentialHdCamClassifier(config)
    assert classifier.acquire(np.zeros(config.samples_per_acquisition)) == []


def test_the_cam_path_and_the_table_path_agree():
    config = make_config()
    signal = make_signal(config, 3, doppler_hz=-500.0, code_phase=88)
    through_cam = DifferentialHdCamClassifier(config, search_mode="cam")
    through_table = DifferentialHdCamClassifier(config, search_mode="table")
    assert through_cam.acquire(signal) == through_table.acquire(signal)
