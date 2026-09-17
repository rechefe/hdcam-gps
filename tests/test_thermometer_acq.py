"""Directed tests for the thermometer family.

The unary code is checked against hand computable answers, the bit domain
rotation against the sample domain one it stands in for, and the gain against
the record it has to come from. The sampling rate is deliberately low, so a
whole acquisition run is cheap.

The acquisition tests ask that the planted satellite comes back with the right
numbers, not that nothing else does; at an uncalibrated threshold this family
also reports PRNs the record does not contain.

What is not covered here: whether the 1.4 dB this design is built for is
actually there. That is a measurement, and notebooks/family_screen.ipynb owns it.
"""

import numpy as np
import pytest

from hdcam_gps.acq_base import AcqConfig, PrnResult
from hdcam_gps.ca_code import sampled_ca_code
from hdcam_gps.cam_acq import QUERY_ROTATIONS
from hdcam_gps.hdcam_acq import OneBitHdCamClassifier
from hdcam_gps.thermometer_acq import (
    DEFAULT_LEVELS,
    ThermometerHdCamClassifier,
    component_scale,
    level_boundaries,
    levels_of,
    negate_thermometer,
    quantize_thermometer,
    rotate_thermometer,
    thermometer,
)

FS_HZ = 204.6e3  # 204 samples per code period, enough to stay correct and cheap
SAMPLES_PER_CODE = 204
OFF_AXIS_PHASE = 0.7  # So a planted replica does not land on an axis


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
    carrier_phase: float = OFF_AXIS_PHASE,
) -> np.ndarray:
    """A noise free replica at a known Doppler, code phase and carrier phase."""
    code = np.tile(sampled_ca_code(prn, config.fs_hz), config.n_codes)
    code = np.roll(code, code_phase).astype(complex)
    time_s = np.arange(config.samples_per_acquisition) / config.fs_hz
    return code * np.exp(2j * np.pi * doppler_hz * time_s + 1j * carrier_phase)


# --------------------------------------------------------------------------
# the unary code - Hamming distance is level difference
# --------------------------------------------------------------------------


def test_a_level_is_that_many_ones():
    assert thermometer(np.array([0, 1, 2, 3]), 4).reshape(4, 3).sum(axis=1).tolist() == (
        [0, 1, 2, 3]
    )


def test_the_code_and_the_levels_are_inverse():
    levels = np.array([3, 0, 2, 1, 1])
    assert levels_of(thermometer(levels, 4), 4).tolist() == levels.tolist()


@pytest.mark.parametrize("first", range(DEFAULT_LEVELS))
@pytest.mark.parametrize("second", range(DEFAULT_LEVELS))
def test_hamming_distance_is_the_difference_between_the_levels(first, second):
    # This is the whole reason for the code: a CAM's distance becomes an L1
    # distance, so the comparison it already does is a soft correlation.
    distance = np.count_nonzero(
        thermometer(np.array([first]), 4) != thermometer(np.array([second]), 4)
    )
    assert distance == abs(first - second)


def test_negating_reflects_every_level_about_the_middle():
    levels = np.array([0, 1, 2, 3])
    reflected = levels_of(negate_thermometer(thermometer(levels, 4), 4), 4)
    assert reflected.tolist() == [3, 2, 1, 0]


def test_negating_twice_is_the_identity():
    bits = thermometer(np.array([0, 2, 3, 1]), 4)
    assert np.array_equal(
        negate_thermometer(negate_thermometer(bits, 4), 4), bits
    )


def test_the_boundaries_are_symmetric_about_zero():
    boundaries = level_boundaries(4, 1.0)
    assert boundaries.tolist() == [-1.0, 0.0, 1.0]
    assert level_boundaries(2, 1.0).tolist() == [0.0]


# --------------------------------------------------------------------------
# the quantiser and its gain
# --------------------------------------------------------------------------


def test_quantizing_gives_two_blocks_of_unary_words():
    samples = np.array([2 + 0j, 0 - 2j])
    bits = quantize_thermometer(samples, 4, scale=1.0, step=1.0)
    assert len(bits) == 2 * 3 * 2
    in_phase, quadrature = np.split(bits, 2)
    assert levels_of(in_phase, 4).tolist() == [3, 1]
    assert levels_of(quadrature, 4).tolist() == [1, 0]


@pytest.mark.parametrize("rotations", range(QUERY_ROTATIONS))
def test_the_bit_rotation_is_the_sample_rotation(rotations):
    # The query side has to stay bit operations, so this is the claim that
    # turning the carrier needs no multiplier.
    rng = np.random.default_rng(0)
    samples = rng.normal(size=16) + 1j * rng.normal(size=16)
    scale = component_scale(samples)
    assert np.array_equal(
        rotate_thermometer(quantize_thermometer(samples, 4, scale), rotations, 4),
        quantize_thermometer(samples * 1j**rotations, 4, scale),
    )


def test_the_gain_is_the_record_s_own_component_scale():
    config = make_config()
    classifier = ThermometerHdCamClassifier(config)
    samples = make_signal(config, 1)
    assert classifier.gain(samples) == pytest.approx(component_scale(samples))


def test_the_gain_tracks_the_record_rather_than_the_generator():
    # Scenario.noise_sigma is the generator's parameter. A receiver has an AGC,
    # so scaling the record must leave the quantised query unchanged.
    config = make_config()
    classifier = ThermometerHdCamClassifier(config)
    samples = make_signal(config, 2, doppler_hz=500.0, code_phase=9)
    quiet = classifier.query_variants(samples, 40)[0]
    loud = ThermometerHdCamClassifier(config).query_variants(17.0 * samples, 40)[0]
    assert np.array_equal(quiet, loud)


def test_a_silent_record_still_quantizes():
    config = make_config()
    classifier = ThermometerHdCamClassifier(config)
    samples = np.zeros(config.samples_per_acquisition, dtype=complex)
    assert len(classifier.query_variants(samples, 0)[0]) == classifier.n_columns


# --------------------------------------------------------------------------
# the shape the extra bits cost
# --------------------------------------------------------------------------


def test_four_levels_take_three_bits_per_component():
    classifier = ThermometerHdCamClassifier(make_config())
    assert classifier.n_columns == 2 * 3 * SAMPLES_PER_CODE


def test_the_row_is_three_times_the_one_bit_row():
    # Three times the area and three times the energy, for the 1.4 dB.
    config = make_config()
    assert ThermometerHdCamClassifier(config).n_columns == (
        3 * OneBitHdCamClassifier(config).n_columns
    )


def test_two_levels_is_one_bit_per_component():
    config = make_config()
    classifier = ThermometerHdCamClassifier(config, n_levels=2)
    assert classifier.n_columns == OneBitHdCamClassifier(config).n_columns


def test_the_rows_are_laid_out_as_the_baseline_s_are():
    config = make_config()
    classifier = ThermometerHdCamClassifier(config)
    assert classifier.n_rows == OneBitHdCamClassifier(config).n_rows
    rows = classifier.row_index()
    for prn in config.prn_list:
        for doppler_bin in range(classifier.n_doppler_bins):
            row = classifier.row_of(prn, doppler_bin)
            assert rows.prn[row] == prn
            assert rows.doppler_bin[row] == doppler_bin


@pytest.mark.parametrize("bad", [1, 0, -1])
def test_fewer_than_two_levels_is_rejected(bad):
    with pytest.raises(AssertionError):
        ThermometerHdCamClassifier(make_config(), n_levels=bad)


def test_a_single_code_period_config_is_rejected():
    with pytest.raises(AssertionError):
        ThermometerHdCamClassifier(make_config(n_codes=1))


# --------------------------------------------------------------------------
# the chance floor, which is not the binomial one
# --------------------------------------------------------------------------


def test_the_measured_floor_is_well_below_half_the_word():
    # Measured at 39 % of the width against the 50 % a binomial with p = 0.5
    # predicts, because the bits of one unary word are not independent.
    classifier = ThermometerHdCamClassifier(make_config())
    mean, deviation = classifier.chance_floor(n_draws=128)
    assert mean < 0.45 * classifier.n_columns
    assert deviation > 0


def test_the_default_threshold_sits_below_the_measured_floor():
    # The 1 bit formula would put it above, and every PRN would fire.
    classifier = ThermometerHdCamClassifier(make_config())
    mean, _ = classifier.chance_floor(n_draws=128)
    assert 0 < classifier.hd_threshold < mean


def test_an_explicit_threshold_is_used_as_given():
    classifier = ThermometerHdCamClassifier(make_config(), hd_threshold=123)
    assert classifier.hd_threshold == 123


# --------------------------------------------------------------------------
# the codebook and acquisition
# --------------------------------------------------------------------------


def test_a_codebook_row_is_the_quantized_replica():
    config = make_config()
    classifier = ThermometerHdCamClassifier(config)
    prn, doppler_bin, phase = 2, 2, 1
    time_s = np.arange(config.samples_per_code) / config.fs_hz
    replica = (
        sampled_ca_code(prn, config.fs_hz)
        * np.exp(2j * np.pi * config.doppler_grid_hz[doppler_bin] * time_s)
        * np.exp(2j * np.pi * (phase + 0.5) / classifier.n_phases)
    )
    expected = quantize_thermometer(
        replica, classifier.n_levels, component_scale(replica), classifier.level_step
    )
    row = classifier.row_of(prn, doppler_bin, phase)
    assert np.array_equal(classifier.build_codebook()[row], expected)


@pytest.mark.parametrize("doppler_hz", [-500.0, 0.0, 500.0])
def test_a_planted_satellite_comes_back_at_its_own_cfo(doppler_hz):
    config = make_config()
    classifier = ThermometerHdCamClassifier(config)
    signal = make_signal(config, 2, doppler_hz=doppler_hz, code_phase=55)
    assert (
        PrnResult(prn=2, doppler_hz=doppler_hz, code_phase=55)
        in classifier.acquire(signal)
    )


@pytest.mark.parametrize("code_phase", [0, 17, 150])
def test_the_code_phase_comes_back_from_the_window_offset(code_phase):
    config = make_config()
    classifier = ThermometerHdCamClassifier(config)
    signal = make_signal(config, 1, doppler_hz=500.0, code_phase=code_phase)
    assert (
        PrnResult(prn=1, doppler_hz=500.0, code_phase=code_phase)
        in classifier.acquire(signal)
    )


def test_a_noise_record_acquires_nothing():
    config = make_config()
    classifier = ThermometerHdCamClassifier(config)
    rng = np.random.default_rng(0)
    n_samples = config.samples_per_acquisition
    noise = rng.normal(scale=np.sqrt(0.5), size=n_samples) + 1j * rng.normal(
        scale=np.sqrt(0.5), size=n_samples
    )
    assert classifier.acquire(noise) == []


def test_silence_is_not_a_meaningful_input_to_a_quantiser_with_a_gain():
    # An AGC has nothing to normalise a silent record against, so every sample
    # lands on one level and the query is a constant word. The 1 bit family gets
    # an all zero query there and rejects it; this one can report a PRN. Noise,
    # not silence, is the empty case that means anything here.
    config = make_config()
    classifier = ThermometerHdCamClassifier(config)
    silence = np.zeros(config.samples_per_acquisition, dtype=complex)
    bits = classifier.query_variants(silence, 0)[0]
    assert len(set(levels_of(bits, classifier.n_levels).tolist())) == 1


def test_the_cam_path_and_the_table_path_agree():
    config = make_config()
    signal = make_signal(config, 3, doppler_hz=-500.0, code_phase=88)
    through_cam = ThermometerHdCamClassifier(config, search_mode="cam")
    through_table = ThermometerHdCamClassifier(config, search_mode="table")
    assert through_cam.acquire(signal) == through_table.acquire(signal)
