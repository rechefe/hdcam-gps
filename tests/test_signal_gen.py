"""Directed tests for the labelled signal generator.

The generator's contract is that a scenario's label is its input, so most of
these check that what comes out is exactly what was asked for: the right power,
the right satellites, the right truth, and the same record for the same seed.
"""

import dataclasses
import numpy as np
import pytest

from hdcam_gps.acq_base import AcqConfig, PrnResult
from hdcam_gps.ca_code import sampled_ca_code
from hdcam_gps.fft_acq import FftAcqClassifier
from hdcam_gps.fft_acq import FftAcqClassifier as _Reference
from hdcam_gps.gps_sdr_sim import sources_checked_out
from hdcam_gps.signal_gen import (
    NAV_BIT_PERIOD_S,
    SatelliteTruth,
    Scenario,
    amplitude_for_cn0,
    generate_from_sim,
    generate_synthetic,
    nav_data_bits,
    random_scenario,
    satellite_signal,
    snap_to_doppler_grid,
)


def make_config(**overrides) -> AcqConfig:
    kwargs = {
        "fs_hz": 1.023e6,
        "prn_list": (1, 7, 19),
        "doppler_min_hz": -2000.0,
        "doppler_max_hz": 2000.0,
        "doppler_step_hz": 500.0,
        "n_codes": 4,
    }
    kwargs.update(overrides)
    return AcqConfig(**kwargs)


# --------------------------------------------------------------------------
# amplitude_for_cn0
# --------------------------------------------------------------------------


def test_amplitude_realizes_the_wanted_cn0():
    fs_hz = 4e6
    amplitude = amplitude_for_cn0(45.0, fs_hz)
    # C / N0 with C = amplitude squared and N0 = 1 / fs_hz
    cn0_dbhz = 10 * np.log10(amplitude**2 * fs_hz)
    assert cn0_dbhz == pytest.approx(45.0)


def test_amplitude_grows_with_cn0():
    fs_hz = 4e6
    assert amplitude_for_cn0(50.0, fs_hz) > amplitude_for_cn0(45.0, fs_hz)


def test_ten_db_more_is_ten_times_the_power():
    weak = amplitude_for_cn0(35.0, 4e6)
    strong = amplitude_for_cn0(45.0, 4e6)
    assert strong**2 / weak**2 == pytest.approx(10.0)


def test_amplitude_shrinks_as_the_bandwidth_grows():
    # The same C / N0 spread over more bandwidth means more noise, so a weaker
    # signal relative to the unit power noise this generator uses.
    assert amplitude_for_cn0(45.0, 8e6) < amplitude_for_cn0(45.0, 4e6)


# --------------------------------------------------------------------------
# snap_to_doppler_grid
# --------------------------------------------------------------------------


def test_snapping_leaves_a_grid_value_alone():
    config = make_config()
    for doppler_hz in config.doppler_grid_hz:
        assert snap_to_doppler_grid(config, float(doppler_hz)) == doppler_hz


@pytest.mark.parametrize(
    ("doppler_hz", "expected"), [(510.0, 500.0), (740.0, 500.0), (760.0, 1000.0)]
)
def test_snapping_picks_the_nearest_bin(doppler_hz, expected):
    assert snap_to_doppler_grid(make_config(), doppler_hz) == expected


def test_snapping_clamps_outside_the_grid():
    config = make_config()
    assert snap_to_doppler_grid(config, 99999.0) == 2000.0
    assert snap_to_doppler_grid(config, -99999.0) == -2000.0


# --------------------------------------------------------------------------
# satellite_signal
# --------------------------------------------------------------------------


def test_satellite_signal_has_the_power_its_cn0_asks_for():
    config = make_config()
    satellite = SatelliteTruth(prn=7, doppler_hz=500.0, code_phase=0, cn0_dbhz=45.0)
    signal = satellite_signal(config, satellite, nav_data=False, rng=None)

    expected = amplitude_for_cn0(45.0, config.fs_hz) ** 2
    assert np.mean(np.abs(signal) ** 2) == pytest.approx(expected)


def test_satellite_signal_length_and_dtype():
    config = make_config()
    signal = satellite_signal(
        config, SatelliteTruth(1, 0.0, 0), nav_data=False, rng=None
    )
    assert len(signal) == config.samples_per_acquisition
    assert np.iscomplexobj(signal)


def test_satellite_signal_carries_the_requested_code_and_phase():
    config = make_config()
    satellite = SatelliteTruth(prn=19, doppler_hz=0.0, code_phase=200, cn0_dbhz=45.0)
    signal = satellite_signal(config, satellite, nav_data=False, rng=None)

    expected = np.roll(
        np.tile(sampled_ca_code(19, config.fs_hz), config.n_codes), 200
    ) * amplitude_for_cn0(45.0, config.fs_hz)
    assert np.allclose(np.real(signal), expected)


@pytest.mark.parametrize("bad_phase", [-1, 1023, 5000])
def test_satellite_signal_rejects_a_code_phase_outside_one_period(bad_phase):
    config = make_config()
    with pytest.raises(AssertionError):
        satellite_signal(
            config, SatelliteTruth(1, 0.0, bad_phase), nav_data=False, rng=None
        )


# --------------------------------------------------------------------------
# nav_data_bits
# --------------------------------------------------------------------------


def test_nav_data_bits_are_plus_or_minus_one_of_the_right_length():
    rng = np.random.default_rng(0)
    bits = nav_data_bits(5000, 1.023e6, rng)
    assert len(bits) == 5000
    assert set(np.unique(bits)) <= {-1.0, 1.0}


def test_nav_data_bits_are_held_for_twenty_code_periods():
    fs_hz = 1.023e6
    samples_per_bit = int(fs_hz * NAV_BIT_PERIOD_S)
    bits = nav_data_bits(3 * samples_per_bit, fs_hz, np.random.default_rng(1))
    for start in range(0, 3 * samples_per_bit, samples_per_bit):
        chunk = bits[start : start + samples_per_bit]
        assert len(np.unique(chunk)) == 1  # constant within a bit


def test_nav_data_does_not_change_the_signal_power():
    config = make_config()
    satellite = SatelliteTruth(prn=1, doppler_hz=0.0, code_phase=0, cn0_dbhz=45.0)
    plain = satellite_signal(config, satellite, False, np.random.default_rng(0))
    modulated = satellite_signal(config, satellite, True, np.random.default_rng(0))
    assert np.mean(np.abs(modulated) ** 2) == pytest.approx(
        np.mean(np.abs(plain) ** 2)
    )


# --------------------------------------------------------------------------
# generate
# --------------------------------------------------------------------------


def test_generate_returns_a_record_of_the_configured_length():
    config = make_config()
    scenario = generate_synthetic(config, [SatelliteTruth(1, 0.0, 10)], seed=0)
    assert len(scenario.samples) == config.samples_per_acquisition
    assert np.iscomplexobj(scenario.samples)
    assert scenario.config is config


def test_the_truth_is_what_was_put_in():
    satellites = [SatelliteTruth(7, 500.0, 42, 40.0), SatelliteTruth(1, -500.0, 900)]
    scenario = generate_synthetic(make_config(), satellites, seed=0)
    assert scenario.truth == tuple(satellites)


def test_a_noiseless_empty_scenario_is_silent():
    scenario = generate_synthetic(make_config(), [], add_noise=False)
    assert not scenario.samples.any()
    assert scenario.truth == ()
    assert scenario.noise_sigma == 0.0


def test_a_noiseless_record_is_exactly_the_sum_of_its_satellites():
    config = make_config()
    satellites = [SatelliteTruth(1, 0.0, 5), SatelliteTruth(19, 1000.0, 700)]
    scenario = generate_synthetic(config, satellites, add_noise=False)

    expected = sum(
        satellite_signal(config, satellite, False, None) for satellite in satellites
    )
    assert np.allclose(scenario.samples, expected)


def test_noise_has_unit_power():
    config = make_config(n_codes=20)
    scenario = generate_synthetic(config, [], add_noise=True, seed=3)
    assert scenario.noise_sigma == 1.0
    assert np.mean(np.abs(scenario.samples) ** 2) == pytest.approx(1.0, rel=0.02)


def test_the_record_power_is_the_noise_plus_the_signals():
    config = make_config(n_codes=20)
    satellite = SatelliteTruth(prn=1, doppler_hz=0.0, code_phase=0, cn0_dbhz=60.0)
    scenario = generate_synthetic(config, [satellite], add_noise=True, seed=4)

    expected = 1.0 + amplitude_for_cn0(60.0, config.fs_hz) ** 2
    assert np.mean(np.abs(scenario.samples) ** 2) == pytest.approx(expected, rel=0.02)


def test_the_same_seed_gives_the_same_record():
    config = make_config()
    satellites = [SatelliteTruth(1, 0.0, 10)]
    first = generate_synthetic(config, satellites, seed=7)
    second = generate_synthetic(config, satellites, seed=7)
    assert np.array_equal(first.samples, second.samples)


def test_different_seeds_give_different_noise():
    config = make_config()
    first = generate_synthetic(config, [], seed=1)
    second = generate_synthetic(config, [], seed=2)
    assert not np.array_equal(first.samples, second.samples)


def test_a_stronger_satellite_dominates_a_weaker_one():
    config = make_config()
    strong = SatelliteTruth(1, 0.0, 0, cn0_dbhz=50.0)
    weak = SatelliteTruth(19, 0.0, 0, cn0_dbhz=40.0)
    both = generate_synthetic(config, [strong, weak], add_noise=False)
    only_strong = generate_synthetic(config, [strong], add_noise=False)

    assert np.mean(np.abs(both.samples) ** 2) > np.mean(
        np.abs(only_strong.samples) ** 2
    )


# --------------------------------------------------------------------------
# Scenario.expected_results
# --------------------------------------------------------------------------


def test_expected_results_mirror_the_placed_satellites():
    config = make_config()
    scenario = generate_synthetic(config, [SatelliteTruth(7, 500.0, 123)], add_noise=False)
    assert scenario.expected_results() == [
        PrnResult(prn=7, doppler_hz=500.0, code_phase=123)
    ]


def test_expected_results_follow_the_configuration_order():
    config = make_config()  # prn_list is (1, 7, 19)
    satellites = [SatelliteTruth(19, 0.0, 3), SatelliteTruth(1, 500.0, 8)]
    scenario = generate_synthetic(config, satellites, add_noise=False)

    assert [result.prn for result in scenario.expected_results()] == [1, 19]


def test_expected_results_snap_an_off_grid_doppler():
    config = make_config()
    scenario = generate_synthetic(config, [SatelliteTruth(1, 1490.0, 4)], add_noise=False)
    assert scenario.expected_results()[0].doppler_hz == 1500.0


def test_expected_results_drop_a_prn_the_configuration_does_not_search():
    config = make_config(prn_list=(1, 7))
    satellites = [SatelliteTruth(1, 0.0, 6), SatelliteTruth(19, 0.0, 9)]
    scenario = generate_synthetic(config, satellites, add_noise=False)
    assert [result.prn for result in scenario.expected_results()] == [1]


def test_scenario_and_truth_are_frozen():
    scenario = generate_synthetic(make_config(), [SatelliteTruth(1, 0.0, 0)], add_noise=False)
    with pytest.raises(dataclasses.FrozenInstanceError):
        scenario.config = None
    with pytest.raises(dataclasses.FrozenInstanceError):
        scenario.truth[0].prn = 2


# --------------------------------------------------------------------------
# random_scenario
# --------------------------------------------------------------------------


def test_random_scenario_places_the_requested_number_of_distinct_prns():
    config = make_config()
    scenario = random_scenario(config, n_satellites=2, seed=0)
    prns = [satellite.prn for satellite in scenario.truth]
    assert len(prns) == 2
    assert len(set(prns)) == 2
    assert set(prns) <= set(config.prn_list)


def test_random_scenario_stays_inside_the_search_space():
    config = make_config()
    scenario = random_scenario(config, n_satellites=3, seed=1)
    for satellite in scenario.truth:
        assert satellite.doppler_hz in config.doppler_grid_hz
        assert 0 <= satellite.code_phase < config.samples_per_code


def test_random_scenario_is_reproducible():
    config = make_config()
    first = random_scenario(config, 2, seed=5)
    second = random_scenario(config, 2, seed=5)
    assert first.truth == second.truth
    assert np.array_equal(first.samples, second.samples)


def test_random_scenario_cannot_place_more_than_the_configuration_lists():
    with pytest.raises(AssertionError):
        random_scenario(make_config(), n_satellites=4)


# --------------------------------------------------------------------------
# the point of the whole module: labels a classifier can be scored against
# --------------------------------------------------------------------------


def test_the_reference_classifier_recovers_a_noiseless_scenario():
    config = make_config()
    scenario = generate_synthetic(
        config,
        [SatelliteTruth(7, 1000.0, 456), SatelliteTruth(1, -1500.0, 12)],
        add_noise=False,
    )
    assert FftAcqClassifier(config).acquire(scenario.samples) == (
        scenario.expected_results()
    )


def test_the_reference_classifier_recovers_a_strong_noisy_scenario():
    config = make_config(fs_hz=4e6, n_codes=10)
    scenario = generate_synthetic(
        config, [SatelliteTruth(19, 500.0, 3210, cn0_dbhz=48.0)], seed=11
    )
    assert FftAcqClassifier(config).acquire(scenario.samples) == (
        scenario.expected_results()
    )


def test_a_random_scenario_is_solvable_by_the_reference_classifier():
    config = make_config(fs_hz=4e6, n_codes=10)
    scenario = random_scenario(config, n_satellites=2, cn0_dbhz=48.0, seed=2)
    assert FftAcqClassifier(config).acquire(scenario.samples) == (
        scenario.expected_results()
    )


def test_a_noise_only_record_yields_no_detections():
    config = make_config(fs_hz=4e6, n_codes=10)
    scenario = generate_synthetic(config, [], add_noise=True, seed=9)
    assert scenario.expected_results() == []
    assert FftAcqClassifier(config).acquire(scenario.samples) == []


# --------------------------------------------------------------------------
# the two backends are interchangeable
# --------------------------------------------------------------------------


needs_simulator = pytest.mark.skipif(
    not sources_checked_out(), reason="the gps-sdr-sim submodule is not checked out"
)


def sim_config() -> AcqConfig:
    return AcqConfig(
        fs_hz=4e6,
        prn_list=tuple(range(1, 33)),
        doppler_min_hz=-5000.0,
        doppler_max_hz=5000.0,
        doppler_step_hz=500.0,
        n_codes=10,
    )


@needs_simulator
def test_a_simulated_scenario_has_the_same_shape_as_a_synthetic_one():
    config = sim_config()
    scenario = generate_from_sim(config, add_noise=False)

    assert isinstance(scenario, Scenario)
    assert scenario.config is config
    assert len(scenario.samples) == config.samples_per_acquisition
    assert np.iscomplexobj(scenario.samples)
    assert len(scenario.truth) >= 4  # a normal sky
    for satellite in scenario.truth:
        assert 1 <= satellite.prn <= 32
        assert abs(satellite.doppler_hz) <= 5000.0
        assert 0 <= satellite.code_phase < config.samples_per_code


@needs_simulator
def test_a_simulated_scenario_scores_the_reference_classifier():
    # The truth is the simulator's own reported channel state, so scoring an
    # acquisition against it is fair.
    config = sim_config()
    scenario = generate_from_sim(config, cn0_dbhz=48.0, seed=0)
    results = _Reference(config).acquire(scenario.samples)
    assert scenario.matches(results, code_phase_tolerance=1)


@needs_simulator
def test_a_simulated_scenario_is_scaled_onto_the_synthetic_convention():
    config = sim_config()
    scenario = generate_from_sim(config, cn0_dbhz=45.0, add_noise=False)

    expected = len(scenario.truth) * amplitude_for_cn0(45.0, config.fs_hz) ** 2
    assert np.mean(np.abs(scenario.samples) ** 2) == pytest.approx(expected, rel=0.2)




# --------------------------------------------------------------------------
# Scenario.matches
# --------------------------------------------------------------------------


def labelled_scenario() -> Scenario:
    return generate_synthetic(
        make_config(), [SatelliteTruth(1, 500.0, 100)], add_noise=False
    )


def test_matches_accepts_the_exact_answer():
    scenario = labelled_scenario()
    assert scenario.matches(scenario.expected_results())


def test_matches_rejects_a_missing_or_extra_prn():
    scenario = labelled_scenario()
    assert not scenario.matches([])
    assert not scenario.matches(
        scenario.expected_results() + [PrnResult(7, 0.0, 0)]
    )


def test_matches_rejects_a_wrong_doppler_bin():
    scenario = labelled_scenario()
    assert not scenario.matches([PrnResult(1, 1000.0, 100)])


def test_matches_honours_the_code_phase_tolerance():
    scenario = labelled_scenario()
    off_by_one = [PrnResult(1, 500.0, 101)]
    assert not scenario.matches(off_by_one)
    assert scenario.matches(off_by_one, code_phase_tolerance=1)
    assert not scenario.matches([PrnResult(1, 500.0, 103)], code_phase_tolerance=1)


def test_matches_tolerance_wraps_around_the_code_period():
    config = make_config()
    scenario = generate_synthetic(
        config, [SatelliteTruth(1, 0.0, 0)], add_noise=False
    )
    last = config.samples_per_code - 1  # one sample before zero, circularly
    assert scenario.matches([PrnResult(1, 0.0, last)], code_phase_tolerance=1)
