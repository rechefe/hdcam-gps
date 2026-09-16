"""Directed tests for the scenario bank.

The bank exists to make a comparison a comparison, so these check the two things
that would quietly break one: that every classifier is handed the same samples
array, and that no calibration sky reaches the evaluation set. The synthetic
backend carries most of them, because it needs no simulator.

What is not covered here: whether the skies are a representative sample of the
constellation's geometry. They are 17 minutes apart and that is all.
"""

import numpy as np
import pytest

from hdcam_gps.acq_base import AcqConfig
from hdcam_gps.evaluate import EvalConfig, sky_start_time
from hdcam_gps.gps_sdr_sim import sources_checked_out
from hdcam_gps.scenarios import ScenarioBank

needs_simulator = pytest.mark.skipif(
    not sources_checked_out(), reason="the gps-sdr-sim submodule is not checked out"
)

FS_HZ = 204.6e3  # 204 samples per code period, enough to stay correct and cheap


def make_config(**overrides) -> AcqConfig:
    kwargs = {
        "fs_hz": FS_HZ,
        "prn_list": (1, 2, 3, 4),
        "doppler_min_hz": -500.0,
        "doppler_max_hz": 500.0,
        "doppler_step_hz": 500.0,
        "n_codes": 3,
    }
    kwargs.update(overrides)
    return AcqConfig(**kwargs)


def make_sweep(**overrides) -> EvalConfig:
    kwargs = {
        "n_scenarios": 4,
        "cn0_dbhz": (48.0, 42.0),
        "backend": "synthetic",
        "n_satellites": 2,
        "progress": False,
    }
    kwargs.update(overrides)
    return EvalConfig(**kwargs)


def a_bank(**overrides) -> ScenarioBank:
    return ScenarioBank.build(make_config(), make_sweep(**overrides))


# --------------------------------------------------------------------------
# build - one record per scaling per sky
# --------------------------------------------------------------------------


def test_the_bank_holds_a_record_for_every_scaling_and_sky():
    bank = a_bank()
    assert len(bank.records) == 2 * 4
    assert bank.indices == (0, 1, 2, 3)
    assert bank.cn0_dbhz == (48.0, 42.0)


def test_a_record_is_built_to_the_configuration_it_was_asked_for():
    config = make_config()
    scenario = ScenarioBank.build(config, make_sweep()).get(48.0, 0)
    assert scenario.config is config
    assert len(scenario.samples) == config.samples_per_acquisition


def test_the_skies_can_be_chosen_rather_than_counted():
    bank = ScenarioBank.build(make_config(), make_sweep(), indices=range(10, 13))
    assert bank.indices == (10, 11, 12)


def test_a_bank_needs_a_sky():
    with pytest.raises(AssertionError):
        ScenarioBank.build(make_config(), make_sweep(), indices=[])


# --------------------------------------------------------------------------
# get - the same array, not the same distribution
# --------------------------------------------------------------------------


def test_two_readers_are_handed_the_identical_samples_array():
    # Five families scored on five independent draws is five experiments. The
    # bank hands out the same object so a difference is the design, not the dice.
    bank = a_bank()
    assert bank.get(48.0, 1).samples is bank.get(48.0, 1).samples


def test_a_scaling_the_bank_does_not_hold_is_refused():
    with pytest.raises(AssertionError):
        a_bank().get(36.0, 0)


def test_a_sky_the_bank_does_not_hold_is_refused():
    with pytest.raises(AssertionError):
        a_bank().get(48.0, 99)


def test_the_scalings_differ_in_the_power_they_put_in():
    bank = a_bank()
    loud = float(np.mean(np.abs(bank.get(48.0, 0).samples) ** 2))
    quiet = float(np.mean(np.abs(bank.get(42.0, 0).samples) ** 2))
    assert loud > quiet


# --------------------------------------------------------------------------
# split - the calibration skies never reach the evaluation set
# --------------------------------------------------------------------------


def test_a_split_divides_the_skies_and_shares_none():
    calibration, evaluation = a_bank().split(1)
    assert calibration.indices == (0,)
    assert evaluation.indices == (1, 2, 3)
    assert not set(calibration.start_times) & set(evaluation.start_times)


def test_a_split_keeps_every_scaling_on_both_sides():
    calibration, evaluation = a_bank().split(2)
    assert calibration.cn0_dbhz == evaluation.cn0_dbhz
    assert len(calibration.records) == 2 * 2
    assert len(evaluation.records) == 2 * 2


def test_a_split_shares_the_records_it_keeps():
    bank = a_bank()
    _, evaluation = bank.split(1)
    assert evaluation.get(48.0, 2).samples is bank.get(48.0, 2).samples


@pytest.mark.parametrize("bad", [0, 4, 9])
def test_a_split_that_would_empty_a_side_is_refused(bad):
    with pytest.raises(AssertionError):
        a_bank().split(bad)


def test_every_sky_has_its_own_start_time():
    bank = a_bank()
    assert len(set(bank.start_times)) == len(bank.indices)
    assert bank.start_times[0] == sky_start_time(0)


# --------------------------------------------------------------------------
# the pooled per satellite C/N0, which is the study's x axis
# --------------------------------------------------------------------------


def test_the_pooled_cn0_has_one_entry_per_record_and_satellite():
    bank = a_bank()
    assert bank.satellite_cn0_dbhz().size == 2 * 4 * 2


def test_a_synthetic_satellite_sits_exactly_at_the_scaling():
    bank = a_bank()
    assert all(
        satellite.cn0_dbhz == 48.0 for satellite in bank.get(48.0, 0).truth
    )


# --------------------------------------------------------------------------
# the simulator backend
# --------------------------------------------------------------------------


def sim_config() -> AcqConfig:
    return AcqConfig(
        fs_hz=2.6e6,
        prn_list=tuple(range(1, 33)),
        doppler_min_hz=-5000.0,
        doppler_max_hz=5000.0,
        doppler_step_hz=500.0,
        n_codes=2,
    )


@needs_simulator
def test_one_simulated_sky_serves_every_scaling():
    # A record depends on the place and the time, so the scalings differ only in
    # what was done to it afterwards. The truth is therefore identical but for
    # the C/N0 each satellite ends up at.
    bank = ScenarioBank.build(
        sim_config(),
        make_sweep(n_scenarios=1, backend="simulator", cn0_dbhz=(48.0, 42.0)),
    )
    loud, quiet = bank.get(48.0, 0), bank.get(42.0, 0)
    assert [s.prn for s in loud.truth] == [s.prn for s in quiet.truth]
    assert [s.code_phase for s in loud.truth] == [s.code_phase for s in quiet.truth]


@needs_simulator
def test_a_simulated_sky_spreads_its_satellites_over_several_dB():
    # Path loss and the receiver antenna pattern, so the scaling a record was
    # built at is an average and not any one satellite's C/N0.
    bank = ScenarioBank.build(
        sim_config(), make_sweep(n_scenarios=1, backend="simulator", cn0_dbhz=(45.0,))
    )
    values = np.array([s.cn0_dbhz for s in bank.get(45.0, 0).truth])
    assert values.size >= 4
    assert 2.0 < values.max() - values.min() < 20.0


@needs_simulator
def test_rescaling_a_sky_moves_every_satellite_by_the_same_amount():
    bank = ScenarioBank.build(
        sim_config(),
        make_sweep(n_scenarios=1, backend="simulator", cn0_dbhz=(48.0, 42.0)),
    )
    loud = np.array([s.cn0_dbhz for s in bank.get(48.0, 0).truth])
    quiet = np.array([s.cn0_dbhz for s in bank.get(42.0, 0).truth])
    assert (loud - quiet) == pytest.approx(6.0, abs=0.2)
