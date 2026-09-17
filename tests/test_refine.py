"""Directed tests for the Doppler second stage.

Two families in the study name a PRN and a code phase and cannot name a Doppler.
These check that the stage which finishes their answer does so on the 1 bit
stream, that it leaves the PRN and code phase alone, and that what it costs is
counted rather than modelled.

The sampling rate here is the study's own, because a coherent millisecond has a
frequency resolution of 1 kHz whatever the rate and the grid being resolved is
500 Hz. The records are four code periods long, so the sweep is still cheap.
"""

import numpy as np
import pytest

from hdcam_gps.acq_base import AcqConfig, PrnResult
from hdcam_gps.ca_code import sampled_ca_code
from hdcam_gps.diff_acq import DifferentialHdCamClassifier
from hdcam_gps.refine import (
    DopplerRefiner,
    RefinedClassifier,
    RefinerCost,
)
from hdcam_gps.segmented_acq import SegmentedHdCamClassifier

FS_HZ = 1.023e6  # One sample per chip, as the study runs
OFF_AXIS_PHASE = 0.7


def make_config(**overrides) -> AcqConfig:
    kwargs = {
        "fs_hz": FS_HZ,
        "prn_list": (1, 2, 3),
        "doppler_min_hz": -2000.0,
        "doppler_max_hz": 2000.0,
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
    cn0_dbhz: float | None = None,
    seed: int = 0,
) -> np.ndarray:
    """A replica at a known Doppler, optionally buried in unit power noise."""
    n_samples = config.samples_per_acquisition
    code = np.tile(sampled_ca_code(prn, config.fs_hz), config.n_codes)
    code = np.roll(code, code_phase).astype(complex)
    time_s = np.arange(n_samples) / config.fs_hz
    signal = code * np.exp(2j * np.pi * doppler_hz * time_s + 1j * carrier_phase)
    if cn0_dbhz is None:
        return signal
    rng = np.random.default_rng(seed)
    amplitude = np.sqrt(10 ** (cn0_dbhz / 10) / config.fs_hz)
    return amplitude * signal + (
        rng.normal(scale=np.sqrt(0.5), size=n_samples)
        + 1j * rng.normal(scale=np.sqrt(0.5), size=n_samples)
    )


# --------------------------------------------------------------------------
# what the sweep costs
# --------------------------------------------------------------------------


def test_the_cost_is_the_product_it_claims():
    cost = RefinerCost(n_results=10, n_bins=21, n_samples=10230, mixer="quadrant")
    assert cost.n_operations == 10 * 21 * 10230


def test_the_saving_is_against_correlating_every_code_phase():
    # The second stage is a correlator, and what makes it affordable is that it
    # only ever runs on the cells the CAM already found.
    cost = RefinerCost(n_results=10, n_bins=21, n_samples=10230, mixer="quadrant")
    assert cost.against_a_full_search(32, 1023) == pytest.approx(32 * 1023 / 10)


def test_costing_nothing_saves_everything():
    cost = RefinerCost(n_results=0, n_bins=21, n_samples=10230, mixer="quadrant")
    assert cost.n_operations == 0
    assert cost.against_a_full_search(32, 1023) == float("inf")


def test_the_refiner_counts_the_detections_it_saw():
    config = make_config()
    refiner = DopplerRefiner(config)
    assert refiner.cost().n_results == 0
    refiner.refine(make_signal(config, 1), [PrnResult(1, 0.0, 0)] * 3)
    assert refiner.cost().n_results == 3
    assert refiner.cost(n_results=7).n_results == 7


def test_the_cost_names_the_mixer_that_produced_it():
    assert DopplerRefiner(make_config(), mixer="exact").cost().mixer == "exact"


@pytest.mark.parametrize("bad", ["nco", "", "Quadrant"])
def test_an_unknown_mixer_is_rejected(bad):
    with pytest.raises(AssertionError):
        DopplerRefiner(make_config(), mixer=bad)


# --------------------------------------------------------------------------
# the sweep stays on the 1 bit stream
# --------------------------------------------------------------------------


def test_the_quantized_stream_is_built_once_per_record():
    config = make_config()
    refiner = DopplerRefiner(config)
    samples = make_signal(config, 1)
    assert refiner.quadrant_stream(samples) is refiner.quadrant_stream(samples)


def test_a_new_record_gets_its_own_stream():
    config = make_config()
    refiner = DopplerRefiner(config)
    first = refiner.quadrant_stream(make_signal(config, 1, doppler_hz=500.0))
    second = refiner.quadrant_stream(make_signal(config, 2, doppler_hz=-500.0))
    assert not np.array_equal(first, second)


def test_the_quadrant_mixer_leaves_unit_magnitude_samples():
    # A 1 bit front end has no magnitude to carry, which is the point: the
    # second stage is a correlator but not a wider one.
    config = make_config()
    refiner = DopplerRefiner(config)
    wiped = refiner.wipe_doppler(make_signal(config, 1, doppler_hz=500.0), 0)
    assert np.allclose(np.abs(wiped), np.sqrt(2.0))


def test_wiping_zero_doppler_leaves_the_stream_as_it_was():
    config = make_config()
    refiner = DopplerRefiner(config)
    samples = make_signal(config, 1, doppler_hz=0.0)
    zero_bin = list(config.doppler_grid_hz).index(0.0)
    wiped = refiner.wipe_doppler(samples, zero_bin)
    assert np.array_equal(np.real(wiped) > 0, np.real(samples) > 0)
    assert np.array_equal(np.imag(wiped) > 0, np.imag(samples) > 0)


# --------------------------------------------------------------------------
# resolving the Doppler
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mixer", ["quadrant", "exact"])
def test_the_scores_peak_at_the_true_bin(mixer):
    config = make_config()
    refiner = DopplerRefiner(config, mixer=mixer)
    signal = make_signal(config, 2, doppler_hz=1000.0, code_phase=313)
    powers = refiner.scores(signal, 2, 313)
    assert len(powers) == len(config.doppler_grid_hz)
    assert config.doppler_grid_hz[int(np.argmax(powers))] == 1000.0


@pytest.mark.parametrize("mixer", ["quadrant", "exact"])
@pytest.mark.parametrize("doppler_hz", [-2000.0, -500.0, 0.0, 1500.0])
def test_a_clean_satellite_is_resolved_to_its_own_bin(mixer, doppler_hz):
    config = make_config()
    refiner = DopplerRefiner(config, mixer=mixer)
    signal = make_signal(config, 2, doppler_hz=doppler_hz, code_phase=313)
    assert refiner.resolve(signal, 2, 313) == doppler_hz


@pytest.mark.parametrize("seed", range(3))
def test_the_1_bit_sweep_still_resolves_a_satellite_under_noise(seed):
    config = make_config()
    refiner = DopplerRefiner(config)
    signal = make_signal(
        config, 2, doppler_hz=500.0, code_phase=313, cn0_dbhz=45.0, seed=seed
    )
    assert refiner.resolve(signal, 2, 313) == 500.0


def test_a_resolved_doppler_is_always_on_the_search_grid():
    config = make_config()
    refiner = DopplerRefiner(config)
    signal = make_signal(config, 1, doppler_hz=260.0, code_phase=5)
    assert refiner.resolve(signal, 1, 5) in set(config.doppler_grid_hz)


def test_refining_keeps_the_prn_and_the_code_phase():
    config = make_config()
    refiner = DopplerRefiner(config)
    signal = make_signal(config, 3, doppler_hz=-1500.0, code_phase=77)
    blind = [PrnResult(prn=3, doppler_hz=0.0, code_phase=77)]
    (refined,) = refiner.refine(signal, blind)
    assert (refined.prn, refined.code_phase) == (3, 77)
    assert refined.doppler_hz == -1500.0


def test_refining_nothing_returns_nothing():
    config = make_config()
    assert DopplerRefiner(config).refine(make_signal(config, 1), []) == []


# --------------------------------------------------------------------------
# the composition
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "build",
    [
        lambda config: SegmentedHdCamClassifier(config, search_mode="table"),
        lambda config: DifferentialHdCamClassifier(config, search_mode="table"),
    ],
    ids=["segmented", "differential"],
)
@pytest.mark.parametrize("doppler_hz", [-1500.0, 0.0, 2000.0])
def test_a_doppler_blind_family_finishes_its_answer(build, doppler_hz):
    config = make_config()
    classifier = RefinedClassifier(build(config))
    signal = make_signal(config, 2, doppler_hz=doppler_hz, code_phase=313)
    assert PrnResult(prn=2, doppler_hz=doppler_hz, code_phase=313) in (
        classifier.acquire(signal)
    )


def test_the_composition_carries_the_inner_configuration():
    config = make_config()
    inner = DifferentialHdCamClassifier(config)
    classifier = RefinedClassifier(inner)
    assert classifier.config is config
    assert classifier.inner is inner


def test_the_composition_still_asserts_on_the_sample_count():
    config = make_config()
    classifier = RefinedClassifier(DifferentialHdCamClassifier(config))
    with pytest.raises(AssertionError):
        classifier.acquire(np.zeros(config.samples_per_acquisition + 1, dtype=complex))


def test_a_silent_record_gives_the_second_stage_nothing_to_do():
    config = make_config()
    classifier = RefinedClassifier(DifferentialHdCamClassifier(config))
    assert classifier.acquire(np.zeros(config.samples_per_acquisition)) == []
    assert classifier.refiner.cost().n_results == 0
