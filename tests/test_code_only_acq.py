"""Directed tests for the code-only family.

The signals are synthetic replicas at a Doppler that sits on the search grid.
The sampling rate is deliberately low, so a whole acquisition run is cheap;
these check the wiring - the quadrant arithmetic, the row and query layout, and
that a planted satellite comes back at the right CFO - not how well the design
performs.

The acquisition tests ask that the planted satellite comes back with the right
numbers, not that nothing else does. At an uncalibrated threshold this family
also reports PRNs the record does not contain, which is the failure mode section
3 predicts for it and has a test of its own below.

What is not covered here: the ~0.9 dB the quadrant mixer costs against the exact
one, or how much of the false alarm rate survives calibration. Those are
measurements, and notebooks/family_screen.ipynb owns them.
"""

import numpy as np
import pytest

from hdcam_gps.acq_base import AcqConfig, PrnResult
from hdcam_gps.ca_code import sampled_ca_code
from hdcam_gps.cam_acq import QUERY_ROTATIONS, quantize_iq, rotate_quarter_turns
from hdcam_gps.code_only_acq import (
    DEFAULT_CODEBOOK_PHASES,
    CodeOnlyHdCamClassifier,
    bits_from_quadrants,
    quadrants,
    rotate_quadrants,
)
from hdcam_gps.hdcam_acq import OneBitHdCamClassifier

FS_HZ = 204.6e3  # 204 samples per code period, enough to stay correct and cheap
SAMPLES_PER_CODE = 204
N_COLUMNS = 2 * SAMPLES_PER_CODE


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
    carrier_phase: float = 0.0,
) -> np.ndarray:
    """A noise free replica at a known Doppler, code phase and carrier phase."""
    code = np.tile(sampled_ca_code(prn, config.fs_hz), config.n_codes)
    code = np.roll(code, code_phase).astype(complex)
    time_s = np.arange(config.samples_per_acquisition) / config.fs_hz
    return code * np.exp(2j * np.pi * doppler_hz * time_s + 1j * carrier_phase)


def sample_bits(n: int = 32, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return quantize_iq(rng.normal(size=n) + 1j * rng.normal(size=n))


# --------------------------------------------------------------------------
# the quadrant stream - a 1 bit sample is a 2 bit phase
# --------------------------------------------------------------------------


def test_the_four_quadrants_are_numbered_anticlockwise():
    bits = quantize_iq(np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]))
    assert quadrants(bits).tolist() == [0, 1, 2, 3]


def test_quadrants_and_bits_are_inverse():
    bits = sample_bits()
    assert np.array_equal(bits_from_quadrants(quadrants(bits)), bits)


@pytest.mark.parametrize("turn", range(QUERY_ROTATIONS))
def test_adding_one_to_the_quadrant_is_one_quarter_turn(turn):
    # This is what buys the cheap mixer: turning a 1 bit sample is an add.
    bits = sample_bits()
    assert np.array_equal(
        rotate_quadrants(bits, np.full(len(bits) // 2, turn)),
        rotate_quarter_turns(bits, turn),
    )


def test_the_turn_count_wraps():
    bits = sample_bits()
    assert np.array_equal(
        rotate_quadrants(bits, np.full(len(bits) // 2, 5)),
        rotate_quarter_turns(bits, 1),
    )


def test_every_sample_turns_by_its_own_count():
    bits = quantize_iq(np.array([1 + 1j, 1 + 1j]))
    turned = rotate_quadrants(bits, np.array([0, 2]))
    assert quadrants(turned).tolist() == [0, 2]


# --------------------------------------------------------------------------
# construction, and the shape the design is for
# --------------------------------------------------------------------------


def test_the_codebook_holds_one_row_per_prn_and_stored_phase():
    classifier = CodeOnlyHdCamClassifier(make_config())
    assert classifier.n_rows == 3 * DEFAULT_CODEBOOK_PHASES
    assert classifier.n_columns == N_COLUMNS


def test_the_rows_drop_by_the_number_of_doppler_bins():
    # The whole point of the family: 21 bins of the study's grid is 21 times
    # less CAM.
    config = make_config(doppler_min_hz=-5000.0, doppler_max_hz=5000.0)
    baseline = OneBitHdCamClassifier(config)
    code_only = CodeOnlyHdCamClassifier(config)
    assert baseline.n_rows == code_only.n_rows * code_only.n_doppler_bins


def test_the_doppler_bin_comes_from_the_query():
    classifier = CodeOnlyHdCamClassifier(make_config())
    assert classifier.doppler_is_in_the_query
    assert (classifier.row_index().doppler_bin == -1).all()


def test_there_is_one_query_per_start_and_bin():
    config = make_config()
    classifier = CodeOnlyHdCamClassifier(config)
    n_starts = config.samples_per_acquisition - config.samples_per_code + 1
    assert classifier.n_queries == n_starts * classifier.n_doppler_bins


def test_the_searches_rise_as_far_as_the_rows_fell():
    # Area for latency, at the same energy: 21 times fewer rows searched 21
    # times more often is the same bit comparisons.
    config = make_config(doppler_min_hz=-5000.0, doppler_max_hz=5000.0)
    baseline = OneBitHdCamClassifier(config)
    code_only = CodeOnlyHdCamClassifier(config)
    assert code_only.n_queries * code_only.n_rows == (
        baseline.n_queries * baseline.n_rows
    )


@pytest.mark.parametrize("bad", ["nco", "", "Quadrant"])
def test_an_unknown_mixer_is_rejected(bad):
    with pytest.raises(AssertionError):
        CodeOnlyHdCamClassifier(make_config(), mixer=bad)


def test_a_single_code_period_config_is_rejected():
    with pytest.raises(AssertionError):
        CodeOnlyHdCamClassifier(make_config(n_codes=1))


# --------------------------------------------------------------------------
# the codebook is the zero-Doppler replica
# --------------------------------------------------------------------------


def test_a_codebook_row_is_the_quantized_zero_doppler_replica():
    config = make_config()
    classifier = CodeOnlyHdCamClassifier(config)
    phase = 1
    expected = quantize_iq(
        sampled_ca_code(2, config.fs_hz)
        * np.exp(2j * np.pi * (phase + 0.5) / classifier.n_phases)
    )
    assert np.array_equal(
        classifier.build_codebook()[classifier.row_of(2, phase)], expected
    )


def test_row_of_and_the_row_index_agree():
    classifier = CodeOnlyHdCamClassifier(make_config())
    rows = classifier.row_index()
    for prn in classifier.config.prn_list:
        for phase in range(classifier.n_codebook_phases):
            assert rows.prn[classifier.row_of(prn, phase)] == prn


def test_the_zero_doppler_row_still_carries_both_bit_planes():
    classifier = CodeOnlyHdCamClassifier(make_config())
    row = classifier.build_codebook()[classifier.row_of(1)]
    in_phase, quadrature = np.split(row, 2)
    assert in_phase.any() and not in_phase.all()
    assert quadrature.any() and not quadrature.all()


# --------------------------------------------------------------------------
# the mixer
# --------------------------------------------------------------------------


def test_the_two_mixers_agree_at_zero_doppler():
    # At zero Doppler the ramp is flat and the NCO is one, so the only thing
    # that could differ is a bug.
    config = make_config()
    samples = make_signal(config, 1, doppler_hz=500.0, code_phase=7)
    zero_bin = list(config.doppler_grid_hz).index(0.0)
    quadrant = CodeOnlyHdCamClassifier(config, mixer="quadrant")
    exact = CodeOnlyHdCamClassifier(config, mixer="exact")
    assert np.array_equal(
        quadrant.wipe_doppler(samples, 13, zero_bin),
        exact.wipe_doppler(samples, 13, zero_bin),
    )


def test_wiping_zero_doppler_leaves_the_window_as_it_was():
    config = make_config()
    samples = make_signal(config, 1, doppler_hz=0.0, code_phase=3)
    zero_bin = list(config.doppler_grid_hz).index(0.0)
    classifier = CodeOnlyHdCamClassifier(config)
    assert np.array_equal(
        classifier.wipe_doppler(samples, 20, zero_bin),
        quantize_iq(classifier.query_window(samples, 20)),
    )


def test_the_quadrant_mixer_stays_within_a_quarter_turn_of_the_exact_one():
    # Rounding each rotation to a quadrant is the whole cost of keeping the
    # front end 1 bit, and it can only move a sample by one quadrant.
    config = make_config()
    samples = make_signal(config, 2, doppler_hz=500.0, code_phase=11, carrier_phase=1.0)
    quadrant = CodeOnlyHdCamClassifier(config, mixer="quadrant")
    exact = CodeOnlyHdCamClassifier(config, mixer="exact")
    for doppler_bin in range(quadrant.n_doppler_bins):
        difference = (
            quadrants(quadrant.wipe_doppler(samples, 40, doppler_bin))
            - quadrants(exact.wipe_doppler(samples, 40, doppler_bin))
        ) % QUERY_ROTATIONS
        assert set(np.unique(difference).tolist()) <= {0, 1, 3}


def test_a_query_has_one_variant_per_quarter_turn():
    config = make_config()
    classifier = CodeOnlyHdCamClassifier(config)
    variants = classifier.query_variants(make_signal(config, 1), 0)
    assert len(variants) == QUERY_ROTATIONS
    assert all(len(bits) == N_COLUMNS for bits in variants)


def test_a_window_past_the_end_of_the_record_is_rejected():
    config = make_config()
    classifier = CodeOnlyHdCamClassifier(config)
    samples = make_signal(config, 1)
    last = len(samples) - config.samples_per_code
    classifier.query_window(samples, last)
    with pytest.raises(AssertionError):
        classifier.query_window(samples, last + 1)


# --------------------------------------------------------------------------
# acquisition
# --------------------------------------------------------------------------


# A carrier phase the wiped query does not land on an axis with. See
# test_a_wiped_query_on_the_real_axis_loses_its_quadrature_plane for why zero
# will not do.
OFF_AXIS_PHASE = 0.7


@pytest.mark.parametrize("mixer", ["quadrant", "exact"])
@pytest.mark.parametrize("doppler_hz", [-500.0, 0.0, 500.0])
def test_a_planted_satellite_comes_back_at_its_own_cfo(mixer, doppler_hz):
    config = make_config()
    classifier = CodeOnlyHdCamClassifier(config, mixer=mixer)
    signal = make_signal(
        config, 2, doppler_hz=doppler_hz, code_phase=55,
        carrier_phase=OFF_AXIS_PHASE,
    )
    assert (
        PrnResult(prn=2, doppler_hz=doppler_hz, code_phase=55)
        in classifier.acquire(signal)
    )


@pytest.mark.parametrize("code_phase", [0, 17, 150])
def test_the_code_phase_comes_back_from_the_window_offset(code_phase):
    config = make_config()
    classifier = CodeOnlyHdCamClassifier(config)
    signal = make_signal(
        config, 1, doppler_hz=500.0, code_phase=code_phase,
        carrier_phase=OFF_AXIS_PHASE,
    )
    assert (
        PrnResult(prn=1, doppler_hz=500.0, code_phase=code_phase)
        in classifier.acquire(signal)
    )


def test_a_wiped_query_on_the_real_axis_loses_its_quadrature_plane():
    # Wiping the CFO off a satellite whose carrier phase is zero leaves a query
    # on the real axis, whose quadrature bits are the sign of numerical dust.
    # Half the word then carries nothing and the distance to the right row is
    # about a quarter of it. Real signals do not sit on the axis; planted
    # replicas do, which is why every test here plants an off axis one.
    config = make_config()
    classifier = CodeOnlyHdCamClassifier(config, mixer="exact")
    true_bin = list(config.doppler_grid_hz).index(500.0)
    rows = [classifier.row_of(1, phase) for phase in range(classifier.n_codebook_phases)]

    def distance(carrier_phase):
        signal = make_signal(
            config, 1, doppler_hz=500.0, code_phase=0, carrier_phase=carrier_phase
        )
        bits = classifier.wipe_doppler(signal, 0, true_bin)
        return classifier.row_distances(bits)[rows].min()

    assert distance(0.0) > N_COLUMNS / 8
    assert distance(OFF_AXIS_PHASE) < N_COLUMNS / 8


def test_a_silent_record_acquires_nothing():
    config = make_config()
    classifier = CodeOnlyHdCamClassifier(config)
    assert classifier.acquire(np.zeros(config.samples_per_acquisition)) == []


def test_an_unlisted_prn_acquires_nothing():
    config = make_config()
    classifier = CodeOnlyHdCamClassifier(config)
    assert classifier.acquire(make_signal(config, 19, code_phase=5)) == []


def test_the_cam_path_and_the_table_path_agree():
    config = make_config()
    signal = make_signal(config, 3, doppler_hz=-500.0, code_phase=88, carrier_phase=2.0)
    through_cam = CodeOnlyHdCamClassifier(config, search_mode="cam")
    through_table = CodeOnlyHdCamClassifier(config, search_mode="table")
    assert through_cam.acquire(signal) == through_table.acquire(signal)



# --------------------------------------------------------------------------
# what the family is expected to fail at
# --------------------------------------------------------------------------


def test_the_quadrant_mixer_reports_prns_the_exact_one_does_not():
    # The predicted failure mode of this family, at the study's own sampling
    # rate: rounding every rotation to a quadrant puts steps in the query that
    # correlate with codes the record does not contain. The true satellite is
    # still found in every case; it is the company it keeps that differs. How
    # much of this survives calibration is phase 3's question, not this file's.
    config = make_config(fs_hz=1.023e6, prn_list=(1, 2), n_codes=2)
    signal = make_signal(
        config, 2, doppler_hz=500.0, code_phase=17, carrier_phase=OFF_AXIS_PHASE
    )
    wanted = PrnResult(prn=2, doppler_hz=500.0, code_phase=17)

    quadrant = CodeOnlyHdCamClassifier(config, mixer="quadrant").acquire(signal)
    exact = CodeOnlyHdCamClassifier(config, mixer="exact").acquire(signal)
    assert wanted in quadrant and wanted in exact
    assert len(quadrant) > len(exact)
