"""Directed tests for the 1 bit HdCam acquisition.

The signals are synthetic replicas at a Doppler that sits on the search grid.
The sampling rate is deliberately low, so that a whole acquisition run is cheap;
these tests check the wiring of the classifier - the bit encoding, the codebook
layout, the CAM it drives, the thresholds and the vote reduction - not how well
the scheme performs.
"""

import numpy as np
import pytest

from hdcam_gps.acq_base import AcqConfig, PrnResult
from hdcam_gps.ca_code import sampled_ca_code
from hdcam_gps.hdcam import HdCam
from hdcam_gps.hdcam_acq import (
    DEFAULT_CODEBOOK_PHASES,
    DEFAULT_FALSE_ALARM_RATE,
    QUERY_ROTATIONS,
    OneBitHdCamClassifier,
    hd_threshold_for_false_alarm,
    quantize_iq,
    rotate_quarter_turns,
)

FS_HZ = 204.6e3  # 204 samples per code period, enough to stay correct and cheap
SAMPLES_PER_CODE = 204
N_COLUMNS = 2 * SAMPLES_PER_CODE


def make_config(**overrides) -> AcqConfig:
    kwargs = {
        "fs_hz": FS_HZ,
        "prn_list": (1, 2),
        "doppler_min_hz": -500.0,
        "doppler_max_hz": 500.0,
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
    carrier_phase: float = 0.0,
) -> np.ndarray:
    """A noise free replica at a known Doppler, code phase and carrier phase."""
    code = np.tile(sampled_ca_code(prn_idx, config.fs_hz), config.n_codes)
    code = np.roll(code, code_phase).astype(complex)
    time_s = np.arange(config.samples_per_acquisition) / config.fs_hz
    return code * np.exp(2j * np.pi * doppler_hz * time_s + 1j * carrier_phase)


# --------------------------------------------------------------------------
# quantize_iq
# --------------------------------------------------------------------------


def test_quantize_iq_splits_into_two_sign_planes():
    samples = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j])
    bits = quantize_iq(samples)
    assert bits.dtype == bool
    assert len(bits) == 2 * len(samples)
    assert list(bits[:4]) == [True, False, False, True]  # sign of I
    assert list(bits[4:]) == [True, True, False, False]  # sign of Q


def test_quantize_iq_puts_zero_on_the_negative_side():
    assert list(quantize_iq(np.array([0 + 0j]))) == [False, False]


def test_quantize_iq_only_looks_at_the_signs():
    small = quantize_iq(np.array([1e-9 - 1e-9j]))
    large = quantize_iq(np.array([1e9 - 1e9j]))
    assert np.array_equal(small, large)


# --------------------------------------------------------------------------
# rotate_quarter_turns - the only rotations a one bit query can express
# --------------------------------------------------------------------------


def sample_bits() -> np.ndarray:
    rng = np.random.default_rng(0)
    samples = rng.normal(size=16) + 1j * rng.normal(size=16)
    return quantize_iq(samples), samples


def test_zero_rotations_is_the_identity():
    bits, _ = sample_bits()
    assert np.array_equal(rotate_quarter_turns(bits, 0), bits)


def test_four_rotations_return_to_the_start():
    bits, _ = sample_bits()
    assert np.array_equal(rotate_quarter_turns(bits, 4), bits)


def test_two_rotations_are_the_complement():
    bits, _ = sample_bits()
    assert np.array_equal(rotate_quarter_turns(bits, 2), ~bits)


def test_rotations_compose():
    bits, _ = sample_bits()
    once = rotate_quarter_turns(bits, 1)
    assert np.array_equal(rotate_quarter_turns(once, 1), rotate_quarter_turns(bits, 2))


def test_rotation_count_wraps():
    bits, _ = sample_bits()
    assert np.array_equal(rotate_quarter_turns(bits, 5), rotate_quarter_turns(bits, 1))


@pytest.mark.parametrize("rotations", [0, 1, 2, 3])
def test_rotation_matches_multiplying_the_samples_by_j(rotations):
    bits, samples = sample_bits()
    turned = samples * (1j**rotations)
    assert np.array_equal(rotate_quarter_turns(bits, rotations), quantize_iq(turned))


def test_rotation_uses_no_information_beyond_the_bits():
    # Two records with the same signs but different magnitudes rotate alike.
    a = np.array([3 + 0.1j, -0.2 + 5j])
    b = np.array([0.9 + 0.9j, -0.7 + 0.1j])
    assert np.array_equal(quantize_iq(a), quantize_iq(b))
    assert np.array_equal(
        rotate_quarter_turns(quantize_iq(a), 3), rotate_quarter_turns(quantize_iq(b), 3)
    )


# --------------------------------------------------------------------------
# threshold helpers
# --------------------------------------------------------------------------


def test_false_alarm_threshold_sits_below_chance():
    threshold = hd_threshold_for_false_alarm(408, 1e-6)
    assert 0 < threshold < 408 // 2


def test_false_alarm_threshold_tightens_with_a_rarer_target():
    loose = hd_threshold_for_false_alarm(408, 1e-2)
    tight = hd_threshold_for_false_alarm(408, 1e-9)
    assert tight < loose


@pytest.mark.parametrize("bad_rate", [0.0, 0.5, 1.0, -0.1])
def test_false_alarm_threshold_rejects_an_impossible_rate(bad_rate):
    with pytest.raises(AssertionError):
        hd_threshold_for_false_alarm(408, bad_rate)


# --------------------------------------------------------------------------
# construction and the CAM it builds
# --------------------------------------------------------------------------


def test_classifier_shape():
    classifier = OneBitHdCamClassifier(make_config())
    assert classifier.n_doppler_bins == 3
    assert classifier.n_codebook_phases == DEFAULT_CODEBOOK_PHASES
    assert classifier.n_phases == QUERY_ROTATIONS * DEFAULT_CODEBOOK_PHASES
    assert classifier.n_rows == 2 * 3 * DEFAULT_CODEBOOK_PHASES
    assert classifier.n_columns == N_COLUMNS


def test_classifier_drives_a_real_hdcam_holding_the_codebook():
    classifier = OneBitHdCamClassifier(make_config())
    assert isinstance(classifier.cam, HdCam)
    assert classifier.cam.n_rows == classifier.n_rows
    assert classifier.cam.n_columns == classifier.n_columns
    assert np.array_equal(classifier.cam.grid, classifier.build_codebook())


def test_default_threshold_sits_below_the_chance_floor():
    classifier = OneBitHdCamClassifier(make_config())
    chance, sigma = N_COLUMNS / 2, np.sqrt(N_COLUMNS) / 2
    assert 2.0 < (chance - classifier.hd_threshold) / sigma < 7.0


def test_a_larger_search_space_tightens_the_threshold():
    # More hypotheses means more chances to draw a low distance by luck, so the
    # same false alarm budget has to be spread thinner.
    narrow = OneBitHdCamClassifier(make_config(prn_list=(1,)))
    wide = OneBitHdCamClassifier(make_config(prn_list=(1, 2)))
    assert wide.n_cells > narrow.n_cells
    assert wide.hd_threshold < narrow.hd_threshold


def test_one_code_period_of_looks_needs_every_vote():
    classifier = OneBitHdCamClassifier(make_config(n_codes=2))
    assert classifier.n_looks == 1
    assert classifier.min_votes == 1


def test_more_code_periods_give_more_looks_and_a_looser_threshold():
    # This is the accumulation: a hypothesis has to match repeatedly, so a single
    # look is allowed to be far less certain.
    few = OneBitHdCamClassifier(make_config(n_codes=2))
    many = OneBitHdCamClassifier(make_config(n_codes=10))
    assert many.n_looks == 9 and few.n_looks == 1
    assert many.min_votes > few.min_votes
    assert many.hd_threshold > few.hd_threshold


def test_min_votes_cannot_exceed_the_looks_available():
    with pytest.raises(AssertionError):
        OneBitHdCamClassifier(make_config(n_codes=2), min_votes=2)


def test_an_explicit_threshold_is_used_as_given():
    classifier = OneBitHdCamClassifier(make_config(), hd_threshold=123)
    assert classifier.hd_threshold == 123
    assert classifier.cam.hd_threshold == 123


def test_set_hd_threshold_retunes_the_cam_in_place():
    classifier = OneBitHdCamClassifier(make_config())
    codebook = classifier.cam.grid.copy()

    classifier.set_hd_threshold(200)
    assert classifier.hd_threshold == 200
    assert classifier.cam.hd_threshold == 200
    assert np.array_equal(classifier.cam.grid, codebook)  # codebook untouched


def test_set_hd_threshold_rejects_a_value_outside_the_row_width():
    classifier = OneBitHdCamClassifier(make_config())
    with pytest.raises(AssertionError):
        classifier.set_hd_threshold(N_COLUMNS)


def test_more_codebook_phases_add_rows():
    two = OneBitHdCamClassifier(make_config(), n_codebook_phases=2)
    four = OneBitHdCamClassifier(make_config(), n_codebook_phases=4)
    assert four.n_rows == 2 * two.n_rows


def test_single_code_period_config_is_rejected():
    with pytest.raises(AssertionError):
        OneBitHdCamClassifier(make_config(n_codes=1))


@pytest.mark.parametrize("bad_phases", [0, -1])
def test_at_least_one_codebook_phase_is_required(bad_phases):
    with pytest.raises(AssertionError):
        OneBitHdCamClassifier(make_config(), n_codebook_phases=bad_phases)


@pytest.mark.parametrize("bad_votes", [0, -1])
def test_at_least_one_vote_is_required(bad_votes):
    with pytest.raises(AssertionError):
        OneBitHdCamClassifier(make_config(), min_votes=bad_votes)


def test_the_vote_rule_and_the_per_look_rate_agree():
    # The per look rate is whatever makes the vote rule hit the wanted budget.
    from hdcam_gps.hdcam_acq import binomial_tail, per_look_false_alarm

    n_looks, min_votes, n_cells, budget = 9, 3, 30_000, 1e-2
    per_look = per_look_false_alarm(n_looks, min_votes, n_cells, budget)
    assert binomial_tail(n_looks, min_votes, per_look) == pytest.approx(
        budget / n_cells, rel=1e-3
    )


def test_voting_lets_a_single_look_be_looser():
    from hdcam_gps.hdcam_acq import per_look_false_alarm

    one_shot = per_look_false_alarm(1, 1, 30_000, 1e-2)
    voted = per_look_false_alarm(9, 3, 30_000, 1e-2)
    assert voted > one_shot


# --------------------------------------------------------------------------
# codebook layout and content
# --------------------------------------------------------------------------


def test_codebook_shape_and_dtype():
    classifier = OneBitHdCamClassifier(make_config())
    codebook = classifier.build_codebook()
    assert codebook.shape == (classifier.n_rows, N_COLUMNS)
    assert codebook.dtype == bool


def test_every_prn_and_cfo_pair_has_its_own_codebook_row():
    # Not every row is distinct: at zero Doppler the carrier does not rotate, so
    # every stored phase inside one quadrant lands on the same bit pattern. What
    # has to differ is the hypotheses the search is actually choosing between.
    classifier = OneBitHdCamClassifier(make_config())
    codebook = classifier.build_codebook()
    rows = {
        codebook[classifier.row_of(prn, doppler_bin)].tobytes()
        for prn in classifier.config.prn_list
        for doppler_bin in range(classifier.n_doppler_bins)
    }
    assert len(rows) == len(classifier.config.prn_list) * classifier.n_doppler_bins


def test_the_zero_doppler_row_still_carries_both_bit_planes():
    # A replica exactly on an axis would have a quadrature plane of all zeros,
    # which matches anything. The stored phases are offset to prevent that.
    classifier = OneBitHdCamClassifier(make_config())
    zero_bin = list(classifier.config.doppler_grid_hz).index(0.0)
    row = classifier.build_codebook()[classifier.row_of(1, zero_bin)]
    in_phase, quadrature = np.split(row, 2)
    assert in_phase.any() and not in_phase.all()
    assert quadrature.any() and not quadrature.all()


def test_row_of_and_hypothesis_of_are_inverse():
    classifier = OneBitHdCamClassifier(make_config())
    config = classifier.config
    for prn in config.prn_list:
        for doppler_bin in range(classifier.n_doppler_bins):
            for phase in range(classifier.n_codebook_phases):
                row = classifier.row_of(prn, doppler_bin, phase)
                assert classifier.hypothesis_of(row) == (
                    prn,
                    float(config.doppler_grid_hz[doppler_bin]),
                )


def test_every_row_index_maps_to_a_hypothesis():
    classifier = OneBitHdCamClassifier(make_config())
    seen = [classifier.hypothesis_of(row) for row in range(classifier.n_rows)]
    assert len(set(seen)) == classifier.n_doppler_bins * len(classifier.config.prn_list)


def test_codebook_row_is_the_quantized_replica():
    classifier = OneBitHdCamClassifier(make_config())
    config = classifier.config
    time_s = np.arange(config.samples_per_code) / config.fs_hz
    prn, doppler_bin, phase = 2, 2, 1

    doppler_hz = config.doppler_grid_hz[doppler_bin]
    expected = quantize_iq(
        sampled_ca_code(prn, config.fs_hz)
        * np.exp(2j * np.pi * doppler_hz * time_s)
        * np.exp(2j * np.pi * (phase + 0.5) / classifier.n_phases)
    )
    row = classifier.row_of(prn, doppler_bin, phase)
    assert np.array_equal(classifier.build_codebook()[row], expected)


def test_the_stored_phases_subdivide_a_quadrant():
    # Phase 1 of 2 is 45 degrees, so it must differ from a quarter turn of phase 0.
    classifier = OneBitHdCamClassifier(make_config())
    codebook = classifier.build_codebook()
    base = codebook[classifier.row_of(1, 0, 0)]
    offset = codebook[classifier.row_of(1, 0, 1)]
    assert not np.array_equal(offset, base)
    for rotations in range(QUERY_ROTATIONS):
        assert not np.array_equal(offset, rotate_quarter_turns(base, rotations))


# --------------------------------------------------------------------------
# query_window
# --------------------------------------------------------------------------


def test_query_window_is_a_contiguous_slice():
    config = make_config()
    classifier = OneBitHdCamClassifier(config)
    samples = np.arange(config.samples_per_acquisition).astype(complex)

    window = classifier.query_window(samples, 30)
    assert len(window) == config.samples_per_code
    assert np.array_equal(window, samples[30 : 30 + config.samples_per_code])


def test_query_window_never_wraps_past_the_end():
    config = make_config()
    classifier = OneBitHdCamClassifier(config)
    samples = np.zeros(config.samples_per_acquisition, dtype=complex)

    last = config.samples_per_acquisition - config.samples_per_code
    classifier.query_window(samples, last)  # the last window that fits
    with pytest.raises(AssertionError):
        classifier.query_window(samples, last + 1)


# --------------------------------------------------------------------------
# tightest_match - the CAM's only way of measuring a distance
# --------------------------------------------------------------------------


def test_tightest_match_of_an_exact_match_is_zero():
    classifier = OneBitHdCamClassifier(make_config())
    row = classifier.row_of(1, doppler_bin=0)
    assert classifier.tightest_match(row, classifier.cam.grid[row].copy()) == 0


@pytest.mark.parametrize("distance", [1, 5, 40, 200])
def test_tightest_match_finds_a_planted_distance(distance):
    classifier = OneBitHdCamClassifier(make_config())
    row = classifier.row_of(2, doppler_bin=1)
    query = classifier.cam.grid[row].copy()
    query[:distance] = ~query[:distance]
    assert classifier.tightest_match(row, query) == distance


def test_tightest_match_leaves_the_cam_threshold_alone():
    classifier = OneBitHdCamClassifier(make_config(), hd_threshold=123)
    row = classifier.row_of(1, doppler_bin=0)
    classifier.tightest_match(row, classifier.cam.grid[row].copy())
    assert classifier.cam.hd_threshold == 123


# --------------------------------------------------------------------------
# shortlist
# --------------------------------------------------------------------------


def test_shortlist_finds_the_planted_hypothesis():
    config = make_config()
    classifier = OneBitHdCamClassifier(config)
    signal = make_signal(config, 1, doppler_hz=500.0, code_phase=33)

    candidates = classifier.shortlist(signal)
    row = classifier.row_of(1, doppler_bin=2)
    assert any(key[0] in (row, row + 1) and key[1] == 33 for key in candidates)


def test_shortlist_is_empty_for_silence():
    config = make_config()
    classifier = OneBitHdCamClassifier(config)
    assert classifier.shortlist(np.zeros(config.samples_per_acquisition)) == {}


def test_min_votes_thins_the_shortlist():
    config = make_config(n_codes=6)
    signal = make_signal(config, 1, doppler_hz=500.0, code_phase=33)
    lenient = OneBitHdCamClassifier(config, hd_threshold=200, min_votes=1)
    strict = OneBitHdCamClassifier(config, hd_threshold=200, min_votes=5)
    assert len(strict.shortlist(signal)) <= len(lenient.shortlist(signal))


# --------------------------------------------------------------------------
# _best_per_prn, driven directly
# --------------------------------------------------------------------------


def test_no_candidates_gives_no_results():
    classifier = OneBitHdCamClassifier(make_config())
    assert classifier._best_per_prn({}) == []


def test_a_single_candidate_names_its_prn_doppler_and_code_phase():
    classifier = OneBitHdCamClassifier(make_config())
    row = classifier.row_of(2, doppler_bin=2, codebook_phase=1)
    assert classifier._best_per_prn({(row, 77): 10}) == [
        PrnResult(prn=2, doppler_hz=500.0, code_phase=77)
    ]


def test_results_come_back_in_configuration_order():
    classifier = OneBitHdCamClassifier(make_config())
    distances = {
        (classifier.row_of(2, doppler_bin=0), 5): 10,
        (classifier.row_of(1, doppler_bin=1), 9): 10,
    }
    assert classifier._best_per_prn(distances) == [
        PrnResult(prn=1, doppler_hz=0.0, code_phase=9),
        PrnResult(prn=2, doppler_hz=-500.0, code_phase=5),
    ]


def test_the_closest_hypothesis_of_a_prn_wins():
    # This is the whole point of the second stage: a nearer row beats a further
    # one, which a set of matches on its own could never tell apart.
    classifier = OneBitHdCamClassifier(make_config())
    distances = {
        (classifier.row_of(1, doppler_bin=0), 11): 300,
        (classifier.row_of(1, doppler_bin=2), 60): 120,  # the winner
    }
    assert classifier._best_per_prn(distances) == [
        PrnResult(prn=1, doppler_hz=500.0, code_phase=60)
    ]


def test_one_result_at_most_per_prn():
    classifier = OneBitHdCamClassifier(make_config())
    distances = {
        (classifier.row_of(1, doppler_bin=bin_index), 3): 100 + bin_index
        for bin_index in range(3)
    }
    assert [r.prn for r in classifier._best_per_prn(distances)] == [1]


# --------------------------------------------------------------------------
# end to end
# --------------------------------------------------------------------------


@pytest.mark.parametrize("code_phase", [0, 1, 100, SAMPLES_PER_CODE - 1])
def test_acquire_recovers_the_code_phase(code_phase):
    config = make_config()
    signal = make_signal(config, 1, doppler_hz=0.0, code_phase=code_phase)
    assert OneBitHdCamClassifier(config).acquire(signal) == [
        PrnResult(prn=1, doppler_hz=0.0, code_phase=code_phase)
    ]


@pytest.mark.parametrize("doppler_hz", [-500.0, 0.0, 500.0])
def test_acquire_recovers_the_doppler_bin(doppler_hz):
    config = make_config()
    signal = make_signal(config, 2, doppler_hz=doppler_hz, code_phase=50)
    assert OneBitHdCamClassifier(config).acquire(signal) == [
        PrnResult(prn=2, doppler_hz=doppler_hz, code_phase=50)
    ]


@pytest.mark.parametrize("carrier_phase", np.linspace(0, 2 * np.pi, 8, endpoint=False))
def test_acquire_works_at_any_carrier_phase(carrier_phase):
    config = make_config()
    signal = make_signal(config, 1, 500.0, 33, carrier_phase=float(carrier_phase))
    assert OneBitHdCamClassifier(config).acquire(signal) == [
        PrnResult(prn=1, doppler_hz=500.0, code_phase=33)
    ]


def test_acquire_returns_nothing_for_a_silent_input():
    config = make_config()
    samples = np.zeros(config.samples_per_acquisition, dtype=complex)
    assert OneBitHdCamClassifier(config).acquire(samples) == []


def test_acquire_returns_nothing_for_an_unlisted_prn():
    config = make_config(prn_list=(1, 2))
    signal = make_signal(config, 20, doppler_hz=0.0, code_phase=40)
    assert OneBitHdCamClassifier(config).acquire(signal) == []


def test_acquire_still_asserts_on_the_sample_count():
    config = make_config()
    classifier = OneBitHdCamClassifier(config)
    with pytest.raises(AssertionError):
        classifier.acquire(np.zeros(config.samples_per_acquisition + 1, dtype=complex))


def test_a_tight_threshold_still_finds_a_clean_signal():
    # A noise free replica sits far below the chance floor, so it survives a
    # threshold far tighter than the default one.
    config = make_config()
    signal = make_signal(config, 1, doppler_hz=500.0, code_phase=20)
    assert OneBitHdCamClassifier(config, hd_threshold=60).acquire(signal) == [
        PrnResult(prn=1, doppler_hz=500.0, code_phase=20)
    ]


def test_a_zero_threshold_rejects_a_swept_carrier_at_an_unstored_phase():
    # With a Doppler the carrier sweeps across quantization boundaries, so a
    # carrier phase that is not one of the stored ones no longer matches exactly.
    config = make_config()
    signal = make_signal(config, 1, doppler_hz=500.0, code_phase=20, carrier_phase=0.3)
    assert OneBitHdCamClassifier(config, hd_threshold=0).acquire(signal) == []


def test_a_constant_carrier_quantizes_a_whole_quadrant_the_same_way():
    # At zero Doppler the carrier does not rotate, so every carrier phase inside
    # one quadrant lands on the same sign bits and matches the same row exactly.
    config = make_config()
    classifier = OneBitHdCamClassifier(config, hd_threshold=0)
    for carrier_phase in (0.1, 0.3, 0.5):
        signal = make_signal(config, 1, 0.0, 20, carrier_phase=carrier_phase)
        assert classifier.acquire(signal) == [
            PrnResult(prn=1, doppler_hz=0.0, code_phase=20)
        ]
