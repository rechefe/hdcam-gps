"""Directed tests for the counted CAM cost model.

The counters are checked against the closed form of the baseline's first pass,
and the derived quantities against the JSSC figures by hand. The sampling rate
is deliberately low, so a whole acquisition run is cheap; one test builds a
CamCost at the study's real geometry without running anything, to check the
headline numbers of docs/CAM_FAMILY_STUDY.md come out of this arithmetic.

What is not covered here: whether 0.19 fJ/bit and 8 ns/search describe a CAM
holding a 2046 bit word. They are that paper's 64 bit macro, applied unchanged.
"""

import numpy as np
import pytest

from hdcam_gps.acq_base import AcqConfig
from hdcam_gps.cam_acq import QUERY_ROTATIONS
from hdcam_gps.cam_cost import (
    ENERGY_PER_BIT_FJ,
    LATENCY_PER_SEARCH_NS,
    CamCost,
    CountingHdCam,
    measure_cost,
    noise_record,
)
from hdcam_gps.hdcam_acq import OneBitHdCamClassifier
from hdcam_gps.hdcam_packed import PackedHdCam
from hdcam_gps.signal_gen import SatelliteTruth, generate_synthetic

FS_HZ = 204.6e3  # 204 samples per code period, enough to stay correct and cheap


def make_config(**overrides) -> AcqConfig:
    kwargs = {
        "fs_hz": FS_HZ,
        "prn_list": (1, 2),
        "doppler_min_hz": -500.0,
        "doppler_max_hz": 500.0,
        "doppler_step_hz": 500.0,
        "n_codes": 3,
    }
    kwargs.update(overrides)
    return AcqConfig(**kwargs)


def first_pass_searches(config: AcqConfig) -> int:
    """The closed form: one search per window per quarter turn."""
    n_starts = config.samples_per_acquisition - config.samples_per_code + 1
    return n_starts * QUERY_ROTATIONS


def a_cost(**overrides) -> CamCost:
    kwargs = dict(
        n_rows=1344,
        n_columns=2046,
        n_searches=36_832,
        n_threshold_writes=0,
        n_first_pass_searches=36_832,
        hits_mean=0.0,
        hits_max=0,
        hd_threshold=920,
        chance_mean=1023.0,
        chance_sd=22.6,
    )
    kwargs.update(overrides)
    return CamCost(**kwargs)


# --------------------------------------------------------------------------
# CountingHdCam - the same answers, plus a tally
# --------------------------------------------------------------------------


def test_a_counting_cam_returns_what_a_packed_cam_returns():
    rng = np.random.default_rng(0)
    grid = rng.random((6, 40)) > 0.5
    packed = PackedHdCam(6, 40, 18)
    counting = CountingHdCam(6, 40, 18)
    packed.write_array(grid)
    counting.write_array(grid)
    for _ in range(5):
        query = rng.random(40) > 0.5
        assert np.array_equal(packed.search_cam(query), counting.search_cam(query))


def test_a_fresh_counting_cam_has_counted_nothing():
    counting = CountingHdCam(4, 8, 2)
    assert counting.n_searches == 0
    assert counting.n_threshold_writes == 0
    assert counting.hits == []


def test_every_search_and_every_retune_is_counted():
    counting = CountingHdCam(4, 8, 2)
    counting.search_cam(np.zeros(8, dtype=bool))
    counting.search_cam(np.ones(8, dtype=bool))
    counting.set_hd_threshold(3)
    assert counting.n_searches == 2
    assert counting.n_threshold_writes == 1


def test_the_hit_count_of_every_search_is_kept():
    counting = CountingHdCam(3, 8, 0)
    counting.write_array([[0] * 8, [0] * 8, [1] * 8])
    counting.search_cam(np.zeros(8, dtype=bool))
    counting.search_cam(np.ones(8, dtype=bool))
    assert counting.hits == [2, 1]


# --------------------------------------------------------------------------
# measure_cost - counted against the closed form
# --------------------------------------------------------------------------


def test_the_counted_first_pass_equals_the_closed_form():
    config = make_config()
    cost = measure_cost(OneBitHdCamClassifier(config))
    assert cost.n_first_pass_searches == first_pass_searches(config)


def test_a_noise_record_shortlists_nothing_and_so_bisects_nothing():
    # With an empty shortlist the second pass never runs, which is the one case
    # where the counted total and the closed form have to agree exactly.
    config = make_config()
    cost = measure_cost(OneBitHdCamClassifier(config))
    assert cost.n_searches == first_pass_searches(config)
    assert cost.n_threshold_writes == 0
    assert cost.hits_max == 0


def test_a_satellite_adds_the_bisections_the_closed_form_misses():
    # PROPOSAL risk 5: tightest_match costs about eleven searches per survivor,
    # and a first pass model would report none of them.
    config = make_config()
    classifier = OneBitHdCamClassifier(config)
    scenario = generate_synthetic(
        config, [SatelliteTruth(1, 500.0, 40, cn0_dbhz=60.0)], seed=0
    )
    cost = measure_cost(classifier, samples=scenario.samples)
    assert cost.n_searches > cost.n_first_pass_searches
    assert cost.n_threshold_writes > 0
    assert cost.hits_max >= 1


def test_measuring_leaves_the_classifier_as_it_was_found():
    classifier = OneBitHdCamClassifier(make_config(), hd_threshold=123)
    original = classifier.cam
    codebook = classifier.cam.grid.copy()

    measure_cost(classifier)
    assert classifier.cam is original
    assert classifier.cam.hd_threshold == 123
    assert np.array_equal(classifier.cam.grid, codebook)


def test_the_classifier_reports_its_own_cost():
    classifier = OneBitHdCamClassifier(make_config())
    assert classifier.cost().n_rows == classifier.n_rows


def test_the_measured_chance_floor_reaches_the_cost():
    classifier = OneBitHdCamClassifier(make_config())
    cost = measure_cost(classifier, n_draws=64)
    assert cost.chance_mean == pytest.approx(classifier.n_columns / 2, rel=0.05)
    assert cost.chance_sd > 0


def test_a_noise_record_is_the_length_the_classifier_acquires():
    classifier = OneBitHdCamClassifier(make_config())
    record = noise_record(classifier)
    assert len(record) == classifier.config.samples_per_acquisition
    assert np.iscomplexobj(record)


# --------------------------------------------------------------------------
# the derived quantities
# --------------------------------------------------------------------------


def test_total_bits_and_bit_comparisons_are_the_products_they_claim():
    cost = a_cost()
    assert cost.total_bits == 1344 * 2046
    assert cost.bit_comparisons == 36_832 * 1344 * 2046


def test_the_study_headline_numbers_come_out_of_this_arithmetic():
    # docs/CAM_FAMILY_STUDY.md predicts 1.0e11 bit comparisons, 19 uJ and 295 us
    # for the baseline. Those are these constants on this geometry.
    cost = a_cost()
    assert cost.bit_comparisons == pytest.approx(1.0e11, rel=0.02)
    assert cost.energy_uj == pytest.approx(19.0, rel=0.05)
    assert cost.latency_us == pytest.approx(295.0, rel=0.01)


def test_energy_and_latency_follow_the_jssc_figures():
    cost = a_cost(n_searches=1000, n_rows=10, n_columns=100)
    assert cost.energy_uj == pytest.approx(
        1000 * 10 * 100 * ENERGY_PER_BIT_FJ * 1e-9
    )
    assert cost.latency_us == pytest.approx(1000 * LATENCY_PER_SEARCH_NS * 1e-3)


def test_the_tolerance_fraction_is_the_threshold_over_the_row_width():
    # This is the number PROPOSAL risk 1 turns on: silicon demonstrates 0.125.
    assert a_cost(hd_threshold=920, n_columns=2046).tolerance_fraction == (
        pytest.approx(920 / 2046)
    )


def test_the_resolution_fraction_is_the_gap_below_the_chance_floor():
    cost = a_cost(hd_threshold=920, chance_mean=1023.0, n_columns=2046)
    assert cost.resolution_fraction == pytest.approx((1023.0 - 920) / 2046)


def test_the_resolution_is_also_reported_in_standard_deviations():
    cost = a_cost(hd_threshold=920, chance_mean=1023.0, chance_sd=22.6)
    assert cost.resolution_sigma == pytest.approx((1023.0 - 920) / 22.6)


def test_a_zero_spread_chance_floor_gives_infinite_resolution():
    assert a_cost(chance_sd=0.0).resolution_sigma == float("inf")


def test_the_table_names_every_quantity():
    printed = a_cost().table()
    for label in ("total bits", "energy", "latency", "tolerance fraction"):
        assert label in printed
    assert len(printed.splitlines()) == 11
