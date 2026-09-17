"""Directed tests for the separation screen.

The screen makes no decision, so what these check is that it finds the right
answer and measures the right population: that true_targets locates a planted
satellite's own (row, query) pairs, that D_true collapses to nothing on a
noiseless replica, and that D_wrong sits at the chance floor. The sampling rate
is deliberately low, so a whole screen runs in a second.

What is not covered here: whether the numbers the screen produces are the ones a
family deserves. That is the study, and notebooks/family_screen.ipynb owns it.
"""

import numpy as np
import pytest

from hdcam_gps.acq_base import AcqConfig
from hdcam_gps.hdcam_acq import OneBitHdCamClassifier
from hdcam_gps.screen import (
    CHANCE_TOLERANCE_FRACTION,
    SILICON_TOLERANCE_FRACTION,
    ScreenBin,
    Separation,
    kill_verdict,
    screen,
    screen_record,
    true_targets,
)
from hdcam_gps.signal_gen import SatelliteTruth, Scenario, generate_synthetic

FS_HZ = 204.6e3  # 204 samples per code period, enough to stay correct and cheap
N_COLUMNS = 408


def make_config(**overrides) -> AcqConfig:
    kwargs = {
        "fs_hz": FS_HZ,
        "prn_list": (1, 2, 3, 4),
        "doppler_min_hz": -500.0,
        "doppler_max_hz": 500.0,
        "doppler_step_hz": 500.0,
        "n_codes": 4,
    }
    kwargs.update(overrides)
    return AcqConfig(**kwargs)


def a_scenario(config: AcqConfig, cn0_dbhz: float = 50.0, seed: int = 0) -> Scenario:
    return generate_synthetic(
        config,
        [SatelliteTruth(prn=2, doppler_hz=500.0, code_phase=37, cn0_dbhz=cn0_dbhz)],
        seed=seed,
    )


def a_separation(**overrides) -> Separation:
    kwargs = dict(
        family="test",
        n_rows=24,
        n_columns=N_COLUMNS,
        n_records=4,
        true_distance=np.array([100, 110, 120, 130] * 10),
        true_cn0_dbhz=np.array([45.0] * 40),
        wrong_mean=204.0,
        wrong_sd=10.0,
        n_wrong=100_000,
        noise_mean=204.0,
        noise_sd=10.1,
    )
    kwargs.update(overrides)
    return Separation(**kwargs)


# --------------------------------------------------------------------------
# true_targets - finding the answer the screen measures the distance to
# --------------------------------------------------------------------------


def test_true_targets_finds_the_rows_of_the_planted_hypothesis():
    config = make_config()
    classifier = OneBitHdCamClassifier(config)
    satellite = SatelliteTruth(prn=2, doppler_hz=500.0, code_phase=37)

    (found, rows, looks), = true_targets(classifier, satellite, code_phase_tolerance=0)
    assert set(rows) == {
        classifier.row_of(2, doppler_bin=2, codebook_phase=phase)
        for phase in range(classifier.n_codebook_phases)
    }
    assert (config.samples_per_code * looks + 37 == classifier.queries().start[found]).all()


def test_true_targets_gives_one_query_per_look():
    config = make_config()
    classifier = OneBitHdCamClassifier(config)
    satellite = SatelliteTruth(prn=1, doppler_hz=0.0, code_phase=5)

    (found, _, looks), = true_targets(classifier, satellite, code_phase_tolerance=0)
    assert len(found) == len(set(looks.tolist()))
    assert len(found) >= classifier.n_looks


@pytest.mark.parametrize("tolerance", [0, 1, 2])
def test_the_tolerance_widens_the_queries_accepted(tolerance):
    # gps_sdr_sim.labels rounds a fractional code phase to a sample, and at one
    # sample per chip the labelled sample is often not the one the signal is at.
    config = make_config()
    classifier = OneBitHdCamClassifier(config)
    satellite = SatelliteTruth(prn=1, doppler_hz=0.0, code_phase=50)

    (found, _, _), = true_targets(classifier, satellite, tolerance)
    phases = classifier.queries().start[found] % config.samples_per_code
    assert set(phases.tolist()) == {50 + d for d in range(-tolerance, tolerance + 1)}


def test_a_code_phase_at_the_wrap_still_finds_its_neighbours():
    config = make_config()
    classifier = OneBitHdCamClassifier(config)
    satellite = SatelliteTruth(prn=1, doppler_hz=0.0, code_phase=0)

    (found, _, _), = true_targets(classifier, satellite, code_phase_tolerance=1)
    phases = set(
        (classifier.queries().start[found] % config.samples_per_code).tolist()
    )
    assert phases == {0, 1, config.samples_per_code - 1}


# --------------------------------------------------------------------------
# screen_record - the two distributions
# --------------------------------------------------------------------------


def test_a_noiseless_replica_sits_on_top_of_its_own_row():
    # No noise and a stored carrier phase to land on, so the only distance left
    # is the quantisation of a replica the codebook holds exactly.
    config = make_config()
    classifier = OneBitHdCamClassifier(config)
    scenario = generate_synthetic(
        config,
        [SatelliteTruth(prn=3, doppler_hz=-500.0, code_phase=11)],
        add_noise=False,
    )
    per_satellite, _ = screen_record(classifier, scenario)
    (_, looks), = per_satellite
    assert looks.size >= classifier.n_looks
    assert looks.max() < N_COLUMNS / 8  # far below the chance floor of 204


def test_the_wrong_rows_sit_near_the_chance_floor():
    # Near, not at: the satellite present cross correlates with the other PRNs'
    # codes, which is exactly why the screen measures this floor rather than
    # assuming n_columns / 2.
    config = make_config()
    classifier = OneBitHdCamClassifier(config)
    _, (total, _, count) = screen_record(classifier, a_scenario(config))
    assert total / count == pytest.approx(N_COLUMNS / 2, rel=0.1)


def test_only_the_absent_prns_count_as_wrong():
    # The satellite that is present must not be in D_wrong, or the floor it
    # defines would include the answer the threshold is meant to let through.
    config = make_config()
    classifier = OneBitHdCamClassifier(config)
    scenario = generate_synthetic(
        config,
        [SatelliteTruth(prn=2, doppler_hz=500.0, code_phase=37, cn0_dbhz=60.0)],
        add_noise=False,
    )
    _, (total, _, count) = screen_record(classifier, scenario)
    rows_per_prn = classifier.n_rows // len(config.prn_list)
    assert count == classifier.n_queries * (classifier.n_rows - rows_per_prn)


def test_a_record_with_no_satellite_measures_only_the_floor():
    config = make_config()
    classifier = OneBitHdCamClassifier(config)
    per_satellite, (_, _, count) = screen_record(
        classifier, generate_synthetic(config, [], seed=1)
    )
    assert per_satellite == []
    assert count == classifier.n_queries * classifier.n_rows


# --------------------------------------------------------------------------
# screen - one family over a set of records
# --------------------------------------------------------------------------


def test_the_screen_pools_every_record_and_satellite():
    config = make_config()
    classifier = OneBitHdCamClassifier(config)
    records = [a_scenario(config, seed=seed) for seed in range(3)]
    separation = screen(classifier, records, progress=False, noise_draws=16)

    assert separation.n_records == 3
    assert separation.family == "OneBitHdCamClassifier"
    assert separation.n_columns == N_COLUMNS
    assert separation.true_distance.size >= 3 * classifier.n_looks
    assert separation.true_distance.size == separation.true_cn0_dbhz.size


def test_the_screen_needs_a_record():
    with pytest.raises(AssertionError):
        screen(OneBitHdCamClassifier(make_config()), [], progress=False)


def test_a_stronger_satellite_sits_closer_to_its_row():
    config = make_config()
    classifier = OneBitHdCamClassifier(config)
    loud = screen(
        classifier,
        [a_scenario(config, cn0_dbhz=55.0, seed=s) for s in range(3)],
        progress=False,
        noise_draws=16,
    )
    quiet = screen(
        classifier,
        [a_scenario(config, cn0_dbhz=35.0, seed=s) for s in range(3)],
        progress=False,
        noise_draws=16,
    )
    assert loud.true_distance.mean() < quiet.true_distance.mean()


def test_the_floor_is_the_rotation_minimum_and_not_the_satellites():
    # The floor sits below chance because distance_table keeps the best of four
    # query rotations, and a minimum over four draws is below their mean. It is
    # not cross correlation: a record holding no satellite at all gives the same
    # floor as one holding a strong one.
    config = make_config()
    classifier = OneBitHdCamClassifier(config)
    loud = screen(
        classifier,
        [a_scenario(config, cn0_dbhz=60.0, seed=s) for s in range(3)],
        progress=False,
        noise_draws=64,
    )
    empty = screen(
        classifier,
        [generate_synthetic(config, [], seed=s) for s in range(3)],
        progress=False,
        noise_draws=64,
    )
    assert loud.wrong_mean == pytest.approx(empty.wrong_mean, rel=0.01)
    assert loud.wrong_mean < 0.98 * (N_COLUMNS / 2)


def test_the_noise_column_measures_one_variant_and_so_lands_at_chance():
    # chance_floor takes every variant separately where distance_table takes
    # their minimum, so the two columns are not comparable and their difference
    # is the rotation minimum.
    classifier = OneBitHdCamClassifier(make_config())
    mean, _ = classifier.chance_floor(n_draws=128)
    assert mean == pytest.approx(N_COLUMNS / 2, rel=0.01)


# --------------------------------------------------------------------------
# the derived numbers
# --------------------------------------------------------------------------


def test_the_tolerance_fraction_is_the_percentile_over_the_row_width():
    entry, = a_separation().bins(width=4.0)
    assert entry.true_percentile == pytest.approx(np.percentile([100, 110, 120, 130] * 10, 90))
    assert entry.tolerance_fraction == pytest.approx(entry.true_percentile / N_COLUMNS)


def test_d_prime_is_the_gap_in_standard_deviations_of_the_floor():
    entry, = a_separation().bins(width=4.0)
    assert entry.d_prime == pytest.approx((204.0 - 115.0) / 10.0)


def test_the_resolution_is_what_is_left_between_the_percentile_and_the_floor():
    entry, = a_separation().bins(width=4.0)
    assert entry.resolution_fraction == pytest.approx(
        (204.0 - entry.true_percentile) / N_COLUMNS
    )
    assert entry.resolution_sigma == pytest.approx((204.0 - entry.true_percentile) / 10.0)


def test_a_bin_with_too_few_looks_is_dropped():
    separation = a_separation(
        true_distance=np.array([100, 110]), true_cn0_dbhz=np.array([45.0, 45.0])
    )
    assert separation.bins(width=4.0, min_looks=20) == []
    assert len(separation.bins(width=4.0, min_looks=2)) == 1


def test_bins_come_back_in_C_over_N0_order():
    separation = a_separation(
        true_distance=np.tile(np.array([100, 110, 120, 130]), 10),
        true_cn0_dbhz=np.repeat([40.0, 50.0], 20),
    )
    centres = [entry.cn0_dbhz for entry in separation.bins(width=4.0, min_looks=5)]
    assert centres == sorted(centres)


def test_at_finds_the_bin_holding_a_given_cn0():
    separation = a_separation()
    assert separation.at(45.0, width=4.0) is not None
    assert separation.at(20.0, width=4.0) is None


def test_a_tolerance_silicon_has_demonstrated_is_marked_buildable():
    assert ScreenBin(45.0, 100, 40.0, 50.0, 5.0, 0.12, 0.3, 4.0).buildable
    assert not ScreenBin(45.0, 100, 40.0, 50.0, 5.0, 0.13, 0.3, 4.0).buildable
    assert SILICON_TOLERANCE_FRACTION < CHANCE_TOLERANCE_FRACTION


def test_the_table_has_a_line_per_bin_and_a_header_naming_the_family():
    lines = a_separation().table(width=4.0).splitlines()
    assert "test" in lines[0]
    assert len(lines) == 4  # family, header, rule, one bin


# --------------------------------------------------------------------------
# the kill rule, fixed before the runs
# --------------------------------------------------------------------------


def test_a_family_below_d_prime_one_is_dead():
    weak = a_separation(family="weak", wrong_mean=120.0)  # gap of 5 over sd 10
    assert "dead" in kill_verdict(weak, width=4.0)


def test_a_family_below_half_the_baseline_is_dead():
    baseline = a_separation(family="baseline", wrong_mean=304.0)  # d' = 18.9
    family = a_separation(family="family", wrong_mean=140.0)  # d' = 2.5
    assert "dead" in kill_verdict(family, baseline, width=4.0)
    assert "survives" in kill_verdict(family, width=4.0)


def test_a_family_that_clears_both_halves_survives():
    baseline = a_separation(family="baseline")
    assert "survives" in kill_verdict(a_separation(), baseline, width=4.0)


def test_a_family_with_nothing_in_the_bin_gets_no_verdict():
    separation = a_separation(true_cn0_dbhz=np.array([20.0] * 40))
    assert "no 45 dB-Hz bin" in kill_verdict(separation, width=4.0)


def test_the_verdict_names_the_number_behind_it():
    assert "d' = " in kill_verdict(a_separation(), width=4.0)
