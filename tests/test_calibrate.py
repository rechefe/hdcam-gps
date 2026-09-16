"""Directed tests for the calibration search.

The statistics are checked against hand computable answers, the replay is checked
against the classifier it is standing in for, and the search itself is run on a
configuration small enough to be cheap.
"""

import numpy as np
import pytest

from hdcam_gps.acq_base import AcqConfig, PrnResult
from hdcam_gps.calibrate import (
    Candidate,
    CalibrationResult,
    OperatingPoint,
    binomial_upper_bound,
    calibrate,
    score_table,
)
from hdcam_gps.hdcam_acq import OneBitHdCamClassifier
from hdcam_gps.signal_gen import SatelliteTruth, generate_synthetic


def make_config(**overrides) -> AcqConfig:
    kwargs = dict(
        fs_hz=204.6e3,
        prn_list=(1, 2),
        doppler_min_hz=-500.0,
        doppler_max_hz=500.0,
        doppler_step_hz=500.0,
        n_codes=3,
    )
    kwargs.update(overrides)
    return AcqConfig(**kwargs)


# --------------------------------------------------------------------------
# binomial_upper_bound
# --------------------------------------------------------------------------


def test_no_events_still_bounds_the_rate_near_three_over_n():
    # The rule of three: seeing nothing in 100 trials only rules out rates above
    # about 3 percent.
    assert binomial_upper_bound(0, 100) == pytest.approx(0.03, abs=0.005)


def test_the_bound_tightens_with_more_trials():
    assert binomial_upper_bound(0, 1000) < binomial_upper_bound(0, 100)


def test_the_bound_loosens_with_more_events():
    assert binomial_upper_bound(5, 100) > binomial_upper_bound(1, 100)


def test_the_bound_is_above_the_point_estimate():
    assert binomial_upper_bound(10, 100) > 0.10


@pytest.mark.parametrize(("successes", "trials"), [(0, 0), (10, 10), (11, 10)])
def test_degenerate_bounds_are_one(successes, trials):
    assert binomial_upper_bound(successes, trials) == 1.0


# --------------------------------------------------------------------------
# Candidate
# --------------------------------------------------------------------------


def a_candidate(**overrides) -> Candidate:
    kwargs = dict(
        hd_threshold=900, min_votes=3, n_trials=100, n_detected=95, n_false=0
    )
    kwargs.update(overrides)
    return Candidate(**kwargs)


def test_the_rates_are_the_ratios_they_claim():
    candidate = a_candidate(n_detected=90, n_false=2)
    assert candidate.pmd == pytest.approx(0.10)
    assert candidate.pfa == pytest.approx(0.02)


def test_a_candidate_meets_a_target_it_satisfies():
    target = OperatingPoint(max_pmd=0.10, max_pfa=0.05)
    assert a_candidate(n_detected=95, n_false=0).meets(target)


def test_too_many_misses_fails_the_target():
    target = OperatingPoint(max_pmd=0.10, max_pfa=0.5)
    assert not a_candidate(n_detected=80).meets(target)


def test_the_false_alarm_side_is_judged_on_the_confidence_bound():
    # Zero false alarms in a hundred trials does not demonstrate a rate of 1e-4,
    # so a demanding target cannot be met just by running few trials.
    strict = OperatingPoint(max_pmd=0.10, max_pfa=1e-4)
    lenient = OperatingPoint(max_pmd=0.10, max_pfa=0.05)
    candidate = a_candidate(n_trials=100, n_detected=95, n_false=0)
    assert candidate.pfa == 0.0
    assert not candidate.meets(strict)
    assert candidate.meets(lenient)


# --------------------------------------------------------------------------
# CalibrationResult
# --------------------------------------------------------------------------


def a_result(candidates, target=None) -> CalibrationResult:
    target = target or OperatingPoint(max_pmd=0.10, max_pfa=0.05)
    return CalibrationResult(
        target=target, candidates=tuple(candidates), n_trials=100
    )


def test_the_best_is_the_fewest_misses_among_those_that_qualify():
    result = a_result(
        [
            a_candidate(hd_threshold=900, n_detected=99, n_false=20),  # too noisy
            a_candidate(hd_threshold=880, n_detected=95, n_false=0),  # qualifies
            a_candidate(hd_threshold=860, n_detected=92, n_false=0),  # worse Pmd
        ]
    )
    assert result.best.hd_threshold == 880


def test_there_is_no_best_when_nothing_qualifies():
    assert a_result([a_candidate(n_detected=50, n_false=50)]).best is None


def test_trials_needed_follows_the_rule_of_three():
    result = a_result([a_candidate()], OperatingPoint(max_pfa=1e-4))
    assert result.trials_needed == 30_000


def test_the_table_lists_every_candidate():
    result = a_result([a_candidate(hd_threshold=900), a_candidate(hd_threshold=880)])
    lines = result.table().splitlines()
    assert len(lines) == 4  # header, rule, two candidates
    assert "900" in lines[2] and "880" in lines[3]


# --------------------------------------------------------------------------
# score_table - the replay has to agree with the classifier
# --------------------------------------------------------------------------


def test_the_replay_reproduces_the_classifier():
    config = make_config()
    classifier = OneBitHdCamClassifier(config)
    scenario = generate_synthetic(
        config, [SatelliteTruth(1, 500.0, 40, cn0_dbhz=60.0)], seed=0
    )
    table = classifier.distance_table(scenario.samples)

    replayed = score_table(
        classifier, table, classifier.hd_threshold, classifier.min_votes
    )
    assert replayed == classifier.acquire(scenario.samples)


def test_the_replay_reproduces_the_classifier_on_silence():
    config = make_config()
    classifier = OneBitHdCamClassifier(config)
    samples = np.zeros(config.samples_per_acquisition, dtype=complex)
    table = classifier.distance_table(samples)
    assert score_table(
        classifier, table, classifier.hd_threshold, classifier.min_votes
    ) == classifier.acquire(samples)


def test_a_tighter_threshold_never_reports_more():
    config = make_config()
    classifier = OneBitHdCamClassifier(config)
    scenario = generate_synthetic(
        config, [SatelliteTruth(1, 0.0, 20, cn0_dbhz=60.0)], seed=1
    )
    table = classifier.distance_table(scenario.samples)
    loose = score_table(classifier, table, classifier.hd_threshold, 1)
    tight = score_table(classifier, table, 20, 1)
    assert len(tight) <= len(loose)


def test_the_distance_table_has_a_row_per_start():
    config = make_config()
    classifier = OneBitHdCamClassifier(config)
    scenario = generate_synthetic(config, [], add_noise=True, seed=2)
    table = classifier.distance_table(scenario.samples)

    n_starts = config.samples_per_acquisition - config.samples_per_code + 1
    assert table.shape == (n_starts, classifier.n_rows)
    assert (table >= 0).all() and (table <= classifier.n_columns).all()


# --------------------------------------------------------------------------
# calibrate
# --------------------------------------------------------------------------


def test_calibrate_scores_every_setting_on_the_same_trials():
    result = calibrate(
        make_config(),
        OperatingPoint(cn0_dbhz=60.0, max_pmd=0.5, max_pfa=0.5),
        n_trials=3,
        sigma_grid=(2.0, 4.0),
        vote_fractions=(0.5,),
        progress=False,
    )
    assert len(result.candidates) == 2
    assert all(c.n_trials == 3 for c in result.candidates)
    assert result.n_trials == 3


def test_calibrate_orders_candidates_by_missed_detections():
    result = calibrate(
        make_config(),
        OperatingPoint(cn0_dbhz=60.0, max_pmd=0.5, max_pfa=0.5),
        n_trials=3,
        sigma_grid=(2.0, 3.0, 4.0),
        vote_fractions=(0.5,),
        progress=False,
    )
    assert [c.pmd for c in result.candidates] == sorted(
        c.pmd for c in result.candidates
    )


def test_calibrate_finds_a_working_setting_for_a_loose_target():
    # A tiny codebook cross correlates badly, so only the detection side is
    # asked for here; the false alarm side is exercised by the unit tests above.
    result = calibrate(
        make_config(),
        OperatingPoint(cn0_dbhz=60.0, max_pmd=0.4, max_pfa=1.0),
        n_trials=4,
        sigma_grid=(2.0, 3.0, 4.0),
        vote_fractions=(0.5,),
        progress=False,
    )
    assert result.best is not None
    assert result.best.pmd <= 0.4


def test_calibrate_reports_no_winner_for_an_unreachable_target():
    result = calibrate(
        make_config(),
        OperatingPoint(cn0_dbhz=10.0, max_pmd=0.0, max_pfa=1e-9),
        n_trials=2,
        sigma_grid=(3.0,),
        vote_fractions=(0.5,),
        progress=False,
    )
    assert result.best is None


def test_calibrate_needs_a_trial():
    with pytest.raises(AssertionError):
        calibrate(make_config(), n_trials=0, progress=False)
