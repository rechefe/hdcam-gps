"""Directed tests for the evaluation harness.

The scoring is driven with a stub classifier that returns exactly what a test
tells it to, so every metric can be checked against a hand counted answer. The
real classifiers appear only in the two end to end tests at the bottom.
"""

import numpy as np
import pytest

from hdcam_gps.acq_base import AcqConfig, GpsL1AcqClassifier, PrnResult
from hdcam_gps.evaluate import (
    EvalConfig,
    EvaluationReport,
    PointMetrics,
    build_scenario,
    code_phase_error,
    evaluate,
    score_scenario,
)
from hdcam_gps.fft_acq import FftAcqClassifier
from hdcam_gps.signal_gen import SatelliteTruth, generate_synthetic


def make_config(**overrides) -> AcqConfig:
    kwargs = {
        "fs_hz": 1.023e6,
        "prn_list": (1, 7, 19),
        "doppler_min_hz": -1000.0,
        "doppler_max_hz": 1000.0,
        "doppler_step_hz": 500.0,
        "n_codes": 2,
    }
    kwargs.update(overrides)
    return AcqConfig(**kwargs)


class StubClassifier(GpsL1AcqClassifier):
    """Returns a fixed answer, so the scoring can be checked exactly."""

    def __init__(self, config: AcqConfig, answer: list[PrnResult]):
        super().__init__(config)
        self.answer = answer
        self.calls = 0

    def _acquire(self, samples: np.ndarray) -> list[PrnResult]:
        self.calls += 1
        return list(self.answer)


def two_satellite_scenario(config: AcqConfig):
    return generate_synthetic(
        config,
        [
            SatelliteTruth(prn=1, doppler_hz=500.0, code_phase=100, cn0_dbhz=50.0),
            SatelliteTruth(prn=7, doppler_hz=-500.0, code_phase=700, cn0_dbhz=50.0),
        ],
        add_noise=False,
    )


# --------------------------------------------------------------------------
# code_phase_error
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reported", "expected", "error"), [(100, 100, 0), (103, 100, 3), (100, 103, 3)]
)
def test_code_phase_error_is_the_plain_difference_in_the_middle(
    reported, expected, error
):
    assert code_phase_error(reported, expected, 1023) == error


def test_code_phase_error_takes_the_short_way_round():
    assert code_phase_error(1022, 0, 1023) == 1
    assert code_phase_error(0, 1022, 1023) == 1


def test_code_phase_error_never_exceeds_half_a_period():
    for reported in range(0, 1023, 37):
        assert code_phase_error(reported, 0, 1023) <= 1023 // 2 + 1


# --------------------------------------------------------------------------
# score_scenario
# --------------------------------------------------------------------------


def test_a_perfect_answer_scores_everything():
    config = make_config()
    scenario = two_satellite_scenario(config)
    classifier = StubClassifier(config, scenario.expected_results())

    score = score_scenario(classifier, scenario, code_phase_tolerance=0)
    assert (score.n_expected, score.n_detected, score.n_correct) == (2, 2, 2)
    assert score.n_false == 0
    assert score.exact
    assert score.doppler_errors_hz == (0.0, 0.0)
    assert score.code_phase_errors == (0, 0)
    assert score.seconds >= 0.0


def test_a_missed_satellite_lowers_detection_but_not_precision():
    config = make_config()
    scenario = two_satellite_scenario(config)
    classifier = StubClassifier(config, scenario.expected_results()[:1])

    score = score_scenario(classifier, scenario, code_phase_tolerance=0)
    assert (score.n_expected, score.n_reported, score.n_detected) == (2, 1, 1)
    assert score.n_false == 0
    assert not score.exact


def test_a_detection_of_an_absent_satellite_is_a_false_alarm():
    config = make_config()
    scenario = two_satellite_scenario(config)
    answer = scenario.expected_results() + [PrnResult(19, 0.0, 0)]
    classifier = StubClassifier(config, answer)

    score = score_scenario(classifier, scenario, code_phase_tolerance=0)
    assert score.n_detected == 2
    assert score.n_false == 1
    assert not score.exact


def test_a_wrong_doppler_counts_as_detected_but_not_correct():
    config = make_config()
    scenario = two_satellite_scenario(config)
    answer = [PrnResult(1, 1000.0, 100), PrnResult(7, -500.0, 700)]
    classifier = StubClassifier(config, answer)

    score = score_scenario(classifier, scenario, code_phase_tolerance=0)
    assert score.n_detected == 2
    assert score.n_correct == 1
    assert 500.0 in [abs(error) for error in score.doppler_errors_hz]


def test_the_code_phase_tolerance_decides_correctness():
    config = make_config()
    scenario = two_satellite_scenario(config)
    answer = [PrnResult(1, 500.0, 101), PrnResult(7, -500.0, 700)]
    classifier = StubClassifier(config, answer)

    tight = score_scenario(classifier, scenario, code_phase_tolerance=0)
    loose = score_scenario(classifier, scenario, code_phase_tolerance=1)
    assert tight.n_correct == 1
    assert loose.n_correct == 2


def test_scoring_a_silent_answer():
    config = make_config()
    scenario = two_satellite_scenario(config)
    score = score_scenario(StubClassifier(config, []), scenario, 0)
    assert (score.n_reported, score.n_detected, score.n_correct) == (0, 0, 0)
    assert not score.exact


# --------------------------------------------------------------------------
# PointMetrics
# --------------------------------------------------------------------------


def a_point(**overrides) -> PointMetrics:
    kwargs = {
        "cn0_dbhz": 45.0, "n_scenarios": 10, "n_expected": 20, "n_reported": 18,
        "n_detected": 16, "n_correct": 15, "n_false": 2, "n_exact": 7,
        "doppler_errors_hz": (0.0, 100.0), "code_phase_errors": (0, 2), "seconds": 5.0,
    }
    kwargs.update(overrides)
    return PointMetrics(**kwargs)


def test_the_rates_are_the_ratios_they_claim():
    point = a_point()
    assert point.detection_rate == pytest.approx(16 / 20)
    assert point.accuracy == pytest.approx(15 / 20)
    assert point.precision == pytest.approx(16 / 18)
    assert point.false_alarms_per_scenario == pytest.approx(2 / 10)
    assert point.exact_rate == pytest.approx(7 / 10)
    assert point.seconds_per_scenario == pytest.approx(0.5)


def test_the_error_metrics_are_root_mean_squares():
    point = a_point(doppler_errors_hz=(3.0, 4.0), code_phase_errors=(0, 4))
    assert point.doppler_rmse_hz == pytest.approx(np.sqrt((9 + 16) / 2))
    assert point.code_phase_rmse == pytest.approx(np.sqrt(16 / 2))


def test_an_empty_point_reports_zeros_rather_than_dividing_by_zero():
    point = a_point(
        n_scenarios=0, n_expected=0, n_reported=0, n_detected=0,
        n_correct=0, n_false=0, n_exact=0,
        doppler_errors_hz=(), code_phase_errors=(), seconds=0.0,
    )
    assert point.detection_rate == 0.0
    assert point.precision == 0.0
    assert point.doppler_rmse_hz == 0.0
    assert point.code_phase_rmse == 0.0
    assert point.seconds_per_scenario == 0.0


# --------------------------------------------------------------------------
# EvaluationReport
# --------------------------------------------------------------------------


def a_report(accuracies) -> EvaluationReport:
    points = tuple(
        a_point(cn0_dbhz=cn0, n_expected=10, n_correct=round(10 * accuracy))
        for cn0, accuracy in accuracies
    )
    return EvaluationReport(points=points, config=EvalConfig())


def test_sensitivity_is_the_weakest_point_still_working():
    report = a_report([(48.0, 1.0), (42.0, 1.0), (36.0, 0.5)])
    assert report.sensitivity_cn0_dbhz(0.9) == 42.0


def test_sensitivity_is_none_when_nothing_works():
    assert a_report([(48.0, 0.2)]).sensitivity_cn0_dbhz(0.9) is None


def test_the_table_has_a_row_per_point():
    report = a_report([(48.0, 1.0), (42.0, 0.5)])
    lines = report.table().splitlines()
    assert len(lines) == 4  # header, rule, two rows
    assert "48" in lines[2] and "42" in lines[3]


# --------------------------------------------------------------------------
# evaluate
# --------------------------------------------------------------------------


def test_evaluate_returns_one_point_per_cn0():
    config = make_config()
    classifier = StubClassifier(config, [])
    sweep = EvalConfig(
        n_scenarios=2, cn0_dbhz=(60.0, 50.0), progress=False, n_satellites=1
    )

    report = evaluate(classifier, sweep)
    assert [point.cn0_dbhz for point in report.points] == [60.0, 50.0]
    assert all(point.n_scenarios == 2 for point in report.points)
    assert classifier.calls == 4
    assert report.config is sweep


@pytest.mark.parametrize(
    "bad", [{"n_scenarios": 0}, {"cn0_dbhz": ()}]
)
def test_evaluate_needs_something_to_run(bad):
    config = make_config()
    with pytest.raises(AssertionError):
        evaluate(StubClassifier(config, []), EvalConfig(progress=False, **bad))


def test_build_scenario_is_reproducible_and_uses_the_classifier_config():
    config = make_config()
    classifier = StubClassifier(config, [])
    sweep = EvalConfig(n_satellites=2, seed=3, progress=False)

    first = build_scenario(classifier, sweep, 45.0, index=1)
    second = build_scenario(classifier, sweep, 45.0, index=1)
    assert first.truth == second.truth
    assert first.config is config
    assert len(first.truth) == 2
    assert all(s.cn0_dbhz == 45.0 for s in first.truth)


def test_build_scenario_varies_with_the_index():
    config = make_config()
    classifier = StubClassifier(config, [])
    sweep = EvalConfig(n_satellites=1, progress=False)
    assert (
        build_scenario(classifier, sweep, 45.0, 0).truth
        != build_scenario(classifier, sweep, 45.0, 1).truth
    )


# --------------------------------------------------------------------------
# end to end, with a real classifier
# --------------------------------------------------------------------------


def test_the_reference_classifier_scores_perfectly_on_strong_signals():
    config = make_config()
    sweep = EvalConfig(
        n_scenarios=3, n_satellites=1, cn0_dbhz=(60.0,), progress=False
    )
    report = evaluate(FftAcqClassifier(config), sweep)

    point = report.points[0]
    assert point.accuracy == 1.0
    assert point.false_alarms_per_scenario == 0.0
    assert point.exact_rate == 1.0


def test_accuracy_falls_away_as_the_signal_weakens():
    config = make_config()
    sweep = EvalConfig(
        n_scenarios=4, n_satellites=1, cn0_dbhz=(60.0, 30.0), progress=False
    )
    report = evaluate(FftAcqClassifier(config), sweep)

    strong, weak = report.points
    assert strong.accuracy == 1.0
    assert weak.accuracy < strong.accuracy
    assert report.sensitivity_cn0_dbhz(0.9) == 60.0
