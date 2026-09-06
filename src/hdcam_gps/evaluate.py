"""Scoring an acquisition classifier against labelled scenarios.

A classifier is anything with the GpsL1AcqClassifier interface: it carries the
AcqConfig it was built for and turns samples into a list of PrnResult. That is
all this module needs, so the reference correlator and the HdCam classifiers go
through exactly the same measurement.

Scenarios come from signal_gen, so every satellite in them has a known PRN,
Doppler and code phase, and nothing is scored against another classifier's
opinion. C/N0 is a sweep rather than a single value, because the useful answer is
the point at which a classifier stops working, not its score at one operating
point.
"""

import datetime
import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from tqdm.auto import tqdm

from hdcam_gps.acq_base import GpsL1AcqClassifier
from hdcam_gps.signal_gen import Scenario, generate_from_sim, random_scenario

Backend = Literal["synthetic", "simulator"]

DEFAULT_CN0_SWEEP: tuple[float, ...] = (48.0, 45.0, 42.0, 39.0, 36.0, 33.0)
# The simulator's sky is fixed by the time and place, so scenarios are spaced out
# in time to get a different constellation for each one.
SCENARIO_SPACING = datetime.timedelta(minutes=17)
SIM_BASE_TIME = datetime.datetime(2022, 1, 1, 0, 0, 0)


@dataclass(frozen=True)
class EvalConfig:
    """What to test a classifier on."""

    n_scenarios: int = 10  # Scenarios per C/N0 point
    cn0_dbhz: tuple[float, ...] = DEFAULT_CN0_SWEEP  # The sweep to walk
    backend: Backend = "synthetic"  # Where the scenarios come from
    n_satellites: int = 1  # Satellites per scenario, synthetic backend only
    nav_data: bool = False  # Modulate with navigation data, synthetic only
    code_phase_tolerance: int = 1  # Samples of code phase error to forgive
    seed: int = 0  # Seed of the scenario draw
    progress: bool = True  # Show a progress bar while the sweep runs
    latitude_deg: float = 32.0  # Receiver position, simulator backend only
    longitude_deg: float = 35.0
    height_m: float = 100.0


@dataclass(frozen=True)
class ScenarioScore:
    """How a classifier did on one scenario."""

    cn0_dbhz: float  # The C/N0 the scenario was generated at
    n_expected: int  # Satellites actually present
    n_reported: int  # Detections the classifier returned
    n_detected: int  # Present satellites it found, whatever it said about them
    n_correct: int  # Found, with the right Doppler bin and code phase
    n_false: int  # Detections of satellites that were not there
    doppler_errors_hz: tuple[float, ...]  # Per found satellite
    code_phase_errors: tuple[int, ...]  # Per found satellite, in samples
    exact: bool  # Every satellite found and nothing extra
    seconds: float  # Wall clock time of the acquisition


@dataclass(frozen=True)
class PointMetrics:
    """The aggregate of every scenario run at one C/N0."""

    cn0_dbhz: float
    n_scenarios: int
    n_expected: int
    n_reported: int
    n_detected: int
    n_correct: int
    n_false: int
    n_exact: int
    doppler_errors_hz: tuple[float, ...] = field(repr=False, default=())
    code_phase_errors: tuple[int, ...] = field(repr=False, default=())
    seconds: float = 0.0

    @property
    def detection_rate(self) -> float:
        """Fraction of present satellites the classifier found."""
        return _ratio(self.n_detected, self.n_expected)

    @property
    def accuracy(self) -> float:
        """Fraction of present satellites found with the right numbers."""
        return _ratio(self.n_correct, self.n_expected)

    @property
    def precision(self) -> float:
        """Fraction of the classifier's detections that were real."""
        return _ratio(self.n_detected, self.n_reported)

    @property
    def false_alarms_per_scenario(self) -> float:
        """Detections of absent satellites, per scenario."""
        return _ratio(self.n_false, self.n_scenarios)

    @property
    def exact_rate(self) -> float:
        """Fraction of scenarios answered completely, with nothing extra."""
        return _ratio(self.n_exact, self.n_scenarios)

    @property
    def doppler_rmse_hz(self) -> float:
        """Root mean square Doppler error over the satellites that were found."""
        return _rmse(self.doppler_errors_hz)

    @property
    def code_phase_rmse(self) -> float:
        """Root mean square code phase error, in samples, over those found."""
        return _rmse(self.code_phase_errors)

    @property
    def seconds_per_scenario(self) -> float:
        """Mean wall clock time of one acquisition."""
        return _ratio(self.seconds, self.n_scenarios)


@dataclass(frozen=True)
class EvaluationReport:
    """Everything measured, one entry per C/N0 of the sweep."""

    points: tuple[PointMetrics, ...]
    config: EvalConfig

    def sensitivity_cn0_dbhz(self, min_accuracy: float = 0.9) -> float | None:
        """The weakest C/N0 the classifier still works at.

        Args:
            min_accuracy (float): The accuracy that counts as working.

        Returns:
            float | None: The lowest C/N0 of the sweep meeting it, or None when
                no point did.
        """
        meeting = [p.cn0_dbhz for p in self.points if p.accuracy >= min_accuracy]
        return min(meeting) if meeting else None

    def table(self) -> str:
        """Renders the report as a fixed width table.

        Returns:
            str: One row per C/N0 of the sweep.
        """
        header = (
            f"{'C/N0':>6} {'detect':>7} {'accur':>7} {'precis':>7} "
            f"{'exact':>7} {'FA/scen':>8} {'dopp RMSE':>10} {'phase RMSE':>11} "
            f"{'s/scen':>8}"
        )
        rows = [header, "-" * len(header)]
        for point in self.points:
            rows.append(
                f"{point.cn0_dbhz:6.0f} {point.detection_rate:7.2f} "
                f"{point.accuracy:7.2f} {point.precision:7.2f} "
                f"{point.exact_rate:7.2f} {point.false_alarms_per_scenario:8.2f} "
                f"{point.doppler_rmse_hz:9.1f}H {point.code_phase_rmse:10.2f}s "
                f"{point.seconds_per_scenario:8.3f}"
            )
        return "\n".join(rows)


def _ratio(numerator: float, denominator: float) -> float:
    """Divides, treating an empty denominator as nothing rather than an error."""
    return float(numerator) / float(denominator) if denominator else 0.0


def _rmse(errors) -> float:
    """Root mean square of a run of errors, zero when there are none."""
    return float(np.sqrt(np.mean(np.square(errors)))) if len(errors) else 0.0


def code_phase_error(reported: int, expected: int, samples_per_code: int) -> int:
    """The code phase error, taking the shorter way round the code period.

    Args:
        reported (int): What the classifier said.
        expected (int): What the scenario holds.
        samples_per_code (int): The length of a code period in samples.

    Returns:
        int: The error in samples, never more than half a code period.
    """
    error = abs(reported - expected)
    return min(error, samples_per_code - error)


def score_scenario(
    classifier: GpsL1AcqClassifier, scenario: Scenario, code_phase_tolerance: int
) -> ScenarioScore:
    """Runs one scenario through a classifier and scores the answer.

    Args:
        classifier (GpsL1AcqClassifier): The classifier under test.
        scenario (Scenario): A labelled record to acquire.
        code_phase_tolerance (int): Samples of code phase error to forgive.

    Returns:
        ScenarioScore: What the classifier got right and wrong.
    """
    started = time.perf_counter()
    results = classifier.acquire(scenario.samples)
    seconds = time.perf_counter() - started

    expected = {satellite.prn: satellite for satellite in scenario.truth}
    wanted = {result.prn: result for result in scenario.expected_results()}
    reported = {result.prn: result for result in results}
    samples_per_code = scenario.config.samples_per_code

    n_correct = 0
    doppler_errors: list[float] = []
    code_phase_errors: list[int] = []
    for prn, result in reported.items():
        if prn not in expected:
            continue
        doppler_errors.append(result.doppler_hz - expected[prn].doppler_hz)
        error = code_phase_error(
            result.code_phase, expected[prn].code_phase, samples_per_code
        )
        code_phase_errors.append(error)
        if (
            result.doppler_hz == wanted[prn].doppler_hz
            and error <= code_phase_tolerance
        ):
            n_correct += 1

    detected = set(reported) & set(wanted)
    return ScenarioScore(
        cn0_dbhz=scenario.truth[0].cn0_dbhz if scenario.truth else float("nan"),
        n_expected=len(wanted),
        n_reported=len(reported),
        n_detected=len(detected),
        n_correct=n_correct,
        n_false=len(set(reported) - set(wanted)),
        doppler_errors_hz=tuple(doppler_errors),
        code_phase_errors=tuple(code_phase_errors),
        exact=scenario.matches(results, code_phase_tolerance),
        seconds=seconds,
    )


def build_scenario(
    classifier: GpsL1AcqClassifier,
    eval_config: EvalConfig,
    cn0_dbhz: float,
    index: int,
) -> Scenario:
    """Builds the index'th scenario of a C/N0 point.

    Args:
        classifier (GpsL1AcqClassifier): Supplies the AcqConfig to build for.
        eval_config (EvalConfig): What kind of scenario to build.
        cn0_dbhz (float): The C/N0 to build it at.
        index (int): Which scenario of the point this is.

    Returns:
        Scenario: A labelled record.
    """
    seed = eval_config.seed + index
    if eval_config.backend == "synthetic":
        return random_scenario(
            classifier.config,
            n_satellites=eval_config.n_satellites,
            cn0_dbhz=cn0_dbhz,
            seed=seed,
        )
    start = SIM_BASE_TIME + index * SCENARIO_SPACING
    return generate_from_sim(
        classifier.config,
        latitude_deg=eval_config.latitude_deg,
        longitude_deg=eval_config.longitude_deg,
        height_m=eval_config.height_m,
        cn0_dbhz=cn0_dbhz,
        seed=seed,
        start_time=start.strftime("%Y/%m/%d,%H:%M:%S"),
    )


def evaluate(
    classifier: GpsL1AcqClassifier, eval_config: EvalConfig
) -> EvaluationReport:
    """Runs a classifier over a sweep of scenarios and measures how it did.

    Args:
        classifier (GpsL1AcqClassifier): The classifier under test. The AcqConfig
            it carries defines the record the scenarios are built to.
        eval_config (EvalConfig): How many scenarios, at what C/N0, from where,
            and whether to show a progress bar.

    Returns:
        EvaluationReport: One PointMetrics per C/N0 of the sweep.
    """
    if eval_config is None:
        eval_config = EvalConfig()
    assert eval_config.n_scenarios >= 1, "At least one scenario is needed."
    assert eval_config.cn0_dbhz, "At least one C/N0 is needed."

    points = []
    total = len(eval_config.cn0_dbhz) * eval_config.n_scenarios
    bar = tqdm(
        total=total,
        disable=not eval_config.progress,
        unit="scenario",
        desc=type(classifier).__name__,
        leave=False,
    )
    for cn0_dbhz in eval_config.cn0_dbhz:
        bar.set_postfix_str(f"{cn0_dbhz:.0f} dB-Hz")
        scores = []
        for index in range(eval_config.n_scenarios):
            scores.append(
                score_scenario(
                    classifier,
                    build_scenario(classifier, eval_config, cn0_dbhz, index),
                    eval_config.code_phase_tolerance,
                )
            )
            bar.update(1)
        points.append(
            PointMetrics(
                cn0_dbhz=cn0_dbhz,
                n_scenarios=len(scores),
                n_expected=sum(s.n_expected for s in scores),
                n_reported=sum(s.n_reported for s in scores),
                n_detected=sum(s.n_detected for s in scores),
                n_correct=sum(s.n_correct for s in scores),
                n_false=sum(s.n_false for s in scores),
                n_exact=sum(s.exact for s in scores),
                doppler_errors_hz=tuple(
                    error for s in scores for error in s.doppler_errors_hz
                ),
                code_phase_errors=tuple(
                    error for s in scores for error in s.code_phase_errors
                ),
                seconds=sum(s.seconds for s in scores),
            )
        )
    bar.close()
    return EvaluationReport(points=tuple(points), config=eval_config)
