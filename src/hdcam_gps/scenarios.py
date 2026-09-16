"""Records built once, so every family is measured on the same input.

Five classifiers scored on five independently drawn sets of records is five
experiments, not one comparison: a difference of two dB in sensitivity can come
from the skies rather than from the design. A ScenarioBank materialises the
records first and hands out the same Scenario object to every classifier, so the
samples array they see is the same array.

The second thing it fixes is the calibration split. A threshold tuned on the
records it is then measured on is tuned on the noise as well as on the signal, so
split separates skies by start time: nothing calibrated on set A appears in set
B, and the split asserts it rather than trusting the arithmetic.

Building 60 skies at 7 scalings is 420 records from 60 simulator runs, because
sim_cache keys the run on the place and time alone and scenario_from_record
applies the scaling afterwards.
"""

from dataclasses import dataclass

import numpy as np
from tqdm.auto import tqdm

from hdcam_gps.acq_base import AcqConfig
from hdcam_gps.evaluate import EvalConfig, sky_start_time
from hdcam_gps.gps_sdr_sim import DEFAULT_RINEX
from hdcam_gps.sim_cache import cached_simulate
from hdcam_gps.signal_gen import Scenario, random_scenario, scenario_from_record


@dataclass(frozen=True)
class ScenarioBank:
    """Every record of a sweep, keyed by the C/N0 it was scaled to and its sky."""

    config: AcqConfig
    cn0_dbhz: tuple[float, ...]  # The scalings every sky was built at
    indices: tuple[int, ...]  # The skies, as indices into the start time series
    records: dict[tuple[float, int], Scenario]
    backend: str = "simulator"

    def get(self, cn0_dbhz: float, index: int) -> Scenario:
        """The record of one sky at one scaling.

        The same object every time, so two classifiers given this bank read the
        same samples array rather than two draws of the same distribution.

        Args:
            cn0_dbhz (float): One of the bank's scalings.
            index (int): One of the bank's sky indices.

        Returns:
            Scenario: The record and its truth.
        """
        key = (float(cn0_dbhz), int(index))
        assert key in self.records, (
            f"The bank holds {sorted(self.cn0_dbhz)} dB-Hz over skies "
            f"{self.indices[0]} to {self.indices[-1]}, not {key}."
        )
        return self.records[key]

    @property
    def start_times(self) -> tuple[str, ...]:
        """The simulator start time of every sky in the bank."""
        return tuple(sky_start_time(index) for index in self.indices)

    def split(self, n_calibration: int) -> tuple["ScenarioBank", "ScenarioBank"]:
        """Cuts the bank into a calibration set and an evaluation set.

        Args:
            n_calibration (int): Skies to put in the calibration set.

        Returns:
            tuple[ScenarioBank, ScenarioBank]: The calibration bank and the
                evaluation bank, sharing no sky.
        """
        assert 0 < n_calibration < len(self.indices), (
            f"A split leaves both sides non empty, so it is between 1 and "
            f"{len(self.indices) - 1}, not {n_calibration}."
        )
        first, second = self._subset(self.indices[:n_calibration]), self._subset(
            self.indices[n_calibration:]
        )
        assert not set(first.start_times) & set(second.start_times), (
            "A calibration sky reached the evaluation set."
        )
        return first, second

    def _subset(self, indices: tuple[int, ...]) -> "ScenarioBank":
        """A bank over some of this one's skies, sharing its Scenario objects.

        Args:
            indices (tuple[int, ...]): The skies to keep.

        Returns:
            ScenarioBank: The same records, under a narrower index.
        """
        kept = set(indices)
        return ScenarioBank(
            config=self.config,
            cn0_dbhz=self.cn0_dbhz,
            indices=tuple(indices),
            records={
                key: scenario
                for key, scenario in self.records.items()
                if key[1] in kept
            },
            backend=self.backend,
        )

    @classmethod
    def build(
        cls,
        config: AcqConfig,
        eval_config: EvalConfig,
        indices=None,
        rinex=DEFAULT_RINEX,
        progress: bool | None = None,
    ) -> "ScenarioBank":
        """Materialises every record of a sweep.

        The simulator backend runs once per sky and applies every scaling to that
        one record. The synthetic backend has nothing to cache, so it draws each
        record as evaluate would.

        Args:
            config (AcqConfig): The configuration the records are built to.
            eval_config (EvalConfig): The sweep to cover. Its n_scenarios,
                cn0_dbhz, backend, seed and receiver position are all read.
            indices: The skies to build, defaulting to the first n_scenarios.
            rinex: The RINEX navigation file, simulator backend only.
            progress (bool | None): Show a progress bar. None follows
                eval_config.

        Returns:
            ScenarioBank: One record per (scaling, sky).
        """
        if indices is None:
            indices = range(eval_config.n_scenarios)
        indices = tuple(int(index) for index in indices)
        assert indices, "A bank needs at least one sky."
        assert eval_config.cn0_dbhz, "A bank needs at least one C/N0."
        if progress is None:
            progress = eval_config.progress

        records: dict[tuple[float, int], Scenario] = {}
        bar = tqdm(
            total=len(indices),
            disable=not progress,
            unit="sky",
            desc="building scenarios",
            leave=False,
        )
        for index in indices:
            seed = eval_config.seed + index
            if eval_config.backend == "synthetic":
                for cn0_dbhz in eval_config.cn0_dbhz:
                    records[(float(cn0_dbhz), index)] = random_scenario(
                        config,
                        n_satellites=eval_config.n_satellites,
                        cn0_dbhz=cn0_dbhz,
                        seed=seed,
                    )
            else:
                record = cached_simulate(
                    latitude_deg=eval_config.latitude_deg,
                    longitude_deg=eval_config.longitude_deg,
                    height_m=eval_config.height_m,
                    fs_hz=config.fs_hz,
                    duration_s=config.samples_per_acquisition / config.fs_hz,
                    rinex=rinex,
                    start_time=sky_start_time(index),
                )
                for cn0_dbhz in eval_config.cn0_dbhz:
                    records[(float(cn0_dbhz), index)] = scenario_from_record(
                        config, record, cn0_dbhz=cn0_dbhz, seed=seed
                    )
            bar.update(1)
        bar.close()
        return cls(
            config=config,
            cn0_dbhz=tuple(float(value) for value in eval_config.cn0_dbhz),
            indices=indices,
            records=records,
            backend=eval_config.backend,
        )

    def satellite_cn0_dbhz(self) -> np.ndarray:
        """Every per satellite C/N0 the bank holds, pooled.

        This is the x axis of the study: the sweep variable is the satellite's
        own C/N0, and the scaling a record was built at is only the sampling
        design that populates the bins.

        Returns:
            np.ndarray: One value per (record, satellite) pair.
        """
        return np.array(
            [
                satellite.cn0_dbhz
                for scenario in self.records.values()
                for satellite in scenario.truth
            ]
        )
