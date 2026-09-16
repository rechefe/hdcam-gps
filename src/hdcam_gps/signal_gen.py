"""Labelled GPS L1 C/A test signal generation.

This is the entry point for building records to score a classifier against. Two
backends produce the same Scenario, so they are interchangeable at the call
site:

* generate_synthetic places satellites the caller chooses. The label is the
  input, so it is exact, and the C/N0 of every satellite is set individually.
* generate_from_sim drives the third party simulator through gps_sdr_sim for a
  real constellation from a real ephemeris. A small patch makes the simulator
  report the Doppler and code phase it already holds, so those labels are its
  own state rather than anything we measured or modelled. A satellite can sit at
  a fractional code phase there, which an acquisition can only answer to the
  sample, so compare with matches rather than with equality.

The one label the simulator cannot supply is C/N0, so measure_cn0_dbhz reads it
off the clean record per satellite. It has to be per satellite: path loss and the
receiver antenna pattern spread one simulated sky over about 8 dB, and the
cn0_dbhz a record was scaled to is the average of that, not the figure any one
satellite sits at.

Neither backend ever labels a record with a classifier. A truth measured by one
acquisition cannot be used to score another: the classifier that produced it
would score perfectly by definition, and any blind spot the two share would be
invisible.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hdcam_gps.acq_base import CODE_PERIOD_S, AcqConfig, PrnResult
from hdcam_gps.ca_code import sampled_ca_code
from hdcam_gps.gps_sdr_sim import DEFAULT_RINEX, SimulatedRecord, labels, simulate

NAV_BIT_PERIOD_S: float = 20e-3  # A navigation data bit lasts 20 code periods


@dataclass(frozen=True)
class SatelliteTruth:
    """One satellite placed in a scenario, and therefore its label."""

    prn: int  # PRN number
    doppler_hz: float  # Carrier Doppler in Hz, not necessarily on the search grid
    code_phase: int  # Code phase in samples, within one code period
    cn0_dbhz: float = 45.0  # Carrier to noise density ratio in dB-Hz
    carrier_phase_rad: float = 0.0  # Initial carrier phase in radians


@dataclass(frozen=True)
class Scenario:
    """A generated record together with what was put into it."""

    samples: np.ndarray  # Complex baseband samples, samples_per_acquisition long
    truth: tuple[SatelliteTruth, ...]  # What the record contains
    config: AcqConfig  # The configuration the record was built for
    noise_sigma: float = 0.0  # Standard deviation of the complex noise added

    def matches(
        self, results: list[PrnResult], code_phase_tolerance: int = 0
    ) -> bool:
        """Whether an acquisition result agrees with the scenario.

        A tolerance is needed for a scenario whose code phase is derived rather
        than placed, since the derivation and the acquisition round the same
        fractional sample independently and can land either side of it.

        Args:
            results (list[PrnResult]): What a classifier returned.
            code_phase_tolerance (int): Samples of code phase to allow either way.

        Returns:
            bool: True when the PRNs, the Doppler bins and the code phases agree.
        """
        expected = self.expected_results()
        if [r.prn for r in results] != [e.prn for e in expected]:
            return False
        samples_per_code = self.config.samples_per_code
        for result, want in zip(results, expected):
            if result.doppler_hz != want.doppler_hz:
                return False
            error = abs(result.code_phase - want.code_phase)
            error = min(error, samples_per_code - error)  # the phase wraps
            if error > code_phase_tolerance:
                return False
        return True

    def expected_results(self) -> list[PrnResult]:
        """The acquisition result a perfect classifier would return.

        Dopplers are snapped to the configuration's search grid, since that is
        the resolution any classifier can report, and the satellites come back in
        the configuration's PRN order, the order the classifiers emit.

        Returns:
            list[PrnResult]: One result per satellite of the scenario.
        """
        by_prn = {satellite.prn: satellite for satellite in self.truth}
        results = []
        for prn in self.config.prn_list:
            if prn not in by_prn:
                continue
            satellite = by_prn[prn]
            results.append(
                PrnResult(
                    prn=prn,
                    doppler_hz=snap_to_doppler_grid(self.config, satellite.doppler_hz),
                    code_phase=satellite.code_phase,
                )
            )
        return results


def snap_to_doppler_grid(config: AcqConfig, doppler_hz: float) -> float:
    """Rounds a Doppler to the nearest bin of the configuration's search grid.

    Args:
        config (AcqConfig): The acquisition configuration.
        doppler_hz (float): The true Doppler in Hz.

    Returns:
        float: The Doppler of the closest bin.
    """
    grid = config.doppler_grid_hz
    return float(grid[int(np.argmin(np.abs(grid - doppler_hz)))])


def amplitude_for_cn0(cn0_dbhz: float, fs_hz: float) -> float:
    """The signal amplitude that realizes a C/N0 against unit power noise.

    The noise this generator adds is complex with total power one, spread over
    the sampled bandwidth, so its density is N0 = 1 / fs_hz. A code of unit
    power at amplitude A then carries C = A squared, and C / N0 = A squared times
    fs_hz.

    Args:
        cn0_dbhz (float): The wanted carrier to noise density ratio in dB-Hz.
        fs_hz (float): The sampling frequency in Hz.

    Returns:
        float: The amplitude to give the code.
    """
    return float(np.sqrt(10.0 ** (cn0_dbhz / 10.0) / fs_hz))


def nav_data_bits(n_samples: int, fs_hz: float, rng: np.random.Generator) -> np.ndarray:
    """Random navigation data bits, held for 20 ms, as plus or minus one.

    Args:
        n_samples (int): How many samples to cover.
        fs_hz (float): The sampling frequency in Hz.
        rng (np.random.Generator): The source of the bits.

    Returns:
        np.ndarray: The data modulation, one value per sample.
    """
    samples_per_bit = int(fs_hz * NAV_BIT_PERIOD_S)
    n_bits = int(np.ceil(n_samples / samples_per_bit))
    bits = rng.choice([-1.0, 1.0], size=n_bits)
    return np.repeat(bits, samples_per_bit)[:n_samples]


def satellite_signal(
    config: AcqConfig, satellite: SatelliteTruth, nav_data: bool, rng
) -> np.ndarray:
    """Builds the contribution of a single satellite to a record.

    Args:
        config (AcqConfig): The acquisition configuration.
        satellite (SatelliteTruth): The satellite to place.
        nav_data (bool): Whether to modulate with navigation data bits.
        rng (np.random.Generator): Used only for the data bits.

    Returns:
        np.ndarray: The complex contribution, samples_per_acquisition long.
    """
    n_samples = config.samples_per_acquisition
    assert (
        0 <= satellite.code_phase < config.samples_per_code
    ), "Code phase must fall inside one code period."

    code = np.tile(sampled_ca_code(satellite.prn, config.fs_hz), config.n_codes)
    code = np.roll(code, satellite.code_phase).astype(float)
    if nav_data:
        code = code * nav_data_bits(n_samples, config.fs_hz, rng)

    time_s = np.arange(n_samples) / config.fs_hz
    carrier = np.exp(
        2j * np.pi * satellite.doppler_hz * time_s + 1j * satellite.carrier_phase_rad
    )
    return amplitude_for_cn0(satellite.cn0_dbhz, config.fs_hz) * code * carrier


def generate_synthetic(
    config: AcqConfig,
    satellites: list[SatelliteTruth] | tuple[SatelliteTruth, ...] = (),
    add_noise: bool = True,
    nav_data: bool = False,
    seed: int | None = None,
) -> Scenario:
    """Builds a labelled record from the satellites the caller places in it.

    Args:
        config (AcqConfig): The acquisition configuration to build for. Its
            sampling frequency and length define the record.
        satellites (list[SatelliteTruth]): The satellites to place. An empty list
            gives a record of noise only, whose truth is empty.
        add_noise (bool): Whether to add the unit power complex noise the C/N0 of
            each satellite is measured against. Without it the record is
            noiseless and the C/N0 values only set the relative powers.
        nav_data (bool): Whether to modulate the codes with navigation data bits.
        seed (int | None): Seed of the noise and data bit generator.

    Returns:
        Scenario: The record and the satellites that were put into it.
    """
    rng = np.random.default_rng(seed)
    n_samples = config.samples_per_acquisition
    samples = np.zeros(n_samples, dtype=complex)
    for satellite in satellites:
        samples += satellite_signal(config, satellite, nav_data, rng)

    noise_sigma = 1.0 if add_noise else 0.0
    if add_noise:
        # Unit total power, so half the variance in each component.
        samples = samples + (
            rng.normal(scale=np.sqrt(0.5), size=n_samples)
            + 1j * rng.normal(scale=np.sqrt(0.5), size=n_samples)
        )
    return Scenario(
        samples=samples,
        truth=tuple(satellites),
        config=config,
        noise_sigma=noise_sigma,
    )


def random_scenario(
    config: AcqConfig,
    n_satellites: int,
    cn0_dbhz: float = 45.0,
    add_noise: bool = True,
    seed: int | None = None,
) -> Scenario:
    """Places random satellites drawn from the configuration's own search space.

    Every satellite lands on a Doppler bin of the grid and a code phase inside
    one code period, so the scenario is always solvable in principle.

    Args:
        config (AcqConfig): The acquisition configuration to build for.
        n_satellites (int): How many of the configured PRNs to place.
        cn0_dbhz (float): The C/N0 to give each of them.
        add_noise (bool): Whether to add noise.
        seed (int | None): Seed of the draw and of the noise.

    Returns:
        Scenario: The record and its truth.
    """
    assert n_satellites <= len(
        config.prn_list
    ), "Cannot place more satellites than the configuration lists."
    rng = np.random.default_rng(seed)
    prns = rng.choice(np.array(config.prn_list), size=n_satellites, replace=False)
    satellites = [
        SatelliteTruth(
            prn=int(prn),
            doppler_hz=float(rng.choice(config.doppler_grid_hz)),
            code_phase=int(rng.integers(config.samples_per_code)),
            cn0_dbhz=cn0_dbhz,
            carrier_phase_rad=float(rng.uniform(0, 2 * np.pi)),
        )
        for prn in prns
    ]
    return generate_synthetic(
        config, satellites, add_noise=add_noise, seed=int(rng.integers(2**31))
    )


def measure_cn0_dbhz(
    samples: np.ndarray,
    fs_hz: float,
    prn: int,
    doppler_hz: float,
    code_phase: int,
    search_samples: int = 1,
) -> float:
    """Reads one satellite's C/N0 out of a noiseless record.

    This is the inverse of amplitude_for_cn0, on the same unit noise convention:
    correlating against the satellite's own replica over one code period returns
    its amplitude, and C/N0 is that amplitude squared times the sampling rate.
    The correlation is coherent within a code period and not across them, so a
    navigation bit transition costs one period rather than the whole estimate,
    and the median over the periods rides out the one that flipped.

    The code phase is searched over a window rather than trusted, because
    gps_sdr_sim.labels rounds a fractional code phase to the nearest sample and
    at 1.023 MHz one sample is one chip. Correlating at a phase one chip off
    reads the autocorrelation sidelobe instead of the peak: measured on a real
    sky, seven of nine satellites came back 15 to 25 dB low, and PRN 10 at 68
    degrees elevation read weaker than PRN 26 at 12. Over the window the answers
    agree with the simulator's own path loss and antenna gain to 0.6 dB.

    Cross correlation from the rest of the constellation is the remaining error.
    It is about -24 dB, so it moves a strong satellite by under 0.1 dB and a
    satellite 25 dB below the strongest by about 2 dB.

    Args:
        samples (np.ndarray): A noiseless record, on the unit noise convention.
        fs_hz (float): The sampling frequency in Hz.
        prn (int): The PRN number of the satellite to measure.
        doppler_hz (float): Its carrier Doppler in Hz.
        code_phase (int): Its code phase in samples.
        search_samples (int): Samples either side of code_phase to try. Zero
            takes the code phase as given.

    Returns:
        float: The carrier to noise density ratio in dB-Hz.
    """
    samples_per_code = int(fs_hz * CODE_PERIOD_S)
    n_periods = len(samples) // samples_per_code
    assert n_periods >= 1, "A C/N0 estimate needs at least one code period."
    assert search_samples >= 0, "The search window cannot be negative."

    n_used = n_periods * samples_per_code
    time_s = np.arange(n_used) / fs_hz
    carrier = np.exp(2j * np.pi * doppler_hz * time_s)
    wiped = (samples[:n_used] * np.conj(carrier)).reshape(n_periods, samples_per_code)
    code = sampled_ca_code(prn, fs_hz)

    amplitude = 0.0
    for offset in range(-search_samples, search_samples + 1):
        replica = np.roll(code, (code_phase + offset) % samples_per_code)
        products = wiped * replica
        amplitude = max(amplitude, float(np.median(np.abs(products.mean(axis=1)))))
    if amplitude <= 0:
        return float("-inf")
    return float(10.0 * np.log10(amplitude**2 * fs_hz))


def scenario_from_record(
    config: AcqConfig,
    record: SimulatedRecord,
    cn0_dbhz: float = 45.0,
    add_noise: bool = True,
    seed: int | None = None,
    offset: int = 0,
) -> Scenario:
    """Turns a simulated record into a labelled Scenario, without re-simulating.

    The record depends only on the place, the time and the sampling rate, while
    the C/N0 and the noise are applied here. Splitting them is what lets one
    simulator run serve a whole sweep: see sim_cache and scenarios.

    The truth is the simulator's own channel state, reported by the patch under
    third_party/patches and read straight out of its output. The simulator places
    a satellite at a fractional code phase, while an acquisition can only answer
    in whole samples, so compare with scenario.matches(results,
    code_phase_tolerance=1) rather than with equality.

    cn0_dbhz scales the whole record, and is not what any one satellite ends up
    at. The simulator runs with a 0 degree elevation mask and an antenna pattern
    reaching -31.6 dB at the horizon, so one sky spans more than 20 dB. Each
    SatelliteTruth therefore carries its own measured value, and cn0_dbhz names
    only the average the record was scaled to.

    Args:
        config (AcqConfig): The configuration to build for. Its sampling
            frequency and length define the record; its PRN list only filters
            what the scenario reports, not what the sky contains.
        record (SimulatedRecord): A record from a patched simulator.
        cn0_dbhz (float): The average per satellite C/N0 to scale the record to.
        add_noise (bool): Whether to add the unit power complex noise.
        seed (int | None): Seed of the noise generator.
        offset (int): First sample of the simulated record to take.

    Returns:
        Scenario: The record and the constellation that produced it.
    """
    n_samples = config.samples_per_acquisition
    window = record.samples[offset : offset + n_samples]
    assert len(window) == n_samples, "The simulator returned too short a record."

    # Onto the convention of generate_synthetic: unit noise, calibrated signal.
    total_power = len(record.prns) * amplitude_for_cn0(cn0_dbhz, config.fs_hz) ** 2
    measured_power = float(np.mean(np.abs(window) ** 2))
    if measured_power > 0:
        window = window * np.sqrt(total_power / measured_power)

    truth = tuple(
        SatelliteTruth(
            prn=prn,
            doppler_hz=doppler_hz,
            code_phase=code_phase,
            cn0_dbhz=measure_cn0_dbhz(
                window, config.fs_hz, prn, doppler_hz, code_phase
            ),
        )
        for prn, (doppler_hz, code_phase) in sorted(labels(record, offset).items())
    )

    if add_noise:
        rng = np.random.default_rng(seed)
        window = window + (
            rng.normal(scale=np.sqrt(0.5), size=n_samples)
            + 1j * rng.normal(scale=np.sqrt(0.5), size=n_samples)
        )
    return Scenario(
        samples=window,
        truth=truth,
        config=config,
        noise_sigma=1.0 if add_noise else 0.0,
    )


def generate_from_sim(
    config: AcqConfig,
    latitude_deg: float = 32.0,
    longitude_deg: float = 35.0,
    height_m: float = 100.0,
    cn0_dbhz: float = 45.0,
    add_noise: bool = True,
    seed: int | None = None,
    offset: int = 0,
    rinex: Path = DEFAULT_RINEX,
    start_time: str | None = None,
) -> Scenario:
    """Builds a Scenario from a real constellation, via the third party simulator.

    This is simulate followed by scenario_from_record, which is the expensive
    half followed by the cheap one. A sweep that re-runs it per C/N0 point pays
    for the same record over and over; scenarios.ScenarioBank does not.

    Args:
        config (AcqConfig): The configuration to build for. Its sampling
            frequency and length define the record; its PRN list only filters
            what the scenario reports, not what the sky contains.
        latitude_deg (float): Receiver latitude in degrees.
        longitude_deg (float): Receiver longitude in degrees.
        height_m (float): Receiver height in metres.
        cn0_dbhz (float): The average per satellite C/N0 to scale the record to.
        add_noise (bool): Whether to add the unit power complex noise.
        seed (int | None): Seed of the noise generator.
        offset (int): First sample of the simulated record to take.
        rinex (Path): The RINEX navigation file to take ephemerides from.
        start_time (str | None): Scenario start as "YYYY/MM/DD,hh:mm:ss".

    Returns:
        Scenario: The record and the constellation that produced it.
    """
    record = simulate(
        latitude_deg=latitude_deg,
        longitude_deg=longitude_deg,
        height_m=height_m,
        fs_hz=config.fs_hz,
        duration_s=(offset + config.samples_per_acquisition) / config.fs_hz,
        rinex=rinex,
        start_time=start_time,
    )
    return scenario_from_record(
        config,
        record,
        cn0_dbhz=cn0_dbhz,
        add_noise=add_noise,
        seed=seed,
        offset=offset,
    )
