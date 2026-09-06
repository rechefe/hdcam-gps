"""Wrapper around the third party gps-sdr-sim simulator.

The simulator lives as a git submodule under third_party/gps-sdr-sim and is
built on demand with its own Makefile. It produces a far more realistic record
than signal_gen does, from a real broadcast ephemeris and a real receiver
position, with the right constellation, the right relative powers and real
navigation data.

What it does not produce is a usable label. Its channel structure carries the
Doppler and the code phase of every satellite, but the only per satellite output
it prints is PRN, azimuth, elevation, range and ionospheric delay. So a record
from here comes with the set of satellites as hard truth, and nothing finer.
Stock, it does not print the Doppler or the code phase, though it holds both per
channel. A small patch under third_party/patches makes it report them, which is
the plainest way to get the values the simulator itself intends:

    if (verb==TRUE)
        fprintf(stderr, "TRUTH %d %02d %.9f %.9f\n",
            iumd, chan[i].prn, chan[i].f_carr, chan[i].code_phase);

placed straight after computeCodePhase, so the numbers are the state the tick's
samples are generated from. The submodule itself is never modified: build copies
its sources into third_party/build, patches the copy and compiles there, so
git submodule status stays clean.

If upstream ever moves the code the patch sits in, the patch stops applying and
the build fails loudly rather than reporting something different.

The user facing entry point is signal_gen.generate_from_sim, which wraps a
record from here into the same Scenario that the synthetic generator returns.
"""

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hdcam_gps.acq_base import CHIPS_PER_CODE, CODE_PERIOD_S

THIRD_PARTY: Path = Path(__file__).resolve().parents[2] / "third_party"
SIM_ROOT: Path = THIRD_PARTY / "gps-sdr-sim"  # The submodule, never written to
PATCH_DIR: Path = THIRD_PARTY / "patches"
BUILD_ROOT: Path = THIRD_PARTY / "build"  # The patched copy that actually builds
SIM_BINARY: Path = BUILD_ROOT / "gps-sdr-sim"
SIM_SOURCES: tuple[str, ...] = (
    "gpssim.c",
    "gpssim.h",
    "getopt.c",
    "getopt.h",
    "Makefile",
)
DEFAULT_RINEX: Path = SIM_ROOT / "brdc0010.22n"  # Ships with the simulator

# The simulator works in whole tenths of a second and writes nothing for the
# first of them, so a run of d seconds yields floor(d / 0.1) - 1 tenths. simulate
# asks for enough whole ticks to cover the request and trims back, to keep its
# own contract exact.
SIM_TICK_S: float = 0.1
SIM_WARMUP_S: float = SIM_TICK_S

# "08  316.3  36.0  22340972.8   2.4" - prn, azimuth, elevation, range, iono delay
CHANNEL_LINE = re.compile(
    r"^(\d{2})\s+(-?[\d.]+)\s+(-?[\d.]+)\s+([\d.]+)\s+(-?[\d.]+)\s*$"
)
# "TRUTH 1 08 2619.123456789 438.041234567" - tick, prn, Doppler Hz, code chips
TRUTH_LINE = re.compile(r"^TRUTH\s+(\d+)\s+(\d+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s*$")


@dataclass(frozen=True)
class VisibleSatellite:
    """One satellite the simulator reported as visible."""

    prn: int  # PRN number
    azimuth_deg: float  # Azimuth in degrees
    elevation_deg: float  # Elevation in degrees
    range_m: float  # Geometric range in metres


@dataclass(frozen=True)
class ChannelState:
    """What a channel was set to at the start of one 0.1 second tick."""

    tick: int  # Tick index, 1 for the first tick of output
    prn: int  # PRN number
    doppler_hz: float  # Carrier Doppler in Hz, the simulator's f_carr
    code_phase_chips: float  # Chips of the code already elapsed


@dataclass(frozen=True)
class SimulatedRecord:
    """A record produced by the simulator, with the labels it can supply."""

    samples: np.ndarray  # Complex baseband samples, normalized to unit RMS
    visible: tuple[VisibleSatellite, ...]  # The constellation, hard truth
    fs_hz: float  # The sampling frequency of the record
    channel_states: tuple[ChannelState, ...] = ()  # Reported by the patch

    @property
    def prns(self) -> tuple[int, ...]:
        """Returns the PRN numbers present in the record."""
        return tuple(satellite.prn for satellite in self.visible)


def sources_checked_out() -> bool:
    """Reports whether the simulator's sources are present to build from.

    False in a clone made without --recurse-submodules, where the submodule
    directory exists but is empty.

    Returns:
        bool: True when the submodule has been checked out.
    """
    return (SIM_ROOT / "gpssim.c").is_file()


def patches() -> list[Path]:
    """Returns the patches applied to the simulator, in order.

    Returns:
        list[Path]: The patch files under third_party/patches.
    """
    return sorted(PATCH_DIR.glob("*.patch"))


def build(force: bool = False) -> Path:
    """Builds a patched copy of the simulator, leaving the submodule untouched.

    The sources are copied into third_party/build, the patches are applied there
    and the build runs there, so the submodule never becomes dirty.

    Args:
        force (bool): Rebuild even when the binary is present and up to date.

    Returns:
        Path: The path of the simulator binary.
    """
    assert (
        sources_checked_out()
    ), f"{SIM_ROOT} is empty. Run: git submodule update --init --recursive"
    inputs = [SIM_ROOT / name for name in SIM_SOURCES] + patches()
    newest = max(path.stat().st_mtime for path in inputs if path.is_file())
    if SIM_BINARY.is_file() and not force and SIM_BINARY.stat().st_mtime > newest:
        return SIM_BINARY

    assert shutil.which("make") and shutil.which("gcc"), "make and gcc are required."
    assert shutil.which("patch"), "the patch command is required."
    if BUILD_ROOT.exists():
        shutil.rmtree(BUILD_ROOT)
    BUILD_ROOT.mkdir(parents=True)
    for name in SIM_SOURCES:
        source = SIM_ROOT / name
        if source.is_file():
            shutil.copy2(source, BUILD_ROOT / name)
    for patch in patches():
        subprocess.run(
            ["patch", "-p1", "-i", str(patch)],
            cwd=BUILD_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    subprocess.run(["make"], cwd=BUILD_ROOT, check=True, capture_output=True)
    return SIM_BINARY


def parse_visible(stderr: str) -> tuple[VisibleSatellite, ...]:
    """Reads the simulator's channel report out of its stderr.

    Args:
        stderr (str): Everything the simulator wrote to stderr.

    Returns:
        tuple[VisibleSatellite, ...]: One entry per satellite, PRN ordered.
    """
    satellites = {}
    for line in stderr.splitlines():
        match = CHANNEL_LINE.match(line.strip())
        if match is None:
            continue
        prn, azimuth, elevation, range_m, _iono = match.groups()
        satellites[int(prn)] = VisibleSatellite(
            prn=int(prn),
            azimuth_deg=float(azimuth),
            elevation_deg=float(elevation),
            range_m=float(range_m),
        )
    return tuple(satellites[prn] for prn in sorted(satellites))


def simulator_duration(duration_s: float) -> float:
    """The duration to ask the simulator for, to get duration_s of samples back.

    Args:
        duration_s (float): The wanted length of the record in seconds.

    Returns:
        float: The length to pass to the simulator, a whole number of ticks.
    """
    ticks = int(np.ceil(duration_s / SIM_TICK_S)) + 1  # one tick for the warm-up
    return round(ticks * SIM_TICK_S, 6)


def parse_truth(stderr: str) -> tuple[ChannelState, ...]:
    """Reads the channel states the patched simulator reports.

    Args:
        stderr (str): Everything the simulator wrote to stderr.

    Returns:
        tuple[ChannelState, ...]: One entry per satellite per tick.
    """
    states = []
    for line in stderr.splitlines():
        match = TRUTH_LINE.search(line.strip())
        if match is None:
            continue
        tick, prn, doppler_hz, code_phase_chips = match.groups()
        states.append(
            ChannelState(
                tick=int(tick),
                prn=int(prn),
                doppler_hz=float(doppler_hz),
                code_phase_chips=float(code_phase_chips),
            )
        )
    return tuple(states)


def read_iq(path: Path, iq_bits: int) -> np.ndarray:
    """Loads an interleaved I/Q file written by the simulator.

    Args:
        path (Path): The file the simulator wrote.
        iq_bits (int): The sample format the simulator was asked for, 8 or 16.

    Returns:
        np.ndarray: Complex samples normalized to unit RMS.
    """
    assert iq_bits in (8, 16), "Only the 8 and 16 bit formats are readable here."
    raw = np.fromfile(path, dtype=np.int8 if iq_bits == 8 else np.int16)
    interleaved = raw.reshape(-1, 2).astype(float)
    samples = interleaved[:, 0] + 1j * interleaved[:, 1]
    if samples.size == 0:
        return samples
    rms = np.sqrt(np.mean(np.abs(samples) ** 2))
    return samples / rms if rms > 0 else samples


def simulate(
    latitude_deg: float = 32.0,
    longitude_deg: float = 35.0,
    height_m: float = 100.0,
    fs_hz: float = 2.6e6,
    duration_s: float = 1.0,
    rinex: Path = DEFAULT_RINEX,
    iq_bits: int = 16,
    output_dir: Path | None = None,
    start_time: str | None = None,
) -> SimulatedRecord:
    """Runs the simulator for a static receiver and loads what it produced.

    Args:
        latitude_deg (float): Receiver latitude in degrees.
        longitude_deg (float): Receiver longitude in degrees.
        height_m (float): Receiver height in metres.
        fs_hz (float): Sampling frequency in Hz. The simulator wants at least
            2.5 MHz.
        duration_s (float): Length of the record in seconds. Exactly this much
            is returned. The file holds 2 * iq_bits / 8 bytes per sample, so this
            grows quickly.
        rinex (Path): The RINEX navigation file to take ephemerides from.
        iq_bits (int): Sample format, 8 or 16.
        output_dir (Path | None): Where to put the intermediate I/Q file. A
            temporary directory is used when None.
        start_time (str | None): Scenario start as "YYYY/MM/DD,hh:mm:ss".

    Returns:
        SimulatedRecord: The samples and the constellation that produced them.
    """
    import tempfile

    binary = build()
    assert Path(rinex).is_file(), f"RINEX file not found: {rinex}"

    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(output_dir) if output_dir else Path(temporary)
        directory.mkdir(parents=True, exist_ok=True)
        iq_path = directory / "gpssim.bin"
        command = [
            str(binary),
            "-e",
            str(rinex),
            "-l",
            f"{latitude_deg},{longitude_deg},{height_m}",
            "-d",
            str(simulator_duration(duration_s)),
            "-s",
            str(int(fs_hz)),
            "-b",
            str(iq_bits),
            "-o",
            str(iq_path),
            "-v",
        ]
        if start_time is not None:
            command += ["-t", start_time]
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        channel_states = parse_truth(completed.stderr)
        samples = read_iq(iq_path, iq_bits)
        n_wanted = int(fs_hz * duration_s)
        assert len(samples) >= n_wanted, (
            f"The simulator returned {len(samples)} samples but {n_wanted} were "
            f"asked for. It was run for {simulator_duration(duration_s)} s."
        )
        return SimulatedRecord(
            samples=samples[:n_wanted],
            visible=parse_visible(completed.stderr),
            fs_hz=fs_hz,
            channel_states=channel_states,
        )


def labels(record: SimulatedRecord, offset: int = 0) -> dict[int, tuple[float, int]]:
    """The Doppler and code phase of every satellite, as the simulator set them.

    Read straight out of the patched simulator's report, so nothing is measured
    and nothing is modelled. The report is per 0.1 second tick, so the tick
    covering the wanted offset is used and the remaining samples of the offset
    are taken off the code phase. Code Doppler drift across that remainder is
    under a sample and is not corrected for.

    The simulator counts chips already elapsed, while an acquisition reports how
    far the replica has to be rolled forward, so the two are complements within a
    code period.

    Args:
        record (SimulatedRecord): A record from a patched simulator.
        offset (int): First sample of the record the labels should describe.

    Returns:
        dict[int, tuple[float, int]]: Doppler in Hz and code phase in samples.
    """
    assert record.channel_states, (
        "The record carries no channel report. Rebuild the simulator so the "
        f"patches in {PATCH_DIR} are applied."
    )
    samples_per_code = int(record.fs_hz * CODE_PERIOD_S)
    samples_per_tick = int(record.fs_hz * SIM_TICK_S)
    tick = 1 + offset // samples_per_tick
    remainder = offset % samples_per_tick

    available = sorted({state.tick for state in record.channel_states})
    assert tick in available, (
        f"The record reports ticks {available[0]} to {available[-1]}, but sample "
        f"{offset} falls in tick {tick}."
    )
    return {
        state.prn: (
            state.doppler_hz,
            round(
                samples_per_code
                - state.code_phase_chips * samples_per_code / CHIPS_PER_CODE
                - remainder
            )
            % samples_per_code,
        )
        for state in record.channel_states
        if state.tick == tick
    }
