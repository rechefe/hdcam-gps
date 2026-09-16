"""Caching the simulator's output, because a record does not depend on C/N0.

gps-sdr-sim produces a noiseless record from a place, a time and a sampling
rate. Everything the study varies on top of that - the scaling to a C/N0, the
noise, which classifier looks at it - is applied afterwards by
signal_gen.scenario_from_record. Re-running the simulator per C/N0 point per
family therefore pays repeatedly for bit identical output: five families over
six points over twenty skies is 600 invocations of 60 distinct records.

So the cache key is exactly the arguments simulate uses to build the samples.
output_dir is not one of them, since it only says where the intermediate I/Q file
goes.

Two layers. An in-process dict, which is what a notebook re-running a cell hits,
and an npz under third_party/cache, which survives the process and is
gitignored. The npz stores the channel report as well as the samples: without it
the record carries no labels and scenario_from_record cannot build a truth.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from hdcam_gps.gps_sdr_sim import (
    THIRD_PARTY,
    ChannelState,
    SimulatedRecord,
    VisibleSatellite,
    simulate,
)

CACHE_ROOT: Path = THIRD_PARTY / "cache"

_MEMORY: dict[str, SimulatedRecord] = {}


def cache_key(**kwargs) -> str:
    """The name a set of simulate arguments is cached under.

    Args:
        **kwargs: The arguments simulate would be called with.

    Returns:
        str: A 16 character hex digest.
    """
    canonical = json.dumps(
        {key: str(value) for key, value in sorted(kwargs.items())}, sort_keys=True
    )
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def save_record(path: Path, record: SimulatedRecord):
    """Writes a record and its labels to an npz.

    Args:
        path (Path): The file to write.
        record (SimulatedRecord): The record to store.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        samples=record.samples,
        fs_hz=np.array([record.fs_hz]),
        visible=np.array(
            [
                [s.prn, s.azimuth_deg, s.elevation_deg, s.range_m]
                for s in record.visible
            ],
            dtype=float,
        ).reshape(-1, 4),
        channel_states=np.array(
            [
                [s.tick, s.prn, s.doppler_hz, s.code_phase_chips]
                for s in record.channel_states
            ],
            dtype=float,
        ).reshape(-1, 4),
    )


def load_record(path: Path) -> SimulatedRecord:
    """Reads back a record written by save_record.

    Args:
        path (Path): The npz to read.

    Returns:
        SimulatedRecord: The record and the labels it was stored with.
    """
    stored = np.load(path)
    return SimulatedRecord(
        samples=stored["samples"],
        visible=tuple(
            VisibleSatellite(
                prn=int(row[0]),
                azimuth_deg=float(row[1]),
                elevation_deg=float(row[2]),
                range_m=float(row[3]),
            )
            for row in stored["visible"]
        ),
        fs_hz=float(stored["fs_hz"][0]),
        channel_states=tuple(
            ChannelState(
                tick=int(row[0]),
                prn=int(row[1]),
                doppler_hz=float(row[2]),
                code_phase_chips=float(row[3]),
            )
            for row in stored["channel_states"]
        ),
    )


def cached_simulate(
    cache_dir: Path | None = None, refresh: bool = False, **kwargs
) -> SimulatedRecord:
    """simulate, answered from the cache when the same record was asked for before.

    Args:
        cache_dir (Path | None): Where the npz files live. None uses CACHE_ROOT.
        refresh (bool): Re-run the simulator and overwrite the cached copy.
        **kwargs: Passed straight to gps_sdr_sim.simulate.

    Returns:
        SimulatedRecord: The samples and the constellation that produced them.
    """
    directory = Path(cache_dir) if cache_dir is not None else CACHE_ROOT
    key = cache_key(**kwargs)
    path = directory / f"{key}.npz"
    memory_key = f"{directory}/{key}"

    if not refresh and memory_key in _MEMORY:
        return _MEMORY[memory_key]
    if not refresh and path.is_file():
        record = load_record(path)
    else:
        record = simulate(**kwargs)
        save_record(path, record)
    _MEMORY[memory_key] = record
    return record


def clear_cache(cache_dir: Path | None = None):
    """Empties both layers of the cache.

    Args:
        cache_dir (Path | None): Where the npz files live. None uses CACHE_ROOT.
    """
    directory = Path(cache_dir) if cache_dir is not None else CACHE_ROOT
    _MEMORY.clear()
    if directory.is_dir():
        for path in directory.glob("*.npz"):
            path.unlink()
