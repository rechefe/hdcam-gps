"""Directed tests for the simulator record cache.

The round trip is checked field by field against a hand built record, so no test
here needs the simulator. The ones that do are marked and skip without it, and a
green run without the submodule therefore proves nothing about them.

What is not covered here: whether two runs of the simulator at the same
arguments produce the same samples. The cache assumes they would, which is the
whole reason it is allowed to answer instead of re-running.
"""

import numpy as np
import pytest

from hdcam_gps import sim_cache
from hdcam_gps.acq_base import AcqConfig
from hdcam_gps.gps_sdr_sim import (
    ChannelState,
    SimulatedRecord,
    VisibleSatellite,
    sources_checked_out,
)
from hdcam_gps.sim_cache import (
    cache_key,
    cached_simulate,
    clear_cache,
    load_record,
    save_record,
)

needs_simulator = pytest.mark.skipif(
    not sources_checked_out(), reason="the gps-sdr-sim submodule is not checked out"
)

FS_HZ = 2.6e6  # The simulator refuses anything under 2.5 MHz


def a_record(n_samples: int = 64) -> SimulatedRecord:
    rng = np.random.default_rng(0)
    samples = rng.normal(size=n_samples) + 1j * rng.normal(size=n_samples)
    return SimulatedRecord(
        samples=samples,
        visible=(
            VisibleSatellite(prn=8, azimuth_deg=316.3, elevation_deg=36.0,
                             range_m=22340972.8),
            VisibleSatellite(prn=19, azimuth_deg=-12.5, elevation_deg=70.2,
                             range_m=20180000.0),
        ),
        fs_hz=FS_HZ,
        channel_states=(
            ChannelState(tick=1, prn=8, doppler_hz=2619.123456789,
                         code_phase_chips=438.041234567),
            ChannelState(tick=1, prn=19, doppler_hz=-1500.5,
                         code_phase_chips=12.25),
        ),
    )


# --------------------------------------------------------------------------
# cache_key - what counts as the same record
# --------------------------------------------------------------------------


def test_the_same_arguments_give_the_same_key():
    assert cache_key(fs_hz=FS_HZ, duration_s=0.2) == cache_key(
        fs_hz=FS_HZ, duration_s=0.2
    )


def test_the_key_does_not_depend_on_the_order_they_were_written_in():
    assert cache_key(fs_hz=FS_HZ, duration_s=0.2) == cache_key(
        duration_s=0.2, fs_hz=FS_HZ
    )


@pytest.mark.parametrize(
    "difference",
    [
        {"fs_hz": 2.7e6},
        {"duration_s": 0.3},
        {"start_time": "2022/01/01,00:17:00"},
        {"latitude_deg": 33.0},
    ],
)
def test_a_different_record_gets_a_different_key(difference):
    base = dict(fs_hz=FS_HZ, duration_s=0.2, start_time=None, latitude_deg=32.0)
    assert cache_key(**base) != cache_key(**{**base, **difference})


# --------------------------------------------------------------------------
# the npz round trip
# --------------------------------------------------------------------------


def test_a_record_survives_the_round_trip(tmp_path):
    original = a_record()
    save_record(tmp_path / "one.npz", original)
    restored = load_record(tmp_path / "one.npz")

    assert np.array_equal(restored.samples, original.samples)
    assert restored.fs_hz == original.fs_hz
    assert restored.visible == original.visible
    assert restored.channel_states == original.channel_states


def test_the_labels_survive_the_round_trip_to_the_ninth_decimal(tmp_path):
    # scenario_from_record cannot build a truth without them, so a cache that
    # dropped the channel report would silently hand back an unusable record.
    save_record(tmp_path / "one.npz", a_record())
    restored = load_record(tmp_path / "one.npz")
    assert restored.channel_states[0].doppler_hz == pytest.approx(
        2619.123456789, abs=1e-9
    )


def test_a_record_with_no_channel_report_round_trips_as_empty(tmp_path):
    bare = SimulatedRecord(samples=np.zeros(4, dtype=complex), visible=(), fs_hz=FS_HZ)
    save_record(tmp_path / "bare.npz", bare)
    restored = load_record(tmp_path / "bare.npz")
    assert restored.visible == ()
    assert restored.channel_states == ()


# --------------------------------------------------------------------------
# cached_simulate - the simulator runs once
# --------------------------------------------------------------------------


@needs_simulator
def test_the_second_call_is_answered_from_the_cache(tmp_path):
    config = AcqConfig(
        fs_hz=FS_HZ,
        prn_list=(1,),
        doppler_min_hz=-500.0,
        doppler_max_hz=500.0,
        doppler_step_hz=500.0,
        n_codes=2,
    )
    arguments = dict(
        fs_hz=config.fs_hz,
        duration_s=config.samples_per_acquisition / config.fs_hz,
    )
    clear_cache(tmp_path)
    first = cached_simulate(cache_dir=tmp_path, **arguments)
    second = cached_simulate(cache_dir=tmp_path, **arguments)

    assert second is first  # the in-process layer, not a reload
    assert len(list(tmp_path.glob("*.npz"))) == 1


@needs_simulator
def test_a_new_process_reads_the_record_back_off_disk(tmp_path):
    arguments = dict(fs_hz=FS_HZ, duration_s=0.2)
    clear_cache(tmp_path)
    written = cached_simulate(cache_dir=tmp_path, **arguments)

    sim_cache._MEMORY.clear()  # as a fresh process would start
    reloaded = cached_simulate(cache_dir=tmp_path, **arguments)
    assert reloaded is not written
    assert np.array_equal(reloaded.samples, written.samples)
    assert reloaded.channel_states == written.channel_states


@needs_simulator
def test_clearing_the_cache_removes_the_files(tmp_path):
    cached_simulate(cache_dir=tmp_path, fs_hz=FS_HZ, duration_s=0.2)
    assert list(tmp_path.glob("*.npz"))
    clear_cache(tmp_path)
    assert not list(tmp_path.glob("*.npz"))
