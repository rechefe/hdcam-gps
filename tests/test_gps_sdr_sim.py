"""Directed tests for the gps-sdr-sim wrapper.

This module is only the driver for the simulator, so these tests cover the
channel report parser, the I/Q reader and the record it returns. The user facing
API built on top of it is tested in test_signal_gen. Tests that actually run the
simulator are skipped when the submodule is missing.
"""

import numpy as np
import pytest

from hdcam_gps.gps_sdr_sim import (
    DEFAULT_RINEX,
    PATCH_DIR,
    SIM_WARMUP_S,
    ChannelState,
    SimulatedRecord,
    VisibleSatellite,
    sources_checked_out,
    labels,
    parse_truth,
    parse_visible,
    patches,
    read_iq,
    simulate,
)

needs_simulator = pytest.mark.skipif(
    not sources_checked_out(), reason="the gps-sdr-sim submodule is not checked out"
)

# A trimmed copy of what the simulator writes to stderr with -v.
TRUTH_SAMPLE = """TRUTH 1 08 2619.629561913 438.038390729
TRUTH 1 27 836.546180892 709.343671851
Time into run =  0.2
TRUTH 2 08 2619.596423737 438.208472725
TRUTH 2 27 836.489331424 709.397969213
"""

STDERR_SAMPLE = """Using static location mode.
xyz =   4435006.3,   3105424.9,   3360484.4
llh =   32.000000,   35.000000,       100.0
Start time = 2022/01/01,00:00:00 (2190:518400)
Duration = 1.0 [sec]
08  316.3  36.0  22340972.8   2.4
10   17.6  68.2  20715692.9   1.6
27  306.6  70.2  20489885.6   1.6
Time into run =  0.2
Done!
"""


# --------------------------------------------------------------------------
# parse_visible
# --------------------------------------------------------------------------


def test_parse_visible_reads_every_channel_line():
    visible = parse_visible(STDERR_SAMPLE)
    assert [satellite.prn for satellite in visible] == [8, 10, 27]


def test_parse_visible_reads_the_fields_of_a_channel():
    satellite = parse_visible(STDERR_SAMPLE)[0]
    assert satellite == VisibleSatellite(
        prn=8, azimuth_deg=316.3, elevation_deg=36.0, range_m=22340972.8
    )


def test_parse_visible_ignores_the_surrounding_chatter():
    assert parse_visible("Done!\nDuration = 1.0 [sec]\nProcess time = 0.1 [sec]\n") == ()


def test_parse_visible_returns_prn_sorted_entries():
    out_of_order = "27  306.6  70.2  20489885.6   1.6\n08  316.3  36.0  22340972.8   2.4"
    assert [satellite.prn for satellite in parse_visible(out_of_order)] == [8, 27]


def test_parse_visible_keeps_one_entry_per_prn():
    # The simulator reprints the channel table as the run progresses.
    repeated = STDERR_SAMPLE + "08  317.0  36.5  22340000.0   2.4\n"
    visible = parse_visible(repeated)
    assert [satellite.prn for satellite in visible] == [8, 10, 27]
    assert visible[0].azimuth_deg == 317.0  # the latest report wins


def test_parse_visible_accepts_a_negative_azimuth():
    assert parse_visible("05  -12.5  40.0  21000000.0   2.0")[0].azimuth_deg == -12.5


def test_parse_visible_accepts_a_negative_azimuth():
    assert parse_visible("05  -12.5  40.0  21000000.0   2.0")[0].azimuth_deg == -12.5


# --------------------------------------------------------------------------
# read_iq
# --------------------------------------------------------------------------


def write_iq(tmp_path, values, dtype):
    path = tmp_path / "record.bin"
    np.array(values, dtype=dtype).tofile(path)
    return path


def test_read_iq_deinterleaves_i_and_q(tmp_path):
    path = write_iq(tmp_path, [3, 4, -3, -4], np.int16)
    samples = read_iq(path, 16)
    assert len(samples) == 2
    # Unit RMS normalization keeps the ratio of the two samples.
    assert samples[0] / samples[1] == pytest.approx(-1.0)
    assert np.angle(samples[0]) == pytest.approx(np.arctan2(4, 3))


def test_read_iq_normalizes_to_unit_rms(tmp_path):
    rng = np.random.default_rng(0)
    raw = rng.integers(-2000, 2000, size=2000)
    samples = read_iq(write_iq(tmp_path, raw, np.int16), 16)
    assert np.mean(np.abs(samples) ** 2) == pytest.approx(1.0)


def test_read_iq_reads_the_eight_bit_format(tmp_path):
    path = write_iq(tmp_path, [1, 0, 0, 1], np.int8)
    samples = read_iq(path, 8)
    assert len(samples) == 2
    assert np.mean(np.abs(samples) ** 2) == pytest.approx(1.0)


def test_read_iq_leaves_an_all_zero_record_alone(tmp_path):
    samples = read_iq(write_iq(tmp_path, [0, 0, 0, 0], np.int16), 16)
    assert not samples.any()


@pytest.mark.parametrize("bad_bits", [1, 4, 32])
def test_read_iq_rejects_an_unsupported_format(tmp_path, bad_bits):
    path = write_iq(tmp_path, [0, 0], np.int16)
    with pytest.raises(AssertionError):
        read_iq(path, bad_bits)


# --------------------------------------------------------------------------
# SimulatedRecord
# --------------------------------------------------------------------------


def make_record(samples, prns, fs_hz) -> SimulatedRecord:
    visible = tuple(
        VisibleSatellite(prn=prn, azimuth_deg=0.0, elevation_deg=45.0, range_m=2.2e7)
        for prn in prns
    )
    return SimulatedRecord(samples=samples, visible=visible, fs_hz=fs_hz)


def test_record_reports_its_prns():
    record = make_record(np.zeros(4, dtype=complex), (8, 10, 27), 4e6)
    assert record.prns == (8, 10, 27)


# --------------------------------------------------------------------------
# the simulator itself
# --------------------------------------------------------------------------


def test_the_shipped_rinex_file_is_present():
    assert not sources_checked_out() or DEFAULT_RINEX.is_file()


@needs_simulator
def test_simulate_produces_a_record_of_the_requested_shape():
    record = simulate(fs_hz=2.6e6, duration_s=0.2)
    assert record.fs_hz == 2.6e6
    assert len(record.samples) == int(2.6e6 * 0.2)  # exactly what was asked for
    assert np.iscomplexobj(record.samples)
    assert np.mean(np.abs(record.samples) ** 2) == pytest.approx(1.0, rel=0.05)


@needs_simulator
def test_simulate_delivers_the_full_duration_at_several_lengths():
    # The wrapper compensates for the simulator's warm-up, so short requests
    # that the simulator alone would return empty still come back full length.
    assert SIM_WARMUP_S > 0
    for duration_s in (0.1, 0.2):
        record = simulate(fs_hz=2.6e6, duration_s=duration_s)
        assert len(record.samples) == int(2.6e6 * duration_s)


@needs_simulator
def test_simulate_reports_a_plausible_constellation():
    record = simulate(fs_hz=2.6e6, duration_s=0.2)
    assert 4 <= len(record.prns) <= 12  # a normal sky
    assert record.prns == tuple(sorted(set(record.prns)))
    for satellite in record.visible:
        assert 1 <= satellite.prn <= 32
        assert 0.0 <= satellite.elevation_deg <= 90.0
        assert 1.9e7 < satellite.range_m < 2.7e7  # GPS orbital range


# --------------------------------------------------------------------------
# the patch, and the channel report it adds
# --------------------------------------------------------------------------


def test_a_patch_is_shipped():
    assert [path.name for path in patches()] == [
        "0001-report-doppler-and-code-phase.patch"
    ]
    assert PATCH_DIR.is_dir()


def test_parse_truth_reads_every_reported_state():
    states = parse_truth(TRUTH_SAMPLE)
    assert len(states) == 4
    assert states[0] == ChannelState(
        tick=1, prn=8, doppler_hz=2619.629561913, code_phase_chips=438.038390729
    )


def test_parse_truth_keeps_the_ticks_apart():
    states = parse_truth(TRUTH_SAMPLE)
    assert {state.tick for state in states} == {1, 2}
    assert [s.prn for s in states if s.tick == 2] == [8, 27]


def test_parse_truth_ignores_everything_else():
    assert parse_truth(STDERR_SAMPLE) == ()


# --------------------------------------------------------------------------
# labels
# --------------------------------------------------------------------------


def truth_record(fs_hz=4e6) -> SimulatedRecord:
    return SimulatedRecord(
        samples=np.zeros(4, dtype=complex),
        visible=(),
        fs_hz=fs_hz,
        channel_states=parse_truth(TRUTH_SAMPLE),
    )


def test_labels_convert_elapsed_chips_into_a_replica_shift():
    record = truth_record()
    samples_per_code = 4000
    doppler_hz, code_phase = labels(record)[8]
    assert doppler_hz == 2619.629561913
    elapsed = 438.038390729 * samples_per_code / 1023
    assert code_phase == round(samples_per_code - elapsed) % samples_per_code


def test_labels_take_the_tick_covering_the_offset():
    record = truth_record()
    first = labels(record, offset=0)[8]
    second = labels(record, offset=int(4e6 * 0.1))[8]
    assert first[0] != second[0]  # a different tick, so a different Doppler
    assert second[0] == 2619.596423737


def test_labels_shift_the_code_phase_by_the_offset_within_a_tick():
    record = truth_record()
    base = labels(record, offset=0)[8][1]
    shifted = labels(record, offset=10)[8][1]
    assert shifted == (base - 10) % 4000


def test_labels_reject_an_offset_past_the_reported_ticks():
    record = truth_record()
    with pytest.raises(AssertionError):
        labels(record, offset=int(4e6 * 0.5))


def test_labels_need_a_patched_simulator():
    record = SimulatedRecord(samples=np.zeros(4, dtype=complex), visible=(), fs_hz=4e6)
    with pytest.raises(AssertionError):
        labels(record)


@needs_simulator
def test_a_real_run_reports_a_state_for_every_visible_satellite():
    record = simulate(fs_hz=4e6, duration_s=0.1)
    assert record.channel_states
    assert set(labels(record)) == set(record.prns)
