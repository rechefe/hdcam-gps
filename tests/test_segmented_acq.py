"""Directed tests for the segmented family.

The one thing worth checking hardest is the claim that makes this design cheap:
a hit on segment k at start s stands for the same code phase as a hit on segment
0 at start s - k*segment_samples, so cutting a row into K pieces multiplies the
rows without multiplying the searches. The sampling rate is deliberately low, so
a whole acquisition run is cheap.

The acquisition tests ask for the PRN and the code phase, not the Doppler. A
128 bit segment is 62 microseconds of carrier, over which 500 Hz turns by 11
degrees, so a single sub-row cannot tell one bin from its neighbour; there is a
test recording that. Restoring it is what the m-of-K rule across segments is
for, and that rule is phase 2.
"""

import numpy as np
import pytest

from hdcam_gps.acq_base import AcqConfig, PrnResult
from hdcam_gps.ca_code import sampled_ca_code
from hdcam_gps.cam_acq import QUERY_ROTATIONS, quantize_iq
from hdcam_gps.hdcam_acq import OneBitHdCamClassifier
from hdcam_gps.segmented_acq import SegmentedHdCamClassifier

FS_HZ = 204.6e3  # 204 samples per code period, enough to stay correct and cheap
SAMPLES_PER_CODE = 204
SEGMENT_BITS = 64  # 32 samples, so the 204 sample period holds six of them
OFF_AXIS_PHASE = 0.7  # So a planted replica does not land on an axis


def make_config(**overrides) -> AcqConfig:
    kwargs = {
        "fs_hz": FS_HZ,
        "prn_list": (1, 2, 3),
        "doppler_min_hz": -500.0,
        "doppler_max_hz": 500.0,
        "doppler_step_hz": 500.0,
        "n_codes": 4,
    }
    kwargs.update(overrides)
    return AcqConfig(**kwargs)


def make_classifier(config: AcqConfig, **overrides) -> SegmentedHdCamClassifier:
    kwargs = {"segment_bits": SEGMENT_BITS}
    kwargs.update(overrides)
    return SegmentedHdCamClassifier(config, **kwargs)


def make_signal(
    config: AcqConfig,
    prn: int,
    doppler_hz: float = 0.0,
    code_phase: int = 0,
    carrier_phase: float = OFF_AXIS_PHASE,
) -> np.ndarray:
    """A noise free replica at a known Doppler, code phase and carrier phase."""
    code = np.tile(sampled_ca_code(prn, config.fs_hz), config.n_codes)
    code = np.roll(code, code_phase).astype(complex)
    time_s = np.arange(config.samples_per_acquisition) / config.fs_hz
    return code * np.exp(2j * np.pi * doppler_hz * time_s + 1j * carrier_phase)


# --------------------------------------------------------------------------
# the shape: more rows, the same bits, the same searches
# --------------------------------------------------------------------------


def test_a_period_is_cut_into_whole_segments():
    classifier = make_classifier(make_config())
    assert classifier.segment_samples == SEGMENT_BITS // 2
    assert classifier.n_segments == SAMPLES_PER_CODE // classifier.segment_samples
    assert classifier.n_columns == SEGMENT_BITS


def test_the_rows_multiply_by_the_segments():
    config = make_config()
    classifier = make_classifier(config)
    assert classifier.n_hypotheses == OneBitHdCamClassifier(config).n_rows
    assert classifier.n_rows == classifier.n_hypotheses * classifier.n_segments


def test_the_stored_bits_do_not_grow():
    # The match line narrows from a code period to segment_bits at no cost in
    # area, which is the whole argument for the design.
    config = make_config()
    classifier = make_classifier(config)
    baseline = OneBitHdCamClassifier(config)
    covered = classifier.n_segments * classifier.segment_samples
    assert classifier.n_rows * classifier.n_columns == (
        baseline.n_rows * 2 * covered
    )
    assert covered <= config.samples_per_code


def test_the_tail_of_a_period_that_does_not_divide_is_left_out():
    # 204 samples hold six 32 sample segments with 12 left over, and 1023 hold
    # fifteen 64 sample ones with 63 left over. Every code phase is still
    # reachable, because the queries slide over all of them.
    classifier = make_classifier(make_config())
    covered = classifier.n_segments * classifier.segment_samples
    assert covered == 192
    assert covered < SAMPLES_PER_CODE


def test_the_searches_do_not_follow_the_rows():
    # One short query sliding over the record visits every segment at every
    # alignment, so the searches rise only because a shorter window starts in
    # more places - 28 % here, 10 % at the study's own rate - while the rows
    # rise by the number of segments.
    config = make_config()
    classifier = make_classifier(config)
    baseline = OneBitHdCamClassifier(config)
    assert classifier.n_rows == baseline.n_rows * classifier.n_segments
    assert classifier.n_queries < baseline.n_queries * 1.5


@pytest.mark.parametrize("bad", [0, -2, 3, 7])
def test_an_odd_or_empty_segment_width_is_rejected(bad):
    with pytest.raises(AssertionError):
        make_classifier(make_config(), segment_bits=bad)


def test_a_segment_longer_than_the_code_period_is_rejected():
    with pytest.raises(AssertionError):
        make_classifier(make_config(), segment_bits=4 * SAMPLES_PER_CODE)


# --------------------------------------------------------------------------
# the mapping that makes one pass enough
# --------------------------------------------------------------------------


def test_a_sub_row_is_its_slice_of_the_replica():
    config = make_config()
    classifier = make_classifier(config)
    prn, doppler_bin, phase, segment = 2, 2, 1, 3
    time_s = np.arange(config.samples_per_code) / config.fs_hz
    replica = (
        sampled_ca_code(prn, config.fs_hz)
        * np.exp(2j * np.pi * config.doppler_grid_hz[doppler_bin] * time_s)
        * np.exp(2j * np.pi * (phase + 0.5) / classifier.n_phases)
    )
    first = segment * classifier.segment_samples
    expected = quantize_iq(replica[first : first + classifier.segment_samples])
    row = classifier.row_of(prn, doppler_bin, segment, phase)
    assert np.array_equal(classifier.build_codebook()[row], expected)


def test_the_segment_offset_is_where_the_sub_row_starts():
    classifier = make_classifier(make_config())
    rows = classifier.row_index()
    for segment in range(classifier.n_segments):
        row = classifier.row_of(1, 0, segment)
        assert rows.segment_offset[row] == segment * classifier.segment_samples
        assert rows.prn[row] == 1
        assert rows.doppler_bin[row] == 0


def test_a_hit_on_a_later_segment_votes_for_an_earlier_code_phase():
    # The claim the design rests on: segment k at start s is segment 0 at start
    # s - k*segment_samples, so both stand for one code phase.
    config = make_config()
    classifier = make_classifier(config)
    rows = classifier.row_index()
    samples_per_code = config.samples_per_code
    for segment in range(classifier.n_segments):
        row = classifier.row_of(1, 0, segment)
        start = 40 + segment * classifier.segment_samples
        code_phase = (start - rows.segment_offset[row]) % samples_per_code
        assert code_phase == 40


def test_a_query_is_one_short_window_turned_four_ways():
    config = make_config()
    classifier = make_classifier(config)
    variants = classifier.query_variants(make_signal(config, 1), 0)
    assert len(variants) == QUERY_ROTATIONS
    assert all(len(bits) == SEGMENT_BITS for bits in variants)


def test_a_window_past_the_end_of_the_record_is_rejected():
    config = make_config()
    classifier = make_classifier(config)
    samples = make_signal(config, 1)
    last = len(samples) - classifier.segment_samples
    classifier.query_variants(samples, last)
    with pytest.raises(AssertionError):
        classifier.query_variants(samples, last + 1)


# --------------------------------------------------------------------------
# acquisition
# --------------------------------------------------------------------------


@pytest.mark.parametrize("code_phase", [0, 55, 150])
def test_a_planted_satellite_comes_back_at_its_own_code_phase(code_phase):
    config = make_config()
    classifier = make_classifier(config)
    signal = make_signal(config, 2, doppler_hz=500.0, code_phase=code_phase)
    reported = [r for r in classifier.acquire(signal) if r.prn == 2]
    assert [r.code_phase for r in reported] == [code_phase]


def test_the_family_cannot_resolve_doppler_at_all():
    # A segment this short has a frequency resolution of 1/T, and the whole
    # search range fits inside one cell of it, so every bin is an equally good
    # answer. The m-of-K rule cannot win this back: it counts segments, and they
    # all match.
    config = make_config()
    classifier = make_classifier(config)
    window_s = classifier.segment_samples / config.fs_hz
    assert 1.0 / window_s > config.doppler_max_hz - config.doppler_min_hz

    reported = {
        classifier.acquire(
            make_signal(config, 2, doppler_hz=float(doppler_hz), code_phase=55)
        )[0].doppler_hz
        for doppler_hz in config.doppler_grid_hz
    }
    assert len(reported) == 1


def test_every_doppler_bin_matches_a_segment_equally_well():
    # The measurement behind the test above: a clean satellite sits at distance
    # zero from its own PRN in every bin, so nothing downstream can choose.
    config = make_config()
    classifier = make_classifier(config, search_mode="table")
    signal = make_signal(config, 2, doppler_hz=500.0, code_phase=17)
    table = classifier.distance_table(signal)
    rows = classifier.row_index()

    closest = set()
    for doppler_bin in range(classifier.n_doppler_bins):
        of_bin = np.flatnonzero((rows.prn == 2) & (rows.doppler_bin == doppler_bin))
        closest.add(int(table[:, of_bin].min()))
    assert closest == {0}


def test_more_segments_per_look_never_shortlists_more():
    config = make_config()
    signal = make_signal(config, 2, doppler_hz=500.0, code_phase=55)
    lenient = make_classifier(config, min_segments=1, search_mode="table")
    strict = make_classifier(config, min_segments=6, search_mode="table")
    table = lenient.distance_table(signal)
    assert set(
        strict.cells_from_table(table, strict.hd_threshold, 1)
    ) <= set(lenient.cells_from_table(table, lenient.hd_threshold, 1))


def test_the_sub_rows_of_one_hypothesis_are_voted_on_together():
    classifier = make_classifier(make_config())
    rows = classifier.rows()
    assert classifier.n_hypotheses == int(rows.hypothesis.max()) + 1
    for segment in range(classifier.n_segments):
        assert rows.hypothesis[classifier.row_of(1, 0, segment)] == (
            rows.hypothesis[classifier.row_of(1, 0, 0)]
        )
    assert rows.hypothesis[classifier.row_of(1, 1, 0)] != (
        rows.hypothesis[classifier.row_of(1, 0, 0)]
    )


@pytest.mark.parametrize("bad", [0, -1, 99])
def test_a_segment_rule_outside_the_sub_rows_available_is_rejected(bad):
    with pytest.raises(AssertionError):
        make_classifier(make_config(), min_segments=bad)


def test_a_silent_record_acquires_nothing():
    config = make_config()
    classifier = make_classifier(config)
    assert classifier.acquire(np.zeros(config.samples_per_acquisition)) == []


def test_the_cam_path_and_the_table_path_agree():
    config = make_config()
    signal = make_signal(config, 3, doppler_hz=-500.0, code_phase=88)
    through_cam = make_classifier(config, search_mode="cam")
    through_table = make_classifier(config, search_mode="table")
    assert through_cam.acquire(signal) == through_table.acquire(signal)
