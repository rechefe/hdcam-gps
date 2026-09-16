"""Directed tests for the shared CAM acquisition base.

These check that the two data paths answer the same question and that the row
and query indices mean what cam_acq's four line mapping says they mean. The
sampling rate is deliberately low, so a whole acquisition run is cheap.

What is not covered here: how well any family performs. tests/test_hdcam_acq.py
owns the 1 bit family's own wiring, and the study's notebooks own sensitivity.
Two families are defined inside this file - one carrying Doppler in the query,
one shifting a row's segment offset - purely to exercise paths the baseline
cannot reach, and neither is a design anybody should use.
"""

import numpy as np
import pytest

from hdcam_gps.acq_base import AcqConfig, PrnResult
from hdcam_gps.ca_code import sampled_ca_code
from hdcam_gps.cam_acq import (
    QUERY_ROTATIONS,
    CamAcqClassifier,
    Cell,
    QueryIndex,
    RowIndex,
    quantize_iq,
    rotate_quarter_turns,
    run_starts,
)
from hdcam_gps.hdcam import HdCam
from hdcam_gps.hdcam_acq import OneBitHdCamClassifier
from hdcam_gps.signal_gen import SatelliteTruth, generate_synthetic

FS_HZ = 204.6e3  # 204 samples per code period, enough to stay correct and cheap
SAMPLES_PER_CODE = 204


def make_config(**overrides) -> AcqConfig:
    kwargs = {
        "fs_hz": FS_HZ,
        "prn_list": (1, 2),
        "doppler_min_hz": -500.0,
        "doppler_max_hz": 500.0,
        "doppler_step_hz": 500.0,
        "n_codes": 3,
    }
    kwargs.update(overrides)
    return AcqConfig(**kwargs)


def make_signal(
    config: AcqConfig,
    prn: int,
    doppler_hz=0.0,
    code_phase=0,
    carrier_phase=0.0,
) -> np.ndarray:
    """A noise free replica at a known Doppler, code phase and carrier phase."""
    code = np.tile(sampled_ca_code(prn, config.fs_hz), config.n_codes)
    code = np.roll(code, code_phase).astype(complex)
    time_s = np.arange(config.samples_per_acquisition) / config.fs_hz
    return code * np.exp(2j * np.pi * doppler_hz * time_s + 1j * carrier_phase)


# The one stored phase of DopplerInQueryClassifier, in radians. A query landing
# on it is what these tests want: they are about the index mapping, and a signal
# at exactly zero carrier phase has an empty quadrature plane.
TOY_STORED_PHASE = np.pi / 8


def naive_table(classifier, samples: np.ndarray) -> np.ndarray:
    """The distance table counted bit by bit, with no GEMM anywhere near it."""
    config = classifier.config
    n_starts = config.samples_per_acquisition - config.samples_per_code + 1
    table = np.empty((n_starts, classifier.n_rows), dtype=np.int32)
    for start in range(n_starts):
        bits = quantize_iq(samples[start : start + config.samples_per_code])
        best = None
        for rotation in range(QUERY_ROTATIONS):
            rotated = rotate_quarter_turns(bits, rotation)
            distances = np.count_nonzero(classifier.cam.grid != rotated, axis=1)
            best = distances if best is None else np.minimum(best, distances)
        table[start] = best
    return table


# --------------------------------------------------------------------------
# run_starts - the reduction every vote goes through
# --------------------------------------------------------------------------


def test_run_starts_groups_every_query_exactly_once():
    group = np.array([2, 0, 1, 0, 2, 2])
    order, first, labels = run_starts(group)

    assert sorted(order.tolist()) == list(range(len(group)))
    assert labels.tolist() == [0, 1, 2]
    assert first.tolist() == [0, 2, 3]


def test_run_starts_keeps_the_queries_of_a_group_together():
    group = np.array([5, 1, 5, 1, 5])
    order, first, labels = run_starts(group)
    bounds = np.append(first, len(order))
    for run, label in enumerate(labels):
        members = order[bounds[run] : bounds[run + 1]]
        assert (group[members] == label).all()


def test_run_starts_handles_one_group():
    order, first, labels = run_starts(np.zeros(4, dtype=int))
    assert first.tolist() == [0] and labels.tolist() == [0]
    assert len(order) == 4


# --------------------------------------------------------------------------
# the GEMM distance table
# --------------------------------------------------------------------------


def test_the_gemm_table_equals_the_counted_table_bit_for_bit():
    # The identity is exact, not an approximation: every dot product is an even
    # integer below 2**24, so float32 carries it without rounding.
    config = make_config()
    classifier = OneBitHdCamClassifier(config)
    scenario = generate_synthetic(
        config, [SatelliteTruth(1, 500.0, 40, cn0_dbhz=45.0)], seed=0
    )
    assert np.array_equal(
        classifier.distance_table(scenario.samples),
        naive_table(classifier, scenario.samples),
    )


def test_the_gemm_table_is_exact_on_noise_too():
    config = make_config()
    classifier = OneBitHdCamClassifier(config)
    scenario = generate_synthetic(config, [], add_noise=True, seed=7)
    assert np.array_equal(
        classifier.distance_table(scenario.samples),
        naive_table(classifier, scenario.samples),
    )


def test_the_table_has_a_row_per_query_and_a_column_per_codebook_row():
    config = make_config()
    classifier = OneBitHdCamClassifier(config)
    table = classifier.distance_table(np.zeros(config.samples_per_acquisition))
    assert table.shape == (classifier.n_queries, classifier.n_rows)


def test_row_distances_agrees_with_the_table():
    config = make_config()
    classifier = OneBitHdCamClassifier(config)
    bits = classifier.cam.grid[3].copy()
    counted = np.count_nonzero(classifier.cam.grid != bits, axis=1)
    assert np.array_equal(classifier.row_distances(bits), counted)


def test_the_signed_codebook_is_plus_and_minus_one():
    classifier = OneBitHdCamClassifier(make_config())
    signed = classifier.signed_codebook()
    assert set(np.unique(signed)) == {-1.0, 1.0}
    assert np.array_equal(signed > 0, classifier.cam.grid)


# --------------------------------------------------------------------------
# the two data paths answer the same question
# --------------------------------------------------------------------------


@pytest.mark.parametrize("code_phase", [0, 33, 150])
def test_the_cam_path_and_the_table_path_agree_on_a_planted_signal(code_phase):
    config = make_config()
    signal = make_signal(config, 1, doppler_hz=500.0, code_phase=code_phase)
    through_cam = OneBitHdCamClassifier(config, search_mode="cam").acquire(signal)
    through_table = OneBitHdCamClassifier(config, search_mode="table").acquire(signal)
    assert through_cam == through_table


@pytest.mark.parametrize("seed", range(4))
def test_the_cam_path_and_the_table_path_agree_under_noise(seed):
    config = make_config()
    scenario = generate_synthetic(
        config, [SatelliteTruth(2, 0.0, 77, cn0_dbhz=45.0)], seed=seed
    )
    through_cam = OneBitHdCamClassifier(config, search_mode="cam")
    through_table = OneBitHdCamClassifier(config, search_mode="table")
    assert through_cam.acquire(scenario.samples) == through_table.acquire(
        scenario.samples
    )


def test_the_two_paths_agree_on_silence():
    config = make_config()
    samples = np.zeros(config.samples_per_acquisition, dtype=complex)
    assert OneBitHdCamClassifier(config, search_mode="cam").acquire(samples) == []
    assert OneBitHdCamClassifier(config, search_mode="table").acquire(samples) == []


def test_the_matched_table_marks_exactly_the_rows_inside_the_threshold():
    config = make_config()
    classifier = OneBitHdCamClassifier(config)
    scenario = generate_synthetic(
        config, [SatelliteTruth(1, 500.0, 12, cn0_dbhz=50.0)], seed=1
    )
    matched = classifier.matched_table(scenario.samples)
    table = classifier.distance_table(scenario.samples)
    assert np.array_equal(matched, table <= classifier.hd_threshold)


def test_an_unknown_search_mode_is_rejected():
    with pytest.raises(AssertionError):
        OneBitHdCamClassifier(make_config(), search_mode="whatever")


def test_the_default_cam_is_still_an_hdcam():
    assert isinstance(OneBitHdCamClassifier(make_config()).cam, HdCam)


# --------------------------------------------------------------------------
# the row and query indices
# --------------------------------------------------------------------------


def test_the_row_index_names_the_prn_and_doppler_bin_of_every_row():
    classifier = OneBitHdCamClassifier(make_config())
    rows = classifier.row_index()
    assert len(rows.prn) == classifier.n_rows
    for row in range(classifier.n_rows):
        prn, doppler_hz = classifier.hypothesis_of(row)
        assert rows.prn[row] == prn
        assert classifier.config.doppler_grid_hz[rows.doppler_bin[row]] == doppler_hz


def test_the_baseline_carries_its_doppler_in_the_row():
    classifier = OneBitHdCamClassifier(make_config())
    assert not classifier.doppler_is_in_the_query
    assert (classifier.query_index().doppler_bin == -1).all()


def test_the_query_index_covers_every_contiguous_window():
    config = make_config()
    classifier = OneBitHdCamClassifier(config)
    queries = classifier.query_index()
    n_starts = config.samples_per_acquisition - config.samples_per_code + 1
    assert queries.start.tolist() == list(range(n_starts))


def test_the_indices_are_built_once_and_kept():
    classifier = OneBitHdCamClassifier(make_config())
    assert classifier.rows() is classifier.rows()
    assert classifier.queries() is classifier.queries()


def test_a_query_has_one_variant_per_quarter_turn():
    config = make_config()
    classifier = OneBitHdCamClassifier(config)
    samples = make_signal(config, 1)
    variants = classifier.query_variants(samples, 0)
    assert len(variants) == QUERY_ROTATIONS
    assert all(len(bits) == classifier.n_columns for bits in variants)


# --------------------------------------------------------------------------
# the decision rule over cells
# --------------------------------------------------------------------------


def test_decide_is_the_cells_it_finds_ranked():
    config = make_config()
    classifier = OneBitHdCamClassifier(config)
    scenario = generate_synthetic(
        config, [SatelliteTruth(1, 0.0, 60, cn0_dbhz=50.0)], seed=2
    )
    table = classifier.distance_table(scenario.samples)
    cells = classifier.cells_from_table(table, classifier.hd_threshold, 1)
    assert classifier.decide(table, classifier.hd_threshold, 1) == (
        classifier.rank_cells(cells)
    )


def test_a_tighter_threshold_never_finds_more_cells():
    config = make_config()
    classifier = OneBitHdCamClassifier(config)
    scenario = generate_synthetic(
        config, [SatelliteTruth(1, 0.0, 60, cn0_dbhz=50.0)], seed=3
    )
    table = classifier.distance_table(scenario.samples)
    loose = classifier.cells_from_table(table, classifier.hd_threshold, 1)
    tight = classifier.cells_from_table(table, classifier.hd_threshold - 20, 1)
    assert set(tight) <= set(loose)


def test_more_votes_never_find_more_cells():
    config = make_config(n_codes=6)
    classifier = OneBitHdCamClassifier(config)
    scenario = generate_synthetic(
        config, [SatelliteTruth(1, 0.0, 60, cn0_dbhz=50.0)], seed=4
    )
    table = classifier.distance_table(scenario.samples)
    assert set(classifier.cells_from_table(table, classifier.hd_threshold, 4)) <= set(
        classifier.cells_from_table(table, classifier.hd_threshold, 1)
    )


def test_a_shortlisted_cell_keeps_every_look_that_matched():
    # The vote counts looks, so the queries kept for a cell are exactly the looks
    # that put it there.
    config = make_config(n_codes=6)
    classifier = OneBitHdCamClassifier(config)
    scenario = generate_synthetic(
        config, [SatelliteTruth(1, 0.0, 60, cn0_dbhz=60.0)], seed=5
    )
    matched = classifier.matched_table(scenario.samples)
    cells = classifier.shortlist_cells(matched, 1)
    assert cells
    for cell, queries in cells.items():
        assert queries == sorted(queries)
        assert all(matched[query, cell.row] for query in queries)
        assert len(queries) >= 1


def test_rank_cells_keeps_the_closest_cell_of_each_prn():
    classifier = OneBitHdCamClassifier(make_config())
    near = Cell(classifier.row_of(1, doppler_bin=2), 2, 60)
    far = Cell(classifier.row_of(1, doppler_bin=0), 0, 11)
    assert classifier.rank_cells({far: 300, near: 120}) == [
        PrnResult(
            prn=1,
            doppler_hz=float(classifier.config.doppler_grid_hz[2]),
            code_phase=60,
        )
    ]


def test_rank_cells_returns_prns_in_configuration_order():
    classifier = OneBitHdCamClassifier(make_config())
    distances = {
        Cell(classifier.row_of(2, doppler_bin=0), 0, 5): 10,
        Cell(classifier.row_of(1, doppler_bin=1), 1, 9): 10,
    }
    assert [r.prn for r in classifier.rank_cells(distances)] == [1, 2]


def test_no_cells_gives_no_results():
    assert OneBitHdCamClassifier(make_config()).rank_cells({}) == []


# --------------------------------------------------------------------------
# the measured chance floor
# --------------------------------------------------------------------------


def test_the_measured_chance_floor_matches_the_binomial_model_for_one_bit():
    # The p = 0.5 model is right for a 1 bit codebook. It is measured anyway,
    # because it is wrong for a thermometer one.
    classifier = OneBitHdCamClassifier(make_config())
    mean, deviation = classifier.chance_floor(n_draws=128)
    assert mean == pytest.approx(classifier.n_columns / 2, rel=0.02)
    assert deviation == pytest.approx(np.sqrt(classifier.n_columns) / 2, rel=0.2)


def test_the_chance_floor_is_repeatable_for_a_seed():
    classifier = OneBitHdCamClassifier(make_config())
    assert classifier.chance_floor(n_draws=32, seed=1) == classifier.chance_floor(
        n_draws=32, seed=1
    )


# --------------------------------------------------------------------------
# a family carrying its Doppler in the query
# --------------------------------------------------------------------------


class DopplerInQueryClassifier(OneBitHdCamClassifier):
    """The code-only arrangement: one row per PRN, the CFO wiped off the query.

    This is not the family docs/CAM_FAMILY_STUDY.md will ship - it mixes in full
    precision, so the front end is no longer 1 bit. It exists here to drive the
    base class down the path where a hit's Doppler bin comes from the query.
    """

    @property
    def n_rows(self) -> int:
        return len(self.config.prn_list)

    def row_of(self, prn: int, doppler_bin: int = 0, codebook_phase: int = 0) -> int:
        return self.config.prn_list.index(prn)

    def build_codebook(self) -> np.ndarray:
        codebook = np.empty((self.n_rows, self.n_columns), dtype=bool)
        for position, prn in enumerate(self.config.prn_list):
            code = sampled_ca_code(prn, self.config.fs_hz).astype(complex)
            codebook[position] = quantize_iq(code * np.exp(1j * np.pi / 8))
        return codebook

    def row_index(self) -> RowIndex:
        return RowIndex(
            prn=np.array(self.config.prn_list),
            doppler_bin=np.full(self.n_rows, -1, dtype=int),
            segment_offset=np.zeros(self.n_rows, dtype=int),
        )

    def query_index(self, n_samples: int | None = None) -> QueryIndex:
        if n_samples is None:
            n_samples = self.config.samples_per_acquisition
        starts = np.arange(n_samples - self.config.samples_per_code + 1)
        bins = np.arange(self.n_doppler_bins)
        return QueryIndex(
            start=np.repeat(starts, len(bins)),
            doppler_bin=np.tile(bins, len(starts)),
        )

    def query_variants(self, samples: np.ndarray, query: int) -> list[np.ndarray]:
        index = self.queries(len(samples))
        start = int(index.start[query])
        doppler_hz = self.config.doppler_grid_hz[int(index.doppler_bin[query])]
        window = self.query_window(samples, start)
        time_s = (start + np.arange(len(window))) / self.config.fs_hz
        wiped = window * np.exp(-2j * np.pi * doppler_hz * time_s)
        bits = quantize_iq(wiped)
        return [
            rotate_quarter_turns(bits, rotation) for rotation in range(QUERY_ROTATIONS)
        ]


def test_moving_the_doppler_into_the_query_keeps_one_row_per_prn():
    classifier = DopplerInQueryClassifier(make_config(), hd_threshold=40)
    assert classifier.n_rows == 2
    assert classifier.doppler_is_in_the_query
    assert classifier.n_queries == (
        classifier.n_doppler_bins
        * (
            classifier.config.samples_per_acquisition
            - classifier.config.samples_per_code
            + 1
        )
    )


@pytest.mark.parametrize("doppler_hz", [-500.0, 0.0, 500.0])
def test_a_query_carried_doppler_bin_reaches_the_result(doppler_hz):
    config = make_config()
    classifier = DopplerInQueryClassifier(config, hd_threshold=40, min_votes=1)
    signal = make_signal(
        config, 2, doppler_hz=doppler_hz, code_phase=55,
        carrier_phase=TOY_STORED_PHASE,
    )
    assert classifier.acquire(signal) == [
        PrnResult(prn=2, doppler_hz=doppler_hz, code_phase=55)
    ]


def test_the_two_paths_agree_when_the_query_carries_the_doppler():
    config = make_config()
    signal = make_signal(
        config, 1, doppler_hz=500.0, code_phase=20, carrier_phase=TOY_STORED_PHASE
    )
    through_cam = DopplerInQueryClassifier(config, hd_threshold=40, min_votes=1)
    through_table = DopplerInQueryClassifier(
        config, hd_threshold=40, min_votes=1, search_mode="table"
    )
    assert through_cam.acquire(signal) == through_table.acquire(signal)


def test_a_query_carried_doppler_multiplies_the_cells_being_voted_on():
    baseline = OneBitHdCamClassifier(make_config())
    code_only = DopplerInQueryClassifier(make_config(), hd_threshold=40)
    assert code_only.n_cells == code_only.n_rows * baseline.n_doppler_bins * (
        code_only.config.samples_per_code
    )


# --------------------------------------------------------------------------
# a row that starts part way into the code period
# --------------------------------------------------------------------------


class OffsetRowClassifier(OneBitHdCamClassifier):
    """The baseline with every row declared to start SHIFT samples in.

    The codebook is untouched, so this says nothing about segmented acquisition.
    It says only that the mapping subtracts the offset from the window start.
    """

    SHIFT = 7

    def row_index(self) -> RowIndex:
        rows = super().row_index()
        return RowIndex(
            prn=rows.prn,
            doppler_bin=rows.doppler_bin,
            segment_offset=np.full(self.n_rows, self.SHIFT, dtype=int),
        )


def test_a_segment_offset_shifts_the_code_phase_a_hit_reports():
    config = make_config()
    signal = make_signal(config, 1, doppler_hz=500.0, code_phase=33)
    plain = OneBitHdCamClassifier(config, search_mode="table")
    offset = OffsetRowClassifier(config, search_mode="table")

    reported = plain.acquire(signal)
    shifted = offset.acquire(signal)
    assert [r.prn for r in reported] == [r.prn for r in shifted]
    for before, after in zip(reported, shifted):
        assert after.code_phase == (
            before.code_phase - OffsetRowClassifier.SHIFT
        ) % config.samples_per_code


class TwoOffsetClassifier(OffsetRowClassifier):
    """Half the rows shifted and half not, so the vote runs twice."""

    def row_index(self) -> RowIndex:
        rows = OneBitHdCamClassifier.row_index(self)
        offsets = np.zeros(self.n_rows, dtype=int)
        offsets[::2] = self.SHIFT
        return RowIndex(
            prn=rows.prn, doppler_bin=rows.doppler_bin, segment_offset=offsets
        )


def test_rows_at_two_offsets_are_voted_on_separately():
    # Every row still gets a verdict, which is what a segmented family needs: its
    # sub-rows read different code phases out of one window.
    config = make_config()
    classifier = TwoOffsetClassifier(config, search_mode="table", min_votes=1)
    signal = make_signal(config, 1, doppler_hz=500.0, code_phase=33)
    table = classifier.distance_table(signal)
    cells = classifier.cells_from_table(table, classifier.hd_threshold, 1)

    offsets = classifier.row_index().segment_offset
    phases = {int(offsets[cell.row]): cell.code_phase for cell in cells}
    assert set(phases) == {0, TwoOffsetClassifier.SHIFT}
    assert (
        phases[0] - phases[TwoOffsetClassifier.SHIFT]
    ) % config.samples_per_code == TwoOffsetClassifier.SHIFT


# --------------------------------------------------------------------------
# the hooks a family has to supply
# --------------------------------------------------------------------------


class BareClassifier(CamAcqClassifier):
    """Nothing overridden, so every hook should refuse."""

    n_rows = 1
    n_columns = 8


@pytest.mark.parametrize(
    "call",
    [
        lambda c: c.build_codebook(),
        lambda c: c.row_index(),
        lambda c: c.query_index(),
        lambda c: c.query_variants(np.zeros(8), 0),
    ],
)
def test_an_unimplemented_hook_says_so(call):
    bare = CamAcqClassifier.__new__(BareClassifier)
    bare.config = make_config()
    with pytest.raises(NotImplementedError):
        call(bare)
