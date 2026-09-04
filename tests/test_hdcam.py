"""Directed tests for the HdCam simulator.

All CAMs here are deliberately small (a handful of rows/columns) so that every
expected result can be checked by hand.
"""

import random

import numpy as np
import pytest

from hdcam_gps.hdcam import HdCam

N_ROWS = 4
N_COLUMNS = 5

# A small, hand-checkable array. Rows are distinct and their pairwise
# distances are easy to reason about.
ARRAY = [
    [0, 0, 0, 0, 0],  # row 0 - all zeros
    [1, 0, 0, 0, 0],  # row 1 - distance 1 from row 0
    [1, 1, 1, 0, 0],  # row 2 - distance 3 from row 0
    [1, 1, 1, 1, 1],  # row 3 - all ones, distance 5 from row 0
]


def make_cam(hd_threshold=0):
    cam = HdCam(N_ROWS, N_COLUMNS, hd_threshold)
    cam.write_array(ARRAY)
    return cam


def hamming(a, b):
    return sum(x != y for x, y in zip(a, b))


# --------------------------------------------------------------------------
# initialization
# --------------------------------------------------------------------------


def test_init_creates_empty_grid_of_the_requested_size():
    cam = HdCam(N_ROWS, N_COLUMNS, 0)
    assert cam.n_rows == N_ROWS
    assert cam.n_columns == N_COLUMNS
    assert cam.grid.shape == (N_ROWS, N_COLUMNS)
    assert cam.grid.dtype == bool
    assert not cam.grid.any()


def test_init_stores_the_hamming_threshold():
    cam = HdCam(N_ROWS, N_COLUMNS, 3)
    assert cam.hd_threshold == 3


@pytest.mark.parametrize("bad_threshold", [-1, N_COLUMNS, N_COLUMNS + 1])
def test_init_rejects_out_of_range_threshold(bad_threshold):
    with pytest.raises(AssertionError):
        HdCam(N_ROWS, N_COLUMNS, bad_threshold)


def test_init_accepts_the_boundary_thresholds():
    assert HdCam(N_ROWS, N_COLUMNS, 0).hd_threshold == 0
    assert HdCam(N_ROWS, N_COLUMNS, N_COLUMNS - 1).hd_threshold == N_COLUMNS - 1


# --------------------------------------------------------------------------
# write_row
# --------------------------------------------------------------------------


def test_write_row_writes_the_first_row():
    cam = HdCam(N_ROWS, N_COLUMNS, 0)
    cam.write_row(0, [1, 0, 1, 0, 1])
    assert list(cam.grid[0]) == [True, False, True, False, True]
    # every other row is untouched
    assert not cam.grid[1:].any()


def test_write_row_writes_the_last_row():
    cam = HdCam(N_ROWS, N_COLUMNS, 0)
    cam.write_row(N_ROWS - 1, [1, 1, 0, 0, 1])
    assert list(cam.grid[N_ROWS - 1]) == [True, True, False, False, True]
    # every other row is untouched
    assert not cam.grid[: N_ROWS - 1].any()


def test_write_row_writes_each_row_at_its_own_address():
    cam = HdCam(N_ROWS, N_COLUMNS, 0)
    for address, row in enumerate(ARRAY):
        cam.write_row(address, row)
    assert cam.grid.tolist() == np.array(ARRAY, dtype=bool).tolist()


def test_write_row_overwrites_an_existing_row():
    cam = make_cam()
    cam.write_row(2, [0, 0, 0, 0, 0])
    assert not cam.grid[2].any()
    # neighbors survive
    assert list(cam.grid[1]) == [True, False, False, False, False]
    assert cam.grid[3].all()


def test_write_row_coerces_non_zero_values_to_true():
    cam = HdCam(N_ROWS, N_COLUMNS, 0)
    cam.write_row(0, [2, 0, -1, 0, 1])
    assert list(cam.grid[0]) == [True, False, True, False, True]


@pytest.mark.parametrize("bad_address", [-1, N_ROWS, N_ROWS + 1])
def test_write_row_rejects_out_of_range_address(bad_address):
    cam = HdCam(N_ROWS, N_COLUMNS, 0)
    with pytest.raises(AssertionError):
        cam.write_row(bad_address, [0] * N_COLUMNS)


@pytest.mark.parametrize("bad_length", [N_COLUMNS - 1, N_COLUMNS + 1])
def test_write_row_rejects_wrong_data_length(bad_length):
    cam = HdCam(N_ROWS, N_COLUMNS, 0)
    with pytest.raises(AssertionError):
        cam.write_row(0, [0] * bad_length)


# --------------------------------------------------------------------------
# write_array
# --------------------------------------------------------------------------


def test_write_array_writes_the_whole_grid():
    cam = HdCam(N_ROWS, N_COLUMNS, 0)
    cam.write_array(ARRAY)
    assert cam.grid.shape == (N_ROWS, N_COLUMNS)
    assert cam.grid.dtype == bool
    assert cam.grid.tolist() == np.array(ARRAY, dtype=bool).tolist()


def test_write_array_writes_the_first_and_last_rows():
    cam = make_cam()
    assert list(cam.grid[0]) == [False] * N_COLUMNS
    assert list(cam.grid[N_ROWS - 1]) == [True] * N_COLUMNS


def test_write_array_matches_row_by_row_writes():
    by_array = make_cam()
    by_row = HdCam(N_ROWS, N_COLUMNS, 0)
    for address, row in enumerate(ARRAY):
        by_row.write_row(address, row)
    assert by_array.grid.tolist() == by_row.grid.tolist()


@pytest.mark.parametrize("bad_rows", [N_ROWS - 1, N_ROWS + 1])
def test_write_array_rejects_wrong_number_of_rows(bad_rows):
    cam = HdCam(N_ROWS, N_COLUMNS, 0)
    with pytest.raises(AssertionError):
        cam.write_array([[0] * N_COLUMNS for _ in range(bad_rows)])


def test_write_array_rejects_a_row_of_the_wrong_length():
    cam = HdCam(N_ROWS, N_COLUMNS, 0)
    data = [[0] * N_COLUMNS for _ in range(N_ROWS)]
    data[N_ROWS - 1] = [0] * (N_COLUMNS + 1)
    with pytest.raises(AssertionError):
        cam.write_array(data)


# --------------------------------------------------------------------------
# set_hd_threshold
# --------------------------------------------------------------------------


def test_set_hd_threshold_replaces_the_initial_value():
    cam = HdCam(N_ROWS, N_COLUMNS, 0)
    cam.set_hd_threshold(3)
    assert cam.hd_threshold == 3


def test_set_hd_threshold_changes_the_search_result():
    cam = make_cam(hd_threshold=0)
    query = [0, 0, 0, 0, 0]  # exactly row 0; row 1 is at distance 1

    assert list(cam.search_cam(query)) == [0]

    cam.set_hd_threshold(1)
    assert list(cam.search_cam(query)) == [0, 1]

    cam.set_hd_threshold(3)
    assert list(cam.search_cam(query)) == [0, 1, 2]

    cam.set_hd_threshold(N_COLUMNS - 1)  # row 3 is at distance 5, still out
    assert list(cam.search_cam(query)) == [0, 1, 2]

    cam.set_hd_threshold(0)  # and back down again
    assert list(cam.search_cam(query)) == [0]


@pytest.mark.parametrize("bad_threshold", [-1, N_COLUMNS, N_COLUMNS + 1])
def test_set_hd_threshold_rejects_out_of_range_values(bad_threshold):
    cam = HdCam(N_ROWS, N_COLUMNS, 2)
    with pytest.raises(AssertionError):
        cam.set_hd_threshold(bad_threshold)
    assert cam.hd_threshold == 2  # unchanged after a rejected write


# --------------------------------------------------------------------------
# search_cam - threshold 0
# --------------------------------------------------------------------------


@pytest.mark.parametrize("address", range(N_ROWS))
def test_search_with_threshold_zero_matches_only_the_exact_row(address):
    cam = make_cam(hd_threshold=0)
    assert list(cam.search_cam(ARRAY[address])) == [address]


def test_search_with_threshold_zero_misses_a_one_bit_difference():
    cam = make_cam(hd_threshold=0)
    query = [0, 1, 0, 0, 0]  # distance 1 from row 0, distance 2 from row 1
    assert list(cam.search_cam(query)) == []


def test_search_with_threshold_zero_matches_every_duplicate_row():
    cam = HdCam(3, N_COLUMNS, 0)
    cam.write_array(
        [
            [1, 0, 1, 0, 1],
            [0, 0, 0, 0, 0],
            [1, 0, 1, 0, 1],
        ]
    )
    assert list(cam.search_cam([1, 0, 1, 0, 1])) == [0, 2]


# --------------------------------------------------------------------------
# search_cam - maximal threshold (n_columns - 1)
# --------------------------------------------------------------------------


def test_search_with_max_threshold_matches_everything_but_the_exact_opposite():
    cam = make_cam(hd_threshold=N_COLUMNS - 1)
    # query == row 3 (all ones); row 0 (all zeros) is its complete opposite
    assert list(cam.search_cam([1, 1, 1, 1, 1])) == [1, 2, 3]


def test_search_with_max_threshold_matches_every_row_when_none_is_opposite():
    cam = make_cam(hd_threshold=N_COLUMNS - 1)
    # distance to rows 0..3 is 1, 2, 2, 4 - none reaches n_columns
    assert list(cam.search_cam([0, 1, 0, 0, 0])) == [0, 1, 2, 3]


def test_search_with_max_threshold_drops_only_the_complementary_rows():
    cam = make_cam(hd_threshold=N_COLUMNS - 1)
    for address, row in enumerate(ARRAY):
        opposite = [1 - bit for bit in row]
        matches = list(cam.search_cam(opposite))
        assert address not in matches
        expected = [
            other
            for other, other_row in enumerate(ARRAY)
            if hamming(other_row, opposite) < N_COLUMNS
        ]
        assert matches == expected


# --------------------------------------------------------------------------
# search_cam - mid thresholds, checked against a brute-force reference
# --------------------------------------------------------------------------


@pytest.mark.parametrize("hd_threshold", [1, 2, 3])
def test_search_with_mid_threshold_matches_the_rows_within_range(hd_threshold):
    cam = make_cam(hd_threshold=hd_threshold)
    query = [1, 1, 0, 0, 0]  # distances to rows 0..3: 2, 1, 1, 3
    expected = [
        address
        for address, row in enumerate(ARRAY)
        if hamming(row, query) <= hd_threshold
    ]
    assert list(cam.search_cam(query)) == expected


def test_search_boundary_row_is_included_at_its_exact_distance():
    cam = make_cam(hd_threshold=2)
    query = [1, 1, 0, 0, 0]  # row 0 is at distance exactly 2
    assert 0 in cam.search_cam(query)

    cam.set_hd_threshold(1)  # one below that distance
    assert 0 not in cam.search_cam(query)


def test_search_over_random_rows_and_thresholds_matches_brute_force():
    rng = random.Random(20260904)
    n_rows, n_columns = 8, 6
    for _ in range(50):
        rows = [[rng.randint(0, 1) for _ in range(n_columns)] for _ in range(n_rows)]
        query = [rng.randint(0, 1) for _ in range(n_columns)]
        hd_threshold = rng.randint(0, n_columns - 1)

        cam = HdCam(n_rows, n_columns, hd_threshold)
        cam.write_array(rows)

        expected = [
            address
            for address, row in enumerate(rows)
            if hamming(row, query) <= hd_threshold
        ]
        assert list(cam.search_cam(query)) == expected, (
            rows,
            query,
            hd_threshold,
        )


def test_search_returns_row_addresses_in_ascending_order():
    cam = make_cam(hd_threshold=N_COLUMNS - 1)
    matches = cam.search_cam([0, 1, 0, 1, 0])
    assert list(matches) == sorted(matches)


def test_search_on_an_untouched_grid_matches_every_row_for_an_all_zero_query():
    cam = HdCam(N_ROWS, N_COLUMNS, 0)
    assert list(cam.search_cam([0] * N_COLUMNS)) == list(range(N_ROWS))


@pytest.mark.parametrize("bad_length", [N_COLUMNS - 1, N_COLUMNS + 1])
def test_search_rejects_a_query_of_the_wrong_length(bad_length):
    cam = make_cam()
    with pytest.raises(AssertionError):
        cam.search_cam([0] * bad_length)
