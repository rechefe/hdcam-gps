"""Directed tests for the bit packed HdCam.

The packed CAM exists only to answer faster, so every test here compares it
against the plain HdCam rather than against a hand computed answer. The widths
are chosen to straddle a byte boundary, which is where packing can go wrong.
"""

import numpy as np
import pytest

from hdcam_gps.hdcam import HdCam
from hdcam_gps.hdcam_packed import PackedHdCam

N_ROWS = 4
N_COLUMNS = 5

# The same hand-checkable array tests/test_hdcam.py uses.
ARRAY = [
    [0, 0, 0, 0, 0],  # row 0 - all zeros
    [1, 0, 0, 0, 0],  # row 1 - distance 1 from row 0
    [1, 1, 1, 0, 0],  # row 2 - distance 3 from row 0
    [1, 1, 1, 1, 1],  # row 3 - all ones, distance 5 from row 0
]


def make_cam(hd_threshold=0) -> PackedHdCam:
    cam = PackedHdCam(N_ROWS, N_COLUMNS, hd_threshold)
    cam.write_array(ARRAY)
    return cam


def random_pair(n_rows: int, n_columns: int, hd_threshold: int, seed: int):
    """The same random grid written into a plain CAM and a packed one."""
    rng = np.random.default_rng(seed)
    grid = rng.random((n_rows, n_columns)) > 0.5
    plain = HdCam(n_rows, n_columns, hd_threshold)
    packed = PackedHdCam(n_rows, n_columns, hd_threshold)
    plain.write_array(grid)
    packed.write_array(grid)
    return plain, packed, rng


# --------------------------------------------------------------------------
# it is an HdCam, with the same state
# --------------------------------------------------------------------------


def test_a_packed_cam_is_an_hdcam():
    assert isinstance(make_cam(), HdCam)


def test_the_boolean_grid_is_still_there():
    # Callers read cam.grid directly, so packing has to be an addition to the
    # model rather than a replacement for it.
    cam = make_cam()
    assert np.array_equal(cam.grid, np.array(ARRAY, dtype=bool))


def test_an_empty_packed_cam_holds_zeros():
    cam = PackedHdCam(N_ROWS, N_COLUMNS, 0)
    assert not cam.grid.any()
    assert cam.search_cam(np.zeros(N_COLUMNS, dtype=bool)).tolist() == [0, 1, 2, 3]


def test_the_threshold_asserts_are_inherited():
    with pytest.raises(AssertionError):
        PackedHdCam(N_ROWS, N_COLUMNS, N_COLUMNS)


def test_a_query_of_the_wrong_width_is_rejected():
    with pytest.raises(AssertionError):
        make_cam().search_cam(np.zeros(N_COLUMNS + 1, dtype=bool))


# --------------------------------------------------------------------------
# search_cam - the same answer as the plain CAM
# --------------------------------------------------------------------------


@pytest.mark.parametrize("hd_threshold", range(N_COLUMNS))
def test_the_two_cams_agree_on_the_hand_checkable_array(hd_threshold):
    plain = HdCam(N_ROWS, N_COLUMNS, hd_threshold)
    plain.write_array(ARRAY)
    packed = make_cam(hd_threshold)
    for query in ARRAY + [[1, 0, 1, 0, 1], [0, 1, 0, 1, 0]]:
        assert np.array_equal(plain.search_cam(query), packed.search_cam(query))


# Widths either side of a byte, so the zero padding of the last byte is exercised.
@pytest.mark.parametrize("n_columns", [1, 7, 8, 9, 15, 16, 17, 64, 65, 408])
def test_the_two_cams_agree_at_any_row_width(n_columns):
    plain, packed, rng = random_pair(6, n_columns, n_columns // 2, seed=n_columns)
    for _ in range(5):
        query = rng.random(n_columns) > 0.5
        assert np.array_equal(plain.search_cam(query), packed.search_cam(query))


@pytest.mark.parametrize("seed", range(8))
def test_the_two_cams_agree_on_random_grids_and_thresholds(seed):
    rng = np.random.default_rng(seed)
    n_rows, n_columns = int(rng.integers(1, 40)), int(rng.integers(1, 70))
    hd_threshold = int(rng.integers(0, n_columns))
    plain, packed, rng = random_pair(n_rows, n_columns, hd_threshold, seed)
    for _ in range(5):
        query = rng.random(n_columns) > 0.5
        assert np.array_equal(plain.search_cam(query), packed.search_cam(query))


def test_the_padding_bits_never_count():
    # 5 columns pack into one byte with 3 bits of padding. A row of all ones and
    # a query of all ones are 0 apart, not 3, so the padding is not compared.
    cam = PackedHdCam(1, 5, 0)
    cam.write_array([[1, 1, 1, 1, 1]])
    assert cam.search_cam(np.ones(5, dtype=bool)).tolist() == [0]


# --------------------------------------------------------------------------
# writes keep the packed copy in step
# --------------------------------------------------------------------------


def test_write_row_updates_what_a_search_sees():
    cam = make_cam(hd_threshold=0)
    cam.write_row(0, [1, 1, 1, 1, 1])
    assert cam.search_cam(np.ones(5, dtype=bool)).tolist() == [0, 3]


def test_write_array_updates_what_a_search_sees():
    cam = make_cam(hd_threshold=0)
    cam.write_array([[1, 1, 1, 1, 1]] * N_ROWS)
    assert cam.search_cam(np.ones(5, dtype=bool)).tolist() == [0, 1, 2, 3]


def test_set_hd_threshold_needs_no_repacking():
    cam = make_cam(hd_threshold=0)
    cam.set_hd_threshold(3)
    assert cam.search_cam(np.zeros(5, dtype=bool)).tolist() == [0, 1, 2]
