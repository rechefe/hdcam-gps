"""A bit packed HdCam, for simulations that are otherwise too slow to run.

HdCam.search_cam compares a query against a boolean grid one byte per bit, which
costs 4.0 ms for the 1344 by 2046 codebook of a full 32 PRN search. An
acquisition issues 36832 of those, so a single record takes 150 seconds and a
study over hundreds of records is a week of compute.

Packing the same grid eight bits to the byte and counting set bits with
np.bitwise_count answers the identical question in 0.30 ms. Nothing about the
model changes: the rows, the threshold and the returned row indices are the same,
and tests/test_hdcam_packed.py asserts the two agree on random grids.

np.packbits zero pads the last byte of every row, and the query is padded the
same way, so the padding never contributes to a distance.
"""

import numpy as np

from hdcam_gps.hdcam import HdCam


class PackedHdCam(HdCam):
    """An HdCam answering search_cam from a bit packed copy of its grid."""

    def __init__(self, n_rows: int, n_columns: int, hd_threshold: int):
        """Builds the CAM and the packed copy its searches read.

        Args:
            n_rows (int): Amount of rows in the grid.
            n_columns (int): Amount of columns in the grid.
            hd_threshold (int): The hamming distance threshold for the HdCam.
        """
        super().__init__(n_rows, n_columns, hd_threshold)
        self._repack()

    def _repack(self):
        """Refreshes the packed copy from the boolean grid."""
        self._packed = np.packbits(self.grid, axis=1)

    def write_row(self, address: int, data):
        """Writes a single row, and repacks it.

        Args:
            address (int): The row to write to.
            data (list[int]): The data to write.
        """
        super().write_row(address, data)
        self._packed[address] = np.packbits(self.grid[address])

    def write_array(self, data):
        """Writes the entire array, and repacks all of it.

        Args:
            data (list[list[int]]): The data to write.
        """
        super().write_array(data)
        self._repack()

    def search_cam(self, query):
        """Returns every row within the threshold, from the packed grid.

        Args:
            query (list[int]): The query to search for.

        Returns:
            np.ndarray: The indices of the matching rows.
        """
        assert (
            len(query) == self.n_columns
        ), "Query length must match the number of columns."
        packed_query = np.packbits(np.asarray(query, dtype=bool))
        differing = np.bitwise_xor(self._packed, packed_query)
        distances = np.bitwise_count(differing).sum(axis=1, dtype=np.int64)
        return np.where(distances <= self.hd_threshold)[0]
