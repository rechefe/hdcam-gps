import numpy as np


class HdCam:
    """HdCam object - used to simulate HdCam behavior
    allows writing/initializing the memory

    """

    def __init__(self, n_rows: int, n_columns: int, hd_threshold: int):
        """Initializes and HdCam module with an empty grid of the given size.
        Args:
            n_rows (int): Amount of rows in the grid.
            n_columns (int): Amount of columns in the grid.
            hd_threshold (int): The hamming distance threshold for the HdCam.
        """
        self.n_rows = n_rows
        self.n_columns = n_columns
        assert (
            0 <= hd_threshold < n_columns
        ), "Hd threshold must be within the number of columns."
        self.hd_threshold = hd_threshold
        self.grid = np.zeros((n_rows, n_columns), dtype=bool)

    def write_row(self, address: int, data):
        """Writes single row of the HdCam array

        Args:
            address (int): The row to write to.
            data (list[int]): The data to write.
        """
        assert 0 <= address < self.n_rows, "Address must be within the number of rows."
        assert (
            len(data) == self.n_columns
        ), "Data length must match the number of columns."
        self.grid[address] = np.array(data, dtype=bool)

    def write_array(self, data):
        """Writes the entire HdCam array

        Args:
            data (list[list[int]]): The data to write.
        """
        assert len(data) == self.n_rows, "Data length must match the number of rows."
        for row in data:
            assert (
                len(row) == self.n_columns
            ), "Each row's length must match the number of columns."
        self.grid = np.array(data, dtype=bool)

    def set_hd_threshold(self, hd_threshold: int):
        """Sets the hamming distance threshold for the HdCam.

        Args:
            hd_threshold (int): The new hamming distance threshold.
        """
        assert (
            0 <= hd_threshold < self.n_columns
        ), "Hd threshold must be within the number of columns."
        self.hd_threshold = hd_threshold

    def search_cam(self, query):
        """Searches the HdCam for a given query.

        Args:
            query (list[int]): The query to search for.
        """
        assert (
            len(query) == self.n_columns
        ), "Query length must match the number of columns."
        query_array = np.array(query, dtype=bool)
        hamming_distances = np.sum(self.grid != query_array, axis=1)
        return np.where(hamming_distances <= self.hd_threshold)[0]