"""What every CAM acquisition family shares, and the one decision rule they run.

A CAM acquisition is a codebook of stored replicas, a stream of queries taken
from the record, and a rule that turns hits into PRN results. Only the first two
differ between families. Where the baseline puts a (PRN, CFO, stored phase) in
every row, a code-only design moves the CFO into the query, a segmented design
cuts a row into sub-rows, and a thermometer design widens the word. The rule that
reads those hits - vote across looks, then rank the survivors by distance - is
the same in all of them, and is implemented once here.

Two data paths produce the hits, and they are the one thing a reader would
otherwise assume are two implementations of the rule:

* search_mode="cam" issues a search_cam per query, which is what the hardware
  does, and measures a shortlisted cell with tightest_match.
* search_mode="table" computes every distance at once and reads them off.

They return the same PrnResult list. The table path exists because a search_cam
per query is 300 us and an acquisition issues 36832 of them, so replaying a grid
of thresholds over hundreds of records is only affordable from a table. The
identity it rests on is exact rather than approximate: on +-1 valued vectors of
n_columns entries, the Hamming distance is (n_columns - s.b) / 2, and every
partial sum of that inner product is an integer below 2**24, so a float32 GEMM
carries it without rounding.

The mapping from a (row, query) hit to a hypothesis is four lines, and covers
every family:

    prn         = row.prn[r]
    doppler_bin = row.doppler_bin[r] if row.doppler_bin[r] >= 0 else
                  query.doppler_bin[q]
    code_phase  = (query.start[q] - row.segment_offset[r]) % samples_per_code
    look        = query.start[q] // samples_per_code

A family that cannot express itself in those four lines needs a new field here,
not a second decision rule.
"""

from dataclasses import dataclass
from math import ceil, comb
from statistics import NormalDist
from typing import Iterator, NamedTuple

import numpy as np

from hdcam_gps.acq_base import AcqConfig, GpsL1AcqClassifier, PrnResult
from hdcam_gps.hdcam_packed import PackedHdCam

QUERY_ROTATIONS: int = 4  # Quarter turns, all a one bit sample can express
DEFAULT_FALSE_ALARM_RATE: float = 1e-2  # Expected false detections per acquisition
DEFAULT_VOTE_FRACTION: float = 1 / 3  # Of the looks a hypothesis gets
# Query bits held in one GEMM, sized to keep that matrix near 32 MB.
GEMM_BATCH_BYTES: int = 32 << 20
# Codebook rows reduced at once, so the permuted copy a vote needs stays small.
VOTE_ROW_CHUNK: int = 512


class Cell(NamedTuple):
    """One hypothesis the vote runs over, and the unit a PrnResult is built from."""

    row: int  # Codebook row
    doppler_bin: int  # Index into the configuration's Doppler grid
    code_phase: int  # Code phase in samples, within one code period


@dataclass(frozen=True)
class RowIndex:
    """What each codebook row stands for, as parallel arrays of n_rows entries."""

    prn: np.ndarray  # PRN number of the row
    doppler_bin: np.ndarray  # Doppler bin, or -1 when the query carries it
    segment_offset: np.ndarray  # Samples into the code period the row starts at


@dataclass(frozen=True)
class QueryIndex:
    """What each query stands for, as parallel arrays of n_queries entries."""

    start: np.ndarray  # First sample of the window inside the record
    doppler_bin: np.ndarray  # Doppler bin, or -1 when the row carries it


def quantize_iq(samples: np.ndarray) -> np.ndarray:
    """Quantizes complex samples to one bit per component.

    The two bit planes are concatenated rather than interleaved, so that a
    quarter turn of the carrier is a swap of the two halves.

    Args:
        samples (np.ndarray): The complex samples to quantize.

    Returns:
        np.ndarray: A boolean array of twice the input length, the sign bits of
            I followed by the sign bits of Q.
    """
    return np.concatenate((np.real(samples) > 0, np.imag(samples) > 0))


def rotate_quarter_turns(bits: np.ndarray, rotations: int) -> np.ndarray:
    """Turns a quantized query by whole quadrants, using bit operations only.

    Multiplying by j maps (I, Q) to (-Q, I), which on sign bits is a swap of the
    two planes with the new leading one complemented. Nothing here needs the
    samples the bits came from, so a one bit front end supports it.

    Args:
        bits (np.ndarray): A quantized query, as produced by quantize_iq.
        rotations (int): How many quarter turns to apply.

    Returns:
        np.ndarray: The rotated query, of the same width.
    """
    for _ in range(rotations % QUERY_ROTATIONS):
        in_phase, quadrature = np.split(bits, 2)
        bits = np.concatenate((~quadrature, in_phase))
    return bits


def binomial_tail(n_looks: int, min_votes: int, per_look: float) -> float:
    """The chance of reaching min_votes out of n_looks at a given per look rate.

    Args:
        n_looks (int): How many independent looks a hypothesis gets.
        min_votes (int): How many of them have to match.
        per_look (float): The chance one look matches.

    Returns:
        float: The probability of min_votes or more matches.
    """
    return sum(
        comb(n_looks, votes) * per_look**votes * (1 - per_look) ** (n_looks - votes)
        for votes in range(min_votes, n_looks + 1)
    )


def per_look_false_alarm(
    n_looks: int, min_votes: int, n_cells: int, false_alarm_rate: float
) -> float:
    """The per look rate that a vote rule turns into a wanted overall rate.

    Voting is what lets a single look be loose. A hypothesis only survives by
    matching repeatedly, so the per look rate may be far higher than the rate the
    acquisition as a whole is allowed.

    Args:
        n_looks (int): How many looks a hypothesis gets.
        min_votes (int): How many of them have to match for it to be shortlisted.
        n_cells (int): How many hypotheses are being voted on.
        false_alarm_rate (float): False shortlistings tolerated per acquisition.

    Returns:
        float: The per look match probability to aim the threshold at.
    """
    wanted = false_alarm_rate / n_cells
    low, high = 1e-12, 0.5
    for _ in range(200):  # bisection, the tail is monotone in the per look rate
        middle = (low + high) / 2
        if binomial_tail(n_looks, min_votes, middle) > wanted:
            high = middle
        else:
            low = middle
    return low


def hd_threshold_for_false_alarm(
    n_columns: int, false_alarm_rate: float, n_tests: int = 1
) -> int:
    """The loosest threshold that still rejects every unrelated codebook row.

    Against a row of a different PRN the sign bits agree at chance, so the
    Hamming distance is binomial with n_columns trials and probability one half.
    The threshold is the normal approximation of that distribution's lower tail.

    An acquisition asks that question once per row, per code phase and per
    rotation, so the tail has to be divided by how many times it is asked.
    Without that, a search over a hundred thousand hypotheses turns a per test
    rate of one in a million into several false detections every run.

    Args:
        n_columns (int): The width of a codebook row in bits.
        false_alarm_rate (float): The wanted number of false detections per
            acquisition, spread over all n_tests of them.
        n_tests (int): How many row comparisons one acquisition makes.

    Returns:
        int: The Hamming distance threshold, within [0, n_columns).
    """
    assert 0 < false_alarm_rate < 0.5, "False alarm rate must be in (0, 0.5)."
    assert n_tests >= 1, "An acquisition makes at least one comparison."
    per_test = false_alarm_rate / n_tests
    z = NormalDist().inv_cdf(per_test)  # negative, chance is n_columns / 2
    threshold = int(np.floor(n_columns / 2 + z * np.sqrt(n_columns) / 2))
    return int(np.clip(threshold, 0, n_columns - 1))


def run_starts(group: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sorts queries by the group they vote in, and finds the runs.

    A vote is a reduction of the queries sharing a group. Sorting once and
    reducing each contiguous run is a hundred times faster than np.add.at on the
    unsorted index, which matters because a calibration replays the reduction for
    every candidate setting.

    Args:
        group (np.ndarray): The group index of every query.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: The sorting permutation, the
            first position of each run within it, and the group each run means.
    """
    order = np.argsort(group, kind="stable")
    sorted_group = group[order]
    first = np.flatnonzero(
        np.concatenate(([True], sorted_group[1:] != sorted_group[:-1]))
    )
    return order, first, sorted_group[first]


class CamAcqClassifier(GpsL1AcqClassifier):
    """A codebook, a query stream and the vote-then-rank rule over their hits.

    A subclass supplies build_codebook, row_index, query_index and
    query_variants, and must set n_rows and n_columns before calling this
    __init__, which uses both to size the CAM.
    """

    def __init__(
        self,
        config: AcqConfig,
        hd_threshold: int | None = None,
        min_votes: int | None = None,
        false_alarm_rate: float = DEFAULT_FALSE_ALARM_RATE,
        search_mode: str = "cam",
        cam_factory=PackedHdCam,
    ):
        """Fixes the vote rule and the threshold, then writes the codebook.

        Args:
            config (AcqConfig): The acquisition configuration to search over.
            hd_threshold (int | None): The per look Hamming distance threshold.
                None derives it from the vote rule and false_alarm_rate.
            min_votes (int | None): How many of its looks a hypothesis has to
                match on. None takes a third of them.
            false_alarm_rate (float): False detections tolerated per acquisition,
                used only when hd_threshold is None.
            search_mode (str): "cam" to issue one search_cam per query, "table"
                to read the same hits off a precomputed distance table.
            cam_factory: The HdCam class to build. The packed one answers the
                same question 12 times faster.
        """
        super().__init__(config)
        assert search_mode in ("cam", "table"), 'search_mode is "cam" or "table".'
        self.search_mode = search_mode
        self._row_cache: RowIndex | None = None
        self._query_cache: dict[int, QueryIndex] = {}
        self._doppler_in_query: bool | None = None
        if min_votes is None:
            min_votes = max(1, ceil(DEFAULT_VOTE_FRACTION * self.n_looks))
        assert 1 <= min_votes <= self.n_looks, (
            f"min_votes must be between 1 and the {self.n_looks} looks a code "
            "phase gets."
        )
        self.min_votes = min_votes
        if hd_threshold is None:
            per_look = per_look_false_alarm(
                self.n_looks, self.min_votes, self.n_cells, false_alarm_rate
            )
            hd_threshold = hd_threshold_for_false_alarm(self.n_columns, per_look)
        self._signed: np.ndarray | None = None
        self.cam = cam_factory(self.n_rows, self.n_columns, hd_threshold)
        self.cam.write_array(self.build_codebook())

    # ----------------------------------------------------------------------
    # the hooks a family overrides
    # ----------------------------------------------------------------------

    def build_codebook(self) -> np.ndarray:
        """The stored replicas, as a boolean array of (n_rows, n_columns)."""
        raise NotImplementedError("A family has to supply its codebook.")

    def row_index(self) -> RowIndex:
        """What each codebook row stands for."""
        raise NotImplementedError("A family has to say what its rows mean.")

    def query_index(self, n_samples: int | None = None) -> QueryIndex:
        """What each query stands for.

        Args:
            n_samples (int | None): Length of the record the queries are taken
                from. None takes the configuration's own acquisition length.
        """
        raise NotImplementedError("A family has to say what its queries mean.")

    def query_variants(self, samples: np.ndarray, query: int) -> list[np.ndarray]:
        """Every bit vector one query is searched with.

        A cell counts the query as a match when any variant matches, so the
        variants are the hypotheses a look does not have to choose between. For
        the 1 bit families they are the four quarter turns of the carrier.

        Args:
            samples (np.ndarray): The full input record.
            query (int): An index into the query index.

        Returns:
            list[np.ndarray]: Boolean arrays, each n_columns wide.
        """
        raise NotImplementedError("A family has to build its queries.")

    # ----------------------------------------------------------------------
    # geometry, derived from the two indices
    # ----------------------------------------------------------------------

    def rows(self) -> RowIndex:
        """The row index, built once and kept.

        Returns:
            RowIndex: What each codebook row stands for.
        """
        if self._row_cache is None:
            self._row_cache = self.row_index()
        return self._row_cache

    def queries(self, n_samples: int | None = None) -> QueryIndex:
        """The query index of a record length, built once per length and kept.

        Args:
            n_samples (int | None): Length of the record. None takes the
                configuration's own acquisition length.

        Returns:
            QueryIndex: What each query stands for.
        """
        if n_samples is None:
            n_samples = self.config.samples_per_acquisition
        if n_samples not in self._query_cache:
            self._query_cache[n_samples] = self.query_index(n_samples)
        return self._query_cache[n_samples]

    @property
    def n_queries(self) -> int:
        """Returns how many queries one acquisition issues, variants aside."""
        return len(self.queries().start)

    @property
    def n_doppler_bins(self) -> int:
        """Returns the number of CFO bins of the configuration."""
        return len(self.config.doppler_grid_hz)

    @property
    def n_looks(self) -> int:
        """Returns how many independent looks every code phase is guaranteed.

        A look is one code period. The last period of the record cannot host a
        full window at every code phase, so one code phase in the record gets an
        extra look and the rest get n_codes - 1.
        """
        return max(1, self.config.n_codes - 1)

    @property
    def n_cells(self) -> int:
        """Returns how many hypotheses the vote runs over."""
        cells = self.n_rows * self.config.samples_per_code
        return cells * self.n_doppler_bins if self.doppler_is_in_the_query else cells

    @property
    def doppler_is_in_the_query(self) -> bool:
        """Whether the Doppler bin of a hit comes from the query, not the row."""
        if self._doppler_in_query is None:
            self._doppler_in_query = bool((self.queries().doppler_bin >= 0).any())
        return self._doppler_in_query

    @property
    def hd_threshold(self) -> int:
        """Returns the Hamming distance threshold currently used by the CAM."""
        return self.cam.hd_threshold

    def set_hd_threshold(self, hd_threshold: int):
        """Retunes the CAM without rebuilding the codebook.

        Args:
            hd_threshold (int): The new Hamming distance threshold.
        """
        self.cam.set_hd_threshold(hd_threshold)

    def build_queries(self, samples: np.ndarray) -> Iterator[tuple[int, np.ndarray]]:
        """Every variant of every query, paired with the query it belongs to.

        Args:
            samples (np.ndarray): The full input record.

        Yields:
            tuple[int, np.ndarray]: A query index and one of its bit vectors.
        """
        for query in range(len(self.queries(len(samples)).start)):
            for bits in self.query_variants(samples, query):
                yield query, bits

    # ----------------------------------------------------------------------
    # the two data paths
    # ----------------------------------------------------------------------

    def signed_codebook(self) -> np.ndarray:
        """The codebook as +-1 float32, which is what the GEMM multiplies.

        Built once from the grid the constructor wrote. Rewriting cam.grid
        afterwards leaves this stale, so nothing does.

        Returns:
            np.ndarray: An array of shape (n_rows, n_columns).
        """
        if self._signed is None:
            self._signed = np.where(self.cam.grid, 1.0, -1.0).astype(np.float32)
        return self._signed

    def row_distances(self, bits: np.ndarray) -> np.ndarray:
        """The Hamming distance from one query to every codebook row.

        Args:
            bits (np.ndarray): A query, n_columns bits wide.

        Returns:
            np.ndarray: One distance per row.
        """
        return self._distances(np.asarray(bits, dtype=bool)[None, :])[0]

    def _distances(self, bits: np.ndarray) -> np.ndarray:
        """Distances of a batch of queries, through one GEMM.

        Args:
            bits (np.ndarray): Queries of shape (n_batch, n_columns).

        Returns:
            np.ndarray: Distances of shape (n_batch, n_rows), as int32.
        """
        signed = np.where(bits, np.float32(1.0), np.float32(-1.0))
        dots = signed @ self.signed_codebook().T
        # Every dot product is an even integer below 2**24, so this is exact.
        return ((self.n_columns - dots) * 0.5).astype(np.int32)

    def distance_table(self, samples: np.ndarray) -> np.ndarray:
        """Every Hamming distance the first pass would ever compare against.

        The value kept per query is the best over its variants, which is what a
        look actually contributes.

        Args:
            samples (np.ndarray): Input samples, samples_per_acquisition long.

        Returns:
            np.ndarray: Distances of shape (n_queries, n_rows).
        """
        n_queries = len(self.queries(len(samples)).start)
        table = np.empty((n_queries, self.n_rows), dtype=np.int32)
        n_variants = max(1, len(self.query_variants(samples, 0)))
        batch = max(1, GEMM_BATCH_BYTES // (4 * self.n_columns * n_variants))
        for low in range(0, n_queries, batch):
            high = min(low + batch, n_queries)
            variants = [self.query_variants(samples, query) for query in range(low, high)]
            counts = np.array([len(group) for group in variants])
            flat = np.array([bits for group in variants for bits in group])
            first = np.concatenate(([0], np.cumsum(counts)[:-1]))
            table[low:high] = np.minimum.reduceat(self._distances(flat), first, axis=0)
        return table

    def matched_table(self, samples: np.ndarray) -> np.ndarray:
        """Which rows each query hits, asked one search_cam at a time.

        This is the CAM path of the same hits distance_table holds, and it is
        what the hardware can answer: a row is inside the threshold or it is not.

        Args:
            samples (np.ndarray): Input samples, samples_per_acquisition long.

        Returns:
            np.ndarray: A boolean array of shape (n_queries, n_rows).
        """
        n_queries = len(self.queries(len(samples)).start)
        matched = np.zeros((n_queries, self.n_rows), dtype=bool)
        for query, bits in self.build_queries(samples):
            matched[query, self.cam.search_cam(bits)] = True
        return matched

    def tightest_match(self, row: int, query: np.ndarray) -> int:
        """The smallest threshold at which a row still matches a query.

        This is the row's Hamming distance to the query, obtained through the
        only question the CAM answers, by bisecting on its threshold. The CAM's
        own threshold is left as it was found.

        Args:
            row (int): The codebook row to measure.
            query (np.ndarray): A quantized query, n_columns bits wide.

        Returns:
            int: The distance, within [0, n_columns).
        """
        original = self.cam.hd_threshold
        low, high = 0, self.n_columns - 1
        while low < high:
            middle = (low + high) // 2
            self.cam.set_hd_threshold(middle)
            if row in self.cam.search_cam(query):
                high = middle
            else:
                low = middle + 1
        self.cam.set_hd_threshold(original)
        return low

    # ----------------------------------------------------------------------
    # the one decision rule
    # ----------------------------------------------------------------------

    def _vote_groups(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """The rows and the per query group index of each vote reduction.

        Rows that start at different points of the code period read a different
        code phase out of the same query, so they cannot share one reduction.
        Every family except the segmented one has a single offset and therefore a
        single group.

        Yields:
            tuple[np.ndarray, np.ndarray]: The rows of this reduction, and the
                group every query votes in.
        """
        rows = self.rows()
        queries = self.queries()
        samples_per_code = self.config.samples_per_code
        for offset in np.unique(rows.segment_offset):
            which = np.flatnonzero(rows.segment_offset == offset)
            code_phase = (queries.start - int(offset)) % samples_per_code
            if self.doppler_is_in_the_query:
                assert (rows.doppler_bin[which] < 0).all(), (
                    "A Doppler bin comes from the row or from the query, never "
                    "from both."
                )
                yield which, queries.doppler_bin * samples_per_code + code_phase
            else:
                yield which, code_phase

    def _cell_of(self, row: int, label: int) -> Cell:
        """The hypothesis a surviving (row, group) pair stands for.

        Args:
            row (int): The codebook row.
            label (int): The group index the vote was counted in.

        Returns:
            Cell: The row, its Doppler bin and its code phase.
        """
        samples_per_code = self.config.samples_per_code
        code_phase = label % samples_per_code
        if self.doppler_is_in_the_query:
            return Cell(row, label // samples_per_code, code_phase)
        return Cell(row, int(self.rows().doppler_bin[row]), code_phase)

    def cells_from_table(
        self, table: np.ndarray, hd_threshold: int, min_votes: int
    ) -> dict[Cell, int]:
        """Every cell reaching min_votes, with the distance of its best look.

        A look that did not match is further away than the threshold, so the
        minimum over all looks of a surviving cell is its minimum over the looks
        that did match. The table path can therefore ignore which ones they were.

        Args:
            table (np.ndarray): Distances of shape (n_queries, n_rows).
            hd_threshold (int): The per look threshold to apply.
            min_votes (int): How many looks have to match.

        Returns:
            dict[Cell, int]: The best distance of each surviving cell.
        """
        found: dict[Cell, int] = {}
        for which, group in self._vote_groups():
            order, first, labels = run_starts(group)
            for start in range(0, which.size, VOTE_ROW_CHUNK):
                chunk = which[start : start + VOTE_ROW_CHUNK]
                block = table[:, chunk][order]
                votes = np.add.reduceat(block <= hd_threshold, first, axis=0)
                best = np.minimum.reduceat(block, first, axis=0)
                for run, position in zip(*np.nonzero(votes >= min_votes)):
                    cell = self._cell_of(int(chunk[position]), int(labels[run]))
                    distance = int(best[run, position])
                    if distance < found.get(cell, self.n_columns + 1):
                        found[cell] = distance
        return found

    def shortlist_cells(
        self, matched: np.ndarray, min_votes: int
    ) -> dict[Cell, list[int]]:
        """Every cell reaching min_votes, with the queries that put it there.

        Args:
            matched (np.ndarray): Hits of shape (n_queries, n_rows).
            min_votes (int): How many looks have to match.

        Returns:
            dict[Cell, list[int]]: The matching query indices of each cell.
        """
        found: dict[Cell, list[int]] = {}
        for which, group in self._vote_groups():
            order, first, labels = run_starts(group)
            bounds = np.append(first, len(order))
            for start in range(0, which.size, VOTE_ROW_CHUNK):
                chunk = which[start : start + VOTE_ROW_CHUNK]
                block = matched[:, chunk][order]
                votes = np.add.reduceat(block, first, axis=0)
                for run, position in zip(*np.nonzero(votes >= min_votes)):
                    row = int(chunk[position])
                    members = order[bounds[run] : bounds[run + 1]]
                    hits = [int(q) for q in members if matched[q, row]]
                    cell = self._cell_of(row, int(labels[run]))
                    found.setdefault(cell, []).extend(hits)
        return found

    def rank_cells(self, distances: dict[Cell, int]) -> list[PrnResult]:
        """Keeps the closest surviving cell of each PRN.

        Args:
            distances (dict[Cell, int]): Hamming distance per surviving cell.

        Returns:
            list[PrnResult]: The winners, in the configuration's PRN order.
        """
        prn_of_row = self.rows().prn
        grid = self.config.doppler_grid_hz
        results = []
        for prn in self.config.prn_list:
            block = [
                (distance, cell)
                for cell, distance in distances.items()
                if prn_of_row[cell.row] == prn
            ]
            if not block:
                continue
            _, cell = min(block)
            results.append(
                PrnResult(
                    prn=prn,
                    doppler_hz=float(grid[cell.doppler_bin]),
                    code_phase=int(cell.code_phase),
                )
            )
        return results

    def _best_per_prn(self, distances: dict[tuple[int, int], int]) -> list[PrnResult]:
        """rank_cells for a family whose rows carry their own Doppler bin.

        Args:
            distances (dict[tuple[int, int], int]): Hamming distance, keyed by
                codebook row and code phase.

        Returns:
            list[PrnResult]: The winners, in the configuration's PRN order.
        """
        doppler_of_row = self.rows().doppler_bin
        return self.rank_cells(
            {
                Cell(row, int(doppler_of_row[row]), code_phase): distance
                for (row, code_phase), distance in distances.items()
            }
        )

    def decide(
        self, table: np.ndarray, hd_threshold: int, min_votes: int
    ) -> list[PrnResult]:
        """The whole decision, read off a distance table.

        Args:
            table (np.ndarray): Distances of shape (n_queries, n_rows).
            hd_threshold (int): The per look threshold to apply.
            min_votes (int): How many looks have to match.

        Returns:
            list[PrnResult]: What the classifier would have returned.
        """
        return self.rank_cells(self.cells_from_table(table, hd_threshold, min_votes))

    def _acquire(self, samples: np.ndarray) -> list[PrnResult]:
        """Shortlists hypotheses, then ranks them by distance.

        Args:
            samples (np.ndarray): Input samples for acquisition.

        Returns:
            list[PrnResult]: One result per acquired PRN, in configuration order.
        """
        if self.search_mode == "table":
            table = self.distance_table(samples)
            return self.decide(table, self.hd_threshold, self.min_votes)
        cells = self.shortlist_cells(self.matched_table(samples), self.min_votes)
        return self.rank_shortlist(samples, cells)

    def rank_shortlist(
        self, samples: np.ndarray, cells: dict[Cell, list[int]]
    ) -> list[PrnResult]:
        """The second pass of the CAM path, measuring each survivor by bisection.

        A cell is measured on every rotation of every look that matched and kept
        at its best, so the ranking sees it at its strongest. A rotation that did
        not match is further away than the threshold, so including it changes
        nothing and saves having to record which ones did.

        Args:
            samples (np.ndarray): Input samples for acquisition.
            cells (dict[Cell, list[int]]): The shortlist and its queries.

        Returns:
            list[PrnResult]: One result per acquired PRN, in configuration order.
        """
        distances = {
            cell: min(
                self.tightest_match(cell.row, bits)
                for query in queries
                for bits in self.query_variants(samples, query)
            )
            for cell, queries in cells.items()
        }
        return self.rank_cells(distances)

    # ----------------------------------------------------------------------
    # what the study measures about the family
    # ----------------------------------------------------------------------

    def chance_floor(self, n_draws: int = 512, seed: int = 0) -> tuple[float, float]:
        """The distance an unrelated row sits at, measured rather than assumed.

        The binomial model with p = 0.5 is right for a 1 bit codebook and wrong
        for a thermometer one, where the unary code makes the bits of a word
        dependent. So the floor is drawn from the family's own queries on noise.

        Args:
            n_draws (int): How many query windows to measure.
            seed (int): Seed of the noise the queries are taken from.

        Returns:
            tuple[float, float]: The mean distance and its standard deviation.
        """
        rng = np.random.default_rng(seed)
        n_samples = self.config.samples_per_acquisition
        noise = rng.normal(scale=np.sqrt(0.5), size=n_samples) + 1j * rng.normal(
            scale=np.sqrt(0.5), size=n_samples
        )
        n_queries = len(self.queries(n_samples).start)
        picks = rng.choice(n_queries, size=min(n_draws, n_queries), replace=False)
        measured = np.concatenate(
            [
                self.row_distances(bits)
                for query in picks
                for bits in self.query_variants(noise, int(query))
            ]
        )
        return float(measured.mean()), float(measured.std())

    def cost(self, samples: np.ndarray | None = None, seed: int = 0):
        """The CAM cost of one acquisition, counted rather than modelled.

        Args:
            samples (np.ndarray | None): The record to count on. None uses a
                noise record, which is the cheapest case.
            seed (int): Seed of that noise record.

        Returns:
            CamCost: The area, energy and latency of one acquisition.
        """
        from hdcam_gps.cam_cost import measure_cost  # cam_cost imports this module

        return measure_cost(self, samples=samples, seed=seed)
