"""1 bit HdCam based GPS L1 C/A acquisition.

The codebook written into the HdCam holds the replicas the acquisition is
searching for. A row is one C/A code period of a (PRN, CFO) hypothesis,
quantized to one bit per component: the sign of I followed by the sign of Q,
which is the carrier phase of every sample rounded to a quadrant.

The two dimensions the codebook cannot span on its own are handled around the
search:

* Code phase, by sliding the query window over the record. The row index of a
  hit gives the PRN and the CFO, the window offset gives the code phase. The
  windows are contiguous, never wrapped: only the code repeats every period, the
  Doppler carrier does not, so the record must hold at least two code periods
  for every code phase to be reachable.

* Carrier phase, which a Hamming distance cannot ignore. A codeword and the same
  codeword received in antiphase sit at opposite ends of the distance scale, and
  the CAM only fires on small distances, so the circle has to be covered
  explicitly. This is split across the two sides of the comparison:

    - The query contributes the four quarter turns, which are all a one bit
      sample can express. They are bit operations, not arithmetic: a quarter
      turn maps (I, Q) to (-Q, I), so it is a swap of the two bit planes with
      one of them complemented.
    - The codebook contributes the phases in between, because it is built
      offline in full precision and can be quantized at any starting phase.

  Four query rotations alone are not enough. Their worst case residual phase of
  45 degrees costs n_columns / 4 in Hamming distance, which is the same order as
  a wrong CFO bin of the same PRN, so the two become indistinguishable. Two
  codebook phases, for eight effective hypotheses, restore the separation.

The decision is made in two stages, because a single search cannot make it. The
CAM answers which rows fall inside a threshold, not how far away they are, so one
search at a tight threshold sees nothing once there is noise, and one search at a
loose threshold sees the right row alongside the neighbouring CFO bins of the same
PRN and cannot tell them apart.

* The first pass votes. Every code period is a fresh look at the same (row, code
  phase) hypothesis, and a hypothesis is shortlisted when it matches on at least
  min_votes of them. That vote count is the non-coherent accumulation: a single
  look only has to be suggestive, not conclusive, so the per look threshold sits
  much closer to chance than a one shot decision could afford, and the evidence
  is combined across the record instead of being thrown away. Carrier phase is
  free inside a look, since a period counts if any of the four rotations matched.
* The second pass ranks the survivors by tightest_match, the smallest threshold a
  row still matches at, found by bisection. That is the Hamming distance read
  through the interface the CAM actually offers, and it is what separates the
  true hypothesis from a neighbouring CFO bin.

The shortlist is small, so the second stage costs almost nothing.
"""

from math import ceil, comb
from statistics import NormalDist

import numpy as np

from hdcam_gps.acq_base import AcqConfig, GpsL1AcqClassifier, PrnResult
from hdcam_gps.ca_code import sampled_ca_code
from hdcam_gps.hdcam import HdCam

QUERY_ROTATIONS: int = 4  # Quarter turns, all a one bit sample can express
DEFAULT_CODEBOOK_PHASES: int = 2  # Extra phases carried by the codebook
DEFAULT_FALSE_ALARM_RATE: float = 1e-2  # Expected false detections per acquisition
DEFAULT_VOTE_FRACTION: float = 1 / 3  # Of the looks a hypothesis gets


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


class OneBitHdCamClassifier(GpsL1AcqClassifier):
    """Acquisition by a single bit codebook lookup in an HdCam."""

    def __init__(
        self,
        config: AcqConfig,
        hd_threshold: int | None = None,
        n_codebook_phases: int = DEFAULT_CODEBOOK_PHASES,
        min_votes: int | None = None,
        false_alarm_rate: float = DEFAULT_FALSE_ALARM_RATE,
    ):
        """Builds the codebook and writes it into a fresh HdCam.

        Args:
            config (AcqConfig): The acquisition configuration. Its PRN list and
                Doppler grid are the two hypothesis dimensions of the codebook.
            hd_threshold (int | None): The per look Hamming distance threshold.
                When left as None it follows from the vote rule: the per look
                rate that turns min_votes out of n_looks into false_alarm_rate,
                converted into a distance through the chance floor.
            n_codebook_phases (int): Carrier phase offsets stored per (PRN, CFO),
                on top of the four quarter turns of the query. One means quarter
                turns only, which does not separate neighbouring CFO bins.
            min_votes (int | None): How many of its looks a hypothesis has to
                match on. None takes a third of them, which is what lets the per
                look threshold sit close to chance.
            false_alarm_rate (float): False detections tolerated per acquisition,
                used only when hd_threshold is None. Lower it to trade
                sensitivity for a cleaner answer.
        """
        super().__init__(config)
        assert n_codebook_phases >= 1, "At least one codebook phase is needed."
        assert config.n_codes >= 2, (
            "At least two code periods are needed, so that a full contiguous "
            "window is available at every code phase."
        )
        self.n_codebook_phases = n_codebook_phases
        self.n_columns = 2 * config.samples_per_code
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
        self.cam = HdCam(self.n_rows, self.n_columns, hd_threshold)
        self.cam.write_array(self.build_codebook())

    @property
    def n_phases(self) -> int:
        """Returns the carrier phase hypotheses covering the circle."""
        return QUERY_ROTATIONS * self.n_codebook_phases

    @property
    def n_doppler_bins(self) -> int:
        """Returns the number of CFO bins of the configuration."""
        return len(self.config.doppler_grid_hz)

    @property
    def n_looks(self) -> int:
        """Returns how many independent looks one code phase gets.

        A look is one code period, and the last period of the record cannot host
        a full window at every code phase, so it is one short of n_codes.
        """
        return max(1, self.config.n_codes - 1)

    @property
    def n_cells(self) -> int:
        """Returns how many (row, code phase) hypotheses the vote runs over."""
        return self.n_rows * self.config.samples_per_code

    @property
    def n_rows(self) -> int:
        """Returns the number of codebook rows, one per PRN, CFO and phase."""
        return len(self.config.prn_list) * self.n_doppler_bins * self.n_codebook_phases

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

    def row_of(self, prn: int, doppler_bin: int, codebook_phase: int = 0) -> int:
        """Returns the codebook row of one hypothesis.

        Args:
            prn (int): A PRN number from the configuration's PRN list.
            doppler_bin (int): An index into the configuration's Doppler grid.
            codebook_phase (int): Which stored carrier phase offset.

        Returns:
            int: The row index in the CAM.
        """
        prn_position = self.config.prn_list.index(prn)
        hypothesis = prn_position * self.n_doppler_bins + doppler_bin
        return hypothesis * self.n_codebook_phases + codebook_phase

    def hypothesis_of(self, row: int) -> tuple[int, float]:
        """Returns the (PRN, Doppler) hypothesis a codebook row stands for.

        Args:
            row (int): The row index in the CAM.

        Returns:
            tuple[int, float]: The PRN number and the Doppler frequency in Hz.
        """
        hypothesis = row // self.n_codebook_phases
        prn = self.config.prn_list[hypothesis // self.n_doppler_bins]
        doppler_bin = hypothesis % self.n_doppler_bins
        return prn, float(self.config.doppler_grid_hz[doppler_bin])

    def build_codebook(self) -> np.ndarray:
        """Builds the quantized replica of every stored hypothesis.

        The stored phase offsets subdivide a quadrant, since the query already
        supplies the quarter turns. They sit at the middle of their share rather
        than at its edge, so no replica ever lands on an axis: a replica at
        exactly zero phase and zero Doppler would be purely real, and its whole
        quadrature bit plane would be the sign of zero, carrying nothing and
        matching anything.

        Returns:
            np.ndarray: A boolean array of shape (n_rows, 2 * samples_per_code).
        """
        config = self.config
        time_s = np.arange(config.samples_per_code) / config.fs_hz
        codebook = np.empty((self.n_rows, self.n_columns), dtype=bool)
        for prn in config.prn_list:
            code = sampled_ca_code(prn, config.fs_hz)
            for doppler_bin, doppler_hz in enumerate(config.doppler_grid_hz):
                carrier = np.exp(2j * np.pi * doppler_hz * time_s)
                for phase in range(self.n_codebook_phases):
                    offset = np.exp(2j * np.pi * (phase + 0.5) / self.n_phases)
                    row = self.row_of(prn, doppler_bin, phase)
                    codebook[row] = quantize_iq(code * carrier * offset)
        return codebook

    def query_window(self, samples: np.ndarray, start: int) -> np.ndarray:
        """Takes one contiguous code period of samples out of the record.

        Args:
            samples (np.ndarray): The full input record.
            start (int): The first sample of the window.

        Returns:
            np.ndarray: A view of samples_per_code complex samples.
        """
        end = start + self.config.samples_per_code
        assert end <= len(samples), "The query window must fit inside the record."
        return samples[start:end]

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

    def shortlist(
        self, samples: np.ndarray
    ) -> dict[tuple[int, int], list[np.ndarray]]:
        """Collects every hypothesis that falls inside the threshold anywhere.

        A hypothesis is kept when it matched on at least min_votes of the looks
        its code phase gets, which is the non-coherent accumulation.

        Args:
            samples (np.ndarray): Input samples for acquisition.

        Returns:
            dict[tuple[int, int], list[np.ndarray]]: Every query that matched,
                keyed by the codebook row and the code phase it matched at.
        """
        config = self.config
        queries: dict[tuple[int, int], list[np.ndarray]] = {}
        n_starts = config.samples_per_acquisition - config.samples_per_code + 1
        for start in range(n_starts):
            query = quantize_iq(self.query_window(samples, start))
            code_phase = start % config.samples_per_code
            # Carrier phase is free within a look: the period counts once if any
            # rotation matched, so a period is one vote rather than up to four.
            matched: dict[int, np.ndarray] = {}
            for rotation in range(QUERY_ROTATIONS):
                rotated = rotate_quarter_turns(query, rotation)
                for row in self.cam.search_cam(rotated):
                    matched.setdefault(int(row), rotated)
            for row, rotated in matched.items():
                queries.setdefault((row, code_phase), []).append(rotated)
        # The number of queries kept for a cell is the number of looks that
        # matched, so the vote count is just how many were collected.
        return {
            key: matches
            for key, matches in queries.items()
            if len(matches) >= self.min_votes
        }

    def _acquire(self, samples: np.ndarray) -> list[PrnResult]:
        """Shortlists hypotheses with the CAM, then ranks them by distance.

        A shortlisted hypothesis is measured on every look that matched, and kept
        at its best, so the ranking sees a cell at its strongest.

        Args:
            samples (np.ndarray): Input samples for acquisition.

        Returns:
            list[PrnResult]: One result per acquired PRN, in configuration order.
        """
        candidates = self.shortlist(samples)
        distances = {
            key: min(self.tightest_match(key[0], query) for query in matches)
            for key, matches in candidates.items()
        }
        return self._best_per_prn(distances)

    def _best_per_prn(
        self, distances: dict[tuple[int, int], int]
    ) -> list[PrnResult]:
        """Keeps the closest surviving hypothesis of each PRN.

        Args:
            distances (dict[tuple[int, int], int]): Hamming distance, keyed by
                codebook row and code phase.

        Returns:
            list[PrnResult]: The winners, in the configuration's PRN order.
        """
        rows_per_prn = self.n_doppler_bins * self.n_codebook_phases
        results = []
        for prn_position, prn in enumerate(self.config.prn_list):
            first = prn_position * rows_per_prn
            block = [
                (distance, key)
                for key, distance in distances.items()
                if first <= key[0] < first + rows_per_prn
            ]
            if not block:
                continue
            _, (row, code_phase) = min(block)
            doppler_bin = (row - first) // self.n_codebook_phases
            results.append(
                PrnResult(
                    prn=prn,
                    doppler_hz=float(self.config.doppler_grid_hz[doppler_bin]),
                    code_phase=int(code_phase),
                )
            )
        return results

    def distance_table(self, samples: np.ndarray) -> np.ndarray:
        """Every Hamming distance the first pass would ever compare against.

        This is a calibration tool, not part of acquisition. It computes in one
        sweep what the search does one lookup at a time, so that a threshold and
        a vote rule can be scored over many records without re-running the CAM
        for each candidate. The value kept per look is the best over the four
        rotations, which is what a look actually contributes.

        Args:
            samples (np.ndarray): Input samples, samples_per_acquisition long.

        Returns:
            np.ndarray: Distances of shape (n_starts, n_rows).
        """
        config = self.config
        n_starts = config.samples_per_acquisition - config.samples_per_code + 1
        table = np.empty((n_starts, self.n_rows), dtype=np.int32)
        grid = self.cam.grid
        for start in range(n_starts):
            query = quantize_iq(self.query_window(samples, start))
            best = None
            for rotation in range(QUERY_ROTATIONS):
                rotated = rotate_quarter_turns(query, rotation)
                distances = np.count_nonzero(grid != rotated, axis=1)
                best = distances if best is None else np.minimum(best, distances)
            table[start] = best
        return table
