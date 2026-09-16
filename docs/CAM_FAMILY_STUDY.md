# Exploring the CAM algorithm space for GPS L1 C/A acquisition

## Context

`hdcam_acq.py` implements exactly one way to use an HD-CAM for acquisition: one row per
(PRN, CFO, stored phase), a full code period wide, queried with four quarter turns. That
design is not the only one, and `PROPOSAL.md` risk 1 says the decisive question is one the
repo never measures — **what fraction of the word width the HD tolerance has to be**. At
44-47% of 2046 bits, against 12.5% demonstrated in silicon, the current design may simply
be unbuildable, and nobody knows whether a different arrangement of the same CAM does
better.

So: implement four more CAM acquisition designs beside the existing one, measure all five
on gps-sdr-sim records at a matched false alarm rate, and rank them on sensitivity **and**
CAM cost together. A family that dies is a result.

---

## 0. First, make the study affordable

Measured on this machine at the full config (1344 x 2046, 9208 starts):

| | now | after |
|---|---|---|
| whole distance table | ~147 s | **0.9 s** (one BLAS GEMM per rotation) |
| one `search_cam` | 3.98 ms | **0.30 ms** (bit-packed + `np.bitwise_count`) |

Identity: on ±1 floats, `hd = (n_columns - s·bᵀ) / 2`. Exact, not an approximation — a test
asserts it equals the naive table bit for bit. Without this, five families x matched-P_fa
x a C/N0 sweep is roughly a week of compute. With it, about an hour.

- `src/hdcam_gps/hdcam_packed.py` — `PackedHdCam(HdCam)`, `np.packbits` grid. `hdcam.HdCam`
  stays untouched as the reference; a test asserts identical row sets on random grids.

---

## 1. The shared abstraction

**`src/hdcam_gps/cam_acq.py`** — `CamAcqClassifier(GpsL1AcqClassifier)`, holding what all
five share.

```python
class CamAcqClassifier(GpsL1AcqClassifier):
    def __init__(self, config, hd_threshold=None, min_votes=None,
                 false_alarm_rate=1e-2, search_mode="cam"|"table",
                 cam_factory=PackedHdCam): ...

    # hooks a family overrides
    def build_codebook(self) -> np.ndarray          # (n_rows, n_columns) bool
    def row_index(self) -> RowIndex                 # what each row means
    def query_index(self, n_samples) -> QueryIndex  # what each query means
    def build_queries(self, samples) -> Iterator[tuple[int, np.ndarray]]
    def chance_floor(self, n_draws=512) -> tuple[float, float]   # measured mean, sd

    # shared, never overridden
    def distance_table(self, samples) -> np.ndarray  # GEMM, (n_queries, n_rows)
    def decide(self, table, hd_threshold, min_votes) -> list[PrnResult]
    def tightest_match(self, row, query) -> int
    def cost(self) -> CamCost
    def _acquire(self, samples) -> list[PrnResult]
```

`RowIndex` / `QueryIndex` are frozen dataclasses of parallel int arrays. One mapping covers
every family:

```
prn         = row.prn[r]
doppler_bin = row.doppler_bin[r] if row.doppler_bin[r] >= 0 else query.doppler_bin[q]
code_phase  = (query.start[q] - row.segment_offset[r]) % samples_per_code
look        = query.start[q] // samples_per_code
```

**The decision rule stops being implemented twice.** `decide()` is the single
implementation; `search_mode="cam"` builds the same table one `search_cam` at a time.
CLAUDE.md's "implemented twice and must agree" invariant becomes "one rule, two data
paths" — **CLAUDE.md is edited in the same change**, and `tests/test_calibrate.py`'s replay
test becomes a CAM-path-vs-table-path test.

`quantize_iq`, `rotate_quarter_turns`, `binomial_tail`, `per_look_false_alarm`,
`hd_threshold_for_false_alarm`, `tightest_match`, `distance_table`, `shortlist`,
`_best_per_prn` move to `cam_acq.py` and are re-exported from `hdcam_acq.py`.
**`OneBitHdCamClassifier` keeps its name, module and entire public API** and shrinks to
~80 lines of hooks. Acceptance criterion for this phase: `tests/test_hdcam_acq.py` and
`tests/test_calibrate.py` pass **unedited**.

---

## 2. The cost model

**`src/hdcam_gps/cam_cost.py`** — `CamCost` (frozen) + `CountingHdCam(PackedHdCam)` +
`measure_cost(classifier, scenario, hd_threshold)`.

Fields: `n_rows`, `n_columns`, `total_bits`, `n_searches` and `n_threshold_writes` (both
**counted** by running one real acquisition through `CountingHdCam`, so `tightest_match`'s
~11 bisections per survivor are included — PROPOSAL risk 5), `bit_comparisons`,
`energy_uj` at 0.19 fJ/bit, `latency_us` at 8 ns/search, `chance_mean`/`chance_sd`
(**measured**, because the binomial p=0.5 model is wrong for the thermometer family), and
the two numbers the study exists for:

- `tolerance_fraction` = calibrated `hd_threshold` / `n_columns`
- `resolution_fraction` = (chance_mean − hd_threshold) / n_columns

Counted searches are asserted against the closed form for the baseline.

Predicted cost, worth stating now because it reframes the comparison:

| family | rows x cols | searches/acq | bit-comparisons | what changes |
|---|---|---|---|---|
| baseline | 1344 x 2046 = 2.75 Mbit | 36 832 | 1.0e11 (19 µJ) | — |
| code-only | 64 x 2046 = 131 kbit | 773 472 | **1.0e11, identical** | 21x less area, 21x more latency, **same energy** |
| differential | 128 x 2046 = 262 kbit | 9 208 | 2.4e9 | 42x less energy — it deletes hypotheses |
| segmented | 21504 x 128 = 2.75 Mbit | 36 832 | 1.0e11 | match line 2046 -> 128 bits |
| thermometer | 1344 x 6138 = 8.25 Mbit | 36 832 | 3.0e11 | 3x area and energy |

Moving Doppler from the row to the query buys area and costs latency; it saves no energy.
Nothing in the repo says that today.

---

## 3. The four new families

### Code-only rows — `code_only_acq.py`, `CodeOnlyHdCamClassifier(mixer=...)`
Rows are the zero-Doppler baseline rows: 32 PRNs x 2 phases = 64 rows. The query wipes
Doppler off; the bin comes from the query. This is the "reduced-row design" PROPOSAL step 4
already names. `mixer="quadrant"` rotates the 1-bit quadrant stream by a 2-bit phase ramp
(mod-4 add, no multiplier) — keeps the front end 1-bit, costs ~0.9 dB. `mixer="exact"` uses
a full-precision NCO and is **no longer a 1-bit front end**; it is reported as an upper
bound only, and the headline number is the quadrant one. Likely failure: `n_tests` rises
21x, so the threshold tightens where risk 1 says there is no headroom — expect false
alarms, not misses.

### Thermometer — `thermometer_acq.py`, `ThermometerHdCamClassifier(n_levels=4)`
I and Q quantised to 4 levels, unary-coded over 3 bits, so Hamming distance approximates L1
distance. `n_columns = 6138`. Expected gain ~1.4 dB (1-bit costs 1.96 dB, 2-bit 0.55 dB) for
3x the area. Needs an AGC estimated from the record itself — using `Scenario.noise_sigma`
would be cheating and must not be done. Likely failure: the row is a noiseless replica while
the query is noise-dominated, so the magnitude term may add a constant without adding
discrimination.

### Segmented — `segmented_acq.py`, `SegmentedHdCamClassifier(segment_bits=128, min_segments=None)`
A code period becomes K=16 sub-rows of 128 bits: 21 504 rows, **same total bits**, match
line narrowed to what silicon demonstrates. The non-obvious part: do not search segments
separately — segment k at start s is segment 0 at start s + k·segment_bits/2, so one short
sliding query already visits every alignment, and searches stay at 36 832. A hit on
(hypothesis h, segment k) at start s votes for cell (h, (s − k·segment_bits/2) mod
samples_per_code). Two-level decision: ≥`min_segments` of K per look, then votes across
looks. The tolerance **fraction** does not improve (SNR sets it); only the absolute count
does. Relative spread worsens, σ/width 1.1% -> 4.4%, so expect 1-2 dB lost to replacing one
coherent 2046-bit sum with m-of-K binary integration. Real risk: the **hit flood** — at a
47% threshold on 128 bits, ~25% of rows fall inside every search, ~2e8 hit events per
acquisition. That is a digital-readout problem, not a CAM problem; score this family
through the table path only.

### Differential — `diff_acq.py`, `DifferentialHdCamClassifier(lag_samples=L)`
Rows and queries hold `quantize_iq(x[n]·conj(x[n−L]))`. Carrier phase cancels: no rotations,
no stored phases. Two problems, both found during design and both worth stating before any
code is written:

1. After the delay-multiply, Doppler is a **constant** phase `2πf_d·L/fs`, which one bit
   resolves into 4 classes. Resolving 500 Hz needs L ≈ 256; staying unambiguous over ±5 kHz
   needs L < 102. **Both cannot hold.** So this is structurally a PRN + code-phase detector
   with ambiguous Doppler (128 rows) and needs a second stage. That is why it is 42x
   cheaper — not a free lunch.
2. Per-sample SNR at 40 dB-Hz is −20 dB, and a delay-multiply detector goes as SNR² at low
   SNR, so the noise x noise penalty is on the order of **20 dB**.

Prediction: it dies in the phase-1 screen. Build the screen, not the classifier, first.

---

## 4. What is measured, exactly

Every number below is produced by the same code path for every family, from the same
records. Tolerances are printed with every table.

### 4.1 The events (one vocabulary for P_d, P_fa and everything else)

For a satellite present in a record, with truth `(prn, f_true, τ_true)`:

| event | definition |
|---|---|
| **found** | `prn` reported with `\|f_rep − f_true\| ≤ doppler_step_hz` and wrapped `\|τ_rep − τ_true\| ≤ 1` sample |
| **wrong fix** | `prn` reported, but Doppler or code phase outside tolerance. Counted as a miss **and** as a false alarm — a wrong fix is worse than none |
| **miss** | not found (includes wrong fix) |
| **false alarm** | a PRN reported that is not in the record |

Doppler tolerance is one bin, not half a bin: simulator truth is off-grid and a satellite
at a bin edge is legitimately found in either neighbour. Code-phase tolerance is one sample
because the simulator's truth is fractional (`gps_sdr_sim.labels`). At 1.023 MHz one
sample is one chip.

The differential family cannot resolve Doppler (§3). It is scored on PRN + code phase only,
its `PrnResult.doppler_hz` is the centre of its Doppler class, and every table flags it
**Doppler-blind**. Its sensitivity is not head-to-head comparable until the second stage
that resolves Doppler is costed in.

### 4.2 Per-satellite C/N0 — the x-axis (new; the plan had this wrong)

gps-sdr-sim runs with a 0° elevation mask and an antenna pattern reaching −31.6 dB at the
horizon (`gpssim.c:86`, `:2298`), so the satellites inside one record span >20 dB.
"Record-level average C/N0" therefore cannot be the sweep variable. Instead, measure C/N0
**per satellite**, in `scenario_from_record` before noise is added: correlate the clean,
rescaled record with the truth replica (PRN, Doppler and code phase from `labels`)
coherently over one code period, giving amplitude `a_i`, and set
`SatelliteTruth.cn0_dbhz = 10·log10(a_i² · fs_hz)` — the inverse of `amplitude_for_cn0`,
same unit-noise convention. Error sources: cross-correlation from the other satellites
(≈ −24 dB → < 0.1 dB) and Doppler quantised to the 0.1 s tick. Test: on a synthetic record
the estimator recovers the set C/N0 within 0.2 dB. Synthetic scenarios already carry an
exact per-satellite value.

### 4.3 P_d(C/N0) and sensitivity

- Pool every (record, satellite) pair over all evaluation skies and all record scalings.
- Bin by per-satellite C/N0 in 1 dB bins. `P_d(bin) = found / observations`, with a
  two-sided 95% Clopper-Pearson interval (add a lower bound beside the existing
  `binomial_upper_bound`).
- **Sensitivity** = lowest bin whose P_d point estimate ≥ 0.9, reported with its interval.
  A bin with < 100 observations is greyed out and cannot be the sensitivity.
- Record scalings are chosen to populate the bins that matter: record-average C/N0 from
  36 to 54 dB-Hz in 3 dB steps (7 scalings), which puts the horizon satellites at the
  detection edge and the zenith ones well above it. 40 evaluation skies × ~10 satellites
  × 7 scalings ≈ 2800 observations over ~25 dB ≈ 110 per bin.
- The same pool gives, for found satellites, the Doppler error and code-phase error
  distributions (median, 95th percentile) — the handover-quality metric of PROPOSAL.

### 4.4 P_fa — per acquisition and per absent PRN

On the same records, which all contain satellites (the cross-correlation floor only exists
when they do — `calibrate.py`'s point).

- `P_fa,acq` = records with ≥ 1 false alarm / records. 280 records → zero events bounds it
  at 1.1e-2 (95%).
- `P_fa,prn` = false reports / (records × absent PRNs). ≈ 280 × 22 ≈ 6000 opportunities,
  zero events → ≤ 5e-4. Supporting number only: opportunities within a record share the
  same strong satellites and are not independent.
- Wrong fixes reported as their own count.
- **P_fa is also tabulated against the strongest satellite's C/N0 in the record**, because
  the cross-correlation floor scales with it. A threshold that is clean at 45 dB-Hz
  average can false-alarm at 54; the table has to show where.

### 4.5 Calibration protocol — fixed before anything in 4.3–4.4 is measured

The plan calibrated on single-satellite synthetic trials; a threshold chosen there does not
transfer to a ten-satellite sky. So:

- Split skies by `start_time`: **20 calibration skies** (set A) disjoint from **40
  evaluation skies** (set B). Nothing in 4.3–4.4 is ever computed on set A.
- Per family, one distance table per set-A record; replay the (threshold, votes) grid via
  `decide()`; score with the 4.1 events. For each vote rule take the loosest threshold whose
  `P_fa,acq` upper bound ≤ 1e-2; among those pick the highest pooled P_d in the 38–42 dB-Hz
  bins. Freeze. Print the calibration table for every family.
- **The FFT reference goes through the identical protocol** over a `peak_ratio` grid,
  replayed from cached `correlate()` surfaces. PROPOSAL: "everything at matched P_fa or it
  is not a comparison" — the plan's `compare()` had left it out.

### 4.6 Separation and required tolerance — the screen and the answer to risk 1

From distance tables only, no decision rule, so it is family-agnostic and cheap:

- `D_true`: for each present satellite and each look, the distance between the query at
  the truth code phase and the true row (nearest Doppler bin, best stored phase, best
  rotation). Keep per-look values, so the single-look distribution exists.
- `D_wrong`: distances of every row belonging to no present PRN, at every start — the
  population the threshold must exclude. Its mean and sd are the **measured chance floor
  with satellites present**. The pure-noise floor is measured separately on noise-only
  records, for reference only.
- `d'(C/N0) = (mean D_wrong − mean D_true) / sd D_wrong`, binned by per-satellite C/N0.
- **Required tolerance fraction** `T(x)` = 90th percentile of single-look `D_true` at
  C/N0 x, divided by `n_columns`. **Required resolution** = `(mean D_wrong − T) /
  n_columns`, also in units of `sd D_wrong`. Segmented family: both per 128-bit sub-row,
  plus the m-of-K rule it needed.
- Plot `T(x)` against C/N0 per family, with the 12.5% silicon line and the ~50% ceiling.

### 4.7 CAM cost per family

As in §2, measured on one set-B record at the calibrated threshold through `CountingHdCam`:
`n_searches`, `n_threshold_writes`, `total_bits`, `bit_comparisons`, `energy_uj`,
`latency_us`, plus **hits per search (mean, max)** — the digital-readout load, which is the
segmented family's real risk and is measurable for every family. Simulation wall-clock is
reported separately and labelled as such.

---

## 5. Matched P_fa — code changes

Three surgical changes to `calibrate.py`:

- `score_table(classifier, table, hd_threshold, min_votes)` keeps its signature; the body
  becomes `classifier.decide(...)`. The `_best_per_prn` reach-in disappears.
- `calibrate(..., *, classifier=None, scenarios=None)` — `None` builds
  `OneBitHdCamClassifier(config)` on synthetic trials, so existing calls are unchanged;
  `scenarios=` takes the set-A `ScenarioBank` of §4.5. The threshold grid uses the
  **measured** `D_wrong` floor of §4.6 instead of `n_columns/2`. Scoring uses the §4.1
  events (wrong fix = miss + false alarm), so `Candidate` gains `n_wrong_fix`,
  `n_prn_trials` / `n_prn_false` with `pfa_per_prn_upper`, and per-satellite pooling.
- New `match_false_alarm(result, target_pfa) -> Candidate` implementing the §4.5 pick.
  Every sensitivity number in the comparison comes from that.
- `binomial_lower_bound` beside `binomial_upper_bound`, for the P_d intervals of §4.3.
- A `PeakRatioCalibrator` for `FftAcqClassifier`: same grid replay over cached
  `correlate()` surfaces, so the reference row is matched too.

**The honest target.** Confirming P_fa <= 1e-4 per acquisition needs ~30 000 records by the
rule of three — days per family. So the common target is `max_pfa = 1e-2` per acquisition,
`pfa_upper` is printed in every table, and the notebook and README state that **1e-4 is
not demonstrated**. The per-absent-PRN bound of §4.4 is the supporting number.

---

## 6. The simulator evaluation

`evaluate.build_scenario` re-runs gps-sdr-sim per scenario per C/N0 point — for 5 families
x 6 points x 20 scenarios that is 600 simulator invocations of identical records. A record
depends only on `(lat, lon, height, fs, duration, rinex, start_time)`; C/N0 and noise are
applied after. So caching is exact:

- Split `generate_from_sim` into `simulate(...)` + a new pure
  `scenario_from_record(config, record, cn0_dbhz, add_noise, seed, offset)`. The old
  function becomes their composition. One behaviour change, deliberate: each
  `SatelliteTruth.cn0_dbhz` is the **measured per-satellite value** of §4.2, no longer the
  record average copied onto every satellite.
- `src/hdcam_gps/sim_cache.py` — `cached_simulate(**kwargs)`, in-process dict plus npz under
  `third_party/cache/` (gitignored).
- `src/hdcam_gps/scenarios.py` — `ScenarioBank.build(config, eval_config)` with
  `.get(cn0_dbhz, index)`, materialised once so **all five families see identical records**.
  `evaluate(classifier, eval_config, scenarios=None)` keeps today's default behaviour.

Headline run: `fs=1.023e6`, 32 PRNs, ±5 kHz / 500 Hz, `n_codes=10`, tolerances of §4.1.
**60 skies** at 17-minute spacing — 20 calibration (set A) + 40 evaluation (set B) — each
rescaled to 7 record-average C/N0 values (36…54 dB-Hz, 3 dB steps): 420 records from **60
simulator invocations**, a few minutes. Table-path cost per record ~1 s (baseline,
segmented, differential), ~3 s (thermometer), ~21 s (code-only): roughly 7 min, 20 min and
2.5 h per family respectively for the full 420, so the study fits in an afternoon and
code-only is the long pole. Every table states C/N0 **per satellite** (§4.2) and names the
record-average scaling only as the sampling design. No simulator patch.

---

## 7. Phasing

| phase | work | new tests |
|---|---|---|
| 0 | `cam_acq.py`, `hdcam_packed.py`, `cam_cost.py`, `sim_cache.py`, `scenarios.py`; per-satellite C/N0 in `scenario_from_record` (§4.2); refactor `hdcam_acq.py` onto the base; generalise `calibrate`; `evaluate(..., scenarios=)`; edit CLAUDE.md | `test_cam_acq.py`, `test_cam_cost.py`, `test_hdcam_packed.py`, `test_scenarios.py`, C/N0-estimator test in `test_signal_gen.py` |
| 1 | **cheap screen** — §4.6 `d'(C/N0)` and `T(x)` per family on set A, from distance tables only. Minimal codebook+query per family, no classifier. Minutes each. | `notebooks/family_screen.ipynb` |
| 2 | full classifiers for survivors, in order: code-only, thermometer, segmented, differential | `test_code_only_acq.py`, `test_thermometer_acq.py`, `test_segmented_acq.py`, `test_diff_acq.py` |
| 3 | §4.5 calibration on set A for every survivor **and the FFT reference**, backgrounded; freeze the settings in a checked-in JSON | — |
| 4 | `src/hdcam_gps/compare.py` — `compare(classifiers, scenarios, target_pfa) -> Comparison` running §4.3, §4.4 and §4.7 on set B and joining them into one table | `notebooks/family_comparison.ipynb` |
| 5 | README family table replaces the two-classifier table; PROPOSAL step 4 gets the measured numbers | — |

New test files follow `test_hdcam_acq.py` conventions exactly: module docstring saying what
is and is not covered, local `make_config`, banner comments, `fs_hz=204.6e3`, long
sentence-style names.

---

## 8. Verification

**Phase 0** — the old tests pass unedited, plus: GEMM table equals the naive table bit for
bit; `PackedHdCam` equals `HdCam` on random grids; CAM path and table path return the same
`PrnResult` list; counted `search_cam` calls equal the closed form for the baseline;
`ScenarioBank` hands two classifiers the identical `samples` array.

**Per family** (wiring-level, noise-free): a planted noiseless replica is found at the right
cell; `row_index`/`query_index` round-trip; the quadrant differential equals the exact
differential on noiseless input; the quadrant mixer equals the exact mixer at zero Doppler;
thermometer Hamming distance is monotone in level difference; a segmented hit at segment k
votes for the same code phase as the unsegmented hit.

**The headline plot** (phase 4): `tolerance_fraction` per family as a bar chart, with the
12.5% JSSC-silicon line and the ~50% theoretical ceiling drawn on it. That single plot is
the answer to PROPOSAL risk 1.

**Kill rule, fixed before the runs so it cannot be argued with afterwards.** A family is
dead when any of: (i) phase-1 `d'` in the 45 dB-Hz bin is below 1.0, or below half the
baseline's; (ii) no (threshold, votes) setting on set A reaches `P_d >= 0.9` in the
40 dB-Hz bin at `P_fa,acq` upper bound ≤ 1e-2; (iii) set-B sensitivity is more than 3 dB
worse than the baseline with no cost win paying for it. A dead family still ships its
screen plot and one paragraph naming which test it failed.

**Measurement-level checks** (pytest, noise-free where possible): the §4.2 estimator
recovers a synthetic satellite's C/N0 within 0.2 dB; the §4.1 scorer counts a planted
wrong-Doppler report as both a miss and a false alarm; a Doppler exactly one bin off is
found and two bins off is not; a set-A `start_time` never appears in set B; `P_d` intervals
from `binomial_lower_bound`/`binomial_upper_bound` bracket the point estimate and shrink
with trials.

---

## Not doing

- Touching `AcqConfig`, `HdCam` semantics, or the gps-sdr-sim patch. All five families fit
  inside the existing geometry.
- Per-satellite C/N0 in the simulator (your call: sweep the record-level average as-is).
- Chasing a demonstrated P_fa of 1e-4.
