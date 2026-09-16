# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

A simulation study asking whether GPS L1 C/A acquisition can be done as a lookup in a
Hamming-distance-tolerant CAM instead of a correlator. `README.md` explains the two
algorithms and the public API; `PROPOSAL.md` states the research question and its risks;
`docs/CAM_FAMILY_STUDY.md` is the plan for the five-family comparison now being built,
and names which phase each module belongs to.

## Commands

Everything is driven by uv. There is no Makefile, no CI, and no linter or type checker
configured (`pyproject.toml` has no ruff/mypy/black section), so do not invent a lint
command.

```bash
uv sync --group dev                  # install, incl. pytest, jupyter, matplotlib, weasyprint
uv run pytest                        # full suite (testpaths = ["tests"], run from repo root)
uv run pytest tests/test_hdcam_acq.py                    # one file
uv run pytest tests/test_hdcam_acq.py::test_shortlist_is_empty_for_silence   # one test
uv run pytest -k "shortlist"                             # by name

git submodule update --init --recursive                  # needed for the simulator backend
uv run jupyter lab notebooks/example_run.ipynb           # end-to-end walkthrough
uv run python scripts/make_proposal_pdf.py               # rebuild PROPOSAL.pdf
```

Tests needing the gps-sdr-sim submodule are guarded by `sources_checked_out()` and
**skip** rather than fail when it is missing, so a green run does not prove that path
was exercised.

## Architecture

Six things that are not visible from any single file.

**The classifier contract** (`acq_base.py`). `GpsL1AcqClassifier` carries the
`AcqConfig` it was built for. Public `acquire()` asserts
`len(samples) == config.samples_per_acquisition`, then delegates to `_acquire()`.
**Subclasses override `_acquire`, never `acquire`** — this holds for both
`FftAcqClassifier` and `CamAcqClassifier`. Everything downstream accepts any object
satisfying this interface, which is what makes the classifiers directly comparable.

**A CAM family overrides hooks, not the decision** (`cam_acq.py`). `CamAcqClassifier`
implements `_acquire`; a family supplies `build_codebook`, `row_index`, `query_index`
and `query_variants`, and sets `n_rows` and `n_columns` before calling `super().__init__`.
`RowIndex` and `QueryIndex` are parallel int arrays, and the four-line mapping in the
module docstring turns any (row, query) hit into a (PRN, Doppler bin, code phase).
A `doppler_bin` of -1 means the other side carries it. **A family that will not fit
those four lines needs a new field in the index, not a second `_acquire`.**

**`AcqConfig` is the single source of geometry.** `samples_per_code`,
`samples_per_acquisition`, `observation_time_s` and `doppler_grid_hz` are derived
properties. Classifiers, signal generation and evaluation all size themselves from them,
so changing one config field propagates everywhere.

**The data flow.**

```
signal_gen → Scenario(.samples, .truth) → classifier.acquire() → list[PrnResult]
                                        ↘ Scenario.expected_results() / .matches(...)
```

`evaluate.evaluate()` runs that loop across a C/N0 sweep; `calibrate.calibrate()` runs it
across a grid of classifier settings. Both take an optional `ScenarioBank`, which
materialises the records first and hands every classifier **the same `samples` array** —
five families scored on five independent draws is five experiments, not one comparison.
`ScenarioBank.split()` is what keeps a calibration sky out of the evaluation set.

**Two signal backends, one return type.** `generate_synthetic` and `generate_from_sim`
both return `Scenario`, so call sites are interchangeable. `signal_gen.py` is the
user-facing entry point; `gps_sdr_sim.py` is only the simulator driver. That driver's
`build()` copies the submodule sources into `third_party/build`, applies
`third_party/patches/*.patch` to the **copy** and compiles there — `make` and `gcc` must
be on PATH.

**C/N0 is per satellite, and the record average is only the sampling design.**
`generate_from_sim` is `simulate` then `scenario_from_record`, and the second half gives
each `SatelliteTruth` its own `cn0_dbhz` from `measure_cn0_dbhz`. Path loss and the
receiver antenna pattern spread one simulated sky over about 8 dB, so the `cn0_dbhz` a
record was scaled to is nobody's figure. The estimator searches the code phase over a
sample either way, because `gps_sdr_sim.labels` rounds a fractional phase and at
1.023 MHz one sample is one chip — correlating at the rounded phase read seven of nine
satellites 15 to 25 dB low.

**One decision rule, two data paths.** Vote across looks, then rank the survivors by
distance — `CamAcqClassifier.shortlist_cells` and `rank_cells`, written once.
`search_mode="cam"` gets the hits from one `search_cam` per query and measures a survivor
with `tightest_match`, which is what the hardware can do; `search_mode="table"` reads the
same hits off a GEMM `distance_table`, which is the only affordable way to replay a grid
of thresholds over hundreds of records. `calibrate.score_table` is now a call to
`decide`, so there is no second copy to keep in step. `tests/test_cam_acq.py` asserts the
two paths return the same `PrnResult` list, and the GEMM table equals a counted one bit
for bit.

## Module map

| Module | Role |
|---|---|
| `acq_base.py` | `AcqConfig`, `PrnResult`, `GpsL1AcqClassifier` base, GPS constants |
| `ca_code.py` | C/A code generation (G1/G2 LFSRs) and sampling |
| `hdcam.py` | The CAM itself: a bit grid with a Hamming-distance threshold |
| `hdcam_packed.py` | The same CAM answering from `np.packbits`, 12x faster |
| `cam_acq.py` | `CamAcqClassifier`: the shared codebook/query hooks and the one decision rule |
| `cam_cost.py` | `CamCost`, `CountingHdCam`: area, energy, latency and tolerance fraction, counted |
| `fft_acq.py` | FFT parallel code-phase search, the reference classifier |
| `hdcam_acq.py` | 1-bit HdCam family: codebook, query rotations, the baseline of the study |
| `signal_gen.py` | Labelled `Scenario` generation, both backends, per-satellite C/N0 |
| `gps_sdr_sim.py` | Driver for the vendored simulator: build, run, parse |
| `sim_cache.py` | Caches a simulator record, which does not depend on C/N0 |
| `scenarios.py` | `ScenarioBank`: records built once, split into calibration and evaluation skies |
| `evaluate.py` | Scores any classifier over a C/N0 sweep; `score_events` is the study's event vocabulary |
| `calibrate.py` | Searches a setting against a target operating point, for a family or the FFT reference |

Tests mirror sources 1:1 as `tests/test_<module>.py`, so a change has an obvious test home.
