# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

A simulation study asking whether GPS L1 C/A acquisition can be done as a lookup in a
Hamming-distance-tolerant CAM instead of a correlator. `README.md` explains the two
algorithms and the public API; `PROPOSAL.md` states the research question and its risks.

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

Five things that are not visible from any single file.

**The classifier contract** (`acq_base.py`). `GpsL1AcqClassifier` carries the
`AcqConfig` it was built for. Public `acquire()` asserts
`len(samples) == config.samples_per_acquisition`, then delegates to `_acquire()`.
**Subclasses override `_acquire`, never `acquire`** — this holds for both
`FftAcqClassifier` and `OneBitHdCamClassifier`. Everything downstream accepts any object
satisfying this interface, which is what makes the two classifiers directly comparable.

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
across a grid of classifier settings.

**Two signal backends, one return type.** `generate_synthetic` and `generate_from_sim`
both return `Scenario`, so call sites are interchangeable. `signal_gen.py` is the
user-facing entry point; `gps_sdr_sim.py` is only the simulator driver. That driver's
`build()` copies the submodule sources into `third_party/build`, applies
`third_party/patches/*.patch` to the **copy** and compiles there — `make` and `gcc` must
be on PATH.

**The HdCam decision rule is implemented twice and the two must agree.**
`OneBitHdCamClassifier._acquire` shortlists by votes then ranks by `tightest_match`;
`calibrate.score_table` replays the same decision from a cached `distance_table`, so a
whole grid of settings can be scored from one sweep per record. Change one and you must
change the other — `tests/test_calibrate.py` asserts the replay reproduces `acquire()`.

## Module map

| Module | Role |
|---|---|
| `acq_base.py` | `AcqConfig`, `PrnResult`, `GpsL1AcqClassifier` base, GPS constants |
| `ca_code.py` | C/A code generation (G1/G2 LFSRs) and sampling |
| `hdcam.py` | The CAM itself: a bit grid with a Hamming-distance threshold |
| `fft_acq.py` | FFT parallel code-phase search, the reference classifier |
| `hdcam_acq.py` | 1-bit HdCam classifier: codebook, query rotations, votes, ranking |
| `signal_gen.py` | Labelled `Scenario` generation, both backends |
| `gps_sdr_sim.py` | Driver for the vendored simulator: build, run, parse |
| `evaluate.py` | Scores any classifier over a C/N0 sweep |
| `calibrate.py` | Searches threshold and vote rule against a target operating point |

Tests mirror sources 1:1 as `tests/test_<module>.py`, so a change has an obvious test home.
