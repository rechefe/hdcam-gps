# hdcam-gps

## Setup

```bash
uv sync
```

The third party GPS signal simulator is a submodule, so clone with:

```bash
git clone --recurse-submodules <url>
# or, in an existing clone
git submodule update --init --recursive
```

## Two ways to acquire

Both classifiers answer the same question — which satellites are present, at what
Doppler and at what code phase — but they parallelise it along different axes.

**The FFT reference** (`fft_acq`) is the textbook parallel code phase search. For
each PRN and each Doppler bin it corrects the carrier with an NCO, then one FFT,
one multiply and one inverse FFT produce the correlation against the local
replica at *every code phase at once*. The resulting power is summed
non-coherently across code periods, and a PRN is declared when the peak of that
grid stands far enough above its own second peak. It is full precision complex
arithmetic throughout, and its cost is dominated by the transforms: one pair per
PRN, Doppler bin and code period.

**The HdCam classifier** (`hdcam_acq`) replaces the correlation with a memory
lookup. The codebook holds one row per (PRN, CFO) hypothesis, quantised to one
bit per component, so a row is the carrier phase of each sample rounded to a
quadrant. A query is the same quantisation of one code period of input, and the
CAM returns every row within a Hamming distance threshold in a single search — so
the *PRN and Doppler* dimensions are the ones resolved in parallel here, not code
phase. Code phase comes from sliding the query window, carrier phase from the
four quarter turns a 1-bit sample can express, accumulation from voting across
code periods, and the final ranking from bisecting the CAM threshold to recover
each survivor's distance.

The trade is arithmetic for memory. The FFT needs multipliers and transforms but
loses nothing to quantisation; the HdCam needs neither, but keeps only the sign of
each sample (about 2 dB) and only eight carrier phases (about another 2.5 dB).
Measured on identical scenarios at `n_codes=10`, the reference works down to
38 dB-Hz and the HdCam classifier to 40 dB-Hz.

| | FFT reference | 1-bit HdCam |
|---|---|---|
| parallel in | code phase | PRN and CFO |
| serial in | PRN, CFO | code phase |
| arithmetic | complex, full precision | Hamming distance on bits |
| detection statistic | peak to second peak ratio | votes, then distance |
| sensitivity (n_codes=10) | 38 dB-Hz | 40 dB-Hz |

Note that in this repository the CAM is *simulated* one lookup at a time, which
makes the HdCam path some two orders of magnitude slower per acquisition. That
ordering is an artefact of the simulation: in hardware the search is the single
parallel primitive, and it is the reason the approach is interesting at all.

## Generating labelled test signals

`signal_gen` is the single entry point. Both backends return a fully labelled
`Scenario`.

```python
from hdcam_gps.acq_base import AcqConfig
from hdcam_gps.signal_gen import SatelliteTruth, generate_synthetic, generate_from_sim

config = AcqConfig(fs_hz=4e6, prn_list=tuple(range(1, 33)), doppler_min_hz=-5000,
                   doppler_max_hz=5000, doppler_step_hz=500, n_codes=10)

# Samples from the third party simulator: a real constellation from a real
# ephemeris, labelled with each satellite's Doppler and code phase.
scenario = generate_from_sim(config, latitude_deg=32.0, longitude_deg=35.0,
                             cn0_dbhz=45, seed=0)

scenario.samples                                  # complex baseband
scenario.expected_results()                       # PRN, Doppler, code phase
scenario.matches(classifier.acquire(scenario.samples), code_phase_tolerance=1)

# Or place satellites yourself, for a scenario you control exactly.
scenario = generate_synthetic(config, [
    SatelliteTruth(prn=7, doppler_hz=1500, code_phase=1234, cn0_dbhz=45),
], seed=0)
assert classifier.acquire(scenario.samples) == scenario.expected_results()
```

Both share one convention: noise has unit power, so `N0 = 1 / fs`, and
`cn0_dbhz` sets the signal against it. `add_noise=False` gives a noiseless
record.

### Where the simulated labels come from

gps-sdr-sim already knows every satellite's Doppler and code phase; it just does
not print them. A seven line patch under `third_party/patches` makes it say so,
right after `computeCodePhase`, so the numbers are the state each tick's samples
are generated from:

```c
if (verb==TRUE)
    fprintf(stderr, "TRUTH %d %02d %.9f %.9f\n",
        iumd, chan[i].prn, chan[i].f_carr, chan[i].code_phase);
```

Nothing is measured from the samples and nothing is modelled, so a classifier can
be scored against these labels.

**The submodule is never modified.** `build()` copies the sources into
`third_party/build`, applies the patches to the copy and compiles there, so
`git submodule status` stays clean and upstream can be updated normally. If
upstream moves the code the patch sits in, the patch stops applying and the build
fails rather than silently reporting something else.

The simulator places a satellite at a fractional code phase while an acquisition
can only answer in whole samples, hence
`matches(..., code_phase_tolerance=1)`.

## Scoring a classifier

`evaluate` takes any classifier with the `GpsL1AcqClassifier` interface and a
test configuration, sweeps C/N0 and reports quality metrics. Both classifiers go
through the same measurement on the same scenarios.

```python
from hdcam_gps.evaluate import EvalConfig, evaluate

report = evaluate(FftAcqClassifier(config), EvalConfig(
    n_scenarios=10, cn0_dbhz=(48, 45, 42, 39, 36), n_satellites=1,
))
print(report.table())
report.sensitivity_cn0_dbhz(min_accuracy=0.9)
```

Per C/N0 it reports detection rate, accuracy (right PRN, Doppler bin and code
phase), precision, false alarms per scenario, the fraction of scenarios answered
completely, Doppler and code phase RMSE, and seconds per scenario. Set
`backend="simulator"` to run against real skies instead of synthetic ones; each
scenario then uses a different scenario time, so the constellation varies.

## Example notebook

`notebooks/example_run.ipynb` walks the whole repository once, with plots: the
Gold code correlation properties, a labelled signal, the FFT reference search
grid, the HdCam codebook, a scored sweep of both classifiers, and a real
simulated sky.

```bash
uv run jupyter lab notebooks/example_run.ipynb
```
