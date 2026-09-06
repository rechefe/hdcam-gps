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
