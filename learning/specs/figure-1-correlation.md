# Spec: Figure 1, correlation search for one hypothesis

Figure 1 sits in Chapter 2 of `learning/explainer/index.html`. It teaches the reader how an acquisition finds a satellite before the CAM is introduced.

## What the reader learns

1. The right (PRN, Doppler) hypothesis produces a peak at the true code phase. A wrong PRN produces only noise.
2. A Doppler error attenuates the peak power by sinc²(Δf·T), with T = 1 ms. The peak vanishes at 1 kHz of error, which is why the grid steps by 500 Hz.
3. Summing code periods non-coherently narrows the noise spread. It does not lower the noise mean. Narrower noise lets a weaker peak clear the decision line.
4. The FFT reference reports a satellite when the peak is at least 2.0 times the second peak.

## Problems in the current figure

| Problem | Cause |
|---|---|
| The plot jumps vertically when a control changes | Control labels change length ("1 ms (coherent only)" → "10 ms", "1500 Hz" → "−5000 Hz"). The control row re-wraps and pushes the canvases 18 px. Measured at 1000 and 1280 px viewports. |
| The display looks broken when Doppler changes | A wrong Doppler row shows only noise and a moving dashed line. Nothing on screen says why the peak went away. |
| The heatmap is unreadable | 1023 one-chip columns at ~700 px: the peak is one column and noise dominates the colour scale. |

## Layout

- **Controls** sit in a CSS grid with columns of fixed minimum width (`repeat(auto-fill, minmax(200px, 1fr))`). Each control has its label on one line and its value in a fixed-width output (`min-width` in `ch`, tabular figures). Wrapping depends on the viewport width only, never on a value.
- **Panel A, correlation trace**: power against code phase 0–1022 for the selected hypothesis. Fixed height 200 px.
- **Panel B, Doppler profile**: power at the true code phase (400) against Doppler, −5 000 to +5 000 Hz. Measured value at each of the 21 bins as dots; the theoretical curve 1 + K·sinc²((f − 1 600 Hz)·1 ms) as a line, where K is the expected coherent peak; the selected bin highlighted. Fixed height 150 px.
- **Readout**: two fixed lines with a reserved height (`min-height`), so changing text never moves the caption.
- No heatmap.

## Scales

- Both panels plot **power ÷ mean noise power**. Each period's correlation power is divided by N = 1023, the expected noise power. Summed periods are averaged, not added. Noise averages 1 at every setting.
- Both panels share one fixed y range, **0 to 35**, ticks every 10. Nothing rescales with the data. Values above 35 are clipped at the top edge.
- C/N₀ runs 30–45 dB-Hz. At 45 dB-Hz the expected peak is 1 + 31.6 = 32.6, inside the range.

## Controls

| Control | Values | Default | Effect |
|---|---|---|---|
| Local replica PRN | 1–32 | 7 (the satellite present) | Recompute |
| C/N₀ | 30–45 dB-Hz, step 1 | 42 | Recompute |
| Doppler bin | 21 bins, −5 000 to +5 000 Hz, step 500 | 1 500 Hz | Redraw only |
| Code periods summed | 1–10 | 1 | Redraw only |
| New noise | button | — | Recompute with the next seed |

The record holds PRN 7 at code phase 400, Doppler 1 600 Hz, carrier phase 0.7 rad, unit-power complex noise, 10 periods at 1.023 MHz.

## Computation

- **Recompute** builds the correlation power for all 21 bins × 10 periods × 1 023 code phases with a 4 096-point FFT per (bin, period), and stores it. Budget: 300 ms.
- **Redraw** reads from the stored grid: average the first `nc` periods of the selected bin and draw. Budget: 30 ms.
- Slider input events are coalesced with `requestAnimationFrame`.

## Decision shown

- The decision applies to the hypothesis on screen: peak = maximum of the trace; second peak = maximum more than 1 sample away (circular); ratio = peak / second peak; detected when ratio ≥ 2.0.
- Panel A draws the dashed line at 2 × second peak and marks the true phase when the PRN is 7.
- Readout line 1: hypothesis, Doppler error, sinc² attenuation, expected peak.
- Readout line 2: measured peak position and value, ratio, and a pill: "detected, right cell", "false alarm" (detected at a wrong cell), or "not detected".

## Test hooks

The `figure#fig1` element carries `data-ymax`, `data-peak-phase`, `data-ratio` and `data-detected` after every draw.

## Acceptance tests (`learning/explainer/tests/figure1.test.mjs`)

1. At viewports 1280, 1000, 820 and 400 px, the top and height of both canvases and of the figure are identical, to the pixel, across every Doppler bin, every `nc` and every C/N₀ value.
2. `data-ymax` is 35 after every interaction.
3. Doppler and `nc` changes finish within 30 ms; PRN, C/N₀ and new-noise changes within 300 ms.
4. PRN 7, 1 500 Hz, 45 dB-Hz, 10 periods: peak at code phase 400, detected.
5. PRN 12 at every `nc` with the default seed: not detected.
6. PRN 7, 2 500 Hz bin (900 Hz error): not detected at 42 dB-Hz.
7. No console errors and no horizontal page overflow at any tested width.

## Out of scope

The other figures. Their control rows can reflow in the same way; fix them after this one is accepted.
