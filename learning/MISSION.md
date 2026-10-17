# Mission: Take over the HD-CAM GPS acquisition study

## Why
You are taking over this study at the end of phase 3. You need to understand it from the bottom up: the signal, the algorithm, every CAM family, the measurements, and the code. Then you can make the phase 4 and phase 5 decisions yourself and defend them.

## Success looks like
- Explain, without notes, why the HD-CAM classifier makes PRN and Doppler parallel while the FFT reference makes code phase parallel, and what each one pays for it.
- Read `docs/screen_results.json` and `docs/calibration.json` and say what each number means and which kill rule it feeds.
- Name the role of every module in `src/hdcam_gps/` and find the code that implements any step of the decision rule.
- State the open decisions for phase 4 and phase 5, with the evidence for each option.
- Place the study against its baselines and the related literature.

## Constraints
- Solid DSP and correlation background; new to this codebase and to CAM hardware vocabulary.
- Learns best from interactive figures for anything algorithmic.
- Writing follows the `technical-writing` skill the user invoked.

## Out of scope
- Circuit-level design of the HD-CAM macro (the study treats the CAM as ideal).
- Tracking loops and navigation solution; the study stops at acquisition.
