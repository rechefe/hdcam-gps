# Handoff: where the CAM family study stands

Written at the end of phase 3, for whoever picks the study up next.
`docs/CAM_FAMILY_STUDY.md` is the plan and stays the authority; this says which
parts of it are now measured, which two parts of it were wrong, and what the next
phase has to decide before it starts.

## State

Branch `cam-algorithm-exploration` at `48809e9`, pushed. `origin/master` is at
`f78cf5c` and has none of this work. No pull request exists, and none should be
opened without being asked for.

`uv run pytest` is 925 tests and green. There is no linter and no type checker.

Phases 0, 1, 2 and 3 are done. Phases 4 and 5 are not started.

## What phase 1 measured

`docs/screen_results.json`, plotted by `notebooks/family_screen.ipynb`.

Every arrangement of the CAM needs between 36% and 50% of its word width as
Hamming tolerance, against the 12.5% a 65 nm macro demonstrates. That answers
PROPOSAL risk 1, and the factor of three to four does not close with C/N0, with
word width, or with any of the five arrangements tried.

The thermometer family needs the least, at 36.1%, and pays three times the area
for it.

## What phase 3 measured

`docs/calibration.json`, written by `scripts/run_family_calibration.py`. 140
records: 20 set-A skies at seven record scalings from 36 to 54 dB-Hz. 275
satellites fall inside the 38-42 dB-Hz band the operating point compares in. The
run took 4.1 hours.

| family | setting | P_fa,acq | pooled P_d | P_d in 38-42 dB-Hz |
|---|---|---|---|---|
| FFT reference | ratio 2.00 | 0.0071 | 0.829 | **0.985** |
| thermometer | thr 2264/6138, votes 6 | 0.0071 | 0.679 | **0.498** |
| baseline | thr 944/2046, votes 4 | 0.0071 | 0.626 | **0.273** |
| code-only (quadrant) | thr 944/2046, votes 4 | 0.0000 | 0.587 | **0.164** |
| segmented | thr 50/128, votes 3 | 0.0000 | 0.295 | **0.000** |
| differential | — | — | — | dead at phase 1, never calibrated |

**Kill rule (ii) kills every CAM family, the baseline included.** It asks for
`P_d >= 0.9` at 40 dB-Hz while the rate holds, and the best is 0.498. The rule
was fixed before the runs, so do not reinterpret it now that it has killed the
study's own reference design. The FFT reference reaches 0.985 on the same
records through the same `score_events`, the same one sample code phase
tolerance and the same one bin Doppler tolerance, which is the control that rules
out the measurement as the explanation.

Three things the table does not show.

**The false alarm column is bimodal.** A setting false alarms on essentially
every record or on essentially none. The baseline goes from 1.0000 at
`thr 955, votes 3` to 0.0071 at `thr 944, votes 4`. The segmented family goes
from 1.0000 at 53 to 0.0000 at 50, which is three bits of 128. This is §4.6's
narrow usable band arriving as a decision rule cliff rather than as a distance
statistic.

**Phase 1's ranking did not survive, and phase 1 named the reason.** The
segmented family was second best on required tolerance, at 38.3% against the
baseline's 45.7%, with the plan warning that the figure was a lower bound because
its `D_true` is a minimum over 360 distances against the baseline's 24. It is
last here, at zero. `d'` transferred and the tolerance fraction did not.

**Code-only picked the baseline's exact setting.** Both families have 1 374 912
cells and the same measured floor of 997.5 ± 13.6 on 2046 bits, so they share a
grid. Its 0.164 against the baseline's 0.273 is the 1.2 dB phase 1 measured the
quadrant mixer to cost.

No family reported a single wrong fix, the segmented one included, which is what
`calibrate.RefinedCamCalibrator` exists to make true.

## Two corrections already applied

Do not re-derive these. Both are in the code and in the plan.

**`D_wrong`'s 25.5 bit offset from chance is the minimum over four query
rotations, not cross correlation.** `screen.py`'s docstring claimed a strong
satellite cross correlates with the other PRNs by a fixed amount. Measured on
records containing no satellites at all, the floor is 997.46 ± 13.62 against the
screen's 997.479 ± 13.630 on 60 real skies, so the cross correlation contribution
is under 0.05 bits. The choice of statistic was right and the explanation
attached to it was wrong.

**§4.5's gate cannot be met on set A by any setting of any family.** It asked for
the pick to be gated on the 95% Clopper-Pearson upper bound. With zero events in
`n` records that bound is 3/n whatever the setting, so it passes every zero event
setting or none according to `n` alone. Set A holds 140 records, where the bound
is 2.1e-2; reaching 1e-2 needs 300. `match_false_alarm` gained
`bound="measured"`, phase 3 picks on the measured rate, `trials_needed` prints
what the bound would have required, and the default of the function is unchanged.
§4.4's bound belongs on set B, where its 280 records are.

## What phase 4 has to build

`src/hdcam_gps/compare.py`: `compare(classifiers, scenarios, target_pfa) ->
Comparison`, running §4.3, §4.4 and §4.7 on set B and joining them into one
table. It reads the frozen settings with `calibrate.load_calibration` and applies
them with `FrozenSetting.apply`. `notebooks/family_comparison.ipynb` plots it.
Set B has not been touched and must stay untouched until then.

**Decide the scale before starting.** Set B is 40 skies at seven scalings, 280
records, which is twice phase 3. Per record on a four core container: code-only
25 s, segmented 64 s, thermometer 12 s, baseline 9 s, the FFT reference 0.5 s.
That is about eight hours. Either commit to it with per family checkpointing, as
`run_family_calibration.py` does, or cut the scaling list deliberately and say in
the plan that you did.

Two things make it cheaper than it looks. `third_party/cache/` holds the
simulator records, keyed on place and time alone, so a scaling costs nothing
extra. `decide_grid` is not needed here, because phase 4 runs one setting per
family rather than a grid.

## Conventions that are not obvious

`CLAUDE.md` carries the architecture rules and is current. Beyond it:

- The `technical-writing` skill in `.claude/skills/` governs every word,
  including commit messages.
- Tests mirror sources one to one as `tests/test_<module>.py`.
- Both study runs are checked in so that nobody repeats them. Read the JSON.
- Commit messages end with the `Co-Authored-By` and `Claude-Session` trailers.
  No model identifier goes into anything pushed to the repo.

## What to tell the human

Phase 3 changed the story, and phase 5 should lead with both halves of it. Phase
1 said the tolerance is three to four times too wide for silicon. Phase 3 adds
that the baseline CAM detects 27% of satellites where the correlator detects
98.5% at the same false alarm rate, which is a sensitivity result rather than a
buildability one.

The operating point is the obvious lever neither phase has pulled. The records
are 10 ms and the decision has nine looks. `d'` is 3.01 and 3.48 in the two bins
inside 38-42 dB-Hz, against the 6.25 the phase 1 verdict table was read at. A
longer dwell buys `d'`, and the plan should say so rather than leave it as an
implication.
