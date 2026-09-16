# GPS L1 C/A acquisition as a 1-bit HD-CAM lookup

## Idea

Acquisition is a search over (PRN, Doppler, code phase). We store every (PRN, Doppler)
replica, quantized to the sign of I and Q, as one row of a Hamming-distance-tolerant CAM
(HD-CAM: concept in Garzón et al., IEEE Access 2022 [18]; silicon in Garzón et al.,
JSSC 60(8):3009, Aug 2025 -- a 128-kbit macro in 65 nm, 512 x 64-bit words per bank,
125 MHz, 0.19 fJ/bit/search, HD tolerance 1-8 bits). The received record is quantized the same way and one code
period at a time is presented as a query; the CAM returns every row within a Hamming
threshold in a single parallel search, and the window offset gives the code phase.
Carrier phase is covered by four bit-level quarter-turn rotations of the query plus two
stored phases per row. Votes across code periods give non-coherent accumulation; a second
pass ranks survivors by bisecting on the threshold.

The claim to be tested is not algorithmic. Per query the CAM performs an exhaustive 1-bit
time-domain correlation against all rows, so its bit-operation count exceeds an FFT search
by roughly N / log2 N (about 190x at 2046 bits per row). The bet is that in-memory search at
sub-fJ/bit beats digital FFT/correlator ASICs by more than that factor, with no data
movement. **If it does not, there is no result.**

## The two algorithms

**FFT parallel code-phase search (the nominal baseline).** For each Doppler hypothesis the
record is mixed down by a local carrier and one forward FFT is taken per code period; one
complex multiply by the conjugate code spectrum and one inverse FFT then yield the
correlation at *all N code phases at once*. Magnitudes are squared and summed over the
code periods, and a PRN is declared when the peak of the (Doppler x code phase) grid
exceeds its own second peak by a set ratio. Code phase is obtained for free by the
transform; PRN and Doppler are the loop. Arithmetic is full-precision complex throughout,
so nothing is lost to quantization, and the cost is O(n_PRN x n_Doppler x n_codes x N log N).

```
code_fft[prn] = conj(FFT(replica(prn)))                 # once per PRN, offline

for prn in PRNs:                                        # serial
    for f in doppler_grid:                              # serial
        z      = samples * exp(-j2pi f t)               # carrier wipe-off
        blocks = reshape(z, n_codes, N)
        corr   = IFFT(FFT(blocks) * code_fft[prn])      # <-- all N code phases at once
        grid[f, :] = sum_over_blocks(|corr|^2)          # non-coherent accumulation
    f*, tau* = argmax(grid)
    if grid[f*, tau*] / second_peak(grid[f*, :]) >= ratio:
        report(prn, f*, tau*)
```

**1-bit HD-CAM lookup (proposed).** One code period of the record is quantized to
sign(I) || sign(Q) and presented as a query; a single search returns every (PRN, Doppler)
row inside the Hamming threshold, so *that* dimension is the one obtained for free. Code
phase becomes the loop instead: the window slides one sample at a time over n_starts
positions (9208 for a 10 ms record at 1.023 MHz), each presented in 4 bit-level quarter
turns to cover carrier phase. Matches are accumulated by voting across code periods, and
the survivors are ranked by bisecting the threshold to recover their distances. Arithmetic
is Hamming distance on bits: no multipliers, no transforms.

```
for prn, f, phi in PRNs x doppler_grid x stored_phases:                  # offline
    CAM[row(prn, f, phi)] = quantize(replica(prn) * exp(j2pi f t + j phi))   # 2N bits

for start in 0 .. n_starts-1:                           # serial: code phase
    q   = quantize(samples[start : start+N])
    tau = start mod N
    hits = union over r in 0..3 of CAM.search(rotate(q, r))   # <-- all PRN x Doppler at once
    for row in hits:
        votes[row, tau] += 1                            # one vote per code period
        kept[row, tau].append(rotate(q, r))

for row, tau where votes[row, tau] >= min_votes:        # survivors only, a handful
    d[row, tau] = min over kept[row, tau] of bisect_threshold(row, query)

for prn in PRNs:
    row*, tau* = argmin d over the rows of prn
    report(prn, doppler_of(row*), tau*)
```

The two therefore exchange which axis is parallel — the FFT collapses code phase, the CAM
collapses PRN and Doppler — and that exchange is the whole proposal. It is not an
algorithmic win: as noted above, the CAM does an exhaustive time-domain correlation per
query and pays roughly N / log2 N more bit-operations than the transform. It is a bet that
those operations, performed in-memory with no data movement, are cheap enough per bit to
win anyway. The quantization is the price paid for making them 1-bit: about 2 dB for
keeping only the sign, about 2.5 dB for resolving carrier phase into only 8 hypotheses.

## Where we stand (honest)

- Simulation only, NumPy, ideal CAM (exact Hamming distance, exact threshold). Nothing
  here has been through a circuit model.
- Evaluated on 3 PRNs, ±1 kHz Doppler, 1 satellite. This is not statistically meaningful
  and not the full 32-PRN, ±5 kHz problem.
- Detection is close to the reference once votes accumulate across code periods: 38 dB-Hz
  for the FFT baseline against 40 dB-Hz for the CAM at 10 ms, a gap of about 2 dB, which is
  the order the 1-bit quantization should cost.
- False alarms are the open problem, and the analytic threshold model is the reason. It
  assumes chance-level cross-correlation, which is wrong when a strong satellite is
  present, and it chose a vote rule (3 of 9) that raises a false alarm on *every* trial;
  measurement found 6 of 9 with none in 40 trials, at 7.5 % missed detections at 40 dB-Hz.
  The model could not see that cliff. `calibrate.py` now searches the threshold and vote
  rule against a stated (C/N0, P_md, P_fa) operating point instead of assuming one.
- Confirming P_fa <= 1e-4 by counting events needs about 30 000 trials (rule of three).
  40 trials only bound it at 7 %, so the target is currently **not demonstrated**, and the
  calibrator correctly refuses to claim it.
- The FFT and CAM classifiers have still not been compared at matched P_fa, so the 2 dB
  figure above is provisional.

## Risks the proposal must answer

Numbers below are from the JSSC 2025 silicon unless stated otherwise.

1. **The required HD tolerance is a large fraction of the word. This is the decisive
   question.** A 1-bit quantized GPS signal disagrees with its own replica on close to
   half the bits, because per-sample SNR is far below 0 dB: measured true-match distances
   are 856/2046 (42 %) at 48 dB-Hz and 902/2046 (44 %) at 45 dB-Hz, and the threshold in
   use sits at 956/2046 (47 %). Even with no noise at all, carrier-phase quantization alone
   puts a worst-placed true match at 512/2046 (25 %). The silicon demonstrates a tolerance
   of 8 bits in a 64-bit word (12.5 %), with [18] arguing a theoretical ceiling of about
   half the word width. So this application needs the extreme end of the theoretical range,
   3-4x beyond anything measured, and precisely in the regime the paper identifies as
   fragile: large tolerance requires low Veval, and as Veval approaches the threshold
   voltage of Meval the cell becomes variation-sensitive and sensitivity falls (measured
   sensitivity std dev rises from 1.6 at HD 2 to 3.5 at HD 5).
   **Shortening the word does not help** — the fraction is set by SNR, so a 64-bit word
   would need a tolerance of about 27 bits rather than 8. If tolerance near half the word
   width is not reachable in silicon, the approach does not work for weak-signal GPS at any
   word width, and the project should stop here.

2. **Row width and ML integration.** The fabricated bank is 512 rows x 64 bits. We need
   2046-bit rows (1 sample/chip) or 8000-bit rows (4 MHz), a 32x to 125x wider ML with
   proportionally larger capacitance and slower discharge. This cannot be assembled by
   tiling 64-bit banks: the ML is a wired NOR whose discharge integrates every mismatching
   cell in the word, so per-bank results would have to be summed digitally, which discards
   the analog HD summation that is the whole point. Separately, the *resolution* needed is
   to distinguish HD 956 from the 1023 chance floor, i.e. 67 bits in 2046 (3.3 % of the
   word), against a measured transition width of roughly 3-4 bits in 64 (5-6 %). Relative
   precision is therefore in the right ballpark but not yet sufficient, and whether it holds
   at 32x width is unknown. Needs a Monte Carlo ML model at the target width, not a
   Python `<=`.

3. **Area.** Full search: 32 PRN x 21 bins x 2 phases = 1344 rows x 2046 bits = 2.75 Mbit.
   At the 3.24 um^2 bitcell that is 8.9 mm^2 of array; at the measured macro density
   (0.21 mm^2 per 32-kbit bank, peripherals included) it is 17.6 mm^2, against 0.64 mm^2
   for the fabricated 128-kbit macro. At 4 MHz it is 69 mm^2. A ±10 kHz cold-start grid
   doubles it again. This may already exclude the target application, and is the strongest
   argument for the reduced-row designs in step 4 of the plan.

4. **Energy.** The measured figure is 0.19 fJ/bit/search (0.76 mW per bank search at
   125 MHz), and unlike the earlier estimate it *includes* search-line drive: the reported
   breakdown is 37.4 % SL circuitry, 54.5 % ML sensing and precharge, 6.4 % array and write,
   1.75 % leakage. One 10 ms acquisition is 9208 windows x 4 rotations = 36 832 searches
   over 2.75 Mbit, so about **19 uJ**, at a latency of 36 832 x 8 ns = 295 us. Against an
   RF front end costing roughly 300 uJ for the same 10 ms, acquisition energy would be a
   small fraction of the receiver budget. This is the encouraging number in the proposal,
   and it is worth nothing at all unless risks 1-3 are answered first.

5. **Second pass.** `tightest_match` reprograms the analog tolerance about 11 times per
   candidate by moving Veval or Vref. The paper notes that dynamic adjustment of these
   voltages carries energy and latency overhead beyond static tuning, because it requires
   real-time sensitivity tracking. Settling time and threshold repeatability under
   reprogramming are unmodelled here.

6. **Signal effects not yet modelled.** 1-bit quantization loss (~2 dB theory, more with a
   1-bit replica), navigation-bit transitions inside the 10 ms record, multi-satellite
   near-far cross-correlation, and receiver clock error moving the true Doppler off-grid.

7. **Prior art.** 1-bit correlation and memory-based parallel correlators are standard in
   commercial receivers; the novelty is only the CAM. A patent/literature search for
   CAM- or in-memory-based GNSS acquisition is required before any claim of novelty.

## Baselines to compare against

| Baseline | Why |
|---|---|
| FFT parallel code-phase search (Van Nee & Coenen), same 1-bit input, same 10 ms | Algorithmic reference, already in repo |
| Dedicated 1-bit matched-filter / FFT acquisition ASIC at 65 nm (literature energy/area, e.g. 1-3.5 mW/channel class) | Fair hardware comparison |
| Snapshot receiver with cloud acquisition (SnapperGPS) | The low-power system-level alternative: shows whether acquisition energy is the bottleneck at all |
| Commercial receiver acquisition engine (u-blox class, ~67 mW / 50 channels) | Sanity bound |

## Metrics

- **Sensitivity**: C/N0 at 90 % detection and P_fa <= 1e-4 per acquisition, per integration time (1, 10, 20 ms). Everything at matched P_fa or it is not a comparison.
- **Energy per acquisition (uJ)** and **area (mm^2)** for the full 32-PRN, ±5 kHz search, from a circuit-level model that includes search-line energy and the second pass.
- **Latency** (time to first fix contribution) at a stated clock.
- **Robustness**: P_d / P_fa degradation vs process corner and threshold error (bits).
- Secondary: Doppler and code-phase RMSE, handover quality to tracking.

## Targets and plan

1. **Statistical evidence** (now): full 32 PRN, ±5 kHz, >= 1000 scenarios per point via
   `distance_table`, calibrated threshold and vote rule, matched-P_fa comparison with FFT.
   Target: within 1 dB of the 1-bit FFT reference at equal P_fa; if worse by > 2 dB, stop.
2. **Tolerance feasibility, in parallel with step 1 and gating everything after it**: can
   an HD tolerance of about 45 % of the word width be reached at 2046 bits, at what Veval,
   and with what sensitivity spread across corners? Risk 1 makes this a stop condition, and
   it is a circuit question that does not wait on more Python. Then the full
   hardware-faithful model: threshold error and variation sampled from silicon
   measurements, extended to >= 2046-bit words. Target: sensitivity loss < 1 dB at the TT
   corner, < 2 dB across corners.
3. **Energy/area model** vs the 65 nm FFT ASIC baseline. Go/no-go: >= 10x lower energy per
   acquisition at <= 2x area, including the front end in the system budget. Below that
   the work is not worth a tape-out.
4. Only then: reduced-row designs (Doppler compensation in the query, fewer stored phases)
   to attack the area problem, and a real-sky dataset.
