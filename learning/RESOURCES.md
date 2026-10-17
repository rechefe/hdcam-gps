# HD-CAM GPS acquisition: resources

Verified through search-index excerpts of publisher, arXiv and repository pages on 2026-10-17. Full texts could not be opened from the cloud session, so items marked "check full text" still need it.

## Knowledge

- [Book: Borre et al., _A Software-Defined GPS and Galileo Receiver_ (Birkhäuser, 2007)](https://doi.org/10.1007/978-0-8176-4540-3)
  The standard reference for serial, parallel-frequency and parallel-code-phase acquisition. Use for: anything about correlation-based acquisition, C/N0, integration.
- [Paper: van Nee & Coenen, "New fast GPS code-acquisition technique using FFT", Electronics Letters 27(2), 1991]
  The FFT parallel code-phase search that `fft_acq.py` implements. Use for: the reference classifier.
- [Paper: Garzón et al., "HD-CAM for DNA Classification", IEEE Access 10, 2022](https://doi.org/10.1109/ACCESS.2022.3158305)
  The HD-CAM concept, simulation only; preprint claims tolerance of ~60% of the pattern. Use for: what an HD-CAM is and how it senses distance.
- [Paper: Garzón et al., "A 128-kbit Approximate Search-Capable CAM With Tunable Hamming Distance", IEEE JSSC 60(8), 2025](https://doi.org/10.1109/JSSC.2025.3529715)
  The 65 nm silicon the study measures against. Use for: tolerance, energy and latency figures. Check full text for 12.5% (8 of 64 bits), 512 × 64 banks, 0.19 fJ/bit.
- [Review: Garzón et al., "Approximate Content-Addressable Memories: A Review", Chips 2(2), 2023](https://doi.org/10.3390/chips2020005)
  Survey of approximate CAMs. Use for: the design space beyond this group's macro.
- [Paper: Harary et al., "PiC-BNN", arXiv:2601.19920, 2026](https://arxiv.org/abs/2601.19920)
  Same HD-CAM silicon used as a statistical classifier. Use for: the closest published use of HD tolerance as a detector.
- [Paper: Beuchert & Rogers, "SnapperGPS", ACM SenSys 2021](https://doi.org/10.1145/3485730.3485931) and [J. Open Hardware 2023](https://doi.org/10.5334/joh.48)
  Snapshot receiver: 12 ms, 4 MHz, 1-bit, cloud solve. Use for: the system-level low-power baseline.
- [Press release: CCSA, ISSCC 2023, Shanghai Jiao Tong University](https://english.seiee.sjtu.edu.cn/english/info/35546.htm)
  Mixed-signal GPS acquisition accelerator, 28 nm, 114–394 TOPS/W. Use for: the strongest silicon baseline and analog-correlation prior art. DOI not found.
- [Paper: Lin Wei et al., in-memory navigation signal acquisition, J. Guangdong Univ. of Technology](https://xbzrb.gdut.edu.cn/en/article/doi/10.12052/gdutxb.250157)
  RRAM-crossbar GPS acquisition, 0.94 mW. Use for: prior art on in-memory acquisition.
- [Patent: Trimble US 6,091,785, memory-based search](https://patents.google.com/patent/US6091785)
  Store 1 ms, compare against stored replicas, threshold. Use for: the conceptual ancestor of the CAM lookup.
- [Survey: Grenier et al., "A Survey on Low-Power GNSS", IEEE COMST 25(3), 2023](https://doi.org/10.1109/COMST.2023.3265841)
  Receiver energy, onboard vs snapshot. Use for: where acquisition sits in the power budget.
- [Paper: Stein et al., arXiv:1412.3964](https://arxiv.org/abs/1412.3964) and MITRE report 09-3995
  Hard-limiting loss 2/π = 1.96 dB (2 bits: 0.55 dB). Use for: quantisation losses.

## Wisdom (Communities)

- Not yet chosen. Candidates to evaluate: the ION GNSS+ conference community, and the authors of the HD-CAM papers (Bar-Ilan University) for the silicon questions in PROPOSAL step 2.

## Gaps

- A 65 nm FFT or matched-filter acquisition ASIC with a published energy or mW/channel figure. PROPOSAL's "1–3.5 mW/channel, 65 nm" has no matching source.
- The JSSC 2025 full text, for every silicon number the study relies on.
- A patent search for CAM-based GNSS acquisition (PROPOSAL risk 7).
