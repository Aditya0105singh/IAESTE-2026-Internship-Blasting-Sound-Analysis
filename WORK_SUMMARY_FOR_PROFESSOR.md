# IAESTE 2026 Internship — Summary of Work

**Intern:** Aditya Singh
**Supervisor:** Prof. Tomas Fryza, Department of Radio Electronics, Brno University of Technology
**Host institution:** VŠB-TU Ostrava, Czech Republic
**Project:** Signal Processing Pipeline for Industrial Blasting Sound Analysis
**Date:** 2026-09-03

---

## 1. What the project set out to do

Build a complete Python signal-processing pipeline that analyzes audio and vibration recordings of an abrasive blasting machine, and use those recordings to estimate the operating conditions (pressure, mix ratio, nozzle size, abrasive type) from sound and vibration alone.

The dataset: **1,568 WAV recordings** (~5.6 GB) from 4 sensors recorded simultaneously — two microphones (Gras 147EB, Gras 46BE) and two accelerometers (Brüel&Kjaer 4507, radial and axial) — across 7 measurement sessions covering 4 abrasive materials, 2 nozzle diameters, 8 pressure settings, and 7 mix ratios.

---

## 2. Work completed, phase by phase

### Phase 1 — Data infrastructure
- Filename parser extracting material, nozzle size, pressure, mix ratio, and sensor from each recording's name.
- WAV loader normalizing 16-/24-bit audio to float32 in [-1, 1].
- Dataset scanner producing a metadata CSV covering all 1,568 files.
- **13 unit tests**, all passing.

### Phase 2 — Preprocessing
- Pipeline: trim 2-second start/end transients → sensor-specific bandpass filter → peak normalize.
- Microphones filtered 20 Hz–20 kHz; accelerometers 1 Hz–6 kHz, reflecting each sensor's real frequency range.
- **17 unit tests**, all passing.

### Phase 3 — Feature extraction
- 35 features per recording, spanning three domains:
  - **Time domain (5):** RMS, zero-crossing rate, peak amplitude, crest factor, kurtosis.
  - **Frequency domain (4):** spectral centroid, bandwidth, 85% rolloff, dominant frequency.
  - **Time-frequency (26):** 13 MFCC coefficients × (mean + std), built from scratch (STFT → mel filterbank → log → DCT) using only `scipy`, with no dependency on external audio libraries.
- **25 unit tests**, all passing.

### Phase 4 — Visualization
- Waveform and spectrogram comparisons across pressure settings.
- RMS and spectral centroid vs. pressure and mix ratio, across all 4 sensors.
- Cross-sensor comparison plots for the same physical event.
- 5 analysis plots generated (`plots/*.png`).

### Phase 5 — Machine learning
- Random Forest models with 5-fold cross-validation across all 1,568 files × 35 features.
- **Material classification (4 classes): 79.5% ± 4.5% accuracy** (chance level = 25%).
- **Pressure prediction: R² = 0.53 ± 0.07, RMSE = 0.97 bar.**
- Mix ratio prediction: R² = 0.44.
- Multi-model comparison (Random Forest vs. Gradient Boosting vs. SVM): SVM (RBF) was the strongest performer — R² = 0.59 for pressure, 86.9% accuracy for material — ahead of Random Forest and Gradient Boosting.
- Sensor comparison: **Mic147EB is the best single sensor for pressure** (R² = 0.59); **AccRadial4507 is the best single sensor for material identification** (92.1% accuracy).
- Cross-session generalisation (leave-one-session-out): pressure prediction generalises well across recording days (R² = 0.54 ± 0.18); material classification's LOSO is not meaningful here since each session recorded only one material, making it a zero-shot rather than a domain-adaptation problem — 5-fold CV is the valid evaluation for that task instead.
- Feature importance: MFCC coefficients dominate — `mfcc_2_mean` alone carries 23% of the predictive weight for pressure, meaning the fine time-frequency texture of the sound is more informative than raw loudness.
- **35 unit tests**, all passing.

### Phase 6 — Technical report
- `report.ipynb` — 37 cells, fully executed, 5.7 MB, covering dataset overview, signal visualization, preprocessing, feature extraction, analysis plots, ML results, sensor comparison, a PCA feature-space view, and the advanced evaluation (confusion matrix, LOSO, multi-model comparison).
- Exported to `report.html` for viewing without Jupyter.

**Total test suite: 90/90 passing.**

---

## 3. Final verification and quality pass (this week)

Before this update, I ran a complete, independent audit of the entire repository — every claim, every script, every notebook — rather than taking the existing results at face value.

**What I confirmed as already correct:**
- All 90 unit tests pass.
- The 35-feature extraction matches its documentation exactly.
- The dataset count (1,568 files) matches the metadata CSV exactly.
- The headline ML numbers — 79.5% material accuracy, R² = 0.53 for pressure, and the full confusion matrix — were **independently re-derived from the raw feature matrix and matched exactly**, cell for cell, confirming these results are genuine and reproducible, not just reported.
- The official `report.ipynb` was correct and fully executed from the start.

**What I found and fixed**, all confined to newer supplementary demonstration notebooks built to visually prove the pipeline's inner workings (not the official report, and not the core results above):

1. A file-scanning bug that double-counted every WAV file (1,568 reported as 3,136) due to a case-insensitive filesystem matching the same files twice.
2. A memory-management bug — matplotlib figures were never released after being drawn, causing crashes under load.
3. A numeric-overflow bug where an unstable high-order filter produced an infinite value that crashed the plotting engine.
4. A code bug comparing NumPy arrays with Python's `and` operator instead of explicit `is not None` checks.
5. A block of leftover, broken code with mismatched brackets causing a syntax error.
6. A genuine filter-theory error in a validation test: it asserted an elliptical filter is always sharper than a Bessel filter in the stopband, tested at a point where that happens to be false (elliptical filters plateau at their design attenuation, while Bessel keeps decaying) — verified numerically and corrected to test at the point where the claim is actually true.
7. An impractically large workload in the multi-file demonstration notebooks (millions of redundant computations) — scaled down to a representative sample so the notebooks run in reasonable time without changing what they prove.
8. Two outdated lines in the documentation (a stale test count, and a file-duration range narrower than the actual measured data).

All fixes are verified: **13 of 14 notebooks now execute end-to-end with zero errors and zero failed internal checks.** The 14th (`validation_report.ipynb`) is code-correct and was proven to run cleanly once already; its most recent run was interrupted by a memory-constrained machine, not a code issue, and will be finalized shortly.

---

## 4. Repository status

Both repositories are up to date:
- My own repository, with one clean, fully-documented commit covering this verification pass.
- Your repository (`tomas-fryza/iaeste26-blasting-sound`), updated with the corrected `README.md`, `PROGRESS_REPORT.md`, `blasting_analysis.ipynb`, `proof_of_plots.ipynb`, all 11 validation notebooks, and the underlying results CSVs.

---

## 5. In short

The pipeline works, the results are real and reproducible, and I've now verified that end-to-end rather than assuming it. Material type can be identified from sound/vibration with 79.5–92% accuracy depending on sensor, and blasting pressure can be predicted with R² ≈ 0.53–0.59, with the microphone sensors outperforming accelerometers for pressure and the radial accelerometer outperforming both microphones for material identification. MFCC-based time-frequency features are consistently the most informative signal representation across every task tested.
