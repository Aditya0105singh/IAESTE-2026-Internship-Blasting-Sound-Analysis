# IAESTE 2026 — Blasting Sound Project
## Final Progress Report | Date: 2026-06-22

**Intern:** Aditya Singh (adityasingh01517@gmail.com)
**Institution:** VŠB-TU Ostrava, Czech Republic
**Project:** Signal Processing Pipeline for Industrial Blasting Sound Analysis

---

## Project Overview

Complete Python signal-processing and machine-learning pipeline for analyzing
audio and vibration recordings of an abrasive blasting machine. Recordings were
made under varying conditions (material, pressure, mix ratio, nozzle size) to
determine whether blasting parameters can be estimated from sound alone.

---

## Dataset Summary

| Property | Value |
|---|---|
| Total recordings | 1,568 WAV files |
| Total size | ~6 GB |
| Sessions | v24, v25, v26, v27, v28, v30, v31 (v29 not measured) |
| Format | Mono WAV, 48 kHz, 16-bit |
| Duration per file | 30–90 seconds |

### Session → Material Mapping
| Sessions | Material | Notes |
|---|---|---|
| v24, v25 | G80 (steel grit) | 2 sessions → enables cross-day validation |
| v26, v27 | GH40 | 2 sessions |
| v28 | GH18 | 1 session only |
| v30, v31 | GH120 | 2 sessions → enables cross-day validation |

### Sensors (4 per test)
| Sensor | Type | Frequency Range |
|---|---|---|
| Mic147EB | Microphone | 3.15 Hz – 20 kHz |
| Mic46BE | Microphone | 4 Hz – 80 kHz |
| AccAxial4507 | Accelerometer (axial) | 0.3 Hz – 6 kHz |
| AccRadial4507 | Accelerometer (radial) | 0.3 Hz – 6 kHz |

---

## Phase Completion Summary

### Phase 1 — Data Infrastructure ✅
**Files:** `src/iaeste26/parser.py`, `src/iaeste26/dataset.py`

- Filename parser extracts: material, nozzle_mm, pressure_bar, mix_ratio_pct, sensor
- WAV loader normalizes int16/int32 to float32 in [-1.0, 1.0]
- Dataset scanner produces a 1,568-row metadata CSV
- **Tests:** 13/13 passing

---

### Phase 2 — Preprocessing ✅
**File:** `src/iaeste26/preprocessing.py`

Pipeline: **trim 2s edges → bandpass filter → peak normalize**

| Sensor type | Bandpass range |
|---|---|
| Microphones | 20 Hz – 20,000 Hz |
| Accelerometers | 1 Hz – 6,000 Hz |

- **Tests:** 17/17 passing

---

### Phase 3 — Feature Extraction ✅
**File:** `src/iaeste26/features.py`

35 features per recording:

| Domain | Features | Count |
|---|---|---|
| Time | RMS, ZCR, Peak, Crest Factor, Kurtosis | 5 |
| Frequency | Centroid, Bandwidth, Rolloff-85%, Dominant Freq | 4 |
| MFCC | 13 coefs × (mean + std) | 26 |

MFCC built from scratch: scipy STFT → mel filterbank → log → DCT.
**Tests:** 25/25 passing

---

### Phase 4 — Visualization ✅
**File:** `src/iaeste26/visualization.py`
**Outputs:** `plots/*.png` (5 plots)

- Waveform + spectrogram comparison at 3 / 5 / 7.5 bar
- RMS vs Pressure (all sensors)
- RMS vs Mix Ratio (all sensors)
- Spectral Centroid vs Pressure
- Sensor comparison (same test, all 4 sensors)

---

### Phase 5 — Machine Learning ✅
**File:** `src/iaeste26/ml.py`
**Outputs:** `results/*.csv`

All models: Random Forest, 5-fold cross-validation, 1,568 samples × 35 features.

#### Core Results

| Task | Model | Metric | Score |
|---|---|---|---|
| Material classification (4 classes) | Random Forest | Accuracy | **79.5% ± 4.5%** |
| Pressure prediction | Random Forest | R² | **0.53 ± 0.07** |
| Mix ratio prediction | Random Forest | R² | 0.44 |

#### Multi-Model Comparison (5-fold CV)

| Model | Pressure R² | Material Accuracy |
|---|---|---|
| SVM (RBF) | **0.59** | **86.9%** |
| Random Forest | 0.53 | 79.5% |
| Gradient Boosting | 0.52 | 80.2% |

#### Sensor Comparison

| Task | Best Sensor | Score |
|---|---|---|
| Pressure prediction | Mic147EB | R²=0.59 |
| Material classification | AccRadial4507 | 92.1% accuracy |

#### Confusion Matrix (material classification)
```
           G80   GH120   GH18   GH40
G80        353      62      5     28    (78.8% correct)
GH120       51     379      4     14    (84.6% correct)
GH18        17      11    148     48    (66.1% correct)  ← hardest
GH40        39      21     21    367    (81.9% correct)
```

#### Cross-Session Generalisation (LOSO)

Key finding: each session recorded ONE material only, so full LOSO for material
classification is zero-shot (model never trained on test class) — not applicable.

Valid LOSO results:

| Evaluation | R² / Accuracy |
|---|---|
| Pressure: full LOSO (7 sessions) | R²=0.54 ± 0.18 — **generalises well** |
| Pressure: within-material LOSO (G80, GH120) | R²=0.31–0.46 — moderate drop |
| Mix ratio: within-material LOSO | R²=0.03–0.35 — session-sensitive |

#### Feature Importance
Top feature for pressure: **mfcc_2_mean** (23% importance).
MFCC features dominate — time-frequency texture is more informative than raw energy.

---

### Phase 6 — Technical Report ✅
**Output:** `report.ipynb` (37 cells, fully executed, 5.7 MB) + `report.html`

Sections:
1. Dataset Overview
2. Signal Visualization
3. Preprocessing
4. Feature Extraction
5. Analysis Plots
6. ML Results
7. Sensor Comparison
8. Feature Space (PCA)
9. Advanced Evaluation (confusion matrix, LOSO, multi-model comparison)
10. Conclusions

---

## Test Suite

| Module | Tests | Status |
|---|---|---|
| parser.py | 13 | ✅ All passing |
| preprocessing.py | 17 | ✅ All passing |
| features.py | 25 | ✅ All passing |
| ml.py | 35 | ✅ All passing |
| **Total** | **90** | **✅ 90/90** |

---

## File Structure

```
iaeste26-blasting-sound-main/
├── data/
│   ├── v24/WAV/   (224 files — G80)
│   ├── v25/WAV/   (224 files — G80)
│   ├── v26/WAV/   (224 files — GH40)
│   ├── v27/WAV/   (224 files — GH40)
│   ├── v28/WAV/   (224 files — GH18)
│   ├── v30/WAV/   (224 files — GH120)
│   └── v31/WAV/   (224 files — GH120)
├── src/iaeste26/
│   ├── __init__.py
│   ├── parser.py          Phase 1 — filename parsing + WAV loading
│   ├── dataset.py         Phase 1 — dataset scan + CSV export
│   ├── preprocessing.py   Phase 2 — trim, filter, normalize
│   ├── features.py        Phase 3 — 35 features per recording
│   ├── visualization.py   Phase 4 — waveforms, spectrograms, plots
│   └── ml.py              Phase 5 — ML pipeline + evaluation
├── scripts/
│   ├── generate_plots.py       Phase 4 — generate 5 plots
│   ├── run_ml.py               Phase 5 — ML on v24 (fast test)
│   ├── run_ml_full.py          Phase 5 — ML on all 1568 files
│   ├── run_advanced.py         Advanced evaluation
│   ├── run_corrected_loso.py   Corrected cross-session evaluation
│   └── build_notebook.py       Phase 6 — notebook generator
├── tests/
│   ├── test_parser.py          13 tests
│   ├── test_preprocessing.py   17 tests
│   ├── test_features.py        25 tests
│   └── test_ml.py              35 tests
├── plots/                 5 PNG analysis plots
├── results/               CSV results (feature matrix, ML, sensor comparison)
├── report.ipynb           Final technical report (executed, 37 cells)
├── report.html            HTML export of report (for sharing/submission)
├── dataset_metadata.csv   1,568-row metadata
├── requirements.txt       All dependencies
├── README.md              Quick-start guide
└── PROGRESS_REPORT.md     This document
```

---

## Environment

| Item | Value |
|---|---|
| Python | 3.12.3 |
| Key libraries | numpy, scipy, pandas, scikit-learn, matplotlib, jupyter |
| Activate | `.\.venv\Scripts\Activate.ps1` + `$env:PYTHONPATH = "src"` |
| Run tests | `pytest tests/ -v` |
| Open report | `jupyter notebook report.ipynb` |
| View report | Open `report.html` in any browser (no Jupyter needed) |

---

## Key Technical Decisions

1. **MFCC from scratch** — scipy only (no librosa), making the math transparent and dependencies minimal.
2. **Sensor-specific bandpass filters** — mic/accelerometer have different useful frequency ranges.
3. **2-second edge trimming** — removes start/stop transients from blasting.
4. **15-second middle window for ML** — balances speed (vs full 80s files) with feature stability.
5. **SVM recommended over Random Forest** — SVM (RBF) consistently outperforms RF and GBM.
6. **5-fold CV is the valid evaluation for material classification** — LOSO is inapplicable because each session = one material (zero-shot, not domain adaptation).
7. **Pressure LOSO is valid** — pressure varies in all sessions, R²=0.54 confirms cross-session generalisation.
