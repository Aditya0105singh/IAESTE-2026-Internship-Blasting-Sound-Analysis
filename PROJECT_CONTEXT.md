# IAESTE 2026 Internship — Blasting Sound Project
## Context Document (for AI / new sessions)

---

## 1. What This Project Is

This is an **IAESTE internship project** at **VŠB-TU Ostrava, Czech Republic**.

The goal is to build a **Python signal-processing pipeline** that analyzes audio and vibration recordings of an **abrasive blasting machine** operating under different conditions — and use those recordings to estimate operating conditions (pressure, mix ratio, nozzle size, abrasive type) via signal features and machine learning.

---

## 2. Dataset

**Location:** `data/` folder — 8 session folders (`v24` to `v31`; `v29` was not measured)

**Total:** 1568 WAV files, ~6 GB

### Audio Format
- Format: WAV, Mono
- Sample rate: 48 kHz (some 96 kHz)
- Bit depth: 16-bit (some 24-bit)
- Duration: 30–90 seconds per file

### Sensors (4 per recording session)
| Sensor Name | Type | Direction |
|---|---|---|
| `Mic147EB` | Microphone (Gras 147EB) | — |
| `Mic46BE` | Microphone (Gras 46BE) | — |
| `AccAxial4507` | Accelerometer (B&K 4507) | Axial |
| `AccRadial4507` | Accelerometer (B&K 4507) | Radial |

### Experimental Parameters
| Parameter | Values |
|---|---|
| Abrasive material | G80, GH18, GH40, GH120 |
| Nozzle diameter | 6 (6.4 mm), 8 (8.0 mm) |
| Pressure (bar) | 3, 4, 5, 5.5, 6, 6.5, 7, 7.5 |
| Mix ratio (%) | 0, 30, 40, 50, 60, 70, 100 |

### File Naming Convention
```
Material_Nozzle_Pressure_MixRatio_Sensor.wav
```
Example: `G80_8_3_50_Mic46BE.wav`
- Material: G80
- Nozzle: 8 mm
- Pressure: 3 bar
- Mix ratio: 50%
- Sensor: Mic 46BE

> **Important:** The README describes 7 fields in the filename, but actual files only have 5 fields (no repetition serial or extra field). This is a known discrepancy.

---

## 3. Official Assignment (from internship PDF CZ-2026-100002)

### What to build:
- A complete **signal-processing pipeline in Python**
- Work with WAV files from multiple microphones and accelerometers

### Pipeline steps:
1. **Understand the measurement setup** — sensor types, recording conditions, parameter variations
2. **Preprocess raw signals** — filtering, segmentation, normalization, synchronization
3. **Extract features** — in time domain, frequency domain, and time-frequency domain
4. **Condition estimation** — estimate operating conditions (pressure, mix ratio, etc.) from signal features
5. **Change detection** — detect changes in system behavior over time
6. **Sensor comparison** — compare how different sensors capture the same events

### Methods to explore:
- **Supervised** — regression/classification using condition labels (available from filenames)
- **Unsupervised / Statistical** — change-detection methods when labels are not used

### Final deliverables:
- Well-documented, reproducible Python framework
- Set of extracted features + evaluation of their relevance
- Comparative results across sensors and signal representations
- Analysis of method strengths and limitations
- **Concise technical report** — methodology, experiments, results, conclusions, with plots and quantitative metrics

---

## 4. Current Code State

The Python package is an **empty scaffold** — no processing logic exists yet.

```
src/
  iaeste26/
    __init__.py    # just exports run()
    main.py        # run() only prints "iaeste26 package is ready."
scripts/
  run_example.py   # calls run()
```

**What's missing:** Everything — parser, loader, preprocessor, feature extractor, analysis, ML.

**Package layout:** `src/` layout. No `pyproject.toml` yet, so must set `PYTHONPATH=src` to use the package.

---

## 5. Environment Setup (already done)

- **Python:** 3.12.3
- **venv:** `.venv/` in project root (already created)
- **Installed:** `pytest` (only requirement so far)

### To activate venv (run this at start of every session):
```powershell
# In PowerShell, from project root:
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "src"
```

### Still needs to be installed (for actual work):
```powershell
pip install numpy scipy matplotlib pandas librosa scikit-learn jupyter
```

---

## 6. Development Roadmap

### Phase 1 — Parser + Loader (Start Here)
- Parse filename → extract all metadata fields as a Python dict
- Load WAV file → return numpy array + sample rate
- Scan all files → build a metadata DataFrame (CSV export)

### Phase 2 — Preprocessing
- Bandpass filter (keep 20 Hz – 20 kHz for mics; appropriate range for accelerometers)
- Normalize amplitude
- Segment recordings (e.g. ignore first/last 2s to avoid transients)

### Phase 3 — Feature Extraction (core work)
- **Time domain:** RMS energy, Zero-Crossing Rate, peak amplitude, kurtosis
- **Frequency domain:** FFT spectrum, dominant frequency, spectral centroid, spectral bandwidth
- **Time-frequency:** STFT Spectrogram, MFCCs (Mel-Frequency Cepstral Coefficients)

### Phase 4 — Analysis & Visualization
- Plot: how features change with pressure, mix ratio
- Plot: compare mic vs accelerometer responses
- Plot: spectrograms for different conditions

### Phase 5 — ML / Condition Estimation
- Predict pressure / mix ratio from extracted features (regression)
- Classify abrasive type or nozzle size (classification)
- Try: Random Forest, SVM, gradient boosting (scikit-learn)
- Evaluate: MAE, RMSE for regression; accuracy/F1 for classification

### Phase 6 — Report
- Jupyter notebook with all experiments documented
- Final technical report (PDF)

---

## 7. Key Facts to Remember

- Dataset is **self-labeled** via filenames — supervised ML is straightforward
- 4 sensors record simultaneously for each test → can compare them directly
- v29 session was intentionally skipped (large abrasive + small nozzle = flow problems)
- Each "version" (v24–v31) is a separate measurement day/session
- The project emphasizes **systematic experimentation + documentation**, not just code

---

## 8. File Locations

| Item | Path |
|---|---|
| Project root | `D:\IAESTE\iaeste26-blasting-sound-main\` |
| Data | `D:\IAESTE\iaeste26-blasting-sound-main\data\` |
| Source code | `D:\IAESTE\iaeste26-blasting-sound-main\src\iaeste26\` |
| Scripts | `D:\IAESTE\iaeste26-blasting-sound-main\scripts\` |
| venv | `D:\IAESTE\iaeste26-blasting-sound-main\.venv\` |
| Internship PDF | `D:\placement\IAESTE INTERNSHIP CZECH\CZ-2026-100002.pdf` |
