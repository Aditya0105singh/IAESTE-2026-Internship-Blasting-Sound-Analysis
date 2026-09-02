# iaeste26 - Blasting Sound Database

### Overview

All recordings for the development database were captured at the VŠB-TU Ostrava experimental facility in collaboration with researchers from BUT Brno.

Acoustic signals within the 20–20,000 Hz range—as well as adjacent infrasound and ultrasound frequencies—were recorded in two ways: **Acoustic blast sounds** via microphones. and **Nozzle inlet vibrations** via accelerometers.

* **Microphones:** Two models (Gras 147EB and 46BE) with different frequency ranges and sensitivities were used. All sessions recorded from both microphones simultaneously.

* **Accelerometers:** Recordings used a pair of identical sensors 4507, where one positioned radially and the other axially.

* **Data Acquisition:** A 4-channel sound card enabled synchronous recording of every blasting test across all 4 independent sensors (2x microphones, 2x accelerometers).

### Audio Format Specifications

* **Format:** Standard WAV (Mono)
* **Sampling Rate:** 96 kHz or 48 kHz
* **Bit Depth:** 24-bit or 16-bit


## Technical Equipment Specifications


Alongside a laptop, the following hardware was used for data collection:


| Device | Type | Frequency Range | Sensitivity | Datasheet |
|---|---|---|---|---|
| Gras 147EB | Microphone | 3.15 Hz – 20 kHz (±2 dB) | 50 mV/Pa | [www.grasacoustic.com](https://www.google.com/search?q=https://www.grasacoustic.com) |
| Gras 46BE | Microphone | 4 Hz – 80 kHz (±2 dB) | 3.6 mV/Pa | [www.grasacoustic.com](https://www.google.com/search?q=https://www.grasacoustic.com) |
| Brüel&Kjaer 4507-B-004 | Accelerometer | 0.3 Hz – 6 kHz | 1 mV/ms⁻² | [hbkworld.com (PDF)](https://media.hbkworld.com/m/6d9fc3d1d2857ec8/original/Piezoelectric-Accelerometer-Types-4507-and-4508.pdf) |


## Database Structure

Recordings were collected based on variations of three parameters:

* **Abrasive Types:** Steel Grit Gxx.
* **Nozzle Diameters (2):** 6.4 mm and 8.0 mm.
* **Pressure & Mix Ratios:** * Pressure ranged from 3 to 7.5 bars (adjusted in steps of 0.5 or 1 bar).
* The abrasive-to-air mix ratio ranged from 0% (pure compressed air) to 100% (adjusted in steps of 10% or 30%).


### Dataset Summary

File Duration is 28 to 90 seconds (determined by how quickly the abrasive container emptied based on pressure, mix ratio, and nozzle size). Blasting parameters remained constant during each individual recording. Some tests were repeated across different days.


### File Naming Convention

To facilitate automated processing, a standardized file-naming key was implemented:

`111_222_333_444_555_666_777.wav`

* **111:** Abrasive material abbreviation
* **222:** Nozzle diameter (mm)
* **333:** Pressure (bar)
* **444:** Mix ratio (%)
* **555:** Signal / Sensor name
* **666:** Test repetition serial number
* **777:** Optional/additional information

**Example:** A file named `G80_8_3_50_Mic46BE.wav` decodes as:

* **Material:** Steel Grit G80
* **Nozzle Diameter:** 8 mm
* **Pressure:** 3 bar
* **Mix Ratio:** 50%
* **Sensor:** Microphone 46BE

---

## Signal Processing Pipeline (IAESTE 2026)

This repository includes a complete signal-processing and machine-learning pipeline
built during the IAESTE 2026 internship at VŠB-TU Ostrava.

### Quick Start

```powershell
# 1. Create virtual environment and install dependencies
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Set Python path (required for the iaeste26 package)
$env:PYTHONPATH = "src"

# 3. Run the test suite (90 tests)
pytest tests/ -v

# 4. Open the technical report notebook
jupyter notebook report.ipynb
```

### Pipeline Modules

| Module | Description |
|---|---|
| `src/iaeste26/parser.py` | Parse WAV filenames, load and normalize audio |
| `src/iaeste26/dataset.py` | Scan all sessions, export metadata CSV |
| `src/iaeste26/preprocessing.py` | Trim edges, bandpass filter, normalize |
| `src/iaeste26/features.py` | 35 features: RMS, ZCR, MFCC, spectral features |
| `src/iaeste26/visualization.py` | Waveforms, spectrograms, analysis plots |
| `src/iaeste26/ml.py` | Random Forest models with cross-validation |

### Key Results

| Task | Result |
|---|---|
| Material classification (4 classes) | **79.5% accuracy** (chance = 25%) |
| Pressure prediction | R² = 0.53, RMSE = 0.97 bar |
| Best sensor for pressure | Mic147EB (R² = 0.59) |
| Best sensor for material ID | AccRadial4507 (92.1% accuracy) |

### Scripts

```powershell
# Generate analysis plots (plots/ folder)
python scripts/generate_plots.py

# Run ML pipeline on all 1568 files (~8 min)
python scripts/run_ml_full.py
```

### Output Files

- `dataset_metadata.csv` — metadata for all 1,568 recordings
- `plots/*.png` — 5 analysis plots
- `results/*.csv` — feature matrix, ML results, sensor comparison
- `report.ipynb` — fully executed technical report notebook
