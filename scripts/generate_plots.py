"""Phase 4: Generate all analysis plots and save to plots/ folder.

Plots generated:
  1. pressure_comparison.png   — waveform + spectrogram at 3, 5, 7.5 bar
  2. rms_vs_pressure.png       — RMS energy vs pressure for all 4 sensors
  3. rms_vs_mix.png            — RMS energy vs mix ratio for all 4 sensors
  4. sensor_comparison.png     — all 4 sensors for the same test
  5. centroid_vs_pressure.png  — spectral centroid vs pressure
"""

import sys
from pathlib import Path

# Make sure package is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")  # non-interactive backend (saves to file)

import numpy as np
from iaeste26.parser import parse_filename, load_wav
from iaeste26.preprocessing import preprocess
from iaeste26.features import rms, spectral_centroid
from iaeste26.visualization import (
    plot_pressure_comparison,
    plot_rms_vs_pressure,
    plot_rms_vs_mix,
    plot_sensor_comparison,
    plot_spectral_centroid_vs_pressure,
)

DATA_DIR  = Path(__file__).parent.parent / "data"
PLOTS_DIR = Path(__file__).parent.parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

SENSORS = ["Mic147EB", "Mic46BE", "AccAxial4507", "AccRadial4507"]

# Fixed parameters for most plots (G80, nozzle 8mm)
MATERIAL = "G80"
NOZZLE   = "8"
MIX      = "50"   # 50% mix ratio

PRESSURES  = ["3", "4", "5", "5.5", "6", "6.5", "7", "7.5"]
MIX_RATIOS = ["0", "30", "40", "50", "60", "70", "100"]


def find_wav(session: str, material: str, nozzle: str,
             pressure: str, mix: str, sensor: str) -> Path | None:
    """Return path to a WAV file matching the given parameters."""
    for session_dir in sorted((DATA_DIR).iterdir()):
        wav_dir = session_dir / "WAV"
        if not wav_dir.exists():
            continue
        name = f"{material}_{nozzle}_{pressure}_{mix}_{sensor}.wav"
        p = wav_dir / name
        if p.exists():
            return p
    return None


def load_clean(path: Path, sensor: str) -> tuple[np.ndarray, int]:
    audio, sr = load_wav(path)
    audio = preprocess(audio, sr, sensor=sensor)
    return audio, sr


# -----------------------------------------------------------------------
# Plot 1: Waveform + Spectrogram at 3 / 5 / 7.5 bar
# -----------------------------------------------------------------------
print("Plot 1: Pressure comparison (waveform + spectrogram)...")

select_pressures = ["3", "5", "7.5"]
sensor = "Mic46BE"
recordings = []
for p in select_pressures:
    path = find_wav("any", MATERIAL, NOZZLE, p, MIX, sensor)
    if path is None:
        print(f"  WARNING: not found — {MATERIAL}_{NOZZLE}_{p}_{MIX}_{sensor}")
        continue
    audio, sr = load_clean(path, sensor)
    recordings.append({
        "audio": audio[:sr * 20],  # first 20s for clarity
        "sr": sr,
        "label": f"{p} bar",
    })

if recordings:
    plot_pressure_comparison(recordings, PLOTS_DIR / "pressure_comparison.png")


# -----------------------------------------------------------------------
# Plot 2: RMS vs Pressure (all 4 sensors)
# -----------------------------------------------------------------------
print("Plot 2: RMS vs Pressure...")

pressures_float = []
rms_by_sensor   = {s: [] for s in SENSORS}

for p in PRESSURES:
    found_all = True
    temp = {}
    for sensor in SENSORS:
        path = find_wav("any", MATERIAL, NOZZLE, p, MIX, sensor)
        if path is None:
            found_all = False
            break
        audio, sr = load_clean(path, sensor)
        temp[sensor] = rms(audio)
    if found_all:
        pressures_float.append(float(p))
        for sensor in SENSORS:
            rms_by_sensor[sensor].append(temp[sensor])

if pressures_float:
    plot_rms_vs_pressure(
        pressures_float, rms_by_sensor,
        title=f"RMS Energy vs Pressure — {MATERIAL}, {NOZZLE}mm nozzle, {MIX}% mix",
        output_path=PLOTS_DIR / "rms_vs_pressure.png",
    )


# -----------------------------------------------------------------------
# Plot 3: RMS vs Mix Ratio (all 4 sensors, fixed pressure=5 bar)
# -----------------------------------------------------------------------
print("Plot 3: RMS vs Mix Ratio...")

FIXED_PRESSURE = "5"
mix_float    = []
rms_by_mix   = {s: [] for s in SENSORS}

for m in MIX_RATIOS:
    found_all = True
    temp = {}
    for sensor in SENSORS:
        path = find_wav("any", MATERIAL, NOZZLE, FIXED_PRESSURE, m, sensor)
        if path is None:
            found_all = False
            break
        audio, sr = load_clean(path, sensor)
        temp[sensor] = rms(audio)
    if found_all:
        mix_float.append(int(m))
        for sensor in SENSORS:
            rms_by_mix[sensor].append(temp[sensor])

if mix_float:
    plot_rms_vs_mix(
        mix_float, rms_by_mix,
        title=f"RMS Energy vs Mix Ratio — {MATERIAL}, {NOZZLE}mm, {FIXED_PRESSURE} bar",
        output_path=PLOTS_DIR / "rms_vs_mix.png",
    )


# -----------------------------------------------------------------------
# Plot 4: All 4 sensors for same test
# -----------------------------------------------------------------------
print("Plot 4: Sensor comparison...")

COMPARE_PRESSURE = "5"
COMPARE_MIX      = "50"
sensor_recs = []
for sensor in SENSORS:
    path = find_wav("any", MATERIAL, NOZZLE, COMPARE_PRESSURE, COMPARE_MIX, sensor)
    if path:
        audio, sr = load_clean(path, sensor)
        sensor_recs.append({
            "audio": audio[:sr * 15],  # 15s slice
            "sr": sr,
            "label": sensor,
        })

if len(sensor_recs) == 4:
    plot_sensor_comparison(sensor_recs, PLOTS_DIR / "sensor_comparison.png")


# -----------------------------------------------------------------------
# Plot 5: Spectral Centroid vs Pressure
# -----------------------------------------------------------------------
print("Plot 5: Spectral Centroid vs Pressure...")

cent_pressures = []
cent_by_sensor = {s: [] for s in SENSORS}

for p in PRESSURES:
    found_all = True
    temp = {}
    for sensor in SENSORS:
        path = find_wav("any", MATERIAL, NOZZLE, p, MIX, sensor)
        if path is None:
            found_all = False
            break
        audio, sr = load_clean(path, sensor)
        temp[sensor] = spectral_centroid(audio, sr)
    if found_all:
        cent_pressures.append(float(p))
        for sensor in SENSORS:
            cent_by_sensor[sensor].append(temp[sensor])

if cent_pressures:
    plot_spectral_centroid_vs_pressure(
        cent_pressures, cent_by_sensor,
        title=f"Spectral Centroid vs Pressure — {MATERIAL}, {NOZZLE}mm, {MIX}% mix",
        output_path=PLOTS_DIR / "centroid_vs_pressure.png",
    )


# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
saved = sorted(PLOTS_DIR.glob("*.png"))
print(f"\nDone. {len(saved)} plots saved to {PLOTS_DIR}/")
for f in saved:
    size_kb = f.stat().st_size // 1024
    print(f"  {f.name:<40} {size_kb} KB")
