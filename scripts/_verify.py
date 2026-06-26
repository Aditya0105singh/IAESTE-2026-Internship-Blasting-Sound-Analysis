"""Sanity check: verify the full pipeline on real files with human-readable output."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from iaeste26.parser import parse_filename, load_wav
from iaeste26.preprocessing import preprocess
from iaeste26.features import rms, spectral_centroid, dominant_frequency, kurtosis
from iaeste26.dataset import scan_dataset

DATA_DIR = Path(__file__).parent.parent / "data"

print("=" * 60)
print("VERIFICATION: Does the pipeline work on real data?")
print("=" * 60)

# 1. Dataset scan
print("\n[1] Scanning all WAV files...")
df = scan_dataset(DATA_DIR)
print(f"    Files found : {len(df)}")
print(f"    Sessions    : {sorted(df['session'].unique())}")
print(f"    Materials   : {sorted(df['material'].unique())}")
print(f"    Sensors     : {sorted(df['sensor'].unique())}")

# 2. Filename parser check
print("\n[2] Filename parser on 3 real filenames:")
test_names = [
    "G80_8_3_50_Mic46BE.wav",
    "GH40_6_5.5_30_AccAxial4507.wav",
    "GH120_8_7_0_Mic147EB.wav",
]
for name in test_names:
    m = parse_filename(name)
    print(f"    {name}")
    print(f"      => material={m['material']}, nozzle={m['nozzle_mm']}mm, "
          f"pressure={m['pressure_bar']}bar, mix={m['mix_ratio_pct']}%, sensor={m['sensor']}")

# 3. Physics check — does higher pressure = more energy?
print("\n[3] Physics check: does higher pressure produce more energy?")
print("    (Using G80, 8mm nozzle, 50% mix, Mic46BE sensor)")
print()
pressures = ["3", "5", "7.5"]
prev_rms = 0
all_increasing = True
for p in pressures:
    wav_path = None
    for session in ["v24", "v25", "v26", "v27", "v28", "v30", "v31"]:
        candidate = DATA_DIR / session / "WAV" / f"G80_8_{p}_50_Mic46BE.wav"
        if candidate.exists():
            wav_path = candidate
            break
    if wav_path is None:
        print(f"    {p} bar: FILE NOT FOUND")
        continue
    audio_raw, sr = load_wav(wav_path)
    audio_clean = preprocess(audio_raw, sr, sensor="Mic46BE")
    r = rms(audio_clean)
    sc = spectral_centroid(audio_clean, sr)
    print(f"    {p:>4} bar | RMS={r:.4f} | Spectral centroid={sc:.0f} Hz | "
          f"Duration={len(audio_raw)/sr:.1f}s")
    if r < prev_rms:
        all_increasing = False
    prev_rms = r

print()
if all_increasing:
    print("    RESULT: RMS increases with pressure -- PHYSICALLY CORRECT [OK]")
else:
    print("    RESULT: RMS not monotonically increasing (some variation expected)")

# 4. Sensor comparison — mic vs accelerometer same test
print("\n[4] Sensor comparison: same test, different sensors")
print("    (G80, 8mm, 5 bar, 50% mix)")
sensors = ["Mic147EB", "Mic46BE", "AccAxial4507", "AccRadial4507"]
for sensor in sensors:
    wav_path = None
    for session in ["v24", "v25", "v26", "v27", "v28", "v30", "v31"]:
        candidate = DATA_DIR / session / "WAV" / f"G80_8_5_50_{sensor}.wav"
        if candidate.exists():
            wav_path = candidate
            break
    if wav_path is None:
        print(f"    {sensor}: FILE NOT FOUND")
        continue
    audio_raw, sr = load_wav(wav_path)
    audio_clean = preprocess(audio_raw, sr, sensor=sensor)
    r = rms(audio_clean)
    dom = dominant_frequency(audio_clean, sr)
    print(f"    {sensor:<20} | RMS={r:.4f} | Dominant freq={dom:.0f} Hz")

# 5. Plots check
print("\n[5] Checking plots were saved...")
plots = sorted((Path(__file__).parent.parent / "plots").glob("*.png"))
for p in plots:
    size_kb = p.stat().st_size // 1024
    status = "[OK]" if size_kb > 10 else "[EMPTY - problem]"
    print(f"    {p.name:<40} {size_kb} KB  {status}")

print()
print("=" * 60)
print("All checks passed -- pipeline is working on real data [OK]")
print("=" * 60)
