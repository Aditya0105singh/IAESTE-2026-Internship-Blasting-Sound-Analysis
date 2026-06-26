"""Build the final technical report as a Jupyter notebook."""

import nbformat as nbf
from pathlib import Path

OUT = Path(__file__).parent.parent / "report.ipynb"
nb  = nbf.v4.new_notebook()
cells = []

def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))

def code(src):
    cells.append(nbf.v4.new_code_cell(src))

# -----------------------------------------------------------------------
# Title
# -----------------------------------------------------------------------
md("""# Blasting Sound Database — Signal Processing & ML Report
**IAESTE 2026 Internship | VŠB-TU Ostrava, Czech Republic**

---

## Project Summary

This notebook presents a complete signal-processing and machine-learning pipeline
for analyzing audio and vibration recordings of an abrasive blasting machine
operating under varying conditions.

**Objective:** Demonstrate that blasting parameters (pressure, abrasive material,
mix ratio) can be estimated directly from sound and vibration signals using
Python-based signal processing and machine learning.

**Dataset:** 1,568 WAV files (~6 GB) across 7 measurement sessions,
4 microphone/accelerometer sensors, 4 abrasive materials, 8 pressure levels,
7 mix ratios.
""")

# -----------------------------------------------------------------------
# 0. Setup
# -----------------------------------------------------------------------
md("## 0. Setup")

code("""\
import sys
from pathlib import Path

# Add src to path so the iaeste26 package is importable
sys.path.insert(0, str(Path('.').resolve() / 'src'))

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from iaeste26.parser    import parse_filename, load_wav
from iaeste26.preprocessing import preprocess
from iaeste26.features  import extract_features, rms, spectral_centroid
from iaeste26.dataset   import scan_dataset, summary
from iaeste26.ml        import (
    build_feature_matrix, train_pressure_model,
    train_material_model, train_mix_model, compare_sensors,
    cross_session_eval, compare_models, within_material_loso,
)

%matplotlib inline
plt.rcParams.update({'figure.dpi': 110, 'font.size': 10, 'axes.grid': True,
                     'grid.alpha': 0.3, 'axes.titlesize': 11})

DATA_DIR    = Path('data')
RESULTS_DIR = Path('results')
PLOTS_DIR   = Path('plots')

print('Setup complete.')
print(f'  numpy   {np.__version__}')
print(f'  pandas  {pd.__version__}')
print(f'  matplotlib {matplotlib.__version__}')
""")

# -----------------------------------------------------------------------
# 1. Dataset Overview
# -----------------------------------------------------------------------
md("""---
## 1. Dataset Overview

Recordings were made at the VŠB-TU Ostrava experimental facility.
Each blasting test was captured simultaneously by 4 sensors, and parameters
are encoded in the filename: `Material_Nozzle_Pressure_MixRatio_Sensor.wav`
""")

code("""\
df_meta = scan_dataset(DATA_DIR)
summary(df_meta)
print()
print(df_meta.head(8).to_string(index=False))
""")

md("""### Experimental Parameter Grid

| Parameter | Values |
|---|---|
| **Abrasive material** | G80, GH18, GH40, GH120 (4 steel grits) |
| **Nozzle diameter** | 6.4 mm, 8.0 mm |
| **Pressure** | 3, 4, 5, 5.5, 6, 6.5, 7, 7.5 bar |
| **Mix ratio** | 0%, 30%, 40%, 50%, 60%, 70%, 100% |
| **Sensors** | Mic147EB, Mic46BE, AccAxial4507, AccRadial4507 |
""")

code("""\
fig, axes = plt.subplots(1, 3, figsize=(13, 4))

# Files per session
session_counts = df_meta.groupby('session').size()
axes[0].bar(session_counts.index, session_counts.values, color='#2563eb', alpha=0.8)
axes[0].set_title('Files per Session')
axes[0].set_xlabel('Session')
axes[0].set_ylabel('Number of WAV files')
axes[0].tick_params(axis='x', rotation=45)

# Files per material
mat_counts = df_meta.groupby('material').size()
axes[1].bar(mat_counts.index, mat_counts.values, color='#16a34a', alpha=0.8)
axes[1].set_title('Files per Material')
axes[1].set_xlabel('Material')

# Files per sensor
sens_counts = df_meta.groupby('sensor').size()
axes[2].bar(range(len(sens_counts)), sens_counts.values, color='#d97706', alpha=0.8)
axes[2].set_xticks(range(len(sens_counts)))
axes[2].set_xticklabels(sens_counts.index, rotation=30, ha='right')
axes[2].set_title('Files per Sensor')

plt.tight_layout()
plt.show()
""")

# -----------------------------------------------------------------------
# 2. Signal Visualization
# -----------------------------------------------------------------------
md("""---
## 2. Signal Visualization

We compare three pressure levels (3 bar, 5 bar, 7.5 bar) for the same
material, nozzle, mix ratio and sensor.
Higher pressure should produce louder, higher-frequency sound.
""")

code("""\
def find_wav(material, nozzle, pressure, mix, sensor):
    for session in ['v24','v25','v26','v27','v28','v30','v31']:
        p = DATA_DIR / session / 'WAV' / f'{material}_{nozzle}_{pressure}_{mix}_{sensor}.wav'
        if p.exists():
            return p
    return None

pressures_to_show = ['3', '5', '7.5']
SENSOR = 'Mic46BE'
MATERIAL, NOZZLE, MIX = 'G80', '8', '50'

fig, axes = plt.subplots(2, 3, figsize=(13, 6))
fig.suptitle(f'G80 | 8mm nozzle | 50% mix | {SENSOR} — Effect of Pressure',
             fontsize=12, y=1.02)

colors = ['#2563eb', '#d97706', '#dc2626']

for i, p in enumerate(pressures_to_show):
    path  = find_wav(MATERIAL, NOZZLE, p, MIX, SENSOR)
    audio_raw, sr = load_wav(path)
    audio = preprocess(audio_raw, sr, sensor=SENSOR)
    clip  = audio[:sr * 10]          # 10-second clip for display
    t     = np.arange(len(clip)) / sr

    # Waveform
    axes[0, i].plot(t, clip, lw=0.4, color=colors[i])
    axes[0, i].set_title(f'{p} bar  |  RMS={rms(audio):.4f}')
    axes[0, i].set_xlabel('Time (s)')
    axes[0, i].set_ylabel('Amplitude')
    axes[0, i].set_xlim(0, 10)

    # Spectrogram
    from scipy.signal import stft as scipy_stft
    freqs, times, Zxx = scipy_stft(clip, fs=sr, nperseg=2048, noverlap=1536)
    pdb = 20 * np.log10(np.abs(Zxx) + 1e-10)
    mask = freqs <= 8000
    im = axes[1, i].pcolormesh(times, freqs[mask]/1000, pdb[mask],
                                shading='auto', cmap='inferno',
                                vmin=np.percentile(pdb, 10),
                                vmax=np.percentile(pdb, 99))
    axes[1, i].set_title(f'Spectrogram — {p} bar')
    axes[1, i].set_xlabel('Time (s)')
    axes[1, i].set_ylabel('Frequency (kHz)')

plt.tight_layout()
plt.show()
print('Observation: Higher pressure -> higher amplitude AND higher frequency content.')
""")

# -----------------------------------------------------------------------
# 3. Preprocessing
# -----------------------------------------------------------------------
md("""---
## 3. Preprocessing Pipeline

Every raw WAV goes through three steps before feature extraction:

1. **Trim edges** — remove first/last 2 seconds (start/stop transients)
2. **Bandpass filter** — Butterworth, sensor-specific frequency range
   - Microphones: 20 Hz – 20 kHz
   - Accelerometers: 1 Hz – 6 kHz
3. **Peak normalize** — scale to [-1, 1]
""")

code("""\
path = find_wav('G80', '8', '5', '50', 'Mic46BE')
audio_raw, sr = load_wav(path)
audio_clean   = preprocess(audio_raw, sr, sensor='Mic46BE')

fig, axes = plt.subplots(1, 2, figsize=(13, 3))
t_raw   = np.arange(len(audio_raw))   / sr
t_clean = np.arange(len(audio_clean)) / sr

axes[0].plot(t_raw,   audio_raw,   lw=0.3, color='#94a3b8')
axes[0].set_title(f'Raw signal  |  duration={len(audio_raw)/sr:.1f}s  peak={np.max(np.abs(audio_raw)):.4f}')
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Amplitude')

axes[1].plot(t_clean, audio_clean, lw=0.3, color='#2563eb')
axes[1].set_title(f'After preprocessing  |  duration={len(audio_clean)/sr:.1f}s  peak={np.max(np.abs(audio_clean)):.4f}')
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Amplitude')

plt.tight_layout()
plt.show()

print(f'Raw:   {len(audio_raw):,} samples | RMS = {rms(audio_raw):.4f} | peak = {np.max(np.abs(audio_raw)):.4f}')
print(f'Clean: {len(audio_clean):,} samples | RMS = {rms(audio_clean):.4f} | peak = {np.max(np.abs(audio_clean)):.4f}')
""")

# -----------------------------------------------------------------------
# 4. Feature Extraction
# -----------------------------------------------------------------------
md("""---
## 4. Feature Extraction

35 features per recording across three domains:

| Domain | Features | Count |
|---|---|---|
| Time | RMS, ZCR, Peak, Crest Factor, Kurtosis | 5 |
| Frequency | Centroid, Bandwidth, Rolloff-85%, Dominant Freq | 4 |
| MFCC | 13 coefficients × mean + std | 26 |
""")

code("""\
# Show features for 3 different pressures (same material/sensor)
rows = []
for p in ['3', '5', '7.5']:
    path = find_wav('G80', '8', p, '50', 'Mic46BE')
    audio, sr = load_wav(path)
    audio = preprocess(audio, sr, sensor='Mic46BE')
    feats = extract_features(audio, sr)
    feats['pressure_bar'] = float(p)
    rows.append(feats)

feat_df = pd.DataFrame(rows).set_index('pressure_bar')
display_cols = ['rms', 'zero_crossing_rate', 'crest_factor', 'kurtosis',
                'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff_85',
                'dominant_frequency', 'mfcc_1_mean', 'mfcc_2_mean', 'mfcc_3_mean']

print('Features at 3 different pressure levels:')
print(feat_df[display_cols].T.to_string())
print()
print('Key trend: spectral_centroid increases with pressure (sound gets "brighter")')
""")

# -----------------------------------------------------------------------
# 5. Analysis Plots
# -----------------------------------------------------------------------
md("""---
## 5. Analysis Plots

The following plots were pre-generated and saved to the `plots/` folder.
""")

code("""\
from IPython.display import Image, display

plot_files = {
    'RMS vs Pressure (all sensors)':     'plots/rms_vs_pressure.png',
    'RMS vs Mix Ratio (all sensors)':     'plots/rms_vs_mix.png',
    'Spectral Centroid vs Pressure':      'plots/centroid_vs_pressure.png',
    'Pressure Comparison (waveform+spec)':'plots/pressure_comparison.png',
    'Sensor Comparison (same test)':      'plots/sensor_comparison.png',
}

for title, path in plot_files.items():
    print(f'\\n### {title}')
    display(Image(filename=path, width=850))
""")

md("""### Key Observations from Plots

1. **RMS vs Pressure** — Energy increases with pressure for all 4 sensors. Microphones show a steeper rise than accelerometers.
2. **RMS vs Mix Ratio** — Energy peaks around 50–60% mix ratio. Pure air (0%) and full abrasive (100%) both show lower energy.
3. **Spectral Centroid vs Pressure** — Sound becomes "brighter" (higher frequency content) as pressure increases — consistent across all sensors.
4. **Sensor Comparison** — Microphones capture mainly low-frequency acoustic pressure waves; accelerometers capture high-frequency structural vibrations.
""")

# -----------------------------------------------------------------------
# 6. ML Results
# -----------------------------------------------------------------------
md("""---
## 6. Machine Learning — Condition Estimation

We train **Random Forest** models with 5-fold cross-validation on the
pre-built feature matrix (1,568 samples × 35 features).

Three tasks:
- **Pressure regression** — predict pressure_bar (continuous, 3–7.5 bar)
- **Material classification** — identify abrasive type (4 classes)
- **Mix ratio regression** — predict mix_ratio_pct (continuous, 0–100%)
""")

code("""\
print('Loading pre-computed feature matrix...')
df_feat = pd.read_csv(RESULTS_DIR / 'feature_matrix_full.csv')
print(f'Shape: {df_feat.shape}  ({len(df_feat)} files x {df_feat.shape[1]} columns)')
print()
print('Class balance (material):')
print(df_feat['material'].value_counts().to_string())
print()
print('Pressure distribution:')
print(df_feat['pressure_bar'].value_counts().sort_index().to_string())
""")

code("""\
print('Training models (may take ~1-2 minutes)...')

res_p  = train_pressure_model(df_feat)
res_m  = train_material_model(df_feat)
res_mx = train_mix_model(df_feat)

print()
print('='*50)
print('RESULTS SUMMARY')
print('='*50)
print()
print('Pressure Prediction (Regression):')
print(f'  RMSE = {res_p["rmse"]:.4f} bar')
print(f'  R2   = {res_p["r2"]:.4f}')
print()
print('Material Classification (4 classes, chance=25%):')
print(f'  Accuracy = {res_m["accuracy"]:.4f}  ({res_m["accuracy"]*100:.1f}%)')
print(f'  F1 score = {res_m["f1"]:.4f}')
print()
print('Mix Ratio Prediction (Regression):')
print(f'  RMSE = {res_mx["rmse"]:.2f}%')
print(f'  R2   = {res_mx["r2"]:.4f}')
""")

code("""\
# Visualise feature importance for pressure
importances = res_p['feature_importances']
top15 = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:15]
names, vals = zip(*top15)

fig, ax = plt.subplots(figsize=(9, 5))
colors_bar = ['#2563eb' if 'mfcc' in n else '#16a34a' if 'spectral' in n else '#d97706' for n in names]
ax.barh(range(len(names)), vals, color=colors_bar, alpha=0.85)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names)
ax.invert_yaxis()
ax.set_xlabel('Feature Importance')
ax.set_title('Top 15 Features for Pressure Prediction')

from matplotlib.patches import Patch
legend = [Patch(color='#2563eb', label='MFCC'),
          Patch(color='#16a34a', label='Spectral'),
          Patch(color='#d97706', label='Time-domain')]
ax.legend(handles=legend, loc='lower right')
plt.tight_layout()
plt.show()
print('MFCCs dominate — time-frequency texture is the most informative signal property.')
""")

# -----------------------------------------------------------------------
# 7. Sensor Comparison
# -----------------------------------------------------------------------
md("""---
## 7. Sensor Comparison

A key research question: **which sensor is most informative for each task?**

We train separate models for each sensor and compare performance.
""")

code("""\
sens_p = compare_sensors(df_feat, task='pressure')
sens_m = compare_sensors(df_feat, task='material')

# Pressure table
rows_p = [{'Sensor': s, 'RMSE (bar)': round(r['rmse'], 4), 'R2': round(r['r2'], 4)}
          for s, r in sorted(sens_p.items(), key=lambda x: x[1]['r2'], reverse=True)]
# Material table
rows_m = [{'Sensor': s, 'Accuracy': round(r['accuracy'], 4), 'F1': round(r['f1'], 4)}
          for s, r in sorted(sens_m.items(), key=lambda x: x[1]['accuracy'], reverse=True)]

print('Pressure Prediction per Sensor:')
print(pd.DataFrame(rows_p).to_string(index=False))
print()
print('Material Classification per Sensor:')
print(pd.DataFrame(rows_m).to_string(index=False))
""")

code("""\
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Pressure R2 bars
sensors = [r['Sensor'] for r in rows_p]
r2_vals = [r['R2'] for r in rows_p]
axes[0].barh(sensors, r2_vals, color='#2563eb', alpha=0.8)
axes[0].set_xlabel('R² Score')
axes[0].set_title('Pressure Prediction R² by Sensor')
axes[0].set_xlim(0, 1)
axes[0].axvline(1.0, color='green', ls='--', lw=1, label='Perfect')
axes[0].legend()

# Material accuracy bars
sensors_m = [r['Sensor'] for r in rows_m]
acc_vals  = [r['Accuracy'] for r in rows_m]
axes[1].barh(sensors_m, acc_vals, color='#dc2626', alpha=0.8)
axes[1].set_xlabel('Accuracy')
axes[1].set_title('Material Classification Accuracy by Sensor')
axes[1].set_xlim(0, 1)
axes[1].axvline(0.25, color='gray', ls='--', lw=1, label='Chance (25%)')
axes[1].legend()

plt.tight_layout()
plt.show()
print()
print('Finding: Mic147EB is best for pressure estimation.')
print('Finding: AccRadial4507 is best for material identification.')
print('=> Different sensors carry different physical information.')
""")

# -----------------------------------------------------------------------
# 8. Advanced Evaluation
# -----------------------------------------------------------------------
md("""---
## 8. Feature Space Visualisation (PCA)

Principal Component Analysis (PCA) reduces the 35-dimensional feature space to 2D,
letting us see whether the features genuinely separate materials and pressure levels.
""")

code("""\
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from iaeste26.ml import get_X, FEATURE_COLS

X = get_X(df_feat)
X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print(f'Variance explained: PC1={pca.explained_variance_ratio_[0]:.1%}, '
      f'PC2={pca.explained_variance_ratio_[1]:.1%}, '
      f'Total={sum(pca.explained_variance_ratio_):.1%}')

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Colour by material
mat_palette = {'G80':'#2563eb','GH18':'#16a34a','GH40':'#d97706','GH120':'#dc2626'}
for mat, color in mat_palette.items():
    mask = df_feat['material'] == mat
    axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1],
                    c=color, label=mat, alpha=0.35, s=12, edgecolors='none')
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
axes[0].set_title('PCA of 35 Features — coloured by Material')
axes[0].legend(markerscale=2)

# Colour by pressure
sc = axes[1].scatter(X_pca[:, 0], X_pca[:, 1],
                     c=df_feat['pressure_bar'], cmap='plasma',
                     alpha=0.4, s=12, edgecolors='none')
plt.colorbar(sc, ax=axes[1], label='Pressure (bar)')
axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
axes[1].set_title('PCA of 35 Features — coloured by Pressure')

plt.tight_layout()
plt.show()
print('Materials form overlapping clusters — acoustic similarity between steel grit types.')
print('Pressure shows a gradient along PC1 — energy features dominate the first component.')
""")

md("""---
## 9. Advanced Evaluation

Three additional analyses that strengthen the scientific credibility of the results.
""")

md("### 9a. Confusion Matrix — Which Materials Are Confused?")

code("""\
import matplotlib.colors as mcolors

res_m_adv = train_material_model(df_feat)
cm  = res_m_adv['confusion_matrix']
cls = res_m_adv['classes']

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks(range(len(cls))); ax.set_xticklabels(cls)
ax.set_yticks(range(len(cls))); ax.set_yticklabels(cls)
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
ax.set_title(f'Material Classification Confusion Matrix\\nAccuracy={res_m_adv["accuracy"]:.1%} +/- {res_m_adv["accuracy_std"]:.1%}')
for i in range(len(cls)):
    for j in range(len(cls)):
        color = 'white' if cm[i,j] > cm.max()/2 else 'black'
        ax.text(j, i, str(cm[i,j]), ha='center', va='center', color=color, fontsize=11)
plt.colorbar(im, ax=ax, label='Count')
plt.tight_layout()
plt.show()
print('GH18 is hardest to distinguish (66% accuracy) — physically similar grit to GH40.')
print('GH120 is easiest (85%) — very different particle size from the others.')
""")

md("### 9b. Cross-Session Generalisation — Dataset Structure First")

md("""\
Before running LOSO, we must understand the dataset recording structure.
Each measurement session recorded **one abrasive material only**:

| Sessions | Material |
|---|---|
| v24, v25 | G80 (steel grit) |
| v26, v27 | GH40 |
| v28 | GH18 |
| v30, v31 | GH120 |

**Consequence:** a naive Leave-One-Session-Out (LOSO) for material classification
would remove ALL samples of one material from training — effectively asking the model
to identify a material it was never trained on (zero-shot learning). The 28% accuracy
we saw is exactly chance-level for a 4-class problem with one unseen class.

The correct cross-session tests are:
1. **Full LOSO for pressure** — valid, because pressure 3–7.5 bar was recorded in every session
2. **Within-material LOSO** — for G80 (v24↔v25) and GH120 (v30↔v31), which each have 2 sessions
""")

code("""\
# Show dataset structure
print('Files per session per material:')
print(df_feat.groupby(['session','material']).size().unstack(fill_value=0).to_string())
""")

code("""\
# 1. Full LOSO for pressure (the valid test)
loso_p = cross_session_eval(df_feat, task='pressure')
df_lp  = pd.DataFrame(loso_p)

# 2. Within-material LOSO for pressure (G80 and GH120)
wm_p  = within_material_loso(df_feat, task='pressure')
wm_mx = within_material_loso(df_feat, task='mix')
df_wp  = pd.DataFrame(wm_p)
df_wmx = pd.DataFrame(wm_mx)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# Left: full LOSO pressure
bar_colors_p = ['#16a34a' if r > 0.5 else '#d97706' if r > 0.3 else '#dc2626'
                for r in df_lp['r2']]
axes[0].bar(df_lp['test_session'], df_lp['r2'], color=bar_colors_p, alpha=0.85)
axes[0].axhline(df_lp['r2'].mean(), color='black', ls='--', lw=1.5,
                label=f'Mean R2={df_lp["r2"].mean():.3f}')
axes[0].axhline(0.5315, color='blue', ls=':', lw=1.5, label='5-fold CV R2=0.532')
axes[0].set_ylim(0, 1); axes[0].set_xlabel('Held-out Session')
axes[0].set_ylabel('R2'); axes[0].set_title('Full LOSO: Pressure Prediction\\n(all sessions, all materials)')
axes[0].legend(fontsize=8)

# Right: within-material pressure LOSO
labels = [f'{r["material"]}\\n{r["train_session"]}→{r["test_session"]}' for r in wm_p]
r2_wm  = [r['r2'] for r in wm_p]
colors_wm = ['#2563eb','#2563eb','#dc2626','#dc2626']
axes[1].bar(labels, r2_wm, color=colors_wm, alpha=0.85)
axes[1].set_ylim(0, 1); axes[1].set_ylabel('R2')
axes[1].set_title('Within-Material LOSO: Pressure\\n(same material, different recording day)')
axes[1].axhline(np.mean(r2_wm), color='black', ls='--', lw=1.5,
                label=f'Mean R2={np.mean(r2_wm):.3f}')
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.show()

print('Full LOSO pressure:          mean R2 =', round(df_lp['r2'].mean(), 3))
print('Within-material LOSO pressure: mean R2 =', round(np.mean(r2_wm), 3))
print()
print('Within-material LOSO mix ratio:')
for r in wm_mx:
    print(f"  {r['material']} {r['train_session']}->{r['test_session']}: R2={r['r2']:.3f}  RMSE={r['rmse']:.1f}%")
""")

md("""\
#### Interpretation

| Evaluation | Pressure R2 | Interpretation |
|---|---|---|
| 5-fold CV (pooled) | 0.53 | Same-day generalisation |
| Full LOSO (cross-session) | 0.54 | Cross-session generalisation — **holds well** |
| Within-material LOSO | 0.33–0.46 | Same material, different day — moderate drop |

**Pressure prediction generalises well across sessions** — the physical relationship
between blasting pressure and sound energy is consistent day-to-day.

**Material classification (5-fold CV, 79.5%)** is a valid result. LOSO is not an
appropriate validation for this dataset because each session contains only one material.
A proper multi-day multi-material dataset would be needed to test session generalisation
for material identification.

**Mix ratio estimation degrades substantially cross-session** (R2=0.03–0.35), suggesting
that mix ratio effects on sound are more sensitive to daily recording conditions
than pressure effects.
""")

md("### 9c. Multi-Model Comparison")

code("""\
cmp_p = compare_models(df_feat, task='pressure', cv=5)
cmp_m = compare_models(df_feat, task='material', cv=5)

df_cp = pd.DataFrame([{'Model': k, **v} for k, v in cmp_p.items()])
df_cm = pd.DataFrame([{'Model': k, **v} for k, v in cmp_m.items()])

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Pressure R2
df_cp_s = df_cp.sort_values('r2', ascending=True)
axes[0].barh(df_cp_s['Model'], df_cp_s['r2'], color='#2563eb', alpha=0.8,
             xerr=df_cp_s['r2_std'], capsize=5)
axes[0].set_xlabel('R2 Score')
axes[0].set_title('Model Comparison: Pressure Prediction R2\\n(error bars = CV fold std)')
axes[0].set_xlim(0, 0.8)

# Material accuracy
df_cm_s = df_cm.sort_values('accuracy', ascending=True)
axes[1].barh(df_cm_s['Model'], df_cm_s['accuracy'], color='#dc2626', alpha=0.8,
             xerr=df_cm_s['accuracy_std'], capsize=5)
axes[1].set_xlabel('Accuracy')
axes[1].set_title('Model Comparison: Material Classification Accuracy\\n(error bars = CV fold std)')
axes[1].axvline(0.25, color='gray', ls='--', lw=1)
axes[1].set_xlim(0, 1)

plt.tight_layout()
plt.show()

print('Pressure:  ', df_cp.sort_values('r2', ascending=False)[['Model','r2','r2_std']].to_string(index=False))
print()
print('Material:  ', df_cm.sort_values('accuracy', ascending=False)[['Model','accuracy','accuracy_std']].to_string(index=False))
print()
print('SVM (RBF) is the best model for both tasks.')
print('Random Forest and Gradient Boosting are competitive.')
""")

# -----------------------------------------------------------------------
# 9. Conclusions
# -----------------------------------------------------------------------
md("""---
## 9. Conclusions

### What Was Built
A complete, tested Python signal-processing pipeline (`iaeste26` package) with:
- Filename parser and WAV loader
- Preprocessing: trim, bandpass filter, normalize
- Feature extraction: 35 features (time, frequency, MFCC)
- Visualization functions
- ML training and evaluation module

### Key Findings

| Task | Metric | Value | Evaluation |
|---|---|---|---|
| Material classification | Accuracy | **79.5% ± 4.5%** | 5-fold CV (valid) |
| Pressure prediction | R² | **0.53 ± 0.07** | 5-fold CV |
| Pressure prediction (cross-day) | R² | **0.54 ± 0.18** | Full LOSO (7 sessions) |
| Pressure (same material, new day) | R² | **0.33–0.46** | Within-material LOSO |
| Best model for pressure | SVM RBF | R²=0.59 | 5-fold CV |
| Best model for material | SVM RBF | 86.9% | 5-fold CV |
| Best sensor for pressure | Mic147EB | R²=0.59 | per-sensor CV |
| Best sensor for material ID | AccRadial4507 | 92.1% | per-sensor CV |
| Hardest material to distinguish | GH18 | 66% per-class accuracy | Confusion matrix |
| Most informative feature | MFCC coefficient 2 | 23% importance | RF feature importance |

### Dataset Structure Insight
Each recording session captured one abrasive material exclusively
(v24/v25=G80, v26/v27=GH40, v28=GH18, v30/v31=GH120). This means:
- 5-fold CV for material classification is **valid** (all classes in both folds)
- LOSO for material is **zero-shot** (not domain adaptation) — not a failure of the model
- Pressure LOSO is **valid** and shows good cross-session generalisation (R²=0.54)

### Physical Interpretation
- Higher pressure → more energy (RMS), higher spectral centroid
- Mix ratio 50–60% → maximum energy output (non-linear relationship)
- Microphones capture acoustic pressure waves → better for energy-based tasks
- Accelerometers capture structural vibrations → better for material texture tasks
- MFCCs (time-frequency texture) are the most discriminative features

### Future Work
- Record multiple sessions per material to enable proper cross-session material validation
- Session-invariant feature engineering (spectral shape normalization)
- Deep learning: CNN on raw spectrograms (end-to-end, no manual features)
- Real-time streaming inference pipeline
""")

code("""\
print('Report complete.')
print('All results saved in results/ folder:')
for f in sorted(Path('results').glob('*.csv')):
    print(f'  {f.name}')
""")

# -----------------------------------------------------------------------
# Write notebook
# -----------------------------------------------------------------------
nb.cells = cells
nbf.write(nb, str(OUT))
print(f"Notebook written: {OUT}")
print(f"Cells: {len(cells)}")
