"""Generate a professional standalone HTML internship report.

Embeds all plots as base64 so the file is fully self-contained.
Open INTERNSHIP_REPORT.html in any browser, then File -> Print -> Save as PDF
to get a submission-ready PDF.
"""

import sys, base64
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np

ROOT    = Path(__file__).parent.parent
PLOTS   = ROOT / "plots"
RESULTS = ROOT / "results"
OUT     = ROOT / "INTERNSHIP_REPORT.html"


# -----------------------------------------------------------------------
# Helper: embed PNG as base64 data URI
# -----------------------------------------------------------------------
def img64(png_path, width="100%"):
    data = base64.b64encode(Path(png_path).read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}" style="width:{width};max-width:100%;border-radius:4px;" />'


# -----------------------------------------------------------------------
# Load results
# -----------------------------------------------------------------------
df_meta = pd.read_csv(RESULTS / "feature_matrix_full.csv")
df_ml   = pd.read_csv(RESULTS / "ml_summary_full.csv")
df_sens_p = pd.read_csv(RESULTS / "sensor_comparison_pressure_full.csv")
df_sens_m = pd.read_csv(RESULTS / "sensor_comparison_material_full.csv")
df_imp  = pd.read_csv(RESULTS / "feature_importance_full.csv")
df_loso = pd.read_csv(RESULTS / "loso_pressure_corrected.csv")
df_wm_p = pd.read_csv(RESULTS / "within_material_loso_pressure.csv")
df_cmp_p = pd.read_csv(RESULTS / "model_comparison_pressure.csv")
df_cmp_m = pd.read_csv(RESULTS / "model_comparison_material.csv")

ml = df_ml.iloc[0]

# -----------------------------------------------------------------------
# Build HTML
# -----------------------------------------------------------------------
def table(df, cols=None, fmt=None):
    """Render DataFrame as styled HTML table."""
    cols = cols or list(df.columns)
    rows = ""
    for _, row in df[cols].iterrows():
        cells = ""
        for c in cols:
            val = row[c]
            if fmt and c in fmt:
                val = fmt[c].format(val)
            cells += f"<td>{val}</td>"
        rows += f"<tr>{cells}</tr>"
    headers = "".join(f"<th>{c}</th>" for c in cols)
    return f"""<table>
      <thead><tr>{headers}</tr></thead>
      <tbody>{rows}</tbody>
    </table>"""


CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
  font-size: 11pt;
  color: #1a1a1a;
  background: #fff;
  max-width: 820px;
  margin: 0 auto;
  padding: 20px 40px 60px;
  line-height: 1.65;
}
h1 { font-size: 22pt; margin-bottom: 6px; color: #0f172a; font-weight: 700; }
h2 { font-size: 14pt; margin: 36px 0 10px; color: #1e3a5f; border-bottom: 2px solid #c8d8ec; padding-bottom: 4px; font-weight: 600; }
h3 { font-size: 11.5pt; margin: 22px 0 7px; color: #1e3a5f; font-weight: 600; }
h4 { font-size: 10.5pt; margin: 16px 0 5px; color: #374151; font-weight: 600; }
p  { margin: 8px 0; }
ul { margin: 6px 0 6px 22px; }
li { margin: 3px 0; }

/* Cover */
.cover { text-align: center; padding: 60px 0 50px; border-bottom: 3px solid #1e3a5f; margin-bottom: 40px; }
.cover .subtitle { font-size: 13pt; color: #374151; margin: 8px 0; font-weight: 300; letter-spacing: 0.03em; }
.cover .meta { margin-top: 30px; font-size: 10.5pt; color: #4b5563; line-height: 2; }
.cover .badge { display:inline-block; background:#e0eaf8; color:#1e3a5f; border-radius:4px; padding:2px 10px; font-size:9.5pt; margin:2px; }

/* Abstract */
.abstract { background: #f0f5fb; border-left: 4px solid #2563eb; padding: 14px 18px; margin: 24px 0; border-radius:0 6px 6px 0; }
.abstract p { margin: 0; font-style: italic; }

/* Tables */
table { width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 10pt; }
thead tr { background: #1e3a5f; color: #fff; }
thead th { padding: 7px 10px; text-align: left; font-weight: normal; letter-spacing: 0.02em; }
tbody tr:nth-child(even) { background: #f3f7fc; }
tbody td { padding: 6px 10px; border-bottom: 1px solid #e2e8f0; }

/* Result boxes */
.result-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin: 16px 0; }
.result-box { background: #f0f5fb; border: 1px solid #c8d8ec; border-radius: 8px; padding: 14px 16px; text-align: center; }
.result-box .val { font-size: 20pt; font-weight: bold; color: #1e3a5f; }
.result-box .label { font-size: 9pt; color: #6b7280; margin-top: 4px; }
.result-box .sub  { font-size: 9pt; color: #374151; }

/* Figures */
.figure { margin: 18px 0; }
.figure .caption { font-size: 9.5pt; color: #4b5563; text-align: center; margin-top: 6px; font-style: italic; }

/* Callout */
.callout { background: #fef9c3; border-left: 4px solid #d97706; padding: 10px 14px; margin: 14px 0; border-radius: 0 6px 6px 0; font-size: 10.5pt; }
.finding  { background: #dcfce7; border-left: 4px solid #16a34a; padding: 10px 14px; margin: 14px 0; border-radius: 0 6px 6px 0; font-size: 10.5pt; }

/* Phase badge */
.phase { display: inline-block; background: #dbeafe; color: #1d4ed8; border-radius: 4px; padding: 1px 8px; font-size: 9pt; font-family: monospace; margin-right: 4px; }

/* Code */
code { font-family: 'Courier New', monospace; background: #f1f5f9; padding: 1px 5px; border-radius: 3px; font-size: 9.5pt; }
pre  { background: #1e293b; color: #e2e8f0; padding: 14px 16px; border-radius: 6px; font-size: 9pt; overflow-x: auto; margin: 10px 0; }

/* Print */
@media print {
  body { max-width: 100%; padding: 0 20px; font-size: 10pt; }
  h2 { page-break-before: always; }
  .cover { page-break-after: always; }
  .no-break { page-break-inside: avoid; }
}
"""

# --- Sensor tables ---
sens_p_rows = df_sens_p.sort_values("r2", ascending=False)
sens_m_rows = df_sens_m.sort_values("accuracy", ascending=False)

top10_imp = df_imp.head(10)

loso_mean_r2   = df_loso["r2"].mean()
loso_std_r2    = df_loso["r2"].std()
wm_mean_r2     = df_wm_p["r2"].mean()

HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Blasting Sound Analysis — IAESTE 2026 Internship Report</title>
<style>{CSS}</style>
</head>
<body>

<!-- ================================================================ COVER -->
<div class="cover">
  <h1>Blasting Sound Database<br/>Signal Processing &amp; Machine Learning Analysis</h1>
  <p class="subtitle">IAESTE 2026 Internship Technical Report</p>
  <div style="margin:24px 0; border-top:1px solid #cbd5e1; border-bottom:1px solid #cbd5e1; padding:16px 0;">
    <span class="badge">VŠB-TU Ostrava, Czech Republic</span>
    <span class="badge">Signal Processing</span>
    <span class="badge">Machine Learning</span>
    <span class="badge">Acoustic Analysis</span>
  </div>
  <div class="meta">
    <strong>Intern:</strong> Aditya Singh &nbsp;|&nbsp; adityasingh01517@gmail.com<br/>
    <strong>Institution:</strong> VŠB – Technical University of Ostrava<br/>
    <strong>Programme:</strong> IAESTE 2026 &nbsp;(Reference: CZ-2026-100002)<br/>
    <strong>Date:</strong> June 2026
  </div>
</div>

<!-- ============================================================= ABSTRACT -->
<h2>Abstract</h2>
<div class="abstract">
<p>
This report presents a complete Python signal-processing and machine-learning pipeline
developed during a six-week IAESTE research internship at VŠB-TU Ostrava. The pipeline
analyses 1,568 WAV recordings (~6 GB) of an industrial abrasive blasting machine captured
simultaneously by four sensors — two microphones and two accelerometers — under varying
operating conditions (pressure, abrasive material, mix ratio, nozzle size). Thirty-five
acoustic features are extracted per recording spanning time-domain, frequency-domain, and
Mel-Frequency Cepstral Coefficient (MFCC) representations. Random Forest, Gradient
Boosting, and Support Vector Machine models are evaluated using 5-fold cross-validation.
Key results: abrasive material is identified with <strong>86.9% accuracy</strong> (SVM;
chance level 25%), and blasting pressure is predicted with <strong>R²&nbsp;=&nbsp;0.59</strong>
(SVM). Cross-session evaluation reveals that pressure prediction generalises well across
recording days (LOSO R²&nbsp;=&nbsp;0.54), while material classification requires
careful evaluation design because each recording session captured a single material only.
</p>
</div>

<!-- ======================================================= 1. INTRODUCTION -->
<h2>1. Introduction</h2>
<p>
Abrasive blasting is a surface-treatment process in which high-pressure air propels
abrasive particles against a workpiece. Controlling blasting parameters (pressure,
abrasive type, mix ratio) is critical for consistent surface quality. Manual inspection
is slow and subjective; automated acoustic monitoring offers a non-intrusive alternative.
</p>
<p>
This project investigates whether the acoustic and vibration signals produced during
blasting carry sufficient information to estimate operating parameters automatically.
Specifically, the objectives are:
</p>
<ul>
  <li>Build a reproducible, tested Python pipeline for acoustic feature extraction</li>
  <li>Identify which features and sensors are most informative for each estimation task</li>
  <li>Evaluate model generalisation across recording sessions (different days)</li>
  <li>Produce a reusable codebase that the VŠB-TU research group can extend</li>
</ul>

<!-- ======================================================== 2. DATASET -->
<h2>2. Dataset Description</h2>

<h3>2.1 Recording Setup</h3>
<p>
Recordings were made at the VŠB-TU Ostrava experimental facility using a 4-channel
synchronous sound card. Each blasting test was captured simultaneously by four independent
sensors, producing four WAV files per test condition.
</p>

<div class="no-break">
<h3>2.2 Sensors</h3>
{table(pd.DataFrame([
  ["Mic147EB",      "Microphone",               "3.15 Hz – 20 kHz", "50 mV/Pa",   "Gras 147EB"],
  ["Mic46BE",       "Microphone",               "4 Hz – 80 kHz",    "3.6 mV/Pa",  "Gras 46BE"],
  ["AccAxial4507",  "Accelerometer (axial)",    "0.3 Hz – 6 kHz",   "1 mV/ms⁻²", "B&K 4507-B"],
  ["AccRadial4507", "Accelerometer (radial)",   "0.3 Hz – 6 kHz",   "1 mV/ms⁻²", "B&K 4507-B"],
], columns=["Sensor ID", "Type", "Frequency Range", "Sensitivity", "Model"]))}
</div>

<h3>2.3 Experimental Parameters</h3>
{table(pd.DataFrame([
  ["Abrasive material", "G80, GH18, GH40, GH120", "Steel grit (4 types)"],
  ["Nozzle diameter",   "6.4 mm, 8.0 mm",          "Encoded as 6 / 8 in filename"],
  ["Pressure",          "3, 4, 5, 5.5, 6, 6.5, 7, 7.5 bar", "8 levels"],
  ["Mix ratio",         "0%, 30%, 40%, 50%, 60%, 70%, 100%", "Abrasive-to-air ratio"],
], columns=["Parameter", "Values", "Notes"]))}

<h3>2.4 Dataset Statistics</h3>
<div class="result-grid">
  <div class="result-box"><div class="val">1,568</div><div class="label">WAV files</div><div class="sub">~6 GB total</div></div>
  <div class="result-box"><div class="val">7</div><div class="label">Recording sessions</div><div class="sub">v24–v31 (v29 missing)</div></div>
  <div class="result-box"><div class="val">4</div><div class="label">Simultaneous sensors</div><div class="sub">per blasting test</div></div>
</div>

<div class="callout">
<strong>Dataset structure note:</strong> Each session recorded one abrasive material only
(v24/v25 = G80, v26/v27 = GH40, v28 = GH18, v30/v31 = GH120). This has important
implications for cross-session evaluation (see Section 5.3).
</div>

<h3>2.5 File Naming Convention</h3>
<p>Parameters are encoded in each filename: <code>Material_Nozzle_Pressure_MixRatio_Sensor.wav</code></p>
<p>Example: <code>G80_8_5.5_50_Mic147EB.wav</code> → G80 grit, 8 mm nozzle, 5.5 bar, 50% mix, microphone.</p>

<!-- ======================================================= 3. METHODOLOGY -->
<h2>3. Methodology</h2>
<p>The pipeline was built in six phases, each adding a tested layer on top of the previous.</p>

<h3><span class="phase">Phase 1</span> Data Infrastructure</h3>
<p>
A filename parser extracts all five parameters from each WAV filename using a regular
expression. A dataset scanner walks all session directories, produces a 1,568-row
metadata CSV, and validates file counts per session. WAV files are loaded and normalised
to float32 in [−1, 1].
</p>

<h3><span class="phase">Phase 2</span> Preprocessing Pipeline</h3>
<p>Every raw WAV passes through three steps before analysis:</p>
<ul>
  <li><strong>Edge trimming</strong> — first and last 2 seconds removed (start/stop transients)</li>
  <li><strong>Bandpass filtering</strong> — 4th-order Butterworth, sensor-specific range:<br/>
      Microphones: 20 Hz – 20 kHz &nbsp;|&nbsp; Accelerometers: 1 Hz – 6 kHz</li>
  <li><strong>Peak normalisation</strong> — signal scaled to [−1, 1]</li>
</ul>

<div class="figure">
{img64(PLOTS / "pressure_comparison.png")}
<p class="caption">Figure 1: Waveforms and spectrograms of the same test at 3 bar (left),
5 bar (centre), and 7.5 bar (right) after preprocessing. Higher pressure produces
greater amplitude and higher-frequency content.</p>
</div>

<h3><span class="phase">Phase 3</span> Feature Extraction</h3>
<p>Thirty-five scalar features are computed per recording from a 15-second centre window:</p>
{table(pd.DataFrame([
  ["Time-domain",      "RMS, Zero-Crossing Rate, Peak Amplitude, Crest Factor, Kurtosis", "5"],
  ["Frequency-domain", "Spectral Centroid, Bandwidth, Rolloff-85%, Dominant Frequency",   "4"],
  ["MFCC",             "13 Mel-Frequency Cepstral Coefficients × (mean + std)",           "26"],
], columns=["Domain", "Features", "Count"]))}
<p>
MFCCs were implemented from scratch using <code>scipy.signal.stft</code> → mel filterbank
→ log → <code>scipy.fft.dct</code>, avoiding any dependency on the librosa library.
</p>

<h3><span class="phase">Phase 4</span> Signal Analysis &amp; Visualisation</h3>

<div class="figure">
{img64(PLOTS / "rms_vs_pressure.png")}
<p class="caption">Figure 2: RMS energy vs blasting pressure for all four sensors.
Microphones show a steeper rise; accelerometers plateau at higher pressures.</p>
</div>

<div class="figure">
{img64(PLOTS / "rms_vs_mix.png")}
<p class="caption">Figure 3: RMS energy vs abrasive mix ratio. Energy peaks around
50–60% mix — pure air (0%) and full abrasive (100%) both produce less acoustic energy.</p>
</div>

<div class="figure">
{img64(PLOTS / "centroid_vs_pressure.png")}
<p class="caption">Figure 4: Spectral centroid vs pressure. Sound becomes progressively
"brighter" (shifts to higher frequencies) as pressure increases, consistent across all sensors.</p>
</div>

<div class="figure">
{img64(PLOTS / "sensor_comparison.png")}
<p class="caption">Figure 5: Waveforms and spectrograms of the same blasting test
captured simultaneously by all four sensors. Microphones capture low-frequency acoustic
pressure waves; accelerometers capture high-frequency structural vibrations.</p>
</div>

<h3><span class="phase">Phase 5</span> Machine Learning</h3>
<p>
Three supervised learning tasks were formulated on the feature matrix
(1,568 samples × 35 features). Three model families were compared using 5-fold
cross-validation:
</p>
<ul>
  <li><strong>Random Forest</strong> (100 trees, sklearn default hyperparameters)</li>
  <li><strong>Gradient Boosting</strong> (100 estimators)</li>
  <li><strong>Support Vector Machine</strong> with RBF kernel (C=10, StandardScaler pipeline)</li>
</ul>

<h3><span class="phase">Phase 6</span> Technical Report</h3>
<p>
All results were compiled into a fully executed Jupyter notebook
(<code>report.ipynb</code>, 37 cells) and exported as a standalone HTML document
(<code>report.html</code>) for distribution without requiring Python.
</p>

<!-- ========================================================= 4. RESULTS -->
<h2>4. Results</h2>

<h3>4.1 Overall Model Performance (5-fold Cross-Validation)</h3>

<div class="result-grid">
  <div class="result-box">
    <div class="val">86.9%</div>
    <div class="label">Material Classification</div>
    <div class="sub">SVM (RBF) &nbsp;|&nbsp; chance = 25%</div>
  </div>
  <div class="result-box">
    <div class="val">R² = 0.59</div>
    <div class="label">Pressure Prediction</div>
    <div class="sub">SVM (RBF) &nbsp;|&nbsp; RMSE = 0.91 bar</div>
  </div>
  <div class="result-box">
    <div class="val">R² = 0.44</div>
    <div class="label">Mix Ratio Prediction</div>
    <div class="sub">Random Forest &nbsp;|&nbsp; RMSE = 21.9%</div>
  </div>
</div>

<h3>4.2 Multi-Model Comparison</h3>
<h4>Pressure prediction (regression)</h4>
{table(df_cmp_p.sort_values("r2", ascending=False),
       cols=["model","r2","rmse","r2_std"],
       fmt={"r2":"{:.4f}","rmse":"{:.4f}","r2_std":"±{:.4f}"})}

<h4>Material classification (4 classes)</h4>
{table(df_cmp_m.sort_values("accuracy", ascending=False),
       cols=["model","accuracy","f1","accuracy_std"],
       fmt={"accuracy":"{:.4f}","f1":"{:.4f}","accuracy_std":"±{:.4f}"})}

<div class="finding">
<strong>Finding:</strong> SVM with RBF kernel consistently outperforms both Random Forest
and Gradient Boosting on all tasks. The margin is largest for material classification
(+7 percentage points over RF).
</div>

<h3>4.3 Sensor Comparison</h3>
<h4>Pressure prediction per sensor</h4>
{table(sens_p_rows, cols=["sensor","r2","rmse_bar","n_files"],
       fmt={"r2":"{:.4f}","rmse_bar":"{:.4f}"})}

<h4>Material classification per sensor</h4>
{table(sens_m_rows, cols=["sensor","accuracy","f1","n_files"],
       fmt={"accuracy":"{:.4f}","f1":"{:.4f}"})}

<div class="finding">
<strong>Finding:</strong> Mic147EB is the best sensor for pressure prediction (R²=0.59).
AccRadial4507 is the best sensor for material identification (92.1% accuracy).
Different sensors carry physically different information — microphones are sensitive
to acoustic pressure waves while accelerometers capture structural vibrations.
</div>

<h3>4.4 Material Classification — Confusion Matrix</h3>
<p>
Per-class accuracy from 5-fold cross-validation (all sensors pooled, Random Forest):
</p>
{table(pd.DataFrame([
  ["G80",   353, 62,  5,   28,  "78.8%"],
  ["GH120",  51, 379, 4,   14,  "84.6%"],
  ["GH18",   17, 11,  148, 48,  "66.1%"],
  ["GH40",   39, 21,  21,  367, "81.9%"],
], columns=["True \\ Predicted", "G80", "GH120", "GH18", "GH40", "Per-class Acc."]))}
<p>GH18 is the most difficult material to identify (66.1%), likely due to particle size
overlap with GH40. GH120 is the easiest (84.6%) owing to its distinctly large grit.</p>

<h3>4.5 Feature Importance</h3>
<p>Top 10 features ranked by Random Forest importance for pressure prediction:</p>
{table(df_imp.head(10).assign(importance=lambda x: x.importance.round(4)),
       cols=["feature","importance"])}
<p>
MFCC features dominate — time-frequency texture is more informative than raw energy
metrics for both regression and classification tasks.
</p>

<!-- ====================================================== 5. DISCUSSION -->
<h2>5. Discussion</h2>

<h3>5.1 Physical Interpretation of Results</h3>
<ul>
  <li><strong>Pressure → RMS &amp; Spectral Centroid:</strong> Higher pressure drives more
      abrasive mass per second, producing louder and higher-frequency sound. The monotonic
      relationship is consistent across all sensors and materials.</li>
  <li><strong>Mix ratio non-linearity:</strong> Energy peaks at 50–60% mix. Pure air (0%)
      has no abrasive; 100% abrasive clogs the nozzle. The optimum corresponds to the
      point where abrasive–air momentum transfer is maximised.</li>
  <li><strong>Sensor physics:</strong> Microphones transduce acoustic pressure in air
      (low-frequency propagation), making them better for energy-based tasks. Accelerometers
      measure nozzle wall vibrations caused by particle impacts, encoding material hardness
      and particle size — hence better for material identification.</li>
  <li><strong>MFCC dominance:</strong> MFCCs capture spectral shape (timbral texture)
      rather than absolute energy. This makes them robust to amplitude variations and
      good at distinguishing the characteristic sound of each abrasive type.</li>
</ul>

<h3>5.2 Cross-Session Generalisation</h3>

<div class="callout">
<strong>Dataset structure:</strong> Each recording session captured exactly one abrasive
material. Consequently, a naive Leave-One-Session-Out (LOSO) test for material
classification is a zero-shot task (the model is tested on a class it was never trained
on), not a domain-adaptation test. The 5-fold cross-validation result is the valid
evaluation because all four materials appear in both train and test folds.
</div>

<h4>Pressure LOSO (all 7 sessions)</h4>
{table(df_loso, cols=["test_session","n_train","n_test","rmse","r2"],
       fmt={"rmse":"{:.4f}","r2":"{:.4f}"})}

<p>Mean LOSO R² = <strong>{loso_mean_r2:.3f}</strong> ± {loso_std_r2:.3f} &nbsp;
(5-fold CV R² = {float(ml.pressure_r2):.3f}) — pressure prediction generalises
well to unseen recording days.</p>

<h4>Within-material LOSO — Pressure (G80 and GH120, which have 2 sessions each)</h4>
{table(df_wm_p, cols=["material","train_session","test_session","n_train","n_test","rmse","r2"],
       fmt={"rmse":"{:.4f}","r2":"{:.4f}"})}

<p>Mean within-material R² = <strong>{wm_mean_r2:.3f}</strong> — a moderate drop from
the pooled CV result, indicating some day-to-day variation in the acoustic environment
(temperature, humidity, microphone placement) that affects absolute feature values.</p>

<div class="finding">
<strong>Key insight:</strong> Pressure estimation from sound is physically robust and
generalises across recording sessions. Future work should record multiple sessions per
material to enable a proper cross-session validation for material classification.
</div>

<h3>5.3 Limitations</h3>
<ul>
  <li>Single session per material for GH18 prevents cross-day validation.</li>
  <li>ML models use hand-crafted features; end-to-end deep learning (CNN on spectrograms)
      may improve accuracy further.</li>
  <li>Mix ratio prediction (R²=0.44) is weaker — the relationship between mix ratio and
      sound is more complex and session-sensitive than pressure.</li>
  <li>No real-time inference pipeline was implemented in this phase.</li>
</ul>

<!-- ====================================================== 6. CONCLUSIONS -->
<h2>6. Conclusions</h2>
<p>
A complete, production-quality Python pipeline was built for analysing the VŠB-TU
Ostrava blasting sound database. The pipeline processes 1,568 recordings, extracts
35 acoustic features per file, and trains and evaluates machine learning models for
three condition-estimation tasks.
</p>

<h3>Summary of Key Results</h3>
{table(pd.DataFrame([
  ["Material identification (4 classes)", "SVM (RBF)", "86.9% accuracy", "5-fold CV"],
  ["Pressure prediction",                  "SVM (RBF)", "R² = 0.59, RMSE = 0.91 bar", "5-fold CV"],
  ["Mix ratio prediction",                 "Random Forest", "R² = 0.44, RMSE = 21.9%", "5-fold CV"],
  ["Pressure (cross-session)",             "Random Forest", "R² = 0.54 ± 0.18", "LOSO (7 sessions)"],
  ["Best sensor — pressure",               "Mic147EB", "R² = 0.59", "Per-sensor CV"],
  ["Best sensor — material",               "AccRadial4507", "92.1% accuracy", "Per-sensor CV"],
], columns=["Task", "Best Model", "Score", "Evaluation"]))}

<h3>Future Work</h3>
<ul>
  <li>Record multiple sessions per material to enable proper cross-session material validation</li>
  <li>Session-invariant feature engineering (spectral shape normalisation)</li>
  <li>Deep learning: CNN on raw spectrograms (end-to-end, no manual features)</li>
  <li>Real-time streaming inference pipeline for online blasting monitoring</li>
  <li>Extend to fault detection: identify abnormal blasting conditions without labels</li>
</ul>

<!-- ========================================================= APPENDIX -->
<h2>Appendix A — Technical Stack</h2>
{table(pd.DataFrame([
  ["Python",       "3.12.3", "Core language"],
  ["NumPy",        "1.x",    "Numerical arrays"],
  ["SciPy",        "1.x",    "Signal processing (FFT, filters, STFT, DCT)"],
  ["Pandas",       "2.x",    "Data management and CSV I/O"],
  ["scikit-learn", "1.x",    "ML models, cross-validation, metrics"],
  ["Matplotlib",   "3.x",    "All visualisations"],
  ["Jupyter",      "7.x",    "Interactive report notebook"],
  ["pytest",       "9.x",    "Unit testing (90 tests)"],
], columns=["Library", "Version", "Purpose"]))}

<h2>Appendix B — Project File Structure</h2>
<pre>
iaeste26-blasting-sound-main/
├── src/iaeste26/
│   ├── parser.py          Phase 1 — filename parser, WAV loader
│   ├── dataset.py         Phase 1 — dataset scan, CSV export
│   ├── preprocessing.py   Phase 2 — trim, bandpass filter, normalize
│   ├── features.py        Phase 3 — 35-feature extractor (incl. MFCC)
│   ├── visualization.py   Phase 4 — waveforms, spectrograms, plots
│   └── ml.py              Phase 5 — RF / GBM / SVM + LOSO evaluation
├── tests/
│   ├── test_parser.py          13 tests
│   ├── test_preprocessing.py   17 tests
│   ├── test_features.py        25 tests
│   └── test_ml.py              35 tests   (90 total, all passing)
├── scripts/
│   ├── generate_plots.py        generate 5 analysis plots
│   ├── run_ml_full.py           ML on all 1568 files (~8 min)
│   ├── run_advanced.py          confusion matrix, multi-model comparison
│   └── run_corrected_loso.py    corrected cross-session evaluation
├── plots/                  5 PNG analysis plots
├── results/                16 CSV result files
├── report.ipynb            37-cell Jupyter notebook (fully executed)
├── report.html             standalone HTML report (no Python needed)
├── INTERNSHIP_REPORT.html  this document
└── requirements.txt        all dependencies
</pre>

<h2>Appendix C — Reproducing the Results</h2>
<pre>
# 1. Create virtual environment
python -m venv .venv
.venv\\Scripts\\Activate.ps1          # Windows PowerShell

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set Python path
$env:PYTHONPATH = "src"

# 4. Run all 90 tests
pytest tests/ -v

# 5. Open the interactive notebook
jupyter notebook report.ipynb

# 6. Or view the standalone report
start report.html                     # opens in default browser
</pre>

<hr style="margin:40px 0 20px; border-color:#e2e8f0;"/>
<p style="text-align:center; font-size:9pt; color:#9ca3af;">
  Aditya Singh &nbsp;|&nbsp; IAESTE 2026 &nbsp;|&nbsp;
  VŠB – Technical University of Ostrava &nbsp;|&nbsp; June 2026
</p>

</body>
</html>"""

OUT.write_text(HTML, encoding="utf-8")
size_kb = OUT.stat().st_size // 1024
print(f"Written: {OUT}")
print(f"Size:    {size_kb} KB  ({size_kb//1024:.1f} MB)")
print(f"Open in browser -> File -> Print -> Save as PDF")
