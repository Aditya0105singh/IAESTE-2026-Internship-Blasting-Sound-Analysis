"""Phase 5: Train ML models and print results.

Uses session v24 (224 files) for speed.
Each file: 15-second window, 35 features extracted.
Models: Random Forest with 5-fold cross-validation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import time
import pandas as pd

from iaeste26.ml import (
    build_feature_matrix,
    train_pressure_model,
    train_material_model,
    train_mix_model,
    compare_sensors,
    FEATURE_COLS,
)

DATA_DIR    = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SEP = "=" * 60


# -----------------------------------------------------------------------
# Step 1: Build feature matrix (v24 session = 224 files)
# -----------------------------------------------------------------------
print(SEP)
print("PHASE 5 — Machine Learning Pipeline")
print(SEP)
print()
print("Step 1: Extracting features from session v24 (224 files)...")
print("  (Using 15-second window from each recording)")
print()

t0 = time.time()
df = build_feature_matrix(DATA_DIR, session="v24", window_sec=15.0, verbose=True)
elapsed = time.time() - t0

print()
print(f"  Done in {elapsed:.1f}s")
print(f"  Feature matrix: {len(df)} samples x {len(FEATURE_COLS)} features")
print(f"  Targets  -> pressure range: {df['pressure_bar'].min()}–{df['pressure_bar'].max()} bar")
print(f"             materials: {sorted(df['material'].unique())}")
print(f"             mix ratios: {sorted(df['mix_ratio_pct'].unique())}%")

# Save feature matrix
csv_path = RESULTS_DIR / "feature_matrix_v24.csv"
df.to_csv(csv_path, index=False)
print(f"  Saved to: {csv_path}")


# -----------------------------------------------------------------------
# Step 2: Pressure prediction (regression)
# -----------------------------------------------------------------------
print()
print(SEP)
print("Step 2: Pressure Prediction (Regression)")
print(SEP)
print("  Model: Random Forest (100 trees, 5-fold CV)")
print("  Target: predict pressure_bar (3.0 – 7.5 bar)")
print()

res_p = train_pressure_model(df)
print(f"  RMSE  = {res_p['rmse']:.4f} bar   (lower = better; scale: 3–7.5 bar)")
print(f"  R^2   = {res_p['r2']:.4f}        (1.0 = perfect, 0.0 = random guess)")
print()
if res_p['r2'] > 0.85:
    verdict = "EXCELLENT — model learns pressure very well from sound"
elif res_p['r2'] > 0.65:
    verdict = "GOOD — model captures most of the pressure variation"
else:
    verdict = "MODERATE — some signal, but pressure is hard to predict"
print(f"  Verdict: {verdict}")


# -----------------------------------------------------------------------
# Step 3: Material classification
# -----------------------------------------------------------------------
print()
print(SEP)
print("Step 3: Abrasive Material Classification")
print(SEP)
print("  Model: Random Forest (100 trees, 5-fold CV)")
print("  Target: classify material (G80 / GH18 / GH40 / GH120)")
print()

res_m = train_material_model(df)
print(f"  Accuracy = {res_m['accuracy']:.4f}  ({res_m['accuracy']*100:.1f}%)")
print(f"  F1 score = {res_m['f1']:.4f}")
print(f"  Classes  : {res_m['classes']}")
print()
chance = 1 / len(res_m['classes'])
if res_m['accuracy'] > 0.80:
    verdict = f"EXCELLENT — well above chance ({chance:.0%})"
elif res_m['accuracy'] > 0.55:
    verdict = f"GOOD — significantly above chance ({chance:.0%})"
else:
    verdict = f"WEAK — only slightly above chance ({chance:.0%})"
print(f"  Verdict: {verdict}")


# -----------------------------------------------------------------------
# Step 4: Mix ratio prediction (bonus)
# -----------------------------------------------------------------------
print()
print(SEP)
print("Step 4: Mix Ratio Prediction (Regression)  [bonus]")
print(SEP)
print("  Model: Random Forest (100 trees, 5-fold CV)")
print("  Target: predict mix_ratio_pct (0 – 100%)")
print()

res_mx = train_mix_model(df)
print(f"  RMSE  = {res_mx['rmse']:.2f}%   (scale: 0–100%)")
print(f"  R^2   = {res_mx['r2']:.4f}")


# -----------------------------------------------------------------------
# Step 5: Sensor comparison — which sensor is best?
# -----------------------------------------------------------------------
print()
print(SEP)
print("Step 5: Sensor Comparison — which sensor predicts pressure best?")
print(SEP)
print()

sensor_results = compare_sensors(df, task="pressure")
sensor_rows = []
for sensor, res in sorted(sensor_results.items(), key=lambda x: x[1]['r2'], reverse=True):
    n_files = len(df[df["sensor"] == sensor])
    print(f"  {sensor:<22}  RMSE={res['rmse']:.4f} bar   R2={res['r2']:.4f}   (n={n_files})")
    sensor_rows.append({
        "sensor": sensor,
        "rmse_bar": round(res['rmse'], 4),
        "r2": round(res['r2'], 4),
        "n_files": n_files,
    })

print()
best = max(sensor_results.items(), key=lambda x: x[1]['r2'])
worst = min(sensor_results.items(), key=lambda x: x[1]['r2'])
print(f"  Best sensor  : {best[0]}  (R2={best[1]['r2']:.4f})")
print(f"  Worst sensor : {worst[0]}  (R2={worst[1]['r2']:.4f})")


# -----------------------------------------------------------------------
# Step 6: Top 10 most important features
# -----------------------------------------------------------------------
print()
print(SEP)
print("Step 6: Top 10 Most Important Features (for pressure)")
print(SEP)
print("  (Feature importance = how much each feature helps the model)")
print()

importances = res_p["feature_importances"]
top10 = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
for rank, (feat, imp) in enumerate(top10, 1):
    bar = "#" * int(imp * 200)
    print(f"  {rank:>2}. {feat:<28}  {imp:.4f}  {bar}")


# -----------------------------------------------------------------------
# Save results summary
# -----------------------------------------------------------------------
summary = {
    "pressure_rmse": round(res_p['rmse'], 4),
    "pressure_r2": round(res_p['r2'], 4),
    "material_accuracy": round(res_m['accuracy'], 4),
    "material_f1": round(res_m['f1'], 4),
    "mix_rmse": round(res_mx['rmse'], 4),
    "mix_r2": round(res_mx['r2'], 4),
}

pd.DataFrame([summary]).to_csv(RESULTS_DIR / "ml_summary.csv", index=False)
pd.DataFrame(sensor_rows).to_csv(RESULTS_DIR / "sensor_comparison.csv", index=False)
pd.DataFrame(top10, columns=["feature", "importance"]).to_csv(
    RESULTS_DIR / "feature_importance.csv", index=False
)

print()
print(SEP)
print("Results saved to results/ folder:")
for f in sorted(RESULTS_DIR.glob("*.csv")):
    print(f"  {f.name}")
print(SEP)
