"""Phase 5 (full): Train ML models on ALL sessions (1568 files, 4 materials).

Takes ~5-7 minutes. Results will be much stronger than single-session run.
"""

import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
from iaeste26.ml import (
    build_feature_matrix, train_pressure_model, train_material_model,
    train_mix_model, compare_sensors, FEATURE_COLS,
)

DATA_DIR    = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SEP  = "=" * 62
SEP2 = "-" * 62


# -----------------------------------------------------------------------
# Step 1: Build full feature matrix (all sessions)
# -----------------------------------------------------------------------
print(SEP)
print("PHASE 5 (FULL) -- ML on all sessions (1568 files)")
print(SEP)
print()
print("Step 1: Extracting features from all sessions...")
print("  (15-second window per file | estimated ~5-7 minutes)")
print()

t0 = time.time()
df = build_feature_matrix(DATA_DIR, session=None, window_sec=15.0, verbose=True)
elapsed = time.time() - t0

print()
print(f"  Done in {elapsed/60:.1f} min ({elapsed:.0f}s)")
print(f"  Feature matrix : {len(df)} samples x {len(FEATURE_COLS)} features")
print(f"  Materials      : {sorted(df['material'].unique())}")
print(f"  Pressures (bar): {sorted(df['pressure_bar'].unique())}")
print(f"  Mix ratios (%) : {sorted(df['mix_ratio_pct'].unique())}")
print(f"  Sensors        : {sorted(df['sensor'].unique())}")

csv_path = RESULTS_DIR / "feature_matrix_full.csv"
df.to_csv(csv_path, index=False)
print(f"  Saved: {csv_path.name}")


# -----------------------------------------------------------------------
# Step 2: Pressure prediction
# -----------------------------------------------------------------------
print()
print(SEP)
print("Step 2: Pressure Prediction (Regression)")
print(SEP)
print("  Random Forest, 100 trees, 5-fold CV | range: 3.0-7.5 bar")
print()

res_p = train_pressure_model(df)
pct_error = (res_p['rmse'] / (7.5 - 3.0)) * 100
print(f"  RMSE  = {res_p['rmse']:.4f} bar  ({pct_error:.1f}% of full scale)")
print(f"  R^2   = {res_p['r2']:.4f}")
if   res_p['r2'] > 0.85: verdict = "EXCELLENT"
elif res_p['r2'] > 0.65: verdict = "GOOD"
else:                     verdict = "MODERATE"
print(f"  Result: {verdict}")


# -----------------------------------------------------------------------
# Step 3: Material classification (4 classes)
# -----------------------------------------------------------------------
print()
print(SEP)
print("Step 3: Material Classification  (G80 / GH18 / GH40 / GH120)")
print(SEP)
print("  Random Forest, 100 trees, 5-fold CV | 4 classes | chance=25%")
print()

res_m = train_material_model(df)
print(f"  Accuracy = {res_m['accuracy']:.4f}  ({res_m['accuracy']*100:.1f}%)")
print(f"  F1 score = {res_m['f1']:.4f}")
print(f"  Classes  : {res_m['classes']}")
if   res_m['accuracy'] > 0.85: verdict = "EXCELLENT -- well above 25% chance"
elif res_m['accuracy'] > 0.55: verdict = "GOOD -- significantly above 25% chance"
else:                           verdict = "WEAK -- barely above 25% chance"
print(f"  Result: {verdict}")


# -----------------------------------------------------------------------
# Step 4: Mix ratio prediction
# -----------------------------------------------------------------------
print()
print(SEP)
print("Step 4: Mix Ratio Prediction  (0% - 100%)")
print(SEP)
print("  Random Forest, 100 trees, 5-fold CV")
print()

res_mx = train_mix_model(df)
print(f"  RMSE  = {res_mx['rmse']:.2f}%")
print(f"  R^2   = {res_mx['r2']:.4f}")


# -----------------------------------------------------------------------
# Step 5: Sensor comparison (pressure task)
# -----------------------------------------------------------------------
print()
print(SEP)
print("Step 5: Sensor Comparison -- Pressure Prediction per Sensor")
print(SEP)
print()

sensor_res = compare_sensors(df, task="pressure")
rows = []
print(f"  {'Sensor':<22}  {'RMSE (bar)':>10}  {'R^2':>8}  {'n files':>8}")
print(f"  {SEP2}")
for sensor, res in sorted(sensor_res.items(), key=lambda x: x[1]['r2'], reverse=True):
    n = len(df[df["sensor"] == sensor])
    print(f"  {sensor:<22}  {res['rmse']:>10.4f}  {res['r2']:>8.4f}  {n:>8}")
    rows.append({"sensor": sensor, "rmse_bar": round(res['rmse'], 4),
                 "r2": round(res['r2'], 4), "n_files": n})

best  = max(sensor_res.items(), key=lambda x: x[1]['r2'])
worst = min(sensor_res.items(), key=lambda x: x[1]['r2'])
print()
print(f"  Best  sensor: {best[0]}   (R2={best[1]['r2']:.4f})")
print(f"  Worst sensor: {worst[0]}  (R2={worst[1]['r2']:.4f})")


# -----------------------------------------------------------------------
# Step 6: Sensor comparison (material task)
# -----------------------------------------------------------------------
print()
print(SEP)
print("Step 6: Sensor Comparison -- Material Classification per Sensor")
print(SEP)
print()

sensor_mat = compare_sensors(df, task="material")
mat_rows = []
print(f"  {'Sensor':<22}  {'Accuracy':>10}  {'F1':>8}  {'n files':>8}")
print(f"  {SEP2}")
for sensor, res in sorted(sensor_mat.items(), key=lambda x: x[1]['accuracy'], reverse=True):
    n = len(df[df["sensor"] == sensor])
    print(f"  {sensor:<22}  {res['accuracy']:>10.4f}  {res['f1']:>8.4f}  {n:>8}")
    mat_rows.append({"sensor": sensor, "accuracy": round(res['accuracy'], 4),
                     "f1": round(res['f1'], 4), "n_files": n})

best_m  = max(sensor_mat.items(), key=lambda x: x[1]['accuracy'])
worst_m = min(sensor_mat.items(), key=lambda x: x[1]['accuracy'])
print()
print(f"  Best  sensor: {best_m[0]}   (Accuracy={best_m[1]['accuracy']:.4f})")
print(f"  Worst sensor: {worst_m[0]}  (Accuracy={worst_m[1]['accuracy']:.4f})")


# -----------------------------------------------------------------------
# Step 7: Top 10 features
# -----------------------------------------------------------------------
print()
print(SEP)
print("Step 7: Top 10 Most Important Features  (for pressure)")
print(SEP)
print()

importances = res_p["feature_importances"]
top10 = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
for rank, (feat, imp) in enumerate(top10, 1):
    bar = "#" * int(imp * 200)
    print(f"  {rank:>2}. {feat:<28}  {imp:.4f}  {bar}")


# -----------------------------------------------------------------------
# Save everything
# -----------------------------------------------------------------------
summary = {
    "n_files": len(df), "n_features": len(FEATURE_COLS),
    "pressure_rmse": round(res_p['rmse'], 4), "pressure_r2": round(res_p['r2'], 4),
    "material_accuracy": round(res_m['accuracy'], 4), "material_f1": round(res_m['f1'], 4),
    "mix_rmse": round(res_mx['rmse'], 4), "mix_r2": round(res_mx['r2'], 4),
}
pd.DataFrame([summary]).to_csv(RESULTS_DIR / "ml_summary_full.csv", index=False)
pd.DataFrame(rows).to_csv(RESULTS_DIR / "sensor_comparison_pressure_full.csv", index=False)
pd.DataFrame(mat_rows).to_csv(RESULTS_DIR / "sensor_comparison_material_full.csv", index=False)
pd.DataFrame(top10, columns=["feature","importance"]).to_csv(
    RESULTS_DIR / "feature_importance_full.csv", index=False)

print()
print(SEP)
print("All results saved to results/ folder.")
print(SEP)
