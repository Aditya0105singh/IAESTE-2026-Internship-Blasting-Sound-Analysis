"""Corrected cross-session evaluation.

Key finding: each session recorded ONE material only (v24/v25=G80,
v26/v27=GH40, v28=GH18, v30/v31=GH120). Full LOSO for material
is therefore zero-shot, not domain adaptation.

Correct evaluations:
  1. Full LOSO for PRESSURE (valid — pressure varies in all sessions)
  2. Within-material LOSO for G80 (v24<->v25) and GH120 (v30<->v31)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
from iaeste26.ml import cross_session_eval, within_material_loso

RESULTS_DIR = Path(__file__).parent.parent / "results"
SEP  = "=" * 64
SEP2 = "-" * 64

df = pd.read_csv(RESULTS_DIR / "feature_matrix_full.csv")

print(SEP)
print("DATASET STRUCTURE (one material per session)")
print(SEP)
print(df.groupby(['session','material']).size().unstack(fill_value=0).to_string())

print()
print(SEP)
print("1. Full LOSO — Pressure Prediction (all 7 sessions)")
print("   Valid: pressure varies across all sessions")
print(SEP)
loso_p = cross_session_eval(df, task='pressure')
print(f"  {'Session':>10}  {'n_train':>8}  {'n_test':>7}  {'RMSE':>8}  {'R2':>8}")
print(f"  {SEP2}")
for r in loso_p:
    print(f"  {r['test_session']:>10}  {r['n_train']:>8}  {r['n_test']:>7}  "
          f"{r['rmse']:>8.4f}  {r['r2']:>8.4f}")
mean_r2   = np.mean([r['r2']   for r in loso_p])
mean_rmse = np.mean([r['rmse'] for r in loso_p])
std_r2    = np.std( [r['r2']   for r in loso_p])
print(f"  {SEP2}")
print(f"  {'MEAN':>10}  {'':>8}  {'':>7}  {mean_rmse:>8.4f}  {mean_r2:>8.4f}  std={std_r2:.4f}")

print()
print(SEP)
print("2. Within-Material LOSO — Pressure (G80 & GH120 only)")
print("   Proper test: same material, different recording day")
print(SEP)
wm_p = within_material_loso(df, task='pressure')
print(f"  {'Material':>8}  {'Train':>7}  {'Test':>7}  {'n_train':>8}  {'n_test':>7}  {'RMSE':>8}  {'R2':>8}")
print(f"  {SEP2}")
for r in wm_p:
    print(f"  {r['material']:>8}  {r['train_session']:>7}  {r['test_session']:>7}  "
          f"{r['n_train']:>8}  {r['n_test']:>7}  {r['rmse']:>8.4f}  {r['r2']:>8.4f}")

print()
print(SEP)
print("3. Within-Material LOSO — Mix Ratio (G80 & GH120 only)")
print(SEP)
wm_mx = within_material_loso(df, task='mix')
print(f"  {'Material':>8}  {'Train':>7}  {'Test':>7}  {'n_train':>8}  {'n_test':>7}  {'RMSE':>8}  {'R2':>8}")
print(f"  {SEP2}")
for r in wm_mx:
    print(f"  {r['material']:>8}  {r['train_session']:>7}  {r['test_session']:>7}  "
          f"{r['n_train']:>8}  {r['n_test']:>7}  {r['rmse']:>8.4f}  {r['r2']:>8.4f}")

# Save
pd.DataFrame(loso_p).to_csv(RESULTS_DIR / "loso_pressure_corrected.csv", index=False)
pd.DataFrame(wm_p ).to_csv(RESULTS_DIR / "within_material_loso_pressure.csv", index=False)
pd.DataFrame(wm_mx).to_csv(RESULTS_DIR / "within_material_loso_mix.csv", index=False)

print()
print(SEP)
print("Saved corrected LOSO results to results/")
print(SEP)
