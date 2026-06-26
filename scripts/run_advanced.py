"""Advanced evaluation: cross-session validation + multi-model comparison.

Loads the pre-computed feature matrix (results/feature_matrix_full.csv)
so no WAV re-processing is needed. Runs in ~2-3 minutes.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
from iaeste26.ml import (
    train_pressure_model, train_material_model,
    cross_session_eval, compare_models,
)

RESULTS_DIR = Path(__file__).parent.parent / "results"
SEP  = "=" * 64
SEP2 = "-" * 64

print(SEP)
print("ADVANCED EVALUATION")
print("  1. CV variance (are results stable across folds?)")
print("  2. Confusion matrix (which materials get confused?)")
print("  3. Leave-one-session-out (does model generalise to new days?)")
print("  4. Multi-model comparison (RF vs GBM vs SVM)")
print(SEP)

print("\nLoading feature matrix...")
df = pd.read_csv(RESULTS_DIR / "feature_matrix_full.csv")
print(f"  {len(df)} samples x {df.shape[1]} columns")
print(f"  Sessions: {sorted(df['session'].unique())}")


# -----------------------------------------------------------------------
# 1 + 2: CV variance + confusion matrix
# -----------------------------------------------------------------------
print()
print(SEP)
print("1 & 2: Pressure R2 variance across folds + Material confusion matrix")
print(SEP)

res_p = train_pressure_model(df, cv=5)
res_m = train_material_model(df, cv=5)

print(f"\n  Pressure prediction (5-fold CV):")
print(f"    R2  = {res_p['r2']:.4f}  +/- {res_p['r2_std']:.4f}  (std across folds)")
print(f"    RMSE = {res_p['rmse']:.4f} bar")

print(f"\n  Material classification (5-fold CV):")
print(f"    Accuracy = {res_m['accuracy']:.4f}  +/- {res_m['accuracy_std']:.4f}")
print(f"    F1       = {res_m['f1']:.4f}")

print(f"\n  Confusion matrix (rows=true, cols=predicted):")
classes = res_m["classes"]
cm = res_m["confusion_matrix"]
header = f"  {'':>8} " + " ".join(f"{c:>8}" for c in classes)
print(header)
for i, row_cls in enumerate(classes):
    row_str = "  " + f"{row_cls:>8} " + " ".join(f"{cm[i,j]:>8}" for j in range(len(classes)))
    print(row_str)

# Per-class accuracy from confusion matrix
print()
for i, cls in enumerate(classes):
    per_class = cm[i, i] / cm[i].sum()
    print(f"    {cls:>8}: {per_class:.1%} correctly identified")


# -----------------------------------------------------------------------
# 3. Leave-one-session-out
# -----------------------------------------------------------------------
print()
print(SEP)
print("3. Leave-One-Session-Out (LOSO) — realistic generalisation test")
print("   Train on 6 sessions -> test on 1 unseen session")
print(SEP)

print("\n  Pressure prediction (LOSO):")
loso_p = cross_session_eval(df, task="pressure")
print(f"  {'Session':>10}  {'n_train':>8}  {'n_test':>7}  {'RMSE':>8}  {'R2':>8}")
print(f"  {SEP2}")
for r in loso_p:
    print(f"  {r['test_session']:>10}  {r['n_train']:>8}  {r['n_test']:>7}  "
          f"{r['rmse']:>8.4f}  {r['r2']:>8.4f}")
mean_r2   = np.mean([r['r2']   for r in loso_p])
mean_rmse = np.mean([r['rmse'] for r in loso_p])
std_r2    = np.std( [r['r2']   for r in loso_p])
print(f"  {SEP2}")
print(f"  {'MEAN':>10}  {'':>8}  {'':>7}  {mean_rmse:>8.4f}  {mean_r2:>8.4f}  (std R2={std_r2:.4f})")

print("\n  Material classification (LOSO):")
loso_m = cross_session_eval(df, task="material")
print(f"  {'Session':>10}  {'n_train':>8}  {'n_test':>7}  {'Accuracy':>10}  {'F1':>8}")
print(f"  {SEP2}")
for r in loso_m:
    print(f"  {r['test_session']:>10}  {r['n_train']:>8}  {r['n_test']:>7}  "
          f"{r['accuracy']:>10.4f}  {r['f1']:>8.4f}")
mean_acc = np.mean([r['accuracy'] for r in loso_m])
std_acc  = np.std( [r['accuracy'] for r in loso_m])
print(f"  {SEP2}")
print(f"  {'MEAN':>10}  {'':>8}  {'':>7}  {mean_acc:>10.4f}  (std={std_acc:.4f})")


# -----------------------------------------------------------------------
# 4. Multi-model comparison
# -----------------------------------------------------------------------
print()
print(SEP)
print("4. Multi-model comparison (5-fold CV) — RF vs GBM vs SVM")
print(SEP)

print("\n  Pressure prediction:")
cmp_p = compare_models(df, task="pressure", cv=5)
print(f"  {'Model':>20}  {'RMSE':>8}  {'R2':>8}  {'R2 std':>8}")
print(f"  {SEP2}")
for name, res in sorted(cmp_p.items(), key=lambda x: x[1]['r2'], reverse=True):
    print(f"  {name:>20}  {res['rmse']:>8.4f}  {res['r2']:>8.4f}  {res['r2_std']:>8.4f}")

print("\n  Material classification:")
cmp_m = compare_models(df, task="material", cv=5)
print(f"  {'Model':>20}  {'Accuracy':>10}  {'F1':>8}  {'Acc std':>8}")
print(f"  {SEP2}")
for name, res in sorted(cmp_m.items(), key=lambda x: x[1]['accuracy'], reverse=True):
    print(f"  {name:>20}  {res['accuracy']:>10.4f}  {res['f1']:>8.4f}  {res['accuracy_std']:>8.4f}")


# -----------------------------------------------------------------------
# Save
# -----------------------------------------------------------------------
pd.DataFrame(loso_p).to_csv(RESULTS_DIR / "loso_pressure.csv", index=False)
pd.DataFrame(loso_m).to_csv(RESULTS_DIR / "loso_material.csv", index=False)
pd.DataFrame([
    {"model": k, **v} for k, v in cmp_p.items()
]).to_csv(RESULTS_DIR / "model_comparison_pressure.csv", index=False)
pd.DataFrame([
    {"model": k, **v} for k, v in cmp_m.items()
]).to_csv(RESULTS_DIR / "model_comparison_material.csv", index=False)

print()
print(SEP)
print("Saved: loso_pressure.csv, loso_material.csv,")
print("       model_comparison_pressure.csv, model_comparison_material.csv")
print(SEP)
