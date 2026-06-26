"""Tests for ml.py — ML pipeline functions.

Uses synthetic DataFrames so no WAV files are needed.
"""

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from iaeste26.ml import (
    FEATURE_COLS,
    get_X,
    get_feature_names,
    session_normalize,
    train_pressure_model,
    train_material_model,
    cross_session_eval,
    within_material_loso,
    compare_models,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n_per_session=30, seed=0):
    """Synthetic feature DataFrame with 2 sessions and 2 materials."""
    rng = np.random.default_rng(seed)
    sessions  = ["sA", "sB"]
    materials = ["Mat1", "Mat2"]
    pressures = [3.0, 5.0, 7.0]

    rows = []
    for sess in sessions:
        for mat in materials:
            for pres in pressures:
                for _ in range(n_per_session // (len(materials) * len(pressures))):
                    row = {"session": sess, "material": mat,
                           "nozzle_mm": 8.0, "pressure_bar": pres,
                           "mix_ratio_pct": 50.0, "sensor": "Mic46BE"}
                    # Features: mat/pressure signal + noise
                    mat_offset  = 1.0 if mat == "Mat1" else -1.0
                    pres_offset = pres / 10.0
                    for col in FEATURE_COLS:
                        row[col] = mat_offset + pres_offset + rng.normal(0, 0.1)
                    rows.append(row)
    return pd.DataFrame(rows)


def _make_multi_material_df(seed=42):
    """Synthetic DF with 2 sessions per material (G80 and GH120 style)."""
    rng = np.random.default_rng(seed)
    rows = []
    for sess, mat in [("v24","G80"),("v25","G80"),("v30","GH120"),("v31","GH120")]:
        for pres in [3.0, 5.0, 7.0]:
            for _ in range(10):
                row = {"session": sess, "material": mat,
                       "nozzle_mm": 8.0, "pressure_bar": pres,
                       "mix_ratio_pct": 50.0, "sensor": "Mic46BE"}
                for col in FEATURE_COLS:
                    row[col] = pres / 7.5 + rng.normal(0, 0.3)
                rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# get_X / get_feature_names
# ---------------------------------------------------------------------------

class TestGetX:
    def test_shape(self):
        df = _make_df()
        X = get_X(df)
        assert X.shape == (len(df), len(FEATURE_COLS))

    def test_dtype(self):
        df = _make_df()
        X = get_X(df)
        assert X.dtype == np.float32

    def test_feature_names_length(self):
        df = _make_df()
        names = get_feature_names(df)
        assert len(names) == len(FEATURE_COLS)

    def test_feature_names_subset(self):
        df = _make_df()
        df = df.drop(columns=["rms"])
        names = get_feature_names(df)
        assert "rms" not in names
        assert len(names) == len(FEATURE_COLS) - 1


# ---------------------------------------------------------------------------
# session_normalize
# ---------------------------------------------------------------------------

class TestSessionNormalize:
    def test_returns_dataframe(self):
        df = _make_df()
        out = session_normalize(df)
        assert isinstance(out, pd.DataFrame)

    def test_does_not_modify_input(self):
        df = _make_df()
        original_rms = df["rms"].values.copy()
        session_normalize(df)
        np.testing.assert_array_equal(df["rms"].values, original_rms)

    def test_within_session_mean_near_zero(self):
        df = _make_df(n_per_session=60)
        out = session_normalize(df)
        for sess in out["session"].unique():
            sub = out[out["session"] == sess]
            assert abs(sub["rms"].mean()) < 0.2

    def test_metadata_columns_unchanged(self):
        df = _make_df()
        out = session_normalize(df)
        pd.testing.assert_series_equal(df["session"], out["session"])
        pd.testing.assert_series_equal(df["material"], out["material"])

    def test_preserves_shape(self):
        df = _make_df()
        out = session_normalize(df)
        assert out.shape == df.shape


# ---------------------------------------------------------------------------
# train_pressure_model
# ---------------------------------------------------------------------------

class TestTrainPressureModel:
    def test_returns_keys(self):
        df = _make_df()
        res = train_pressure_model(df, n_estimators=10, cv=2)
        for key in ["rmse", "r2", "r2_std", "feature_importances", "model"]:
            assert key in res

    def test_rmse_positive(self):
        df = _make_df()
        res = train_pressure_model(df, n_estimators=10, cv=2)
        assert res["rmse"] > 0

    def test_r2_in_valid_range(self):
        df = _make_df()
        res = train_pressure_model(df, n_estimators=10, cv=2)
        assert -1.0 <= res["r2"] <= 1.0

    def test_r2_std_non_negative(self):
        df = _make_df()
        res = train_pressure_model(df, n_estimators=10, cv=2)
        assert res["r2_std"] >= 0

    def test_feature_importances_sum_to_one(self):
        df = _make_df()
        res = train_pressure_model(df, n_estimators=10, cv=2)
        total = sum(res["feature_importances"].values())
        assert abs(total - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# train_material_model
# ---------------------------------------------------------------------------

class TestTrainMaterialModel:
    def test_returns_keys(self):
        df = _make_df()
        res = train_material_model(df, n_estimators=10, cv=2)
        for key in ["accuracy", "accuracy_std", "f1", "classes",
                    "confusion_matrix", "feature_importances", "model"]:
            assert key in res

    def test_accuracy_between_0_and_1(self):
        df = _make_df()
        res = train_material_model(df, n_estimators=10, cv=2)
        assert 0.0 <= res["accuracy"] <= 1.0

    def test_classes_match_input(self):
        df = _make_df()
        res = train_material_model(df, n_estimators=10, cv=2)
        assert set(res["classes"]) == set(df["material"].unique())

    def test_confusion_matrix_shape(self):
        df = _make_df()
        res = train_material_model(df, n_estimators=10, cv=2)
        n = len(res["classes"])
        assert res["confusion_matrix"].shape == (n, n)

    def test_confusion_matrix_row_sums(self):
        df = _make_df()
        res = train_material_model(df, n_estimators=10, cv=2)
        # Each row sums to the number of true samples for that class
        assert res["confusion_matrix"].sum() == len(df)

    def test_accuracy_std_non_negative(self):
        df = _make_df()
        res = train_material_model(df, n_estimators=10, cv=2)
        assert res["accuracy_std"] >= 0


# ---------------------------------------------------------------------------
# cross_session_eval
# ---------------------------------------------------------------------------

class TestCrossSessionEval:
    def test_returns_one_record_per_session(self):
        df = _make_df()
        records = cross_session_eval(df, task="pressure", n_estimators=10)
        assert len(records) == len(df["session"].unique())

    def test_pressure_keys(self):
        df = _make_df()
        r = cross_session_eval(df, task="pressure", n_estimators=10)[0]
        for key in ["test_session", "n_train", "n_test", "rmse", "r2"]:
            assert key in r

    def test_material_keys(self):
        df = _make_df()
        r = cross_session_eval(df, task="material", n_estimators=10)[0]
        for key in ["test_session", "n_train", "n_test", "accuracy", "f1"]:
            assert key in r

    def test_n_train_plus_n_test_equals_total(self):
        df = _make_df()
        records = cross_session_eval(df, task="pressure", n_estimators=10)
        n_total = len(df)
        for r in records:
            assert r["n_train"] + r["n_test"] == n_total

    def test_normalize_flag_runs(self):
        df = _make_df()
        records = cross_session_eval(df, task="pressure",
                                     n_estimators=10, normalize=True)
        assert len(records) == len(df["session"].unique())


# ---------------------------------------------------------------------------
# within_material_loso
# ---------------------------------------------------------------------------

class TestWithinMaterialLoso:
    def test_returns_records_for_valid_pairs(self):
        df = _make_multi_material_df()
        records = within_material_loso(df, task="pressure", n_estimators=10)
        # G80 (v24<->v25) and GH120 (v30<->v31) each give 2 directions
        assert len(records) == 4

    def test_pressure_keys_present(self):
        df = _make_multi_material_df()
        r = within_material_loso(df, task="pressure", n_estimators=10)[0]
        for key in ["material", "train_session", "test_session",
                    "n_train", "n_test", "rmse", "r2"]:
            assert key in r

    def test_r2_in_valid_range(self):
        df = _make_multi_material_df()
        records = within_material_loso(df, task="pressure", n_estimators=10)
        for r in records:
            assert -5.0 <= r["r2"] <= 1.0

    def test_mix_task_runs(self):
        df = _make_multi_material_df()
        records = within_material_loso(df, task="mix", n_estimators=10)
        assert len(records) == 4


# ---------------------------------------------------------------------------
# compare_models
# ---------------------------------------------------------------------------

class TestCompareModels:
    def test_pressure_returns_three_models(self):
        df = _make_df()
        res = compare_models(df, task="pressure", cv=2)
        assert len(res) == 3

    def test_material_returns_three_models(self):
        df = _make_df()
        res = compare_models(df, task="material", cv=2)
        assert len(res) == 3

    def test_pressure_keys(self):
        df = _make_df()
        res = compare_models(df, task="pressure", cv=2)
        for name, r in res.items():
            assert "rmse" in r and "r2" in r and "r2_std" in r

    def test_material_keys(self):
        df = _make_df()
        res = compare_models(df, task="material", cv=2)
        for name, r in res.items():
            assert "accuracy" in r and "f1" in r and "accuracy_std" in r

    def test_all_r2_in_valid_range(self):
        df = _make_df()
        res = compare_models(df, task="pressure", cv=2)
        for name, r in res.items():
            assert -5.0 <= r["r2"] <= 1.0, f"{name} R2={r['r2']} out of range"

    def test_all_accuracy_in_valid_range(self):
        df = _make_df()
        res = compare_models(df, task="material", cv=2)
        for name, r in res.items():
            assert 0.0 <= r["accuracy"] <= 1.0
