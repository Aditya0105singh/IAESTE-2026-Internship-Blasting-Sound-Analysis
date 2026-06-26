"""Machine learning pipeline for blasting condition estimation.

Tasks:
  1. Pressure regression  — predict pressure_bar from audio features
  2. Material classification — predict abrasive material from audio features
  3. Mix ratio regression — predict abrasive-to-air ratio

All models use Random Forest with 5-fold cross-validation.
Advanced evaluation: leave-one-session-out cross-validation, multi-model comparison.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, f1_score, confusion_matrix,
)

from .parser import parse_filename, load_wav
from .preprocessing import preprocess
from .features import extract_features


# ---------------------------------------------------------------------------
# Feature matrix builder
# ---------------------------------------------------------------------------

def build_feature_matrix(
    data_dir: str | Path,
    session: str | None = None,
    window_sec: float = 15.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """Extract features from all WAV files and return a single DataFrame.

    Each row = one WAV file, with 35 feature columns + 5 metadata columns.

    Args:
        data_dir:   root data directory (contains v24, v25, ... subfolders)
        session:    if given, restrict to this session (e.g. "v24")
        window_sec: seconds of audio to use per file (from middle of recording)
        verbose:    print progress

    Returns:
        DataFrame with columns: session, material, nozzle_mm, pressure_bar,
        mix_ratio_pct, sensor, + 35 feature columns
    """
    data_dir = Path(data_dir)
    wav_files = sorted(data_dir.rglob("*.wav"))

    if session:
        wav_files = [f for f in wav_files if f.parent.parent.name == session]

    records = []
    n = len(wav_files)

    for i, wav_path in enumerate(wav_files, 1):
        try:
            meta = parse_filename(wav_path.name)
        except ValueError:
            continue

        session_name = wav_path.parent.parent.name

        if verbose and (i % 20 == 0 or i == 1 or i == n):
            print(f"  [{i:>3}/{n}] {wav_path.name}")

        try:
            audio, sr = load_wav(wav_path)
            audio = preprocess(audio, sr, sensor=meta["sensor"])

            # Use a fixed-length window from the middle of the clean signal
            n_samples = int(window_sec * sr)
            if len(audio) > n_samples:
                start = (len(audio) - n_samples) // 2
                audio = audio[start: start + n_samples]

            feats = extract_features(audio, sr)

        except Exception as e:
            if verbose:
                print(f"    WARNING: skipping {wav_path.name} — {e}")
            continue

        row = {
            "session":       session_name,
            "material":      meta["material"],
            "nozzle_mm":     meta["nozzle_mm"],
            "pressure_bar":  meta["pressure_bar"],
            "mix_ratio_pct": meta["mix_ratio_pct"],
            "sensor":        meta["sensor"],
            **feats,
        }
        records.append(row)

    df = pd.DataFrame(records)
    return df


# ---------------------------------------------------------------------------
# Feature column helpers
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "rms", "zero_crossing_rate", "peak_amplitude", "crest_factor", "kurtosis",
    "spectral_centroid", "spectral_bandwidth", "spectral_rolloff_85", "dominant_frequency",
    *[f"mfcc_{i}_mean" for i in range(1, 14)],
    *[f"mfcc_{i}_std"  for i in range(1, 14)],
]


def get_X(df: pd.DataFrame) -> np.ndarray:
    """Extract feature matrix from DataFrame."""
    cols = [c for c in FEATURE_COLS if c in df.columns]
    return df[cols].values.astype(np.float32)


def get_feature_names(df: pd.DataFrame) -> list[str]:
    return [c for c in FEATURE_COLS if c in df.columns]


# ---------------------------------------------------------------------------
# Model training + evaluation
# ---------------------------------------------------------------------------

def train_pressure_model(
    df: pd.DataFrame,
    n_estimators: int = 100,
    cv: int = 5,
) -> dict:
    """Train and evaluate pressure regression with Random Forest + CV.

    Returns dict with: rmse, r2, r2_std, feature_importances, model
    """
    X = get_X(df)
    y = df["pressure_bar"].values

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)

    y_pred = cross_val_predict(model, X, y, cv=cv)
    rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
    r2   = float(r2_score(y, y_pred))

    fold_r2 = cross_val_score(model, X, y, cv=cv, scoring="r2")
    r2_std  = float(fold_r2.std())

    model.fit(X, y)
    importances = dict(zip(get_feature_names(df), model.feature_importances_))

    return {"rmse": rmse, "r2": r2, "r2_std": r2_std,
            "feature_importances": importances, "model": model}


def train_material_model(
    df: pd.DataFrame,
    n_estimators: int = 100,
    cv: int = 5,
) -> dict:
    """Train and evaluate material classification with Random Forest + CV.

    Returns dict with: accuracy, accuracy_std, f1, classes, confusion_matrix,
                       feature_importances, model
    """
    X = get_X(df)
    le = LabelEncoder()
    y = le.fit_transform(df["material"].values)

    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)

    y_pred = cross_val_predict(model, X, y, cv=cv)
    accuracy = float(accuracy_score(y, y_pred))
    f1       = float(f1_score(y, y_pred, average="weighted"))
    cm       = confusion_matrix(y, y_pred)

    fold_acc    = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    accuracy_std = float(fold_acc.std())

    model.fit(X, y)
    importances = dict(zip(get_feature_names(df), model.feature_importances_))

    return {
        "accuracy": accuracy,
        "accuracy_std": accuracy_std,
        "f1": f1,
        "classes": list(le.classes_),
        "confusion_matrix": cm,
        "feature_importances": importances,
        "model": model,
    }


def train_mix_model(
    df: pd.DataFrame,
    n_estimators: int = 100,
    cv: int = 5,
) -> dict:
    """Train and evaluate mix ratio regression with Random Forest + CV."""
    X = get_X(df)
    y = df["mix_ratio_pct"].values

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    y_pred = cross_val_predict(model, X, y, cv=cv)
    rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
    r2   = float(r2_score(y, y_pred))

    model.fit(X, y)
    importances = dict(zip(get_feature_names(df), model.feature_importances_))

    return {"rmse": rmse, "r2": r2, "feature_importances": importances, "model": model}


# ---------------------------------------------------------------------------
# Sensor comparison
# ---------------------------------------------------------------------------

def compare_sensors(
    df: pd.DataFrame,
    task: str = "pressure",
    cv: int = 5,
) -> dict[str, dict]:
    """Train separate models for each sensor and compare performance.

    Args:
        df:   full feature DataFrame (all sensors)
        task: "pressure", "material", or "mix"
        cv:   cross-validation folds

    Returns:
        dict mapping sensor_name -> result dict
    """
    results = {}
    for sensor in sorted(df["sensor"].unique()):
        sub = df[df["sensor"] == sensor]
        if len(sub) < cv:
            continue
        if task == "pressure":
            results[sensor] = train_pressure_model(sub, cv=cv)
        elif task == "material":
            results[sensor] = train_material_model(sub, cv=cv)
        elif task == "mix":
            results[sensor] = train_mix_model(sub, cv=cv)
    return results


# ---------------------------------------------------------------------------
# Session-invariant normalisation
# ---------------------------------------------------------------------------

def session_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score each feature within its recording session.

    Removes session-level mean and variance (caused by microphone placement
    drift, room temperature, day-to-day equipment variation).  After this
    transform, features represent relative patterns within a session rather
    than absolute values, which improves leave-one-session-out accuracy.

    Valid for inference: at test time a full session is available, so
    its own mean/std can be computed before predicting.
    """
    df = df.copy()
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    for col in feat_cols:
        group_mean = df.groupby("session")[col].transform("mean")
        group_std  = df.groupby("session")[col].transform("std").replace(0, 1)
        df[col] = (df[col] - group_mean) / group_std
    return df


# ---------------------------------------------------------------------------
# Leave-one-session-out cross-validation
# ---------------------------------------------------------------------------

def cross_session_eval(
    df: pd.DataFrame,
    task: str = "pressure",
    n_estimators: int = 100,
    normalize: bool = False,
) -> list[dict]:
    """Leave-one-session-out (LOSO) evaluation — the realistic test.

    For each session: train on ALL other sessions, test on this one.
    This tests whether the model generalises to new recording days,
    not just new samples from the same day.

    Args:
        normalize: if True, apply session_normalize() before each fold.
                   This makes features session-invariant and improves
                   material classification generalisation.

    Returns list of dicts, one per session:
      pressure task: session, n_train, n_test, rmse, r2
      material task: session, n_train, n_test, accuracy, f1
    """
    sessions = sorted(df["session"].unique())
    le = LabelEncoder().fit(df["material"].values) if task == "material" else None
    records = []

    for test_sess in sessions:
        train_df = df[df["session"] != test_sess].copy()
        test_df  = df[df["session"] == test_sess].copy()

        if normalize:
            # Normalise each split independently (no data leakage)
            train_df = session_normalize(train_df)
            test_df  = session_normalize(test_df)

        X_train = get_X(train_df)
        X_test  = get_X(test_df)

        if task == "pressure":
            y_train = train_df["pressure_bar"].values
            y_test  = test_df["pressure_bar"].values
            model   = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            y_pred  = model.predict(X_test)
            records.append({
                "test_session": test_sess,
                "n_train": len(train_df),
                "n_test":  len(test_df),
                "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                "r2":   float(r2_score(y_test, y_pred)),
            })

        elif task == "material":
            y_train = le.transform(train_df["material"].values)
            y_test  = le.transform(test_df["material"].values)
            model   = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            y_pred  = model.predict(X_test)
            records.append({
                "test_session": test_sess,
                "n_train": len(train_df),
                "n_test":  len(test_df),
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "f1":       float(f1_score(y_test, y_pred, average="weighted")),
            })

    return records


def cross_session_pressure_eval(
    df: pd.DataFrame,
    n_estimators: int = 100,
) -> list[dict]:
    """LOSO for pressure prediction only — the only valid cross-session test.

    Dataset structure note: each session recorded exactly one abrasive material
    (v24/v25=G80, v26/v27=GH40, v28=GH18, v30/v31=GH120), so a full LOSO
    for material classification is equivalent to zero-shot learning (training
    without any samples of the test class).  Pressure prediction does not
    have this problem because pressure varies across all sessions.

    This function runs a proper LOSO for pressure and also tests cross-session
    within each material pair (G80: v24<->v25, GH120: v30<->v31).
    """
    return cross_session_eval(df, task="pressure", n_estimators=n_estimators)


def within_material_loso(
    df: pd.DataFrame,
    task: str = "pressure",
    n_estimators: int = 100,
) -> list[dict]:
    """Cross-session test within materials that have multiple sessions.

    Only G80 (v24, v25) and GH120 (v30, v31) have 2 sessions each.
    For these pairs: train on one session, test on the other.

    This is the proper way to test cross-session generalisation for
    within-material condition estimation (pressure, mix ratio).
    """
    pairs = [
        ("G80",   "v24", "v25"),
        ("G80",   "v25", "v24"),
        ("GH120", "v30", "v31"),
        ("GH120", "v31", "v30"),
    ]
    le = LabelEncoder().fit(df["material"].values) if task == "material" else None
    records = []

    for material, train_sess, test_sess in pairs:
        train_df = df[(df["session"] == train_sess) & (df["material"] == material)]
        test_df  = df[(df["session"] == test_sess)  & (df["material"] == material)]
        if len(train_df) < 5 or len(test_df) < 5:
            continue

        X_train = get_X(train_df)
        X_test  = get_X(test_df)

        if task == "pressure":
            y_train = train_df["pressure_bar"].values
            y_test  = test_df["pressure_bar"].values
            model   = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            y_pred  = model.predict(X_test)
            records.append({
                "material": material,
                "train_session": train_sess,
                "test_session":  test_sess,
                "n_train": len(train_df),
                "n_test":  len(test_df),
                "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                "r2":   float(r2_score(y_test, y_pred)),
            })
        elif task == "mix":
            y_train = train_df["mix_ratio_pct"].values
            y_test  = test_df["mix_ratio_pct"].values
            model   = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            y_pred  = model.predict(X_test)
            records.append({
                "material": material,
                "train_session": train_sess,
                "test_session":  test_sess,
                "n_train": len(train_df),
                "n_test":  len(test_df),
                "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                "r2":   float(r2_score(y_test, y_pred)),
            })

    return records


# ---------------------------------------------------------------------------
# Multi-model comparison
# ---------------------------------------------------------------------------

def compare_models(
    df: pd.DataFrame,
    task: str = "pressure",
    cv: int = 5,
) -> dict[str, dict]:
    """Compare Random Forest, Gradient Boosting, and SVM on the same task.

    SVM uses a StandardScaler pipeline (SVM is not scale-invariant).

    Returns dict mapping model_name -> result dict with same keys as
    train_pressure_model / train_material_model.
    """
    X = get_X(df)

    if task == "pressure":
        y = df["pressure_bar"].values
        models = {
            "RandomForest": RandomForestRegressor(
                n_estimators=100, random_state=42, n_jobs=-1),
            "GradientBoosting": GradientBoostingRegressor(
                n_estimators=100, random_state=42),
            "SVM (RBF)": Pipeline([
                ("scaler", StandardScaler()),
                ("svm", SVR(kernel="rbf", C=10, epsilon=0.1)),
            ]),
        }
        results = {}
        for name, model in models.items():
            y_pred = cross_val_predict(model, X, y, cv=cv)
            rmse   = float(np.sqrt(mean_squared_error(y, y_pred)))
            r2     = float(r2_score(y, y_pred))
            fold_r2 = cross_val_score(model, X, y, cv=cv, scoring="r2")
            results[name] = {"rmse": rmse, "r2": r2, "r2_std": float(fold_r2.std())}
        return results

    elif task == "material":
        le = LabelEncoder()
        y  = le.fit_transform(df["material"].values)
        models = {
            "RandomForest": RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1),
            "GradientBoosting": GradientBoostingClassifier(
                n_estimators=100, random_state=42),
            "SVM (RBF)": Pipeline([
                ("scaler", StandardScaler()),
                ("svm", SVC(kernel="rbf", C=10)),
            ]),
        }
        results = {}
        for name, model in models.items():
            y_pred   = cross_val_predict(model, X, y, cv=cv)
            accuracy = float(accuracy_score(y, y_pred))
            f1       = float(f1_score(y, y_pred, average="weighted"))
            fold_acc = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
            results[name] = {"accuracy": accuracy, "f1": f1,
                             "accuracy_std": float(fold_acc.std())}
        return results

    return {}
