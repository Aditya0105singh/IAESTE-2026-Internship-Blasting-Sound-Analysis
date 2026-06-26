"""Scan the dataset directory and build a structured metadata DataFrame."""

from pathlib import Path

import pandas as pd

from .parser import parse_filename


def scan_dataset(data_dir: str | Path) -> pd.DataFrame:
    """Scan all WAV files under data_dir and return a metadata DataFrame.

    Each row represents one WAV file with its blasting parameters and path.

    Args:
        data_dir: root data directory (the folder containing v24, v25, ... subfolders)

    Returns:
        DataFrame with columns:
            session, material, nozzle_mm, pressure_bar, mix_ratio_pct, sensor, filepath
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    records = []
    skipped = []

    for wav_path in sorted(data_dir.rglob("*.wav")):
        try:
            meta = parse_filename(wav_path.name)
        except ValueError:
            skipped.append(wav_path.name)
            continue

        # Path structure: data/vXX/WAV/filename.wav
        # parent        = .../WAV/
        # parent.parent = .../vXX/
        session = wav_path.parent.parent.name

        records.append({
            "session": session,
            "filepath": str(wav_path),
            **meta,
        })

    if skipped:
        print(f"[scan_dataset] Skipped {len(skipped)} unrecognized files: {skipped[:5]}")

    df = pd.DataFrame(records)

    # Sensible column order
    col_order = ["session", "material", "nozzle_mm", "pressure_bar", "mix_ratio_pct", "sensor", "filepath"]
    df = df[[c for c in col_order if c in df.columns]]

    return df


def export_csv(df: pd.DataFrame, output_path: str | Path) -> None:
    """Save the metadata DataFrame to a CSV file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} records to {output_path}")


def summary(df: pd.DataFrame) -> None:
    """Print a human-readable summary of the dataset."""
    print("=" * 50)
    print(f"  DATASET SUMMARY")
    print("=" * 50)
    print(f"  Total files       : {len(df)}")
    print(f"  Sessions          : {sorted(df['session'].unique())}")
    print(f"  Materials         : {sorted(df['material'].unique())}")
    print(f"  Nozzle diameters  : {sorted(df['nozzle_mm'].unique())} mm")
    print(f"  Pressures (bar)   : {sorted(df['pressure_bar'].unique())}")
    print(f"  Mix ratios (%)    : {sorted(df['mix_ratio_pct'].unique())}")
    print(f"  Sensors           : {sorted(df['sensor'].unique())}")
    print("=" * 50)
