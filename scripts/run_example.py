"""Phase 1 demo: parse filenames, load a WAV, scan full dataset."""

from pathlib import Path
from iaeste26.parser import parse_filename, load_wav
from iaeste26.dataset import scan_dataset, export_csv, summary

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_CSV = Path(__file__).parent.parent / "dataset_metadata.csv"


def demo_parser():
    print("\n--- 1. Filename Parser ---")
    examples = [
        "G80_8_3_50_Mic46BE.wav",
        "GH40_6_5.5_30_AccAxial4507.wav",
        "GH120_8_7_0_Mic147EB.wav",
    ]
    for name in examples:
        meta = parse_filename(name)
        print(f"  {name}")
        print(f"    -> {meta}\n")


def demo_wav_loader():
    print("--- 2. WAV Loader ---")
    wav_files = list(DATA_DIR.rglob("*.wav"))
    if not wav_files:
        print("  No WAV files found in data/")
        return

    sample_file = wav_files[0]
    audio, sr = load_wav(sample_file)
    duration = len(audio) / sr
    print(f"  File    : {sample_file.name}")
    print(f"  Samples : {len(audio):,}")
    print(f"  Rate    : {sr} Hz")
    print(f"  Duration: {duration:.1f} sec")
    print(f"  dtype   : {audio.dtype}")
    print(f"  Range   : [{audio.min():.4f}, {audio.max():.4f}]")


def demo_dataset_scan():
    print("\n--- 3. Dataset Scan ---")
    df = scan_dataset(DATA_DIR)
    summary(df)

    export_csv(df, OUTPUT_CSV)
    print(f"\nFirst 5 rows:")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    demo_parser()
    demo_wav_loader()
    demo_dataset_scan()
