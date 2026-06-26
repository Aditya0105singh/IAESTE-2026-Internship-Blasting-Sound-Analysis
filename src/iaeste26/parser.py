"""Parse blasting-sound filenames and load WAV audio as numpy arrays."""

import re
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wav_io

# Nominal nozzle value in filename -> actual physical diameter (mm)
_NOZZLE_DIAMETER_MM = {6: 6.4, 8: 8.0}

_FILENAME_RE = re.compile(
    r"^(?P<material>[^_]+)"
    r"_(?P<nozzle>\d+)"
    r"_(?P<pressure>[\d.]+)"
    r"_(?P<mix_ratio>\d+)"
    r"_(?P<sensor>[^_.]+)"
    r"(?:_.*)?\.wav$",
    re.IGNORECASE,
)


def parse_filename(filename: str) -> dict:
    """Extract blasting parameters from a WAV filename.

    Filename format: Material_Nozzle_Pressure_MixRatio_Sensor.wav
    Example:         G80_8_3_50_Mic46BE.wav

    Returns:
        dict with keys:
            material       (str)   e.g. "G80"
            nozzle_mm      (float) actual diameter: 6->6.4, 8->8.0
            pressure_bar   (float) e.g. 5.5
            mix_ratio_pct  (int)   e.g. 50
            sensor         (str)   e.g. "Mic46BE"

    Raises:
        ValueError: if filename does not match the expected pattern.
    """
    name = Path(filename).name
    m = _FILENAME_RE.match(name)
    if not m:
        raise ValueError(
            f"Filename does not match expected pattern "
            f"'Material_Nozzle_Pressure_MixRatio_Sensor.wav': {name!r}"
        )

    nozzle_nominal = int(m.group("nozzle"))
    return {
        "material": m.group("material").upper(),
        "nozzle_mm": _NOZZLE_DIAMETER_MM.get(nozzle_nominal, float(nozzle_nominal)),
        "pressure_bar": float(m.group("pressure")),
        "mix_ratio_pct": int(m.group("mix_ratio")),
        "sensor": m.group("sensor"),
    }


def load_wav(filepath: str | Path) -> tuple[np.ndarray, int]:
    """Load a WAV file and return (audio, sample_rate).

    Audio is always returned as float32 normalized to [-1.0, 1.0].

    Args:
        filepath: path to the .wav file

    Returns:
        audio       np.ndarray of shape (n_samples,), dtype float32
        sample_rate int, e.g. 48000
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"WAV file not found: {filepath}")

    sample_rate, data = wav_io.read(filepath)

    # Normalize integer PCM formats to float32 [-1, 1]
    if data.dtype == np.int16:
        audio = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        audio = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.float32:
        audio = data.copy()
    else:
        audio = data.astype(np.float32)

    return audio, sample_rate
