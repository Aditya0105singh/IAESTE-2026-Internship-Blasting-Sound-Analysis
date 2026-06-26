"""Signal preprocessing: trim, bandpass filter, normalize."""

import numpy as np
from scipy.signal import butter, sosfilt

# Practical frequency ranges per sensor (based on datasheets)
# Mic147EB  : 3.15 Hz – 20 kHz
# Mic46BE   : 4 Hz – 80 kHz  (capped at half Nyquist in practice)
# Acc 4507  : 0.3 Hz – 6 kHz
SENSOR_FREQ_RANGES: dict[str, tuple[float, float]] = {
    "Mic147EB":      (20.0,  20_000.0),
    "Mic46BE":       (20.0,  20_000.0),
    "AccAxial4507":  (1.0,    6_000.0),
    "AccRadial4507": (1.0,    6_000.0),
}


def trim_edges(audio: np.ndarray, sample_rate: int, trim_sec: float = 2.0) -> np.ndarray:
    """Remove the first and last trim_sec seconds from audio.

    Removes blasting start/stop transients that are not representative
    of steady-state operation.

    Args:
        audio:       1-D float32 array
        sample_rate: samples per second
        trim_sec:    seconds to cut from each end (default 2.0)

    Returns:
        Trimmed audio array. If recording is too short, returned unchanged.
    """
    n = int(trim_sec * sample_rate)
    if n == 0:
        return audio
    if 2 * n >= len(audio):
        return audio
    return audio[n:-n].copy()


def bandpass_filter(
    audio: np.ndarray,
    sample_rate: int,
    low_hz: float,
    high_hz: float,
    order: int = 4,
) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass filter.

    Uses second-order sections (SOS) for numerical stability.

    Args:
        audio:       1-D float32 array
        sample_rate: samples per second
        low_hz:      lower cutoff frequency (Hz)
        high_hz:     upper cutoff frequency (Hz)
        order:       filter order (default 4)

    Returns:
        Filtered audio as float32.
    """
    nyq = sample_rate / 2.0
    low = max(low_hz / nyq, 1e-6)
    high = min(high_hz / nyq, 0.999)  # must stay below Nyquist

    if low >= high:
        raise ValueError(
            f"Invalid filter range: low={low_hz} Hz >= high={high_hz} Hz "
            f"(Nyquist={nyq:.0f} Hz)"
        )

    sos = butter(order, [low, high], btype="band", output="sos")
    filtered = sosfilt(sos, audio)
    return filtered.astype(np.float32)


def normalize(audio: np.ndarray) -> np.ndarray:
    """Peak-normalize audio so the maximum absolute value is 1.0.

    Returns the original array unchanged if it is silent (all zeros).
    """
    peak = np.max(np.abs(audio))
    if peak == 0.0:
        return audio
    return (audio / peak).astype(np.float32)


def preprocess(
    audio: np.ndarray,
    sample_rate: int,
    sensor: str | None = None,
    trim_sec: float = 2.0,
    low_hz: float | None = None,
    high_hz: float | None = None,
    do_normalize: bool = True,
) -> np.ndarray:
    """Full preprocessing pipeline: trim → bandpass filter → normalize.

    Args:
        audio:        raw audio loaded via load_wav()
        sample_rate:  sample rate in Hz
        sensor:       sensor name (e.g. "Mic46BE") — sets default filter range
        trim_sec:     seconds to trim from each end (default 2.0)
        low_hz:       override lower filter cutoff; ignored if sensor is given
        high_hz:      override upper filter cutoff; ignored if sensor is given
        do_normalize: whether to peak-normalize after filtering (default True)

    Returns:
        Preprocessed audio as float32 numpy array.
    """
    # Step 1: trim transients at start/end
    audio = trim_edges(audio, sample_rate, trim_sec)

    # Step 2: bandpass filter
    if sensor in SENSOR_FREQ_RANGES:
        lo, hi = SENSOR_FREQ_RANGES[sensor]
    else:
        lo = low_hz if low_hz is not None else 20.0
        hi = high_hz if high_hz is not None else 20_000.0

    audio = bandpass_filter(audio, sample_rate, lo, hi)

    # Step 3: normalize
    if do_normalize:
        audio = normalize(audio)

    return audio
