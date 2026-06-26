"""Feature extraction from preprocessed audio signals.

Extracts 35 features per recording across three domains:
  - Time domain  (5 features)
  - Frequency domain (4 features)
  - Time-frequency / MFCC (13 mean + 13 std = 26 features)
"""

import numpy as np
from scipy import stats as sp_stats
from scipy.fft import rfft, rfftfreq, dct
from scipy.signal import stft as scipy_stft


# ---------------------------------------------------------------------------
# Time-domain features
# ---------------------------------------------------------------------------

def rms(audio: np.ndarray) -> float:
    """Root Mean Square — overall signal energy."""
    return float(np.sqrt(np.mean(audio ** 2)))


def zero_crossing_rate(audio: np.ndarray) -> float:
    """Fraction of samples where the signal changes sign.

    High ZCR → noisy/high-frequency content.
    Low ZCR  → tonal/low-frequency content.
    """
    signs = np.sign(audio)
    signs[signs == 0] = 1  # treat exact zero as positive
    crossings = np.sum(signs[:-1] != signs[1:])
    return float(crossings / len(audio))


def peak_amplitude(audio: np.ndarray) -> float:
    """Maximum absolute sample value."""
    return float(np.max(np.abs(audio)))


def crest_factor(audio: np.ndarray) -> float:
    """Ratio of peak to RMS — measures impulsiveness.

    High crest factor → sharp impacts (common in abrasive blasting).
    """
    r = rms(audio)
    if r == 0.0:
        return 0.0
    return float(peak_amplitude(audio) / r)


def kurtosis(audio: np.ndarray) -> float:
    """Statistical kurtosis — peakedness of amplitude distribution.

    Gaussian noise ≈ 0. Impulsive signals >> 0.
    """
    return float(sp_stats.kurtosis(audio))


# ---------------------------------------------------------------------------
# Frequency-domain features
# ---------------------------------------------------------------------------

def _power_spectrum(audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (frequencies, power) for positive frequencies via rfft."""
    fft_vals = rfft(audio)
    power = np.abs(fft_vals) ** 2
    freqs = rfftfreq(len(audio), d=1.0 / sr)
    return freqs, power


def spectral_centroid(audio: np.ndarray, sr: int) -> float:
    """Weighted mean of the frequency spectrum — 'brightness' of the sound."""
    freqs, power = _power_spectrum(audio, sr)
    total = np.sum(power)
    if total == 0.0:
        return 0.0
    return float(np.sum(freqs * power) / total)


def spectral_bandwidth(audio: np.ndarray, sr: int) -> float:
    """Weighted standard deviation around the spectral centroid — spread."""
    freqs, power = _power_spectrum(audio, sr)
    centroid = spectral_centroid(audio, sr)
    total = np.sum(power)
    if total == 0.0:
        return 0.0
    return float(np.sqrt(np.sum(power * (freqs - centroid) ** 2) / total))


def spectral_rolloff(
    audio: np.ndarray, sr: int, roll_percent: float = 0.85
) -> float:
    """Frequency below which roll_percent of total spectral energy is contained."""
    freqs, power = _power_spectrum(audio, sr)
    cumsum = np.cumsum(power)
    total = cumsum[-1]
    if total == 0.0:
        return 0.0
    idx = np.searchsorted(cumsum, roll_percent * total)
    return float(freqs[min(idx, len(freqs) - 1)])


def dominant_frequency(audio: np.ndarray, sr: int) -> float:
    """Frequency with the highest power in the spectrum."""
    freqs, power = _power_spectrum(audio, sr)
    return float(freqs[np.argmax(power)])


# ---------------------------------------------------------------------------
# Time-frequency features: MFCC (implemented via scipy, no extra deps)
# ---------------------------------------------------------------------------

def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(
    sr: int,
    n_fft: int,
    n_mels: int = 40,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> np.ndarray:
    """Build a mel-scale triangular filterbank matrix of shape (n_mels, n_fft//2+1)."""
    if fmax is None:
        fmax = sr / 2.0

    mel_points = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bins = np.clip(
        np.floor((n_fft + 1) * hz_points / sr).astype(int),
        0,
        n_fft // 2,
    )

    n_bins = n_fft // 2 + 1
    fb = np.zeros((n_mels, n_bins), dtype=np.float32)
    for m in range(1, n_mels + 1):
        fl, fc, fr = bins[m - 1], bins[m], bins[m + 1]
        if fc > fl:
            fb[m - 1, fl:fc] = (np.arange(fl, fc) - fl) / (fc - fl)
        if fr > fc:
            fb[m - 1, fc:fr] = (fr - np.arange(fc, fr)) / (fr - fc)

    return fb


def mfcc_features(
    audio: np.ndarray,
    sr: int,
    n_mfcc: int = 13,
    n_mels: int = 40,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Compute MFCCs and return mean + std for each coefficient.

    MFCCs capture the timbral 'texture' of the sound — very informative
    for distinguishing blasting conditions.

    Returns:
        1-D array of length 2*n_mfcc:
            [mean_mfcc_1 ... mean_mfcc_n  std_mfcc_1 ... std_mfcc_n]
    """
    # Short-time Fourier transform
    _, _, Zxx = scipy_stft(
        audio, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length
    )
    power = np.abs(Zxx) ** 2  # shape: (n_fft//2+1, n_frames)

    # Apply mel filterbank
    fb = _mel_filterbank(sr, n_fft, n_mels=n_mels, fmax=sr / 2.0)
    mel_spec = fb @ power  # (n_mels, n_frames)

    # Log compression
    log_mel = np.log(mel_spec + 1e-10)

    # DCT → MFCCs
    coeffs = dct(log_mel, type=2, axis=0, norm="ortho")[:n_mfcc]  # (n_mfcc, n_frames)

    return np.concatenate([coeffs.mean(axis=1), coeffs.std(axis=1)])


# ---------------------------------------------------------------------------
# Master function: extract everything at once
# ---------------------------------------------------------------------------

def extract_features(
    audio: np.ndarray,
    sr: int,
    n_mfcc: int = 13,
) -> dict[str, float]:
    """Extract all 35 features from a preprocessed audio signal.

    Args:
        audio:  preprocessed float32 array (output of preprocessing.preprocess)
        sr:     sample rate in Hz
        n_mfcc: number of MFCC coefficients (default 13)

    Returns:
        Flat dict of {feature_name: float} — 1 row for the feature DataFrame.
    """
    feats: dict[str, float] = {
        # Time domain
        "rms":                 rms(audio),
        "zero_crossing_rate":  zero_crossing_rate(audio),
        "peak_amplitude":      peak_amplitude(audio),
        "crest_factor":        crest_factor(audio),
        "kurtosis":            kurtosis(audio),
        # Frequency domain
        "spectral_centroid":   spectral_centroid(audio, sr),
        "spectral_bandwidth":  spectral_bandwidth(audio, sr),
        "spectral_rolloff_85": spectral_rolloff(audio, sr, roll_percent=0.85),
        "dominant_frequency":  dominant_frequency(audio, sr),
    }

    # MFCC features: mfcc_1_mean … mfcc_13_mean, mfcc_1_std … mfcc_13_std
    mfcc_vals = mfcc_features(audio, sr, n_mfcc=n_mfcc)
    for i in range(n_mfcc):
        feats[f"mfcc_{i + 1}_mean"] = float(mfcc_vals[i])
        feats[f"mfcc_{i + 1}_std"]  = float(mfcc_vals[n_mfcc + i])

    return feats
