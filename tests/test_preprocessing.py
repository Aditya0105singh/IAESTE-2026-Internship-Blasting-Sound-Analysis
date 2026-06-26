"""Tests for the preprocessing pipeline."""

import numpy as np
import pytest

from iaeste26.preprocessing import (
    trim_edges,
    bandpass_filter,
    normalize,
    preprocess,
    SENSOR_FREQ_RANGES,
)

SR = 48_000  # common sample rate used across tests


def make_sine(freq_hz: float, duration_sec: float, sr: int = SR) -> np.ndarray:
    """Generate a pure sine wave as float32."""
    t = np.arange(int(duration_sec * sr)) / sr
    return np.sin(2 * np.pi * freq_hz * t).astype(np.float32)


class TestTrimEdges:
    def test_removes_correct_samples(self):
        audio = np.ones(SR * 10, dtype=np.float32)  # 10 sec
        trimmed = trim_edges(audio, SR, trim_sec=2.0)
        assert len(trimmed) == SR * 6  # 10 - 2 - 2

    def test_returns_unchanged_when_too_short(self):
        audio = np.ones(SR * 3, dtype=np.float32)  # 3 sec
        result = trim_edges(audio, SR, trim_sec=2.0)  # would need 4 sec
        assert len(result) == len(audio)

    def test_zero_trim_returns_full_audio(self):
        audio = np.ones(SR * 5, dtype=np.float32)
        result = trim_edges(audio, SR, trim_sec=0.0)
        assert len(result) == len(audio)


class TestBandpassFilter:
    def test_passes_in_band_signal(self):
        """1 kHz sine should survive a 100–10000 Hz bandpass."""
        sine = make_sine(1_000, duration_sec=1.0)
        filtered = bandpass_filter(sine, SR, 100.0, 10_000.0)
        # Energy should be mostly preserved
        assert np.sqrt(np.mean(filtered ** 2)) > 0.3

    def test_attenuates_out_of_band_signal(self):
        """100 Hz sine should be heavily attenuated by a 1kHz–10kHz bandpass."""
        sine = make_sine(100, duration_sec=1.0)
        filtered = bandpass_filter(sine, SR, 1_000.0, 10_000.0)
        rms_in = np.sqrt(np.mean(sine ** 2))
        rms_out = np.sqrt(np.mean(filtered ** 2))
        assert rms_out < rms_in * 0.1  # at least 90% reduction

    def test_output_is_float32(self):
        sine = make_sine(1_000, duration_sec=0.5)
        result = bandpass_filter(sine, SR, 100.0, 10_000.0)
        assert result.dtype == np.float32

    def test_invalid_range_raises(self):
        sine = make_sine(1_000, duration_sec=0.5)
        with pytest.raises(ValueError, match="Invalid filter range"):
            bandpass_filter(sine, SR, 10_000.0, 100.0)  # low > high


class TestNormalize:
    def test_peak_becomes_one(self):
        audio = np.array([0.0, 0.5, -0.3, 0.8], dtype=np.float32)
        result = normalize(audio)
        assert np.isclose(np.max(np.abs(result)), 1.0)

    def test_silent_audio_unchanged(self):
        audio = np.zeros(100, dtype=np.float32)
        result = normalize(audio)
        assert np.all(result == 0.0)

    def test_output_is_float32(self):
        audio = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        result = normalize(audio)
        assert result.dtype == np.float32

    def test_shape_preserved(self):
        audio = np.random.randn(SR).astype(np.float32)
        result = normalize(audio)
        assert result.shape == audio.shape


class TestPreprocess:
    def test_returns_shorter_than_input(self):
        audio = np.random.randn(SR * 10).astype(np.float32)
        result = preprocess(audio, SR, trim_sec=2.0)
        assert len(result) < len(audio)

    def test_output_normalized(self):
        audio = np.random.randn(SR * 10).astype(np.float32) * 0.1
        result = preprocess(audio, SR, do_normalize=True)
        assert np.isclose(np.max(np.abs(result)), 1.0, atol=1e-5)

    def test_sensor_mic_uses_correct_range(self):
        lo, hi = SENSOR_FREQ_RANGES["Mic46BE"]
        assert lo == 20.0
        assert hi == 20_000.0

    def test_sensor_acc_uses_correct_range(self):
        lo, hi = SENSOR_FREQ_RANGES["AccAxial4507"]
        assert lo == 1.0
        assert hi == 6_000.0

    def test_output_is_float32(self):
        audio = np.random.randn(SR * 10).astype(np.float32)
        result = preprocess(audio, SR)
        assert result.dtype == np.float32

    def test_no_normalize_flag(self):
        audio = np.random.randn(SR * 10).astype(np.float32) * 0.1
        result = preprocess(audio, SR, do_normalize=False)
        # peak should not be 1.0 (since signal was scaled down)
        assert np.max(np.abs(result)) < 0.5
