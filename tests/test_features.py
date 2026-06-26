"""Tests for feature extraction functions."""

import numpy as np
import pytest

from iaeste26.features import (
    rms,
    zero_crossing_rate,
    peak_amplitude,
    crest_factor,
    kurtosis,
    spectral_centroid,
    spectral_bandwidth,
    spectral_rolloff,
    dominant_frequency,
    mfcc_features,
    extract_features,
)

SR = 48_000


def make_sine(freq_hz: float, duration_sec: float = 1.0, sr: int = SR) -> np.ndarray:
    t = np.arange(int(duration_sec * sr)) / sr
    return np.sin(2 * np.pi * freq_hz * t).astype(np.float32)


def make_silence(duration_sec: float = 1.0, sr: int = SR) -> np.ndarray:
    return np.zeros(int(duration_sec * sr), dtype=np.float32)


class TestTimeDomain:
    def test_rms_ones(self):
        audio = np.ones(1000, dtype=np.float32)
        assert np.isclose(rms(audio), 1.0)

    def test_rms_silence(self):
        assert rms(make_silence()) == 0.0

    def test_rms_sine_approx_half(self):
        # RMS of a full-amplitude sine ≈ 1/√2 ≈ 0.707
        sine = make_sine(1000)
        assert np.isclose(rms(sine), 1.0 / np.sqrt(2), atol=0.01)

    def test_zero_crossing_rate_alternating(self):
        # Perfect alternating signal crosses on every sample
        audio = np.array([1, -1, 1, -1, 1, -1], dtype=np.float32)
        assert zero_crossing_rate(audio) > 0.8

    def test_zero_crossing_rate_constant(self):
        # Constant positive signal → no crossings
        audio = np.ones(100, dtype=np.float32)
        assert zero_crossing_rate(audio) == 0.0

    def test_peak_amplitude(self):
        audio = np.array([-0.5, 0.3, 0.9, -0.2], dtype=np.float32)
        assert np.isclose(peak_amplitude(audio), 0.9)

    def test_crest_factor_sine(self):
        # Crest factor of sine ≈ √2 ≈ 1.414
        sine = make_sine(1000)
        assert np.isclose(crest_factor(sine), np.sqrt(2), atol=0.05)

    def test_crest_factor_silence_is_zero(self):
        assert crest_factor(make_silence()) == 0.0

    def test_kurtosis_gaussian_near_zero(self):
        rng = np.random.default_rng(42)
        noise = rng.standard_normal(50_000).astype(np.float32)
        # Excess kurtosis of Gaussian ≈ 0
        assert abs(kurtosis(noise)) < 0.5


class TestFrequencyDomain:
    def test_dominant_frequency_sine(self):
        freq = 1000.0
        sine = make_sine(freq)
        dom = dominant_frequency(sine, SR)
        assert abs(dom - freq) < 5.0  # within 5 Hz

    def test_spectral_centroid_sine(self):
        freq = 2000.0
        sine = make_sine(freq)
        centroid = spectral_centroid(sine, SR)
        # Centroid of a pure sine ≈ its frequency
        assert abs(centroid - freq) < 20.0

    def test_spectral_centroid_silence(self):
        assert spectral_centroid(make_silence(), SR) == 0.0

    def test_spectral_bandwidth_returns_positive(self):
        sine = make_sine(1000)
        bw = spectral_bandwidth(sine, SR)
        assert bw >= 0.0

    def test_spectral_rolloff_below_nyquist(self):
        sine = make_sine(1000)
        rolloff = spectral_rolloff(sine, SR)
        assert rolloff < SR / 2.0

    def test_spectral_rolloff_silence(self):
        assert spectral_rolloff(make_silence(), SR) == 0.0

    def test_higher_freq_sine_has_higher_centroid(self):
        low = spectral_centroid(make_sine(500), SR)
        high = spectral_centroid(make_sine(5000), SR)
        assert high > low


class TestMFCC:
    def test_output_length(self):
        sine = make_sine(1000)
        result = mfcc_features(sine, SR, n_mfcc=13)
        assert len(result) == 26  # 13 means + 13 stds

    def test_output_is_finite(self):
        sine = make_sine(1000)
        result = mfcc_features(sine, SR)
        assert np.all(np.isfinite(result))

    def test_different_signals_give_different_mfcc(self):
        low = mfcc_features(make_sine(500), SR)
        high = mfcc_features(make_sine(5000), SR)
        assert not np.allclose(low, high)


class TestExtractFeatures:
    def test_returns_dict(self):
        audio = make_sine(1000)
        result = extract_features(audio, SR)
        assert isinstance(result, dict)

    def test_correct_number_of_features(self):
        audio = make_sine(1000)
        result = extract_features(audio, SR)
        # 5 time + 4 freq + 13*2 mfcc = 35
        assert len(result) == 35

    def test_all_values_are_floats(self):
        audio = make_sine(1000)
        result = extract_features(audio, SR)
        for k, v in result.items():
            assert isinstance(v, float), f"{k} is not float: {type(v)}"

    def test_all_values_are_finite(self):
        audio = make_sine(1000)
        result = extract_features(audio, SR)
        for k, v in result.items():
            assert np.isfinite(v), f"{k} = {v} is not finite"

    def test_expected_keys_present(self):
        audio = make_sine(1000)
        result = extract_features(audio, SR)
        expected = {"rms", "zero_crossing_rate", "peak_amplitude",
                    "crest_factor", "kurtosis", "spectral_centroid",
                    "spectral_bandwidth", "spectral_rolloff_85",
                    "dominant_frequency", "mfcc_1_mean", "mfcc_13_std"}
        assert expected.issubset(result.keys())

    def test_noise_gives_different_features_than_sine(self):
        rng = np.random.default_rng(0)
        noise = rng.standard_normal(SR).astype(np.float32)
        f_sine  = extract_features(make_sine(1000), SR)
        f_noise = extract_features(noise, SR)
        # At least RMS and ZCR should differ
        assert f_sine["zero_crossing_rate"] != f_noise["zero_crossing_rate"]
