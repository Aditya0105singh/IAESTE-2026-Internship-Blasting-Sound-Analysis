"""Tests for the filename parser and WAV loader."""

import numpy as np
import pytest

from iaeste26.parser import parse_filename, load_wav


class TestParseFilename:
    def test_standard_filename(self):
        result = parse_filename("G80_8_3_50_Mic46BE.wav")
        assert result == {
            "material": "G80",
            "nozzle_mm": 8.0,
            "pressure_bar": 3.0,
            "mix_ratio_pct": 50,
            "sensor": "Mic46BE",
        }

    def test_float_pressure(self):
        result = parse_filename("GH40_6_5.5_30_AccAxial4507.wav")
        assert result["pressure_bar"] == 5.5
        assert result["nozzle_mm"] == 6.4

    def test_zero_mix_ratio(self):
        result = parse_filename("GH120_8_7_0_Mic147EB.wav")
        assert result["mix_ratio_pct"] == 0

    def test_full_mix_ratio(self):
        result = parse_filename("GH18_6_4_100_AccRadial4507.wav")
        assert result["mix_ratio_pct"] == 100

    def test_nozzle_6_maps_to_6_4mm(self):
        result = parse_filename("G80_6_3_50_Mic46BE.wav")
        assert result["nozzle_mm"] == 6.4

    def test_nozzle_8_maps_to_8_0mm(self):
        result = parse_filename("G80_8_3_50_Mic46BE.wav")
        assert result["nozzle_mm"] == 8.0

    def test_material_uppercased(self):
        result = parse_filename("g80_8_3_50_Mic46BE.wav")
        assert result["material"] == "G80"

    def test_accepts_full_path(self):
        result = parse_filename("data/v24/WAV/G80_8_3_50_Mic46BE.wav")
        assert result["material"] == "G80"

    def test_invalid_filename_raises(self):
        with pytest.raises(ValueError, match="does not match"):
            parse_filename("random_audio.wav")

    def test_invalid_too_few_fields_raises(self):
        with pytest.raises(ValueError):
            parse_filename("G80_8_3.wav")


class TestLoadWav:
    def test_returns_float32(self, tmp_path):
        import scipy.io.wavfile as wav_io
        # Write a small int16 WAV
        data = np.array([0, 16384, -16384, 32767], dtype=np.int16)
        path = tmp_path / "test.wav"
        wav_io.write(str(path), 48000, data)

        audio, sr = load_wav(path)
        assert audio.dtype == np.float32
        assert sr == 48000

    def test_int16_normalized_range(self, tmp_path):
        import scipy.io.wavfile as wav_io
        data = np.array([32767, -32768, 0], dtype=np.int16)
        path = tmp_path / "test.wav"
        wav_io.write(str(path), 48000, data)

        audio, _ = load_wav(path)
        assert audio.max() <= 1.0
        assert audio.min() >= -1.0

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_wav(tmp_path / "nonexistent.wav")
