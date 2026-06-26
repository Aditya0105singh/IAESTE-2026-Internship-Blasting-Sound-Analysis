from iaeste26.parser import parse_filename, load_wav
from iaeste26.preprocessing import preprocess
from iaeste26.features import extract_features

wav = 'data/v24/WAV/G80_8_3_50_Mic46BE.wav'
meta = parse_filename(wav)
audio_raw, sr = load_wav(wav)
audio_clean = preprocess(audio_raw, sr, sensor=meta['sensor'])
feats = extract_features(audio_clean, sr)

print(f"File: G80_8_3_50_Mic46BE.wav")
print(f"Params: material={meta['material']}  pressure={meta['pressure_bar']}bar  mix={meta['mix_ratio_pct']}%  sensor={meta['sensor']}")
print()
print("--- 35 EXTRACTED FEATURES ---")
print()
print("Time domain:")
for k in ['rms','zero_crossing_rate','peak_amplitude','crest_factor','kurtosis']:
    print(f"  {k:<25} {feats[k]:>12.6f}")
print()
print("Frequency domain:")
for k in ['spectral_centroid','spectral_bandwidth','spectral_rolloff_85','dominant_frequency']:
    print(f"  {k:<25} {feats[k]:>12.2f} Hz")
print()
print("MFCC (mean of each coeff):")
mfcc_means = [f"{feats[f'mfcc_{i+1}_mean']:7.3f}" for i in range(13)]
print("  [" + "  ".join(mfcc_means) + "]")
print()
print(f"Total features: {len(feats)}")
