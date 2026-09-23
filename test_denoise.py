from pathlib import Path
import numpy as np
from src.dsp.filters import load_and_preprocess_audio, compute_shannon_energy_envelope

# Pick a sample from Cohort B or Cohort A
test_file = next(Path("data/raw/training-b").glob("*.wav"), next(Path("data/raw/training-a").glob("*.wav"), None))

if test_file:
    print(f"Testing on: {test_file}")
    
    # Preprocess with and without noise floor subtraction
    _, y_clean, sr = load_and_preprocess_audio(str(test_file), enable_denoise=True)
    _, y_raw_filt, _ = load_and_preprocess_audio(str(test_file), enable_denoise=False)
    
    env_clean = compute_shannon_energy_envelope(y_clean, fs=sr)
    env_raw = compute_shannon_energy_envelope(y_raw_filt, fs=sr)
    
    ratio_raw = np.max(env_raw) / (np.percentile(env_raw, 10) + 1e-6)
    ratio_clean = np.max(env_clean) / (np.percentile(env_clean, 10) + 1e-6)
    
    print("-" * 50)
    print(f"Shannon Dynamic Ratio (Without Denoising): {ratio_raw:.2f}")
    print(f"Shannon Dynamic Ratio (With Denoising):    {ratio_clean:.2f}")
    print(f"Dynamic Range Improvement:                +{((ratio_clean - ratio_raw) / ratio_raw)*100:.1f}%")
    print("-" * 50)
else:
    print("No audio file found to test.")
