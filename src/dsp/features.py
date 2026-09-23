import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any

from src.dsp.filters import load_and_preprocess_audio, compute_shannon_energy_envelope

def extract_signal_features(
    file_path: str, 
    target_sr: int = 4000, 
    n_mfcc: int = 13
) -> Dict[str, Any]:
    """
    Extracts physically motivated DSP and spectral features from a single audio file.
    """
    # 1. Preprocess: 4th-order zero-phase Butterworth bandpass (50-1200 Hz)
    _, y_filt, fs = load_and_preprocess_audio(file_path, target_sr=target_sr)
    
    # 2. Shannon Energy Dynamics
    env = compute_shannon_energy_envelope(y_filt, fs=fs)
    env_mean = float(np.mean(env))
    env_std = float(np.std(env))
    env_max = float(np.max(env))
    # 10th percentile serves as a robust floor for valley energy
    env_floor = float(np.percentile(env, 10))
    shannon_dynamic_ratio = float(env_max / (env_floor + 1e-6))
    
    # 3. Spectral Features (STFT parameters tuned for 4000 Hz)
    n_fft = 512
    hop_length = 128
    
    # Spectral Centroid
    cent = librosa.feature.spectral_centroid(y=y_filt, sr=fs, n_fft=n_fft, hop_length=hop_length)[0]
    cent_mean = float(np.mean(cent))
    cent_std = float(np.std(cent))
    cent_max = float(np.max(cent))
    
    # Spectral Rolloff (85% energy threshold)
    rolloff = librosa.feature.spectral_rolloff(y=y_filt, sr=fs, n_fft=n_fft, hop_length=hop_length, roll_percent=0.85)[0]
    rolloff_mean = float(np.mean(rolloff))
    rolloff_std = float(np.std(rolloff))
    
    # Spectral Flatness (tonality vs noisiness of the bruit)
    flatness = librosa.feature.spectral_flatness(y=y_filt, n_fft=n_fft, hop_length=hop_length)[0]
    flatness_mean = float(np.mean(flatness))
    
    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y=y_filt, frame_length=n_fft, hop_length=hop_length)[0]
    zcr_mean = float(np.mean(zcr))
    
    # 4. MFCCs (13 coefficients, taking mean and standard deviation across time)
    mfccs = librosa.feature.mfcc(y=y_filt, sr=fs, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    
    features = {
        "record_id": Path(file_path).stem,
        "shannon_mean": env_mean,
        "shannon_std": env_std,
        "shannon_dynamic_ratio": shannon_dynamic_ratio,
        "spectral_centroid_mean": cent_mean,
        "spectral_centroid_std": cent_std,
        "spectral_centroid_max": cent_max,
        "spectral_rolloff_mean": rolloff_mean,
        "spectral_rolloff_std": rolloff_std,
        "spectral_flatness_mean": flatness_mean,
        "zero_crossing_rate_mean": zcr_mean,
    }
    
    # Append MFCC summary statistics
    for i in range(n_mfcc):
        features[f"mfcc_{i+1}_mean"] = float(np.mean(mfccs[i, :]))
        features[f"mfcc_{i+1}_std"] = float(np.std(mfccs[i, :]))
        
    return features

def process_dataset(
    dataset_dir: str, 
    reference_csv: str, 
    output_csv: str
) -> pd.DataFrame:
    """
    Batch processes all .wav files referenced in reference_csv and saves
    the resulting feature matrix to output_csv.
    """
    dataset_path = Path(dataset_dir)
    ref_df = pd.read_csv(reference_csv, header=None, names=["record_id", "label"])
    
    print(f"Extracting features for {len(ref_df)} recordings from {dataset_dir}...")
    
    records = []
    for _, row in tqdm(ref_df.iterrows(), total=len(ref_df), desc="Extracting features"):
        file_path = dataset_path / f"{row['record_id']}.wav"
        if not file_path.exists():
            continue
        try:
            feats = extract_signal_features(str(file_path))
            feats["label"] = int(row["label"]) # 1: normal, -1: abnormal
            records.append(feats)
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            
    df_features = pd.DataFrame(records)
    
    # Save to disk
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(output_path, index=False)
    print(f"Feature extraction complete! Saved {len(df_features)} records to {output_path}")
    
    return df_features

if __name__ == "__main__":
    DATA_DIR = Path("data/raw/training-a")
    REF_CSV = DATA_DIR / "REFERENCE.csv"
    OUTPUT_CSV = Path("data/processed/training_a_features.csv")
    
    if DATA_DIR.exists() and REF_CSV.exists():
        process_dataset(str(DATA_DIR), str(REF_CSV), str(OUTPUT_CSV))
    else:
        print(f"Error: Missing {DATA_DIR} or {REF_CSV}. Run data download first.")
