import sys
import argparse
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from typing import Dict, Any

from src.dsp.filters import load_and_preprocess_audio, compute_shannon_energy_envelope

def extract_signal_features(
    file_path: str, 
    target_sr: int = 4000, 
    n_mfcc: int = 13
) -> Dict[str, Any]:
    """Extracts DSP and spectral features from a single audio file."""
    _, y_filt, fs = load_and_preprocess_audio(file_path, target_sr=target_sr)
    
    # Shannon Energy Dynamics
    env = compute_shannon_energy_envelope(y_filt, fs=fs)
    env_mean = float(np.mean(env))
    env_std = float(np.std(env))
    env_max = float(np.max(env))
    env_floor = float(np.percentile(env, 10))
    shannon_dynamic_ratio = float(env_max / (env_floor + 1e-6))
    
    # Spectral Features (STFT tuned for 4000 Hz)
    n_fft = 512
    hop_length = 128
    
    cent = librosa.feature.spectral_centroid(y=y_filt, sr=fs, n_fft=n_fft, hop_length=hop_length)[0]
    cent_mean = float(np.mean(cent))
    cent_std = float(np.std(cent))
    cent_max = float(np.max(cent))
    
    rolloff = librosa.feature.spectral_rolloff(y=y_filt, sr=fs, n_fft=n_fft, hop_length=hop_length, roll_percent=0.85)[0]
    rolloff_mean = float(np.mean(rolloff))
    rolloff_std = float(np.std(rolloff))
    
    flatness = librosa.feature.spectral_flatness(y=y_filt, n_fft=n_fft, hop_length=hop_length)[0]
    flatness_mean = float(np.mean(flatness))
    
    zcr = librosa.feature.zero_crossing_rate(y=y_filt, frame_length=n_fft, hop_length=hop_length)[0]
    zcr_mean = float(np.mean(zcr))
    
    # MFCCs
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
    
    for i in range(n_mfcc):
        features[f"mfcc_{i+1}_mean"] = float(np.mean(mfccs[i, :]))
        features[f"mfcc_{i+1}_std"] = float(np.std(mfccs[i, :]))
        
    return features

def process_dataset(dataset_dir: str, reference_csv: str, output_csv: str) -> pd.DataFrame:
    """Batch processes audio files in a dataset folder and saves features to CSV."""
    dataset_path = Path(dataset_dir)
    ref_df = pd.read_csv(reference_csv, header=None, names=["record_id", "label"])
    
    print(f"Extracting features for {len(ref_df)} recordings from {dataset_dir}...")
    
    records = []
    for _, row in tqdm(ref_df.iterrows(), total=len(ref_df), desc=f"Processing {dataset_path.name}"):
        file_path = dataset_path / f"{row['record_id']}.wav"
        if not file_path.exists():
            continue
        try:
            feats = extract_signal_features(str(file_path))
            feats["label"] = int(row["label"])
            records.append(feats)
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            
    df_features = pd.DataFrame(records)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(output_path, index=False)
    print(f"Extraction complete! Saved {len(df_features)} records to {output_path}")
    return df_features

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract DSP features from audio dataset")
    parser.add_argument("--set", type=str, default="training-a", help="Dataset folder name (e.g. training-a, training-b)")
    args = parser.parse_args()
    
    DATA_DIR = ROOT_DIR / "data" / "raw" / args.set
    REF_CSV = DATA_DIR / "REFERENCE.csv"
    OUTPUT_CSV = ROOT_DIR / "data" / "processed" / f"{args.set.replace('-', '_')}_features.csv"
    
    if DATA_DIR.exists() and REF_CSV.exists():
        process_dataset(str(DATA_DIR), str(REF_CSV), str(OUTPUT_CSV))
    else:
        print(f"Error: Missing directory {DATA_DIR} or reference file {REF_CSV}.")
