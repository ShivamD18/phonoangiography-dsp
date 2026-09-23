import numpy as np
import scipy.signal as signal
from scipy.integrate import trapezoid
import librosa
from src.dsp.filters import load_and_preprocess_audio, compute_shannon_energy_envelope

def compute_subband_energy_ratio(y: np.ndarray, fs: int = 4000) -> float:
    """
    Computes ratio of turbulent murmur energy (200-600 Hz) to valve closure energy (50-150 Hz).
    Pathological murmurs produce significantly higher ratios.
    """
    freqs, psd = signal.welch(y, fs=fs, nperseg=min(512, len(y)))
    
    valve_band = (freqs >= 50.0) & (freqs <= 150.0)
    murmur_band = (freqs >= 200.0) & (freqs <= 600.0)
    
    valve_energy = trapezoid(psd[valve_band], freqs[valve_band]) if np.any(valve_band) else 1e-6
    murmur_energy = trapezoid(psd[murmur_band], freqs[murmur_band]) if np.any(murmur_band) else 0.0
    
    return float(murmur_energy / (valve_energy + 1e-6))

def extract_signal_features(y_clean: np.ndarray, sr: int = 4000) -> dict:
    """
    Extracts all physical, spectral, and cepstral features directly from 
    a preconditioned, bandpassed audio signal.
    """
    hop_length = 128
    n_fft = 512
    
    # 1. Spectral Moments
    cent = librosa.feature.spectral_centroid(y=y_clean, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y_clean, sr=sr, n_fft=n_fft, hop_length=hop_length, roll_percent=0.85)[0]
    flatness = librosa.feature.spectral_flatness(y=y_clean, n_fft=n_fft, hop_length=hop_length)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y_clean, hop_length=hop_length)[0]
    
    # 2. Shannon Envelope Dynamic Metrics
    envelope = compute_shannon_energy_envelope(y_clean, fs=sr)
    env_max = np.max(envelope)
    env_p10 = np.percentile(envelope, 10)
    dynamic_ratio = float(env_max / (env_p10 + 1e-6))
    
    # 3. Sub-Band Energy Ratio
    subband_ratio = compute_subband_energy_ratio(y_clean, fs=sr)
    
    # 4. MFCCs
    mfccs = librosa.feature.mfcc(y=y_clean, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length)
    
    feats = {
        "spectral_centroid_mean": float(np.mean(cent)),
        "spectral_centroid_std": float(np.std(cent)),
        "spectral_centroid_max": float(np.max(cent)),
        "spectral_centroid_p95": float(np.percentile(cent, 95)),
        "spectral_rolloff_mean": float(np.mean(rolloff)),
        "spectral_rolloff_std": float(np.std(rolloff)),
        "spectral_flatness_mean": float(np.mean(flatness)),
        "spectral_flatness_std": float(np.std(flatness)),
        "zcr_mean": float(np.mean(zcr)),
        "zcr_std": float(np.std(zcr)),
        "shannon_dynamic_ratio": dynamic_ratio,
        "shannon_valley_floor": float(env_p10),
        "murmur_subband_ratio": subband_ratio
    }
    
    for i in range(13):
        feats[f"mfcc_{i+1}_mean"] = float(np.mean(mfccs[i, :]))
        feats[f"mfcc_{i+1}_std"] = float(np.std(mfccs[i, :]))
        
    return feats

def extract_features_from_audio(file_path: str, target_sr: int = 4000) -> dict:
    """Loads audio, applies DSP preprocessing, and extracts features."""
    _, y_clean, sr = load_and_preprocess_audio(file_path, target_sr=target_sr, enable_denoise=True)
    return extract_signal_features(y_clean, sr=sr)
