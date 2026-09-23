import numpy as np
import scipy.signal as signal
import librosa
from typing import Tuple

def butter_bandpass_filter(
    data: np.ndarray, 
    lowcut: float = 50.0, 
    highcut: float = 1200.0, 
    fs: int = 4000, 
    order: int = 2
) -> np.ndarray:
    """
    Applies a zero-phase forward-backward Butterworth bandpass filter.
    order=2 passed to sosfiltfilt produces an effective 4th-order roll-off (-24 dB/octave)
    without phase distortion.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    # Second-order sections (SOS) are numerically more stable than (b, a)
    sos = signal.butter(order, [low, high], btype='bandpass', output='sos')
    y_filtered = signal.sosfiltfilt(sos, data)
    return y_filtered

def load_and_preprocess_audio(
    file_path: str, 
    target_sr: int = 4000, 
    lowcut: float = 50.0, 
    highcut: float = 1200.0
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Loads audio, resamples to target_sr, applies bandpass filter, and normalizes amplitude.
    Returns:
        (raw_normalized, filtered_normalized, sample_rate)
    """
    y_raw, sr = librosa.load(file_path, sr=target_sr, mono=True)
    
    # Peak amplitude normalization [-1.0, 1.0]
    y_raw_norm = y_raw / (np.max(np.abs(y_raw)) + 1e-8)
    
    # Filter
    y_filtered = butter_bandpass_filter(y_raw_norm, lowcut=lowcut, highcut=highcut, fs=sr)
    y_filtered_norm = y_filtered / (np.max(np.abs(y_filtered)) + 1e-8)
    
    return y_raw_norm, y_filtered_norm, sr

def compute_shannon_energy_envelope(
    signal_data: np.ndarray, 
    fs: int = 4000, 
    window_ms: float = 20.0, 
    epsilon: float = 1e-6
) -> np.ndarray:
    """
    Computes the normalized average Shannon energy envelope.
    Window size defaults to 20 ms (80 samples at 4 kHz) for hemodynamic pulse smoothing.
    """
    # 1. Normalize signal to maximum absolute value
    x = signal_data / (np.max(np.abs(signal_data)) + 1e-8)
    
    # 2. Instantaneous Shannon Energy: -x^2 * log(x^2 + eps)
    x_sq = x ** 2
    shannon_energy = - x_sq * np.log(x_sq + epsilon)
    
    # 3. Moving Average smoothing window
    window_samples = int((window_ms / 1000.0) * fs)
    if window_samples < 3:
        window_samples = 3
    kernel = np.ones(window_samples) / window_samples
    
    # Convolve with 'same' to preserve exact sample alignment
    smooth_envelope = np.convolve(shannon_energy, kernel, mode='same')
    
    # 4. Standardize envelope to [0, 1] range
    env_min = np.min(smooth_envelope)
    env_max = np.max(smooth_envelope)
    envelope_norm = (smooth_envelope - env_min) / (env_max - env_min + 1e-8)
    
    return envelope_norm
