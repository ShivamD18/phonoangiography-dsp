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
    order=2 produces an effective 4th-order roll-off (-24 dB/octave).
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    sos = signal.butter(order, [low, high], btype='bandpass', output='sos')
    y_filtered = signal.sosfiltfilt(sos, data)
    return y_filtered

def spectral_noise_subtraction(
    y: np.ndarray, 
    fs: int = 4000, 
    n_fft: int = 512, 
    hop_length: int = 128, 
    floor_percentile: float = 15.0, 
    over_subtraction: float = 1.2
) -> np.ndarray:
    """
    Estimates stationary background noise floor from the lowest energy frames 
    (quiescent valleys) and performs spectral power subtraction.
    """
    # 1. Compute STFT
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    mag, phase = np.abs(D), np.angle(D)
    
    # 2. Identify quiescent frames based on total frame energy
    frame_energies = np.sum(mag ** 2, axis=0)
    thresh = np.percentile(frame_energies, floor_percentile)
    noise_frames = mag[:, frame_energies <= thresh]
    
    if noise_frames.shape[1] == 0:
        noise_profile = np.median(mag, axis=1, keepdims=True)
    else:
        noise_profile = np.median(noise_frames, axis=1, keepdims=True)
        
    # 3. Power spectrum subtraction with spectral floor to avoid musical noise
    power_signal = mag ** 2
    power_noise = (noise_profile ** 2) * over_subtraction
    
    clean_power = np.maximum(power_signal - power_noise, 0.05 * power_signal)
    clean_mag = np.sqrt(clean_power)
    
    # 4. Invert back to time domain
    D_clean = clean_mag * np.exp(1j * phase)
    y_clean = librosa.istft(D_clean, hop_length=hop_length, length=len(y))
    return y_clean

def apply_spectral_whitening(
    y: np.ndarray, 
    fs: int = 4000, 
    n_fft: int = 512, 
    hop_length: int = 128, 
    smoothing_bins: int = 16
) -> np.ndarray:
    """
    Spectral whitening: flattens the long-term spectral tilt / sensor transfer 
    function by dividing the magnitude spectrum by its smoothed envelope.
    """
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    mag, phase = np.abs(D), np.angle(D)
    
    # Estimate the smooth frequency envelope across frequency bins
    kernel = np.ones((smoothing_bins, 1)) / smoothing_bins
    # Pad along frequency axis to maintain dimension
    smooth_env = signal.convolve2d(mag, kernel, mode='same', boundary='symm')
    
    # Normalize magnitude by smooth envelope (prevent division by near-zero)
    whitened_mag = mag / (smooth_env + 1e-6)
    
    D_whitened = whitened_mag * np.exp(1j * phase)
    y_whitened = librosa.istft(D_whitened, hop_length=hop_length, length=len(y))
    return y_whitened

def load_and_preprocess_audio(
    file_path: str, 
    target_sr: int = 4000, 
    lowcut: float = 50.0, 
    highcut: float = 1200.0,
    enable_denoise: bool = True
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Complete DSP frontend:
    Resampling -> 50-1200Hz Bandpass -> Noise Subtraction -> Amplitude Normalization
    """
    y_raw, sr = librosa.load(file_path, sr=target_sr, mono=True)
    y_raw_norm = y_raw / (np.max(np.abs(y_raw)) + 1e-8)
    
    # 1. Bandpass filter
    y_filtered = butter_bandpass_filter(y_raw_norm, lowcut=lowcut, highcut=highcut, fs=sr)
    
    # 2. Subtract stationary noise floor if enabled
    if enable_denoise:
        y_filtered = spectral_noise_subtraction(y_filtered, fs=sr)
        
    y_filtered_norm = y_filtered / (np.max(np.abs(y_filtered)) + 1e-8)
    return y_raw_norm, y_filtered_norm, sr

def compute_shannon_energy_envelope(
    signal_data: np.ndarray, 
    fs: int = 4000, 
    window_ms: float = 20.0, 
    epsilon: float = 1e-6
) -> np.ndarray:
    """Computes normalized average Shannon energy envelope."""
    x = signal_data / (np.max(np.abs(signal_data)) + 1e-8)
    x_sq = x ** 2
    shannon_energy = - x_sq * np.log(x_sq + epsilon)
    
    window_samples = int((window_ms / 1000.0) * fs)
    if window_samples < 3:
        window_samples = 3
    kernel = np.ones(window_samples) / window_samples
    
    smooth_envelope = np.convolve(shannon_energy, kernel, mode='same')
    env_min = np.min(smooth_envelope)
    env_max = np.max(smooth_envelope)
    envelope_norm = (smooth_envelope - env_min) / (env_max - env_min + 1e-8)
    return envelope_norm
