import sys
import tempfile
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import librosa
import librosa.display

from src.inference.predict import PhonoangiographyPredictor
from src.dsp.filters import load_and_preprocess_audio, compute_shannon_energy_envelope

st.set_page_config(
    page_title="Phonoangiography Acoustic Triage",
    page_icon="🩺",
    layout="wide"
)

@st.cache_resource
def load_predictor():
    return PhonoangiographyPredictor()

try:
    predictor = load_predictor()
except Exception as e:
    st.error(f"Failed to load ONNX model or configuration: {e}")
    st.stop()

# Header
st.title("🩺 Phonoangiography Acoustic Telemetry & Triage")
st.markdown(
    "Acoustic monitoring for vascular access and hemodynamic flow. "
    "Classifies **Laminar (Normal)** vs. **Turbulent (Stenotic/Murmur)** signatures using zero-phase DSP and edge ONNX inference."
)

# Sidebar: Input Selection
st.sidebar.header("Audio Input Source")
input_mode = st.sidebar.radio("Choose Input Mode", ["Upload WAV File", "Select Sample from Dataset"])

audio_bytes = None
temp_audio_path = None
file_display_name = ""

if input_mode == "Upload WAV File":
    uploaded_file = st.sidebar.file_uploader("Upload an acoustic .wav file", type=["wav"])
    if uploaded_file is not None:
        audio_bytes = uploaded_file.read()
        file_display_name = uploaded_file.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            temp_audio_path = tmp.name
else:
    sample_options = {}
    for cohort in ["training-a", "training-b", "training-c"]:
        p = ROOT_DIR / "data" / "raw" / cohort
        if p.exists():
            for f in sorted(list(p.glob("*.wav")))[:4]:
                sample_options[f"{cohort}/{f.name}"] = f

    if sample_options:
        selected_sample_key = st.sidebar.selectbox("Choose sample recording", list(sample_options.keys()))
        selected_file_path = sample_options[selected_sample_key]
        file_display_name = selected_file_path.name
        temp_audio_path = str(selected_file_path)
        with open(selected_file_path, "rb") as f:
            audio_bytes = f.read()
    else:
        st.sidebar.warning("No sample files found in data/raw/. Please use the file uploader.")

# Main Dashboard Execution
if temp_audio_path and audio_bytes:
    # 1. Audio Playback Bar
    st.subheader(f"Audio Recording: `{file_display_name}`")
    st.audio(audio_bytes, format="audio/wav")

    # 2. Run Inference
    with st.spinner("Extracting DSP features & running ONNX inference..."):
        try:
            result = predictor.predict(temp_audio_path)
            raw, filtered, sr = load_and_preprocess_audio(temp_audio_path, target_sr=4000)
            envelope = compute_shannon_energy_envelope(filtered, fs=sr)
        except Exception as e:
            st.error(f"Error during audio processing: {e}")
            st.stop()

    # 3. Diagnostic Results Banner
    st.markdown("---")
    res_col1, res_col2, res_col3 = st.columns([1.5, 1.5, 1])

    is_abnormal = result["is_abnormal"]
    p_abnormal = result["prob_abnormal"]
    p_normal = result["prob_normal"]
    tau = result["operating_threshold"]

    with res_col1:
        if is_abnormal:
            st.error("### ⚠️ ABNORMAL / TURBULENT DETECTED")
            st.caption("Acoustic profile indicates flow jetting, stenosis, or pathological murmur.")
        else:
            st.success("### ✅ NORMAL / LAMINAR FLOW")
            st.caption("Acoustic profile exhibits low-frequency baseline and structured valve dynamics.")

    with res_col2:
        st.metric(
            label="Turbulence Probability P(Abnormal)",
            value=f"{p_abnormal * 100:.1f}%",
            delta=f"{(p_abnormal - tau) * 100:+.1f}% vs Cutoff",
            delta_color="inverse"
        )
        st.progress(min(max(p_abnormal, 0.0), 1.0))
        st.caption(f"Calibrated Clinical Operating Threshold: **{tau:.3f}**")

    with res_col3:
        st.metric(label="Sampling Rate", value=f"{sr} Hz")
        st.metric(label="Duration", value=f"{len(filtered) / sr:.2f} s")

    # 4. Acoustic Telemetry KPIs
    st.markdown("#### Real-Time Acoustic Telemetry")
    tel = result["telemetry_summary"]
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Spectral Centroid", f"{tel['spectral_centroid_mean_hz']:.1f} Hz")
    kpi2.metric("Spectral Rolloff (85%)", f"{tel['spectral_rolloff_mean_hz']:.1f} Hz")
    kpi3.metric("Shannon Dynamic Ratio", f"{tel['shannon_dynamic_ratio']:.1f}")
    kpi4.metric("Spectral Flatness", f"{tel['spectral_flatness_mean']:.4f}")

    # 5. Signal Visualizations
    st.markdown("---")
    st.markdown("#### Signal Morphology & Time-Frequency Analysis")

    tab1, tab2 = st.tabs(["Time-Frequency Spectrogram", "Waveform & Shannon Envelope"])

    max_sec = 5.0
    n_plot = min(len(filtered), int(max_sec * sr))
    t = np.linspace(0, n_plot / sr, n_plot)

    with tab1:
        fig_spec, ax_spec = plt.subplots(figsize=(12, 4))
        hop = 128
        n_fft = 512
        S = librosa.feature.melspectrogram(y=filtered[:n_plot], sr=sr, n_fft=n_fft, hop_length=hop, n_mels=64, fmax=1200)
        S_db = librosa.power_to_db(S, ref=np.max)
        
        cent = librosa.feature.spectral_centroid(y=filtered[:n_plot], sr=sr, n_fft=n_fft, hop_length=hop)[0]
        times = librosa.times_like(cent, sr=sr, hop_length=hop)

        img = librosa.display.specshow(S_db, sr=sr, hop_length=hop, x_axis='time', y_axis='mel', fmax=1200, ax=ax_spec, cmap='magma')
        ax_spec.plot(times, cent, color='cyan', lw=1.8, label='Spectral Centroid Track (Hz)')
        ax_spec.set_title("50-1200 Hz Mel-Spectrogram with Centroid Trajectory", fontweight="bold")
        ax_spec.legend(loc="upper right")
        fig_spec.colorbar(img, ax=ax_spec, format="%+2.0f dB")
        plt.tight_layout()
        st.pyplot(fig_spec)
        plt.close(fig_spec)

    with tab2:
        fig_wave, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
        ax1.plot(t, filtered[:n_plot], color="#1f77b4", lw=0.9)
        ax1.set_title("Zero-Phase Bandpassed Acoustic Signal (50-1200 Hz)", fontweight="bold")
        ax1.set_ylabel("Amplitude")
        ax1.grid(True, alpha=0.3)

        ax2.plot(t, envelope[:n_plot], color="#2ca02c", lw=1.2)
        ax2.fill_between(t, 0, envelope[:n_plot], color="#2ca02c", alpha=0.25)
        ax2.set_title("Normalized Shannon Energy Envelope", fontweight="bold")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Energy")
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_wave)
        plt.close(fig_wave)

else:
    st.info("Select a sample recording from the sidebar or upload your own .wav file to run inference.")
