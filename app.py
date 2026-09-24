import sys
import io
import tempfile
from pathlib import Path
from datetime import datetime

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
import plotly.graph_objects as go

from src.inference.predict import PhonoangiographyPredictor
from src.dsp.filters import load_and_preprocess_audio, compute_shannon_energy_envelope

st.set_page_config(
    page_title="Phonoangiography Acoustic Triage",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Global styling
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
        html, body, [class*="css"]  {
            font-family: "Inter", "Segoe UI", sans-serif;
        }
        .block-container { padding-top: 1.6rem; padding-bottom: 3rem; }

        /* Hero header */
        .paa-hero {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 1.1rem 1.5rem;
            border-radius: 14px;
            background: linear-gradient(135deg, #101820 0%, #1c2733 100%);
            border: 1px solid rgba(255,255,255,0.06);
            margin-bottom: 1.2rem;
        }
        .paa-hero h1 { font-size: 1.5rem; margin: 0; color: #fafafa; }
        .paa-hero p { margin: 0.15rem 0 0 0; color: #9aa5b1; font-size: 0.92rem; }
        .paa-badge {
            background: rgba(230,57,70,0.12);
            border: 1px solid rgba(230,57,70,0.4);
            color: #ff8a94;
            padding: 0.3rem 0.75rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.03em;
            white-space: nowrap;
        }

        /* Verdict card */
        .verdict-card {
            border-radius: 16px;
            padding: 1.4rem 1.6rem;
            display: flex;
            flex-direction: column;
            justify-content: center;
            height: 100%;
            border-left: 6px solid;
        }
        .verdict-abnormal {
            background: rgba(230, 57, 70, 0.10);
            border-left-color: #e63946;
        }
        .verdict-normal {
            background: rgba(46, 160, 67, 0.10);
            border-left-color: #2ea043;
        }
        .verdict-title { font-size: 1.35rem; font-weight: 700; margin: 0 0 0.25rem 0; }
        .verdict-abnormal .verdict-title { color: #ff6b74; }
        .verdict-normal .verdict-title { color: #4fd67a; }
        .verdict-sub { color: #b8c0cb; font-size: 0.88rem; margin: 0; }

        /* Metric-ish KPI tiles */
        .kpi-tile {
            border-radius: 12px;
            padding: 0.85rem 1rem;
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.06);
            height: 100%;
        }
        .kpi-label { font-size: 0.74rem; color: #93a0ad; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 0.2rem;}
        .kpi-value { font-size: 1.35rem; font-weight: 700; color: #fafafa; font-variant-numeric: tabular-nums; }
        .kpi-note { font-size: 0.76rem; margin-top: 0.15rem; }
        .note-elevated { color: #ff9f68; }
        .note-normal { color: #4fd67a; }
        .note-neutral { color: #93a0ad; }

        .paa-footer {
            margin-top: 2.2rem;
            padding: 0.9rem 1.1rem;
            border-radius: 10px;
            background: rgba(255,255,255,0.03);
            border: 1px dashed rgba(255,255,255,0.15);
            font-size: 0.78rem;
            color: #93a0ad;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_predictor():
    return PhonoangiographyPredictor()


def audio_to_wav_bytes(audio_array: np.ndarray, sr: int, gain_db: float = 0.0) -> bytes:
    """
    Applies dB gain multiplier and hyperbolic tangent soft-limiting,
    then converts to playable 16-bit PCM WAV bytes.
    """
    norm_audio = audio_array / (np.max(np.abs(audio_array)) + 1e-8)
    linear_gain = 10.0 ** (gain_db / 20.0)
    boosted_audio = norm_audio * linear_gain
    limited_audio = np.tanh(boosted_audio) * 0.95

    buf = io.BytesIO()
    sf.write(buf, limited_audio, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


def kpi_tile(label: str, value: str, note: str = "", note_class: str = "note-neutral"):
    note_html = f'<div class="kpi-note {note_class}">{note}</div>' if note else ""
    st.markdown(
        f"""
        <div class="kpi-tile">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {note_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


try:
    predictor = load_predictor()
except Exception as e:
    st.error(f"Failed to load ONNX model or configuration: {e}")
    st.stop()

# ----------------------------------------------------------------------------
# Hero header
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="paa-hero">
        <div>
            <h1>🩺 Phonoangiography Acoustic Telemetry &amp; Triage</h1>
            <p>Acoustic monitoring for vascular access and hemodynamic flow &mdash;
            Laminar (Normal) vs. Turbulent (Stenotic/Murmur) classification.</p>
        </div>
        <div class="paa-badge">EDGE ONNX INFERENCE</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Configuration")

    with st.expander("🎧 Input Source", expanded=True):
        input_mode = st.radio(
            "Choose Input Mode",
            ["Upload WAV File", "Select Sample from Dataset"],
            label_visibility="collapsed",
        )

        temp_audio_path = None
        file_display_name = ""

        if input_mode == "Upload WAV File":
            uploaded_file = st.file_uploader("Upload an acoustic .wav file", type=["wav"])
            if uploaded_file is not None:
                file_display_name = uploaded_file.name
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(uploaded_file.read())
                    temp_audio_path = tmp.name
        else:
            sample_options = {}
            for cohort in ["training-a", "training-b", "training-c"]:
                p = ROOT_DIR / "data" / "raw" / cohort
                if p.exists():
                    for f in sorted(list(p.glob("*.wav")))[:6]:
                        sample_options[f"{cohort}/{f.name}"] = f

            if sample_options:
                selected_sample_key = st.selectbox("Choose sample recording", list(sample_options.keys()))
                selected_file_path = sample_options[selected_sample_key]
                file_display_name = selected_file_path.name
                temp_audio_path = str(selected_file_path)
            else:
                st.warning("No sample files found in data/raw/. Please upload a .wav file.")

    with st.expander("🔊 Playback Controls", expanded=True):
        gain_db = st.slider(
            "Audio Playback Gain (dB)",
            min_value=0.0,
            max_value=24.0,
            value=6.0,
            step=1.0,
            help="Amplify quiet heart sounds and murmurs for laptop speakers without clipping.",
        )
        st.caption("Soft tanh limiting is applied automatically to prevent harsh digital clipping.")

    with st.expander("ℹ️ About this Model", expanded=False):
        st.caption(
            "Classifies short cardiovascular acoustic recordings as laminar (normal) "
            "or turbulent (stenotic/murmur) flow using a lightweight ONNX model over "
            "zero-phase filtered signals and Shannon-energy envelope features."
        )
        st.caption(f"Session started: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# ----------------------------------------------------------------------------
# Main Dashboard
# ----------------------------------------------------------------------------
if temp_audio_path:
    with st.spinner("Loading audio & applying zero-phase filtering..."):
        try:
            raw, filtered, sr = load_and_preprocess_audio(temp_audio_path, target_sr=4000)
        except Exception as e:
            st.error(f"Error during audio preprocessing: {e}")
            st.stop()

    with st.spinner("Computing Shannon energy envelope..."):
        try:
            envelope = compute_shannon_energy_envelope(filtered, fs=sr)
        except Exception as e:
            st.error(f"Error computing envelope: {e}")
            st.stop()

    with st.spinner("Running ONNX inference..."):
        try:
            result = predictor.predict(temp_audio_path)
        except Exception as e:
            st.error(f"Error during inference: {e}")
            st.stop()

    is_abnormal = result["is_abnormal"]
    p_abnormal = result["prob_abnormal"]
    p_normal = result["prob_normal"]
    tau = result["operating_threshold"]
    tel = result["telemetry_summary"]

    # -- Audio playback --------------------------------------------------
    with st.container(border=True):
        st.markdown(f"**Audio Playback:** `{file_display_name}`")
        play_col1, play_col2 = st.columns([2, 1])
        with play_col1:
            playable_wav = audio_to_wav_bytes(filtered, sr, gain_db=gain_db)
            st.audio(playable_wav, format="audio/wav")
        with play_col2:
            st.caption(
                f"Amplified by **+{gain_db:.0f} dB** with tanh soft-limiting. "
                "Use the sidebar to boost quiet systolic/diastolic intervals."
            )

    st.write("")

    # -- Verdict + summary row -------------------------------------------
    v_col, m_col, s_col = st.columns([1.4, 1.6, 1])

    with v_col:
        verdict_class = "verdict-abnormal" if is_abnormal else "verdict-normal"
        verdict_title = "⚠️ ABNORMAL / TURBULENT" if is_abnormal else "✅ NORMAL / LAMINAR"
        verdict_sub = (
            "Acoustic profile indicates flow jetting, stenosis, or pathological murmur."
            if is_abnormal
            else "Acoustic profile exhibits low-frequency baseline and structured valve dynamics."
        )
        st.markdown(
            f"""
            <div class="verdict-card {verdict_class}">
                <p class="verdict-title">{verdict_title}</p>
                <p class="verdict-sub">{verdict_sub}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m_col:
        st.metric(
            label="Turbulence Probability P(Abnormal)",
            value=f"{p_abnormal * 100:.1f}%",
            delta=f"{(p_abnormal - tau) * 100:+.1f}% vs cutoff",
            delta_color="inverse",
        )
        st.progress(min(max(p_abnormal, 0.0), 1.0))
        st.caption(f"Calibrated clinical cutoff: **{tau:.3f}**  ·  P(Normal): {p_normal * 100:.1f}%")

    with s_col:
        kpi_tile("Sampling Rate", f"{sr} Hz")
        st.write("")
        kpi_tile("Duration", f"{len(filtered) / sr:.2f} s")

    st.write("")

    # -- Telemetry KPIs ----------------------------------------------------
    st.markdown("#### Real-Time Acoustic Telemetry")
    k1, k2, k3, k4 = st.columns(4)

    centroid = tel["spectral_centroid_mean_hz"]
    rolloff = tel["spectral_rolloff_mean_hz"]
    dyn_ratio = tel["shannon_dynamic_ratio"]
    flatness = tel["spectral_flatness_mean"]

    with k1:
        note, cls = ("Elevated", "note-elevated") if centroid > 250 else ("Within normal range", "note-normal")
        kpi_tile("Spectral Centroid", f"{centroid:.1f} Hz", note, cls)
    with k2:
        note, cls = ("Elevated", "note-elevated") if rolloff > 600 else ("Within normal range", "note-normal")
        kpi_tile("Spectral Rolloff (85%)", f"{rolloff:.1f} Hz", note, cls)
    with k3:
        kpi_tile("Shannon Dynamic Ratio", f"{dyn_ratio:.1f}", "", "note-neutral")
    with k4:
        note, cls = ("More noise-like", "note-elevated") if flatness > 0.3 else ("Tonal / structured", "note-normal")
        kpi_tile("Spectral Flatness", f"{flatness:.4f}", note, cls)

    st.write("")

    # -- Visualizations ------------------------------------------------
    st.markdown("#### Signal Morphology & Time-Frequency Analysis")

    tab1, tab2 = st.tabs(["Interactive Spectrogram", "Waveform & Shannon Envelope"])

    start_sample = int(0.1 * sr)
    n_plot = min(len(filtered) - start_sample, int(5.0 * sr))
    sig_plot = filtered[start_sample:start_sample + n_plot]
    env_plot = envelope[start_sample:start_sample + n_plot]
    t = np.linspace(0.1, 0.1 + (n_plot / sr), n_plot)

    with tab1:
        hop = 128
        n_fft = 512
        D = np.abs(librosa.stft(sig_plot, n_fft=n_fft, hop_length=hop))
        D_db = librosa.amplitude_to_db(D, ref=np.max)

        cent = librosa.feature.spectral_centroid(y=sig_plot, sr=sr, n_fft=n_fft, hop_length=hop)[0]
        times = librosa.times_like(cent, sr=sr, hop_length=hop) + 0.1
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

        freq_mask = freqs <= 1200
        spec_times = librosa.frames_to_time(np.arange(D_db.shape[1]), sr=sr, hop_length=hop) + 0.1

        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(
                z=D_db[freq_mask, :],
                x=spec_times,
                y=freqs[freq_mask],
                colorscale="Magma",
                colorbar=dict(title="dB"),
                hovertemplate="Time: %{x:.2f}s<br>Freq: %{y:.0f} Hz<br>%{z:.1f} dB<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=times,
                y=cent,
                mode="lines",
                line=dict(color="cyan", width=2),
                name="Spectral Centroid",
                hovertemplate="Time: %{x:.2f}s<br>Centroid: %{y:.0f} Hz<extra></extra>",
            )
        )
        fig.update_layout(
            title="50–1200 Hz Spectrogram with Spectral Centroid Overlay",
            xaxis_title="Time (s)",
            yaxis_title="Frequency (Hz)",
            template="plotly_dark",
            height=420,
            margin=dict(l=10, r=10, t=50, b=10),
            legend=dict(orientation="h", y=1.08, x=1, xanchor="right"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig_wave, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
        ax1.plot(t, sig_plot, color="#1f77b4", lw=0.9)
        ax1.set_title("Zero-Phase Bandpassed Acoustic Signal (50–1200 Hz)", fontweight="bold")
        ax1.set_ylabel("Amplitude")
        ax1.grid(True, alpha=0.3)

        ax2.plot(t, env_plot, color="#2ca02c", lw=1.2)
        ax2.fill_between(t, 0, env_plot, color="#2ca02c", alpha=0.25)
        ax2.set_title("Normalized Shannon Energy Envelope", fontweight="bold")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Energy")
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_wave)
        plt.close(fig_wave)

    # -- Footer disclaimer --------------------------------------------
    st.markdown(
        """
        <div class="paa-footer">
        ⚕️ <strong>Research / triage support tool — not a diagnostic device.</strong>
        Results should be interpreted alongside clinical judgment and are not a substitute
        for formal diagnostic imaging or specialist evaluation.
        </div>
        """,
        unsafe_allow_html=True,
    )

else:
    st.info("Select a sample recording from the sidebar or upload your own .wav file to run inference.")