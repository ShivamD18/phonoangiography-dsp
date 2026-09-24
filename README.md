# Phonoangiography DSP: Acoustic Telemetry & Hemodynamic Flow Triage

An edge-deployable digital signal processing (DSP) and machine learning pipeline for real-time acoustic phonoangiography. The system characterizes vascular access patency and valvular hemodynamics by isolating turbulent flow regimes (stenotic murmurs, bruits) from laminar valve closures in phonocardiogram (PCG) recordings.

## Clinical Rationale: Beyond Digital Stethoscopes

Commercial digital stethoscopes operate primarily as acoustic amplifiers. While they offer active decibel gain and basic hardware bandpass filtering (typically Bell or Diaphragm modes), they retain critical diagnostic liabilities:

- **Acoustic Subjectivity and Human Error:** Digital stethoscopes rely entirely on subjective auditory interpretation by the clinician. Subtle high-frequency turbulent bruits indicative of early-stage arteriovenous fistula (AVF) stenosis or subclinical valvular defects are easily masked by low-frequency, high-amplitude heart sounds ($S_1$/$S_2$) and room acoustics.
    
- **Absence of Quantitative Spectral Descriptors:** Simple amplification boosts background friction and sensor displacement artifacts alongside physiologic acoustic events. Amplifiers cannot quantify energy distribution or compute turbulence markers across discrete frequency sub-bands.
    
- **Binary Auscultation Limitations:** Human hearing perceives loudness logarithmically and struggles to resolve simultaneous acoustic events with overlapping spectra. Digital stethoscopes provide no automated boundary detection, objective telemetry metrics, or reproducible risk calibration.
    

### The Phonoangiography Approach

Phonoangiography treats the vascular bed as an acoustic fluid-structure system. Fluid flowing through a non-stenosed, compliant vessel remains predominantly laminar, generating low-frequency boundary acoustic signatures during valve coaptation ($50\text{--}150\text{ Hz}$).

When lumen diameter decreases due to stenosis or vessel wall deformation, local flow velocity exceeds the critical Reynolds number ($Re_c$), inducing turbulent vortices and wall vibrations:

$$Re = \frac{\rho v D}{\mu}$$

This turbulence produces high-frequency, broadband acoustic energy ($200\text{--}600\text{ Hz}$) that persists across the cardiac cycle. Rather than simply boosting volume, this system separates acoustic energy states, computes normalized ratio telemetry, and classifies hemodynamic risk using a calibrated machine learning boundary.

## Architectural Pipeline

```
Raw Acoustic Stream (.wav, 4 kHz)
  │
  ▼
[Stage 1: Preconditioning & Denoising]
  ├── Zero-Phase 4th-Order Butterworth Filter (25–800 Hz)
  ├── 100 ms Transient Startup Impulse Truncation
  └── Spectral Subtraction Noise Reduction (Librosa/SciPy)
  │
  ▼
[Stage 2: Deterministic DSP Extraction]
  ├── Sub-Band Energy Ratio: PSD(200–600 Hz) / PSD(50–150 Hz)
  ├── Shannon Energy Envelope Dynamics (Peak-to-Floor Tracking)
  ├── Spectral Moments (Centroid Mean, P95, Rolloff, Flatness)
  └── 13 Mel-Frequency Cepstral Coefficients (MFCCs + Variance)
  │
  ▼
[Stage 3: Quantized Edge Inference (ONNX)]
  └── Multi-Center Calibrated Random Forest Model
  │
  ▼
[Stage 4: Telemetry Output & Dynamic Triage]
  ├── Real-Time Decision Thresholding (Screening vs. Confirmation)
  ├── Linear Time-Frequency Spectrogram with Centroid Tracking
  └── Soft-Limited PCM16 In-Memory Audio Playback
```

## Quantitative Feature Formulation

### 1. Turbulent Murmur Sub-Band Ratio

Turbulence shifts spectral density upward away from fundamental valve motion. Using Welch's method for Power Spectral Density (PSD) estimation, we evaluate the ratio of high-frequency turbulent kinetic dissipation to low-frequency laminar energy:

$$\mathcal{R}_{\text{murmur}} = \frac{\int_{200}^{600} S_{xx}(f) \, df}{\int_{50}^{150} S_{xx}(f) \, df + \epsilon}$$

Pathological recordings exhibit elevated ratios ($\mathcal{R}_{\text{murmur}} \gg 0.35$), whereas physiologic laminar recordings confine energy almost entirely to the denominator.

### 2. Shannon Energy Dynamic Envelope

To detect transient murmur murmurs without false triggering on baseline static noise, the signal is projected through a non-linear Shannon energy transform:

$$E_{\text{Shannon}}[n] = -y[n]^2 \log_e\left(y[n]^2 + \epsilon\right)$$

Dynamic range calculation between envelope peaks and the 10th-percentile valley floor ($P_{10}$) captures whether the recording contains distinct structural pauses or constant baseline turbulence.

### 3. Spectral Moments & Centroid Trajectory

The spectral centroid traces the instantaneous center of mass of the spectrum across short-time Fourier transform (STFT) frames:

$$f_c[t] = \frac{\sum_{k=0}^{K-1} f_k \vert{}X[t, k]\vert{}}{\sum_{k=0}^{K-1} \vert{}X[t, k]\vert{}}$$

The 95th-percentile spectral centroid ($f_{c, 95}$) is monitored to ensure transient, high-frequency systolic/diastolic bruits are detected even if cycle-averaged energy remains low.

## Performance & Diagnostic Operating Modes

The model is trained on multi-center clinical cohorts (PhysioNet/CinC Challenge dataset). Due to operational imbalances in diagnostic triage, the deployment incorporates a configurable decision boundary ($\tau$):

|**Operating Mode**|**Cutoff Range (τ)**|**Target Sensitivity**|**Target Specificity**|**Clinical Utility**|
|---|---|---|---|---|
|**High-Sensitivity Screening**|$0.20 \le \tau \le 0.45$|$>92\%$|$\approx 65\text{--}72\%$|Outpatient triage; early vascular access monitoring; zero tolerated false negatives.|
|**Balanced Clinical Triage**|$0.46 \le \tau \le 0.65$|$\approx 83\%$|$\approx 82\%$|General bedside triage; optimal diagnostic trade-off.|
|**High-Specificity Confirmation**|$0.66 \le \tau \le 0.90$|$\approx 68\%$|$>95\%$|Confirmation prior to secondary Doppler ultrasound or invasive angiogram referral.|

## Repository Structure

```
phonoangiography-dsp/
├── app.py                     # Streamlit telemetry web application
├── packages.txt               # Debian system dependencies (libsndfile1, ffmpeg)
├── requirements.txt           # Python dependency manifest
├── Dockerfile                 # Multi-stage production container configuration
├── docker-compose.yml         # Container orchestration manifest
├── demo_samples/              # Benchmark clinical audio files & cohort manifest
│   ├── manifest.csv
│   └── *.wav
├── models/                    # Serialized model artifacts
│   ├── phonoangiography_rf.onnx
│   └── model_config.json
├── src/
│   ├── dsp/
│   │   ├── filters.py         # Butterworth conditioning & Shannon envelopes
│   │   └── features.py        # Sub-band integration & spectral moment extraction
│   ├── inference/
│   │   └── predict.py         # ONNX Runtime inference wrapper
│   └── models/
│       └── train_multicenter.py
└── docs/
    └── engineering_log.md     # Development history and mathematical logs
```

## Local Installation and Execution

### Running via Docker (Recommended)

Ensure Docker Engine is installed and running:

Bash

```
docker compose up --build -d
```

The application will be accessible at `http://localhost:8501`.

### Running in a Python Environment

1. Ensure Python 3.11 is active.
    
2. Install system audio codecs:
    
    - **Ubuntu/Debian:** `sudo apt-get install -y libsndfile1 ffmpeg`
        
    - **macOS:** `brew install libsndfile ffmpeg`
        
    - **Windows:** Codecs are typically bundled with wheel binaries.
        
3. Install dependencies and start the interface:
    
    Bash
    
    ```
    pip install -r requirements.txt
    streamlit run app.py
    ```
    

## Verification & Testing

To run the offline cohort evaluation script across dataset splits:

Bash

```
python evaluate_cohort_a.py
```

To run feature threshold sweeping across user-defined metrics:

Bash

```
python sweep_thresholds.py
```
