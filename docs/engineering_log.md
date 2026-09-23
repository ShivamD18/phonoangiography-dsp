# Engineering Log: Phonoangiography DSP & Machine Learning Pipeline

## Milestone: Feature Engineering, Model Training, Inversion Correction & Hardware Equalization

### 1. Ingestion & Preprocessing Frontend
* **Automated Data Pipeline (`src/data/download.py`):**
  * Updated dataset ingestion to handle the complete PhysioNet CinC 2016 archive (`training.zip`).
  * Implemented streaming download with custom User-Agent headers, chunked buffer writing, and interactive `tqdm` progress tracking.
* **Core DSP Architecture (`src/dsp/filters.py`):**
  * **Bandpass Filtering:** Implemented a 4th-order zero-phase Butterworth bandpass filter (\(50\text{--}1200\text{ Hz}\)) using Second-Order Sections (`sosfiltfilt`) to reject baseline drift and high-frequency electrical interference without phase distortion.
  * **Envelope Tracking:** Developed the normalized average Shannon energy envelope to accentuate systolic/diastolic flow pulses while suppressing baseline noise.
  * **Adaptive Noise Gating & Spectral Subtraction:** Added quiescent valley noise profiling and spectral power subtraction to suppress stationary background sensor hum and equalize baseline transfer functions across varied stethoscopes.
  * **Spectral Whitening:** Implemented running spectral envelope normalization to mitigate hardware-specific acoustic coloring.

### 2. Feature Extraction Pipeline (`src/dsp/features.py`)
* Built parameter-driven feature extraction capable of batch processing arbitrary dataset splits (`--set training-a`, `--set training-b`, `--set training-c`).
* Extracted 37 physical and timbral acoustic metrics per recording:
  * **Spectral Moments:** Spectral centroid (mean, std, max), spectral roll-off (85% threshold), spectral flatness, and zero-crossing rate.
  * **Dynamic Metrics:** Shannon dynamic energy ratio (peak pulse vs. 10th-percentile valley floor).
  * **Cepstral Coefficients:** MFCCs 1–13 (mean and standard deviation).

### 3. Model Training, Calibration & Clinical Optimization
* **Baseline Modeling (`src/models/train.py`):**
  * Implemented 5-fold Stratified K-Fold cross-validation for Random Forest and XGBoost classifiers.
  * Resolved critical label inversion between PhysioNet raw conventions (`1 = Normal`, `-1 = Abnormal`) and internal ML binary representations (`0 = Normal`, `1 = Abnormal`), ensuring abnormal turbulent bruits correctly align with elevated spectral centroids.
* **Operating Threshold Tuning (`src/models/optimize_threshold.py`):**
  * Built an ROC-constrained optimizer that tunes the probability cutoff \(\tau^*\) to maximize clinical Sensitivity while enforcing Specificity \(\ge 80\%\), heavily penalizing false negatives.
* **Multi-Center / Cross-Hardware Generalization (`src/models/train_multicenter.py`):**
  * Pooled data across multiple recording cohorts (`training-a`, `training-b`, `training-c`) representing distinct hardware (Welch Allyn, Littmann 3200, JABES).
  * Evaluated cross-hardware invariance using Leave-One-Group-Out (LOGO) cross-validation to guard against stethoscope sensor transfer function overfitting.

### 4. Edge Deployment & Inference Runtime
* **ONNX Model Export (`src/models/export_onnx.py`):**
  * Serialized the production Random Forest model into an ONNX binary (`models/phonoangiography_rf.onnx`) with `zipmap=False` for sub-millisecond CPU inference.
  * Exported runtime metadata schema (`models/model_config.json`) embedding feature alignments, sampling parameters, and the empirical operating threshold.
* **Edge Inference Client (`src/inference/predict.py`):**
  * Built an end-to-end client taking arbitrary `.wav` audio, executing DSP filtering, extracting runtime features, and predicting diagnostic class with acoustic telemetry outputs (spectral centroid, roll-off, Shannon dynamic ratio).

## Milestone: Audio Engine, Containerization, and Diagnostic Calibration

### 1. Acoustic Playback & Audibility
* **PCM16 Normalization & In-Memory Streaming:** Converted raw, low-amplitude PCG signals into 16-bit PCM WAV byte buffers in memory using `soundfile`, resolving browser audio decoding failures.
* **Dynamic Digital Gain Boost & Soft Limiting:** Integrated a customizable sidebar gain multiplier (\(0\text{--}24\text{ dB}\)) with hyperbolic tangent (\(\tanh\)) soft clipping to amplify faint low-frequency valve closures and murmurs without digital square-wave distortion.

### 2. Visualization & Boundary Processing
* **Spectrogram Coordinate Realignment:** Migrated time-frequency plots to a linear frequency scale (\(0\text{--}1200\text{ Hz}\)) to align the overlaid cyan spectral centroid trajectory with the underlying STFT magnitude bins.
* **Transient Edge Trimming:** Trimmed the initial \(100\text{ ms}\) filter startup impulse to prevent convolution edge artifacts from skewing visual displays and global energy moments.

### 3. Feature Set Expansion & Numerical Stability
* **Turbulent Murmur Sub-Band Ratio:** Added energy ratio tracking comparing murmur frequencies (\(200\text{--}600\text{ Hz}\)) to fundamental valve closures (\(50\text{--}150\text{ Hz}\)) via Welch power spectral density.
* **NumPy 2.0+ Compatibility:** Migrated numerical integration from the deprecated `np.trapz` to `scipy.integrate.trapezoid`.
* **Transient & Noise Floor Descriptors:** Added the 95th-percentile spectral centroid (`spectral_centroid_p95`) and 10th-percentile envelope floor (`shannon_valley_floor`) to preserve brief bruits and quantify ambient sensor noise.

### 4. Ground-Truth Calibration & Diagnostic Validation
* **PhysioNet Label Realignment:** Aligned training target mappings with the challenge reference standard (`label == 1` \(\rightarrow\) Abnormal/Turbulent, `label == -1` \(\rightarrow\) Normal/Laminar), removing heuristic inversions.
* **Empirical Threshold Sweeps:** Developed automated evaluation tools (`evaluate_cohort_a.py`, `sweep_thresholds.py`) to systematically analyze sensitivity-specificity trade-offs across 409 cohort recordings.

### 5. Production Containerization
* **Debian Slim Audio Container:** Built a production `Dockerfile` providing C-level audio codecs (`libsndfile1`, `ffmpeg`), headless Streamlit runtime flags, and integrated health checks.
* **Docker Compose Orchestration:** Configured read-only host dataset bind-mounts and automated port exposure (`8501:8501`).
