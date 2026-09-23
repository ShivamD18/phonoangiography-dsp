import sys
import json
from pathlib import Path
from typing import Dict, Any, Union

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import onnxruntime as ort
from src.dsp.filters import load_and_preprocess_audio
from src.dsp.features import extract_signal_features

MODELS_DIR = ROOT_DIR / "models"
ONNX_PATH = MODELS_DIR / "phonoangiography_rf.onnx"
CONFIG_PATH = MODELS_DIR / "model_config.json"

class PhonoangiographyPredictor:
    def __init__(self, model_path: Union[str, Path] = ONNX_PATH, config_path: Union[str, Path] = CONFIG_PATH):
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        
        if not self.model_path.exists() or not self.config_path.exists():
            raise FileNotFoundError(f"Missing model binary or config file in {MODELS_DIR}")
            
        with open(self.config_path, "r") as f:
            self.config = json.load(f)
            
        self.feature_names = self.config["feature_names"]
        self.threshold = float(self.config.get("optimal_threshold", 0.50))
        self.target_sr = int(self.config.get("target_sampling_rate", 4000))
        
        # Load ONNX Inference Session
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 2
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(str(self.model_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, audio_input: Union[str, Path, np.ndarray]) -> Dict[str, Any]:
        if isinstance(audio_input, (str, Path)):
            _, y_clean, sr = load_and_preprocess_audio(str(audio_input), target_sr=self.target_sr, enable_denoise=True)
        elif isinstance(audio_input, np.ndarray):
            y_clean = audio_input
            sr = self.target_sr
        else:
            raise TypeError("audio_input must be a file path or numpy array.")

        # 1. Feature extraction
        feature_dict = extract_signal_features(y_clean, sr=sr)
        
        # 2. Vector assembly matching ONNX trained input order
        feature_vector = np.array([[feature_dict[name] for name in self.feature_names]], dtype=np.float32)
        
        # 3. ONNX execution
        outputs = self.session.run(None, {self.input_name: feature_vector})
        
        # Determine output format (probabilities are typically output index 1)
        if len(outputs) > 1 and isinstance(outputs[1], np.ndarray):
            probs = outputs[1][0]
            prob_normal = float(probs[0])
            prob_abnormal = float(probs[1])
        elif isinstance(outputs[0], np.ndarray) and outputs[0].ndim == 2 and outputs[0].shape[1] == 2:
            probs = outputs[0][0]
            prob_normal = float(probs[0])
            prob_abnormal = float(probs[1])
        else:
            prob_abnormal = float(outputs[0][0])
            prob_normal = 1.0 - prob_abnormal
            
        is_abnormal = bool(prob_abnormal >= self.threshold)
        prediction_label = "Abnormal_Turbulent" if is_abnormal else "Normal_Laminar"
        
        return {
            "prediction": prediction_label,
            "is_abnormal": is_abnormal,
            "prob_abnormal": prob_abnormal,
            "prob_normal": prob_normal,
            "operating_threshold": self.threshold,
            "telemetry_summary": {
                "spectral_centroid_mean_hz": feature_dict["spectral_centroid_mean"],
                "spectral_rolloff_mean_hz": feature_dict["spectral_rolloff_mean"],
                "shannon_dynamic_ratio": feature_dict["shannon_dynamic_ratio"],
                "spectral_flatness_mean": feature_dict["spectral_flatness_mean"],
                "murmur_subband_ratio": feature_dict["murmur_subband_ratio"]
            }
        }
