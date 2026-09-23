import sys
import json
import argparse
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import onnxruntime as ort

from src.dsp.features import extract_signal_features

MODELS_DIR = ROOT_DIR / "models"
ONNX_PATH = MODELS_DIR / "phonoangiography_rf.onnx"
CONFIG_PATH = MODELS_DIR / "model_config.json"

class PhonoangiographyPredictor:
    def __init__(self, model_path: Path = ONNX_PATH, config_path: Path = CONFIG_PATH):
        if not model_path.exists() or not config_path.exists():
            raise FileNotFoundError(
                f"Missing model artifacts. Ensure {model_path} and {config_path} exist. "
                "Run src/models/export_onnx.py first."
            )

        # 1. Load Model Configuration Metadata
        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.feature_names = self.config["feature_names"]
        self.threshold = float(self.config["optimal_threshold"])
        self.target_sr = int(self.config.get("target_sampling_rate", 4000))
        self.labels = self.config.get("class_labels", {"0": "Normal_Laminar", "1": "Abnormal_Turbulent"})

        # 2. Initialize Lightweight ONNX Runtime Session
        self.session = ort.InferenceSession(
            str(model_path), 
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, wav_file_path: str) -> dict:
        """
        Executes end-to-end processing:
        WAV -> 50-1200 Hz Bandpass -> Feature Extraction -> ONNX Classification
        """
        path_obj = Path(wav_file_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Audio file not found: {wav_file_path}")

        # 1. Extract Signal & Spectral Features
        features_dict = extract_signal_features(str(path_obj), target_sr=self.target_sr)

        # 2. Build aligned float32 feature array matching training schema
        feature_vector = np.array(
            [[features_dict[feat] for feat in self.feature_names]], 
            dtype=np.float32
        )

        # 3. Run Inference via ONNX
        _, probabilities = self.session.run(None, {self.input_name: feature_vector})
        
        # Binary classification: index 1 represents positive/abnormal
        prob_abnormal = float(probabilities[0, 1])
        prob_normal = float(probabilities[0, 0])
        
        # 4. Calibrated Decision Thresholding
        is_abnormal = prob_abnormal >= self.threshold
        predicted_class = self.labels["1"] if is_abnormal else self.labels["0"]

        return {
            "record_id": path_obj.name,
            "prediction": predicted_class,
            "is_abnormal": bool(is_abnormal),
            "prob_abnormal": prob_abnormal,
            "prob_normal": prob_normal,
            "operating_threshold": self.threshold,
            "telemetry_summary": {
                "spectral_centroid_mean_hz": round(features_dict["spectral_centroid_mean"], 2),
                "spectral_rolloff_mean_hz": round(features_dict["spectral_rolloff_mean"], 2),
                "shannon_dynamic_ratio": round(features_dict["shannon_dynamic_ratio"], 2),
                "spectral_flatness_mean": round(features_dict["spectral_flatness_mean"], 4),
            }
        }

def main():
    parser = argparse.ArgumentParser(description="Edge Acoustic Classification for Vascular Bruits")
    parser.add_argument("audio_path", type=str, help="Path to input .wav audio recording")
    args = parser.parse_args()

    try:
        predictor = PhonoangiographyPredictor()
        result = predictor.predict(args.audio_path)

        print("\n" + "=" * 55)
        print("          PHONOANGIOGRAPHY EDGE INFERENCE")
        print("=" * 55)
        print(f"File:                  {result['record_id']}")
        print(f"Diagnostic Decision:   {result['prediction'].upper()}")
        print(f"P(Abnormal/Turbulent): {result['prob_abnormal'] * 100:.2f}%")
        print(f"P(Normal/Laminar):     {result['prob_normal'] * 100:.2f}%")
        print(f"Calibrated Cutoff:     {result['operating_threshold']:.3f}")
        print("-" * 55)
        print("Acoustic Telemetry:")
        for metric, val in result["telemetry_summary"].items():
            print(f"  - {metric:<26}: {val}")
        print("=" * 55 + "\n")

    except Exception as e:
        print(f"Inference error: {e}")

if __name__ == "__main__":
    main()
