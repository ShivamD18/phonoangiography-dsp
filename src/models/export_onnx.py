import sys
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnx
import onnxruntime as ort
from sklearn.model_selection import StratifiedKFold

from src.models.train import load_data, DATA_PATH
from src.models.optimize_threshold import optimize_clinical_threshold

MODELS_DIR = ROOT_DIR / "models"
ONNX_PATH = MODELS_DIR / "phonoangiography_rf.onnx"
METADATA_PATH = MODELS_DIR / "model_config.json"

def export_model_to_onnx():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data using corrected mapping from train.py
    X, y, feature_cols = load_data(DATA_PATH)
    n_features = X.shape[1]
    
    print(f"Loaded training data: {len(y)} samples.")
    print(f"Count Class 0 (Normal): {np.sum(y == 0)}")
    print(f"Count Class 1 (Abnormal): {np.sum(y == 1)}")
    
    # Check mean spectral centroid to ensure physical orientation
    cent_norm = X.loc[y == 0, "spectral_centroid_mean"].mean()
    cent_abnorm = X.loc[y == 1, "spectral_centroid_mean"].mean()
    print(f"Sanity Check -> Mean Centroid Class 0 (Normal):   {cent_norm:.2f} Hz")
    print(f"Sanity Check -> Mean Centroid Class 1 (Abnormal): {cent_abnorm:.2f} Hz")
    
    # If the labels in CSV are flipped (i.e. Normal has higher centroid than Abnormal), invert y
    if cent_norm > cent_abnorm:
        print("WARNING: Label inversion detected in CSV. Inverting target y to preserve physiological ground truth...")
        y = 1 - y
        print(f"Corrected -> Class 0 (Normal) Centroid: {X.loc[y == 0, 'spectral_centroid_mean'].mean():.2f} Hz")
        print(f"Corrected -> Class 1 (Abnormal) Centroid: {X.loc[y == 1, 'spectral_centroid_mean'].mean():.2f} Hz")

    # 2. Determine Optimal Threshold via Stratified 5-Fold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf = RandomForestClassifier(n_estimators=200, max_depth=6, class_weight="balanced", random_state=42, n_jobs=-1)
    
    oof_probs = np.zeros(len(y))
    for train_idx, val_idx in cv.split(X, y):
        rf.fit(X.iloc[train_idx], y[train_idx])
        oof_probs[val_idx] = rf.predict_proba(X.iloc[val_idx])[:, 1]
        
    optimal_threshold, sens, spec, _, _, _ = optimize_clinical_threshold(y, oof_probs, min_specificity=0.80)
    print(f"Calibrated Threshold: {optimal_threshold:.4f} (Sens: {sens*100:.1f}%, Spec: {spec*100:.1f}%)")

    # 3. Train Final Model on All Data
    print("Fitting production Random Forest...")
    rf_final = RandomForestClassifier(n_estimators=200, max_depth=6, class_weight="balanced", random_state=42, n_jobs=-1)
    rf_final.fit(X, y)

    # 4. Convert Scikit-Learn Model to ONNX
    initial_type = [('float_input', FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(
        rf_final, 
        initial_types=initial_type, 
        target_opset=12,
        options={id(rf_final): {'zipmap': False}}
    )

    with open(ONNX_PATH, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"Overwritten: {ONNX_PATH}")

    # 5. Export JSON Configuration
    config = {
        "model_file": ONNX_PATH.name,
        "n_features": n_features,
        "feature_names": feature_cols,
        "optimal_threshold": float(optimal_threshold),
        "target_sampling_rate": 4000,
        "bandpass_hz": [50.0, 1200.0],
        "class_labels": {
            "0": "Normal_Laminar",
            "1": "Abnormal_Turbulent"
        }
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Overwritten: {METADATA_PATH}")

if __name__ == "__main__":
    export_model_to_onnx()
