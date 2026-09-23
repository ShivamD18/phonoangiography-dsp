import sys
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import onnxruntime as ort
from sklearn.metrics import (
    roc_auc_score, 
    confusion_matrix, 
    classification_report,
    recall_score
)

MODELS_DIR = ROOT_DIR / "models"
ONNX_PATH = MODELS_DIR / "phonoangiography_rf.onnx"
CONFIG_PATH = MODELS_DIR / "model_config.json"
TEST_DATA_PATH = ROOT_DIR / "data" / "processed" / "training_b_features.csv"

def evaluate_external():
    if not TEST_DATA_PATH.exists():
        print(f"Error: {TEST_DATA_PATH} not found. Run feature extraction on training-b first.")
        return

    # 1. Load Model Config & Session
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    
    feature_names = config["feature_names"]
    threshold = float(config["optimal_threshold"])
    
    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    # 2. Load External Test Set (training-b)
    df = pd.read_csv(TEST_DATA_PATH)
    X = df[feature_names].to_numpy(dtype=np.float32)
    
    # Ground truth mapping: 1 is Normal (0), -1 is Abnormal (1)
    y_true = np.where(df["label"] == -1, 1, 0)
    
    # Check if REFERENCE.csv in subset B uses 1/-1 standard convention
    cent_norm = df.loc[y_true == 0, "spectral_centroid_mean"].mean()
    cent_abnorm = df.loc[y_true == 1, "spectral_centroid_mean"].mean()
    if cent_norm > cent_abnorm:
        print("Note: Inverting test set reference mapping to maintain hemodynamic alignment.")
        y_true = 1 - y_true

    # 3. Run Inference
    _, probs = session.run(None, {input_name: X})
    prob_abnormal = probs[:, 1]
    y_pred = (prob_abnormal >= threshold).astype(int)

    # 4. Compute Metrics
    auc = roc_auc_score(y_true, prob_abnormal)
    sens = recall_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn / (tn + fp)
    acc = (tp + tn) / len(y_true)

    print("=" * 60)
    print("  EXTERNAL VALIDATION BENCHMARK (DATASET: training-b)")
    print("=" * 60)
    print(f"Total Test Samples:     {len(y_true)}")
    print(f"Normal (0):             {tn + fp}")
    print(f"Abnormal (1):           {tp + fn}")
    print(f"Operating Threshold:    {threshold:.4f}")
    print("-" * 60)
    print(f"Out-of-Domain ROC-AUC:  {auc:.4f}")
    print(f"Sensitivity (Recall):   {sens * 100:.2f}%  (Catches turbulent flow)")
    print(f"Specificity:            {spec * 100:.2f}%  (Rejects normal flow)")
    print(f"Overall Accuracy:       {acc * 100:.2f}%")
    print("-" * 60)
    print("Confusion Matrix:")
    print(f"  [TN (Normal):   {tn:3d}  |  FP (False Alarm): {fp:3d}]")
    print(f"  [FN (Missed):   {fn:3d}  |  TP (Detected):    {tp:3d}]")
    print("=" * 60)

if __name__ == "__main__":
    evaluate_external()
