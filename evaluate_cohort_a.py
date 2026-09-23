import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, roc_auc_score
from src.inference.predict import PhonoangiographyPredictor

def evaluate_cohort(cohort_dir: Path):
    ref_file = cohort_dir / "REFERENCE.csv"
    if not ref_file.exists():
        print(f"Error: REFERENCE.csv not found in {cohort_dir}")
        return

    # Load ground truth labels
    # PhysioNet convention: 1 = Abnormal, -1 = Normal
    ref_df = pd.read_csv(ref_file, header=None, names=["record_id", "label"])
    
    # Initialize ONNX inference pipeline
    predictor = PhonoangiographyPredictor()
    tau = predictor.threshold
    
    y_true = []
    y_pred = []
    y_prob = []
    skipped = 0

    print("=" * 65)
    print(f"  EVALUATION PIPELINE: {cohort_dir.name.upper()} (409 RECORDINGS)")
    print(f"  Active Clinical Operating Threshold (tau): {tau:.3f}")
    print("=" * 65)

    for _, row in tqdm(ref_df.iterrows(), total=len(ref_df), desc="Inference Progress"):
        rec_id = str(row["record_id"]).strip()
        ground_truth = int(row["label"]) # 1: Abnormal, -1: Normal
        
        wav_path = cohort_dir / f"{rec_id}.wav"
        if not wav_path.exists():
            skipped += 1
            continue

        try:
            res = predictor.predict(str(wav_path))
            prob_abn = res["prob_abnormal"]
            pred_binary = 1 if prob_abn >= tau else 0
            target_binary = 1 if ground_truth == 1 else 0

            y_true.append(target_binary)
            y_pred.append(pred_binary)
            y_prob.append(prob_abn)
        except Exception as e:
            print(f"\nFailed to process {rec_id}: {e}")
            skipped += 1

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    # Compute Confusion Matrix Elements
    # tn: Normal predicted Normal
    # fp: Normal predicted Abnormal
    # fn: Abnormal predicted Normal
    # tp: Abnormal predicted Abnormal
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    roc_auc = roc_auc_score(y_true, y_prob)

    print("\n" + "=" * 65)
    print("                     CLINICAL PERFORMANCE METRICS")
    print("=" * 65)
    print(f"Total Evaluated Records: {len(y_true)}  (Skipped / Missing: {skipped})")
    print(f"Class Breakdown:         Normal (0): {np.sum(y_true == 0)} | Abnormal (1): {np.sum(y_true == 1)}")
    print("-" * 65)
    print(f"Sensitivity (Recall):    {sensitivity * 100:6.2f}%  (TP: {tp} / {tp + fn})")
    print(f"Specificity:             {specificity * 100:6.2f}%  (TN: {tn} / {tn + fp})")
    print(f"Positive Pred. Val (PPV):{precision * 100:6.2f}%  (TP: {tp} / {tp + fp})")
    print(f"Negative Pred. Val (NPV):{npv * 100:6.2f}%  (TN: {tn} / {tn + fn})")
    print(f"Overall Accuracy:        {accuracy * 100:6.2f}%")
    print(f"Area Under ROC (AUC):    {roc_auc:6.4f}")
    print("-" * 65)
    print("CONFUSION MATRIX:")
    print("                     Predicted Normal   Predicted Abnormal")
    print(f"Actual Normal (0):         {tn:<18} {fp:<18}")
    print(f"Actual Abnormal (1):       {fn:<18} {tp:<18}")
    print("=" * 65)

if __name__ == "__main__":
    target_dir = ROOT_DIR / "data" / "raw" / "training-a"
    evaluate_cohort(target_dir)
