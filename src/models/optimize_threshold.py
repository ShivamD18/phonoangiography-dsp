import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score

from src.models.train import load_data, DATA_PATH

def optimize_clinical_threshold(y_true: np.ndarray, y_probs: np.ndarray, min_specificity: float = 0.80):
    """
    Finds the operating probability threshold that maximizes Sensitivity 
    subject to Specificity >= min_specificity.
    """
    # Compute ROC points across all decision boundaries
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    specificities = 1.0 - fpr
    sensitivities = tpr

    # Constraint mask: Specificity >= target
    valid_mask = specificities >= min_specificity

    if not np.any(valid_mask):
        print(f"Warning: No threshold achieves specificity >= {min_specificity * 100:.1f}%.")
        best_idx = np.argmax(specificities)
    else:
        valid_indices = np.where(valid_mask)[0]
        # Choose the candidate that maximizes sensitivity
        best_sub_idx = np.argmax(sensitivities[valid_indices])
        best_idx = valid_indices[best_sub_idx]

    optimal_threshold = thresholds[best_idx]
    optimal_sens = sensitivities[best_idx]
    optimal_spec = specificities[best_idx]

    return optimal_threshold, optimal_sens, optimal_spec, fpr, tpr, thresholds

def evaluate_at_threshold(y_true: np.ndarray, y_probs: np.ndarray, threshold: float, label: str = ""):
    preds = (y_probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    acc = (tp + tn) / len(y_true)

    print(f"\n--- Performance @ Threshold = {threshold:.3f} ({label}) ---")
    print(f"Sensitivity (Recall): {sens * 100:6.2f}%")
    print(f"Specificity:          {spec * 100:6.2f}%")
    print(f"Accuracy:             {acc * 100:6.2f}%")
    print("Confusion Matrix:")
    print(f"  [TN: {tn:3d}  |  FP: {fp:3d}]")
    print(f"  [FN: {fn:3d}  |  TP: {tp:3d}]")
    return sens, spec

def run():
    if not DATA_PATH.exists():
        print(f"Dataset not found at {DATA_PATH}. Run feature extraction first.")
        return

    X, y, _ = load_data(DATA_PATH)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf = RandomForestClassifier(n_estimators=200, max_depth=6, class_weight="balanced", random_state=42, n_jobs=-1)

    # Gather out-of-fold predicted probabilities
    oof_probs = np.zeros(len(y))
    for train_idx, val_idx in cv.split(X, y):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val = X.iloc[val_idx]
        rf.fit(X_train, y_train)
        oof_probs[val_idx] = rf.predict_proba(X_val)[:, 1]

    auc = roc_auc_score(y, oof_probs)
    print(f"5-Fold Cross-Validated ROC-AUC: {auc:.4f}")

    # Baseline comparison at standard 0.5 threshold
    evaluate_at_threshold(y, oof_probs, threshold=0.5, label="Default Cutoff")

    # Optimize threshold for Specificity >= 80%
    tau_opt, sens_opt, spec_opt, fpr, tpr, thresholds = optimize_clinical_threshold(y, oof_probs, min_specificity=0.80)
    evaluate_at_threshold(y, oof_probs, threshold=tau_opt, label="Clinically Optimized (Spec >= 80%)")

    # Plot ROC curve with operating points
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"ROC Curve (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", alpha=0.6)

    # Mark points
    default_idx = np.argmin(np.abs(thresholds - 0.5))
    plt.scatter(fpr[default_idx], tpr[default_idx], color="orange", s=90, zorder=5, label="Default (tau=0.50)")
    plt.scatter(1.0 - spec_opt, sens_opt, color="red", s=110, zorder=6, label=f"Constrained Opt (tau={tau_opt:.2f})")

    plt.axvline(x=0.20, color="green", linestyle=":", alpha=0.7, label="Spec >= 80% (FPR <= 0.20)")
    plt.title("ROC Operating Point Optimization for Vascular Bruit Detection", fontweight="bold")
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    fig_path = ROOT_DIR / "docs" / "threshold_optimization_roc.png"
    plt.savefig(fig_path, dpi=300)
    print(f"\nSaved diagnostic ROC curve plot to: {fig_path}")

if __name__ == "__main__":
    run()
