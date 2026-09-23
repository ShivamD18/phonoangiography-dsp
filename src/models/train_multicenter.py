import sys
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score, recall_score, confusion_matrix
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from src.models.optimize_threshold import optimize_clinical_threshold

PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
ONNX_PATH = MODELS_DIR / "phonoangiography_rf.onnx"
CONFIG_PATH = MODELS_DIR / "model_config.json"

DATASETS = ["training_a", "training_b", "training_c"]

def load_and_harmonize_datasets():
    combined_dfs = []
    
    for ds_name in DATASETS:
        csv_file = PROCESSED_DIR / f"{ds_name}_features.csv"
        if not csv_file.exists():
            print(f"Warning: {csv_file.name} not found. Skipping.")
            continue
            
        df = pd.read_csv(csv_file)
        
        # Ground truth: 1 is Normal (0), -1 is Abnormal (1)
        y = np.where(df["label"] == -1, 1, 0)
        
        # Sanity check: Ensure Abnormal has higher mean centroid
        cent_norm = df.loc[y == 0, "spectral_centroid_mean"].mean()
        cent_abnorm = df.loc[y == 1, "spectral_centroid_mean"].mean()
        if cent_norm > cent_abnorm:
            y = 1 - y
            
        df["target"] = y
        df["source_center"] = ds_name
        combined_dfs.append(df)
        print(f"Loaded {ds_name:<12}: {len(df)} records (Normal: {np.sum(y == 0)}, Abnormal: {np.sum(y == 1)})")
        
    full_df = pd.concat(combined_dfs, ignore_index=True)
    
    # Exclude identifiers and metadata
    drop_cols = ["record_id", "label", "target", "source_center"]
    feature_cols = [c for c in full_df.columns if c not in drop_cols]
    
    X = full_df[feature_cols].copy()
    y = full_df["target"].values
    groups = full_df["source_center"].values
    
    return X, y, groups, feature_cols

def run_multicenter_experiment():
    X, y, groups, feature_cols = load_and_harmonize_datasets()
    print(f"\nPooled Dataset: {X.shape[0]} total samples across {len(np.unique(groups))} recording hardware setups.")
    
    # 1. Leave-One-Group-Out Validation (Evaluating cross-hardware invariance)
    logo = LeaveOneGroupOut()
    oof_probs = np.zeros(len(y))
    
    print("\n" + "=" * 65)
    print("  LEAVE-ONE-GROUP-OUT (OUT-OF-HARDWARE) EVALUATION")
    print("=" * 65)
    
    for train_idx, test_idx in logo.split(X, y, groups=groups):
        held_out_center = groups[test_idx][0]
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_test, y_test = X.iloc[test_idx], y[test_idx]
        
        clf = RandomForestClassifier(
            n_estimators=250, 
            max_depth=6, 
            class_weight="balanced", 
            random_state=42, 
            n_jobs=-1
        )
        clf.fit(X_train, y_train)
        
        probs = clf.predict_proba(X_test)[:, 1]
        oof_probs[test_idx] = probs
        
        fold_auc = roc_auc_score(y_test, probs)
        print(f"Held-out Hardware Setup: {held_out_center:<12} | Test ROC-AUC: {fold_auc:.4f}")
        
    # Aggregate Out-of-Hardware Metrics
    overall_auc = roc_auc_score(y, oof_probs)
    tau_opt, sens_opt, spec_opt, _, _, _ = optimize_clinical_threshold(y, oof_probs, min_specificity=0.80)
    
    print("-" * 65)
    print(f"Overall Multi-Hardware ROC-AUC: {overall_auc:.4f}")
    print(f"Calibrated Multi-Center Cutoff: {tau_opt:.4f}")
    print(f"Cross-Hardware Sensitivity:     {sens_opt * 100:.2f}% (Spec >= {spec_opt * 100:.2f}%)")
    print("=" * 65)
    
    # 2. Train Final Production Model on all Combined Data
    print("\nTraining final multi-center model for ONNX export...")
    rf_final = RandomForestClassifier(
        n_estimators=300, 
        max_depth=7, 
        class_weight="balanced", 
        random_state=42, 
        n_jobs=-1
    )
    rf_final.fit(X, y)
    
    # 3. Export to ONNX
    initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]
    onnx_model = convert_sklearn(
        rf_final, 
        initial_types=initial_type, 
        target_opset=12,
        options={id(rf_final): {'zipmap': False}}
    )
    
    with open(ONNX_PATH, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"Updated ONNX model exported to: {ONNX_PATH}")
    
    # 4. Save Production Config
    config = {
        "model_file": ONNX_PATH.name,
        "n_features": X.shape[1],
        "feature_names": feature_cols,
        "optimal_threshold": float(tau_opt),
        "target_sampling_rate": 4000,
        "bandpass_hz": [50.0, 1200.0],
        "trained_centers": list(np.unique(groups)),
        "class_labels": {
            "0": "Normal_Laminar",
            "1": "Abnormal_Turbulent"
        }
    }
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Updated configuration written to: {CONFIG_PATH}")

if __name__ == "__main__":
    run_multicenter_experiment()
