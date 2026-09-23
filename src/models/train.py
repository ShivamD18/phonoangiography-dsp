import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, 
    recall_score, 
    confusion_matrix, 
    f1_score, 
    precision_score
)
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = ROOT_DIR / "data" / "processed" / "training_a_features.csv"

def load_data(csv_path: Path):
    df = pd.read_csv(csv_path)
    
    drop_cols = ["record_id", "label"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    X = df[feature_cols].copy()
    
    # PhysioNet convention: 1 is Normal, -1 is Abnormal.
    # Standard ML Binary: 0 = Normal/Laminar, 1 = Abnormal/Turbulent (Positive Condition)
    y = np.where(df["label"] == -1, 1, 0)
    
    return X, y, feature_cols

def evaluate_classifier(clf, name: str, X: pd.DataFrame, y: np.ndarray, cv: StratifiedKFold):
    oof_probs = np.zeros(len(y))
    oof_preds = np.zeros(len(y))
    feature_importances = np.zeros(X.shape[1])
    
    for train_idx, val_idx in cv.split(X, y):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]
        
        clf.fit(X_train, y_train)
        
        probs = clf.predict_proba(X_val)[:, 1]
        preds = clf.predict(X_val)
        
        oof_probs[val_idx] = probs
        oof_preds[val_idx] = preds
        
        if hasattr(clf, "feature_importances_"):
            feature_importances += clf.feature_importances_ / cv.n_splits

    auc = roc_auc_score(y, oof_probs)
    f1 = f1_score(y, oof_preds)
    sens = recall_score(y, oof_preds) # True Positive Rate
    
    tn, fp, fn, tp = confusion_matrix(y, oof_preds).ravel()
    spec = tn / (tn + fp)
    
    print("=" * 60)
    print(f"Model: {name}")
    print("=" * 60)
    print(f"ROC-AUC:     {auc:.4f}")
    print(f"Sensitivity: {sens:.4f}  (Ability to catch abnormal bruits)")
    print(f"Specificity: {spec:.4f}  (Ability to identify normal flow)")
    print(f"F1-Score:    {f1:.4f}")
    print("\nConfusion Matrix:")
    print(f"  [TN (Normal):   {tn:3d}  |  FP (False Alarm): {fp:3d}]")
    print(f"  [FN (Missed):   {fn:3d}  |  TP (Detected):    {tp:3d}]")
    print("-" * 60)
    
    return feature_importances, oof_probs

def run_experiment():
    if not DATA_PATH.exists():
        print(f"Error: Could not find {DATA_PATH}. Run feature extraction first.")
        return

    X, y, feature_cols = load_data(DATA_PATH)
    print(f"Dataset Loaded: {X.shape[0]} samples, {X.shape[1]} features.")
    print(f"Class distribution: {np.sum(y == 0)} Normal (0), {np.sum(y == 1)} Abnormal (1)")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rf = RandomForestClassifier(
        n_estimators=200, 
        max_depth=6, 
        class_weight="balanced", 
        random_state=42, 
        n_jobs=-1
    )
    rf_imp, _ = evaluate_classifier(rf, "Random Forest (Balanced)", X, y, cv)

    # Top 10 Features
    feat_series = pd.Series(rf_imp, index=feature_cols).sort_values(ascending=False)
    print("\nTop 10 Most Predictive Features:")
    for rank, (feat, score) in enumerate(feat_series.head(10).items(), 1):
        print(f"  {rank:2d}. {feat:<28} : {score:.4f}")

if __name__ == "__main__":
    run_experiment()
