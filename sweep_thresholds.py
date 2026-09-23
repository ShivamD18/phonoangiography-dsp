import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from tqdm import tqdm
from src.inference.predict import PhonoangiographyPredictor

ref_file = ROOT_DIR / "data" / "raw" / "training-a" / "REFERENCE.csv"
ref_df = pd.read_csv(ref_file, header=None, names=["record_id", "label"])
predictor = PhonoangiographyPredictor()

records = []
for _, row in tqdm(ref_df.iterrows(), total=len(ref_df), desc="Gathering Probabilities"):
    rec_id = str(row["record_id"]).strip()
    wav_path = ROOT_DIR / "data" / "raw" / "training-a" / f"{rec_id}.wav"
    if not wav_path.exists():
        continue
    res = predictor.predict(str(wav_path))
    records.append({
        "record_id": rec_id,
        "y_true": 1 if row["label"] == 1 else 0,
        "prob": res["prob_abnormal"]
    })

df = pd.DataFrame(records)
y_true = df["y_true"].values
probs = df["prob"].values

print("\n" + "=" * 70)
print(f"{'Threshold (tau)':<16} | {'Sensitivity':<14} | {'Specificity':<14} | {'Accuracy':<10}")
print("=" * 70)

for tau in np.arange(0.45, 0.75, 0.02):
    y_pred = (probs >= tau).astype(int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    acc = (tp + tn) / len(y_true)
    
    mark = " <-- Target Spec >= 80%" if spec >= 0.80 and spec - 0.02 < 0.80 else ""
    print(f"{tau:<16.2f} | {sens * 100:6.2f}%       | {spec * 100:6.2f}%       | {acc * 100:6.2f}% {mark}")
print("=" * 70)
