# ml/anomaly_rationale_fast.py
from __future__ import annotations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

DATA = Path("data/features/final_feature_mart.parquet")
IF_PATH = Path("models/ops_anomaly.joblib")
OUT = Path("models/explain/fast/anomaly_top_features.csv")

TOPK = 5
CAP = 5000  # cap how many anomalies we write (for file size)


def run():
    if not IF_PATH.exists():
        raise FileNotFoundError(
            "models/ops_anomaly.joblib not found. Run: python -m ml.train_ops_forecast"
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DATA)
    target = "utilization_index" if "utilization_index" in df.columns else "pay_ratio"
    drop_cols = {"npi", "dom_state", target}
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    X = X.select_dtypes(include=[np.number]).fillna(0.0)

    iso = joblib.load(IF_PATH)
    pred = iso.predict(X)  # -1 anomaly, 1 normal
    anomaly_idx = np.where(pred == -1)[0]
    print(f"Warning: anomalies detected: {len(anomaly_idx)}")

    # z-score matrix
    mu = X.mean(axis=0)
    sd = X.std(axis=0).replace(0, 1.0)
    Z = (X - mu) / sd

    rows = []
    for i in anomaly_idx[:CAP]:
        z = Z.iloc[i].abs().sort_values(ascending=False)
        rows.append(
            {
                "row_idx": int(i),
                "top_features": list(z.index[:TOPK]),
                "top_abs_z": [float(v) for v in z.values[:TOPK]],
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT, index=False)
    print(f"Anomaly rationale saved to {OUT}")


if __name__ == "__main__":
    run()
