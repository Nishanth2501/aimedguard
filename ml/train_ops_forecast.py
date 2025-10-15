# ml/train_ops_forecast.py
from __future__ import annotations
import json
import joblib
import sys
import time
from pathlib import Path

# Add project root to path to find utils module
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.model_selection import train_test_split

from utils.metrics import regression_metrics

DATA = Path("data/features/final_feature_mart.parquet")
OUT_DIR = Path("models")
OUT_DIR.mkdir(exist_ok=True)
LOG = Path(f"logs/training_ops_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
LOG.parent.mkdir(parents=True, exist_ok=True)  # ensure logs/ exists


def run():
    print("Loading feature mart...")
    df = pd.read_parquet(DATA)
    print(f"   Loaded {len(df):,} rows")

    # Target: prefer utilization_index if present; else fall back to pay_ratio
    target = "utilization_index" if "utilization_index" in df.columns else "pay_ratio"
    drop_cols = ["npi", "dom_state", target]

    # Prepare X, y
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    y = df[target].astype(float).fillna(0.0)
    X = X.select_dtypes(include=[np.number]).fillna(0.0)

    if X.shape[1] == 0:
        raise ValueError(
            "No numeric features available for training after filtering. Check your feature mart."
        )

    print(f"Success: Features: {X.shape[1]} columns | Target: {target}")

    # Train/test split (regression)
    print("\nTraining regression model...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    metrics = regression_metrics(y_test, preds)

    # ---- Anomaly detection (IsolationForest) ----
    print("Training anomaly detection with sampling...")
    sample_size = min(100_000, len(X)) if len(X) > 0 else 0
    if sample_size < 10:
        print(
            "   Warning: Not enough rows for anomaly sampling; skipping IsolationForest."
        )
        iso = None
        n_anoms = 0
    else:
        X_sample = X.sample(n=sample_size, random_state=42)
        iso = IsolationForest(contamination=0.02, random_state=42, n_jobs=-1)
        iso.fit(X_sample)

        print("   Predicting anomalies on full dataset...")
        iso_pred = iso.predict(X)  # -1 = anomaly, 1 = normal
        iso_score = iso.decision_function(X)  # higher = more normal
        df["anomaly_flag"] = (iso_pred == -1).astype(int)
        df["anomaly_score"] = iso_score
        n_anoms = int(df["anomaly_flag"].sum())
        # Optional: keep a quick artifact for inspection
        df.loc[df["anomaly_flag"] == 1, ["npi"] if "npi" in df.columns else []].to_csv(
            OUT_DIR / "ops_anomalies_preview.csv", index=False
        )

    # ---- Persist models ----
    joblib.dump(rf, OUT_DIR / "ops_forecast.joblib")
    if iso is not None:
        joblib.dump(iso, OUT_DIR / "ops_anomaly.joblib")

    # ---- Log training summary ----
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "regression_metrics": metrics,
                    "n_features": int(X.shape[1]),
                    "target": target,
                    "n_rows": int(len(X)),
                    "n_anomalies": n_anoms,
                }
            )
            + "\n"
        )

    print(
        f"Success: ops_forecast RF trained | RMSE={metrics['rmse']:.3f} | R2={metrics['r2']:.3f}"
    )
    if iso is not None:
        print(
            f"Warning: anomalies detected: {n_anoms} (preview → models/ops_anomalies_preview.csv)"
        )


if __name__ == "__main__":
    run()
