# ml/evaluate_models.py
from __future__ import annotations
from pathlib import Path
import json, joblib
import pandas as pd
from sklearn.inspection import permutation_importance

DATA = Path("data/features/final_feature_mart.parquet")
MODEL_FRAUD = Path("models/fraud_baseline.joblib")
MODEL_OPS = Path("models/ops_forecast.joblib")
OUT = Path("models/metrics_summary.json")


def run():
    df = pd.read_parquet(DATA)
    df = df.select_dtypes(include=["number"]).fillna(0)

    if MODEL_FRAUD.exists():
        model = joblib.load(MODEL_FRAUD)
        perm = permutation_importance(
            model, df, model.predict(df), n_repeats=3, random_state=42
        )
        importances = (
            pd.DataFrame(
                {"feature": df.columns, "importance_mean": perm.importances_mean}
            )
            .sort_values("importance_mean", ascending=False)
            .head(15)
        )
        importances.to_csv("models/fraud_feature_importances.csv", index=False)
        print("fraud model top features saved to models/fraud_feature_importances.csv")

    if MODEL_OPS.exists():
        joblib.load(MODEL_OPS)
        print("ops_forecast model loaded — ready for dashboard integration")

    summary = {
        "fraud_model": str(MODEL_FRAUD),
        "ops_model": str(MODEL_OPS),
        "feature_importances_csv": "models/fraud_feature_importances.csv",
    }
    with open(OUT, "w") as f:
        json.dump(summary, f, indent=2)
    print("Success: evaluation summary written")


if __name__ == "__main__":
    run()
