# ml/train_fraud_baseline.py
from __future__ import annotations
import pandas as pd, numpy as np, json, joblib, time
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from utils.metrics import classification_metrics

DATA = Path("data/features/final_feature_mart.parquet")
OUT_DIR = Path("models")
OUT_DIR.mkdir(exist_ok=True)
LOG = Path(f"logs/training_fraud_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")


def run():
    df = pd.read_parquet(DATA)
    # Create fraud flag based on realistic patterns:
    # 1. Very low payment ratio (< 10th percentile) - potential overbilling
    # 2. High services per beneficiary (> 95th percentile) - unusual volume
    # 3. High total charges - outlier billing
    pay_ratio_low = df["pay_ratio"].quantile(0.10)
    svc_per_bene_high = df["svc_per_bene"].quantile(0.95)
    df["fraud_flag"] = (
        (df["pay_ratio"] < pay_ratio_low) & (df["svc_per_bene"] > svc_per_bene_high)
    ).astype(int)

    target = "fraud_flag"
    drop_cols = ["npi", "dom_state", "fraud_flag"]
    features = [c for c in df.columns if c not in drop_cols]
    X = df[features].select_dtypes(include=[np.number]).fillna(0)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "logreg": Pipeline(
            [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=500))]
        ),
        "rf": RandomForestClassifier(n_estimators=150, random_state=42),
        "xgb": XGBClassifier(
            n_estimators=250,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
        ),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )
        metrics = classification_metrics(y_test, y_pred, y_prob)
        results[name] = metrics
        joblib.dump(model, OUT_DIR / f"fraud_{name}.joblib")
        print(
            f"Success: {name} trained | F1={metrics['f1']:.3f} | AUC={metrics.get('roc_auc', 0):.3f}"
        )

    best_model = max(results, key=lambda k: results[k]["f1"])
    joblib.dump(models[best_model], OUT_DIR / "fraud_baseline.joblib")

    with open(LOG, "a") as f:
        f.write(json.dumps({"results": results, "best": best_model}) + "\n")

    print(f"Best fraud model: {best_model}")


if __name__ == "__main__":
    run()
