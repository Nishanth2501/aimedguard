from __future__ import annotations
from pathlib import Path
import joblib
from functools import lru_cache

MODELS_DIR = Path("models")


@lru_cache(maxsize=4)
def load_fraud_model():
    path = MODELS_DIR / "fraud_baseline.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Missing model: {path}")
    return joblib.load(path)


@lru_cache(maxsize=4)
def load_ops_forecast():
    path = MODELS_DIR / "ops_forecast.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Missing model: {path}")
    return joblib.load(path)


@lru_cache(maxsize=4)
def load_ops_anomaly():
    path = MODELS_DIR / "ops_anomaly.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Missing model: {path}")
    return joblib.load(path)
