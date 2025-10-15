from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain.tools import tool
from api.model_registry import load_ops_forecast, load_ops_anomaly


@tool("ops_forecast")
def ops_forecast(
    pay_ratio: float = 0.0,
    svc_per_bene: float = 0.0,
    total_beneficiaries: float = 0.0,
    mean_charge: float = 0.0,
    mean_payment: float = 0.0,
    total_services: float = 0.0,
) -> str:
    """
    Forecast operational metrics for a healthcare provider.

    IMPORTANT: Pass each parameter individually. Do NOT pass features_json.
    Example: ops_forecast(total_beneficiaries=120, mean_payment=350.4, mean_charge=620.8)

    Args:
        pay_ratio: Ratio of payment to charges (e.g., 0.85)
        svc_per_bene: Services per beneficiary (e.g., 10)
        total_beneficiaries: Total beneficiaries (e.g., 100)
        mean_charge: Average charge in dollars (e.g., 5000)
        mean_payment: Average payment in dollars (e.g., 4000)
        total_services: Total services provided (e.g., 1000)

    Returns:
        Predicted operational metric value
    """
    model = load_ops_forecast()

    features_json = {
        "pay_ratio": pay_ratio,
        "svc_per_bene": svc_per_bene,
        "total_beneficiaries": total_beneficiaries,
        "mean_charge": mean_charge,
        "mean_payment": mean_payment,
        "total_services": total_services,
    }

    # Get expected features from model and fill missing with 0
    if hasattr(model, "feature_names_in_"):
        expected_features = list(model.feature_names_in_)
        feature_dict = {
            feat: features_json.get(feat, 0.0) for feat in expected_features
        }
        X = pd.DataFrame([feature_dict])
    else:
        X = pd.DataFrame([features_json]).select_dtypes(include=[np.number]).fillna(0.0)

    pred = float(model.predict(X)[0])
    return f"Operational Forecast: {pred:.4f}"


@tool("ops_anomaly")
def ops_anomaly(
    pay_ratio: float = 0.0,
    svc_per_bene: float = 0.0,
    total_beneficiaries: float = 0.0,
    mean_charge: float = 0.0,
    mean_payment: float = 0.0,
    total_services: float = 0.0,
) -> str:
    """
    Detect if a provider's metrics are anomalous (unusual patterns).

    IMPORTANT: Pass each parameter individually. Do NOT pass features_json.
    Example: ops_anomaly(svc_per_bene=22, total_beneficiaries=85, mean_payment=970.2)

    Args:
        pay_ratio: Ratio of payment to charges (e.g., 0.85)
        svc_per_bene: Services per beneficiary (e.g., 10)
        total_beneficiaries: Total beneficiaries (e.g., 100)
        mean_charge: Average charge in dollars (e.g., 5000)
        mean_payment: Average payment in dollars (e.g., 4000)
        total_services: Total services provided (e.g., 1000)

    Returns:
        Whether the provider is anomalous and anomaly score
    """
    iso = load_ops_anomaly()

    features_json = {
        "pay_ratio": pay_ratio,
        "svc_per_bene": svc_per_bene,
        "total_beneficiaries": total_beneficiaries,
        "mean_charge": mean_charge,
        "mean_payment": mean_payment,
        "total_services": total_services,
    }

    # Get expected features from model and fill missing with 0
    if hasattr(iso, "feature_names_in_"):
        expected_features = list(iso.feature_names_in_)
        feature_dict = {
            feat: features_json.get(feat, 0.0) for feat in expected_features
        }
        X = pd.DataFrame([feature_dict])
    else:
        X = pd.DataFrame([features_json]).select_dtypes(include=[np.number]).fillna(0.0)

    pred = int(iso.predict(X)[0])  # -1 anomaly, 1 normal
    score = float(iso.decision_function(X)[0])
    is_anomaly = pred == -1
    status = "ANOMALOUS" if is_anomaly else "NORMAL"
    return f"Status: {status} (anomaly_score={score:.4f}, higher score = more normal)"
