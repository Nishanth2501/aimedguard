from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain.tools import tool
from api.model_registry import load_fraud_model


@tool("fraud_predict")
def fraud_predict(
    pay_ratio: float = 0.0,
    svc_per_bene: float = 0.0,
    total_beneficiaries: float = 0.0,
    mean_charge: float = 0.0,
    mean_payment: float = 0.0,
    total_services: float = 0.0,
) -> str:
    """
    Predict fraud risk for a healthcare provider based on their metrics.

    CRITICAL: This tool requires individual parameters. NEVER use features_json.
    Call format: fraud_predict(pay_ratio=1.25, svc_per_bene=18, total_beneficiaries=42)

    Parameters:
        pay_ratio (float): Ratio of payment to charges, typically 0.5-1.5
        svc_per_bene (float): Average services per beneficiary, typically 5-30
        total_beneficiaries (float): Total number of beneficiaries served
        mean_charge (float): Average charge amount in dollars
        mean_payment (float): Average payment amount in dollars
        total_services (float): Total number of services provided

    Returns:
        Fraud risk assessment with score and risk level
    """
    model = load_fraud_model()

    # Build features dict from parameters
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

    score = (
        float(model.predict_proba(X)[:, 1][0])
        if hasattr(model, "predict_proba")
        else float(model.predict(X)[0])
    )
    label = int(score >= 0.5)
    risk_level = "HIGH RISK" if label == 1 else "LOW RISK"
    return f"Fraud Risk: {risk_level} (score={score:.3f}, threshold=0.5)"
