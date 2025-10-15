from __future__ import annotations
from fastapi import APIRouter, HTTPException
import numpy as np
import pandas as pd
from api.schemas import FraudRequest, FraudResponse
from api.model_registry import load_fraud_model

router = APIRouter(prefix="/predict", tags=["fraud"])


@router.post("/fraud", response_model=FraudResponse)
def predict_fraud(req: FraudRequest):
    model = load_fraud_model()

    # Get expected features from model
    if hasattr(model, "feature_names_in_"):
        expected_features = list(model.feature_names_in_)
    else:
        raise HTTPException(
            status_code=500, detail="Model does not expose feature names"
        )

    # Create DataFrame with all expected features, fill missing with 0
    feature_dict = {feat: req.features.get(feat, 0.0) for feat in expected_features}
    X = pd.DataFrame([feature_dict])

    if hasattr(model, "predict_proba"):
        score = float(model.predict_proba(X)[:, 1][0])
    else:
        score = float(model.predict(X)[0])
    label = int(score >= 0.5)
    return FraudResponse(score=score, label=label)
