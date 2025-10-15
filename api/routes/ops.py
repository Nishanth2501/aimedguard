from __future__ import annotations
from fastapi import APIRouter, HTTPException
import numpy as np
import pandas as pd
from api.schemas import OpsRequest, OpsForecastResponse, OpsAnomalyResponse
from api.model_registry import load_ops_forecast, load_ops_anomaly

router = APIRouter(prefix="/predict", tags=["ops"])


@router.post("/ops_forecast", response_model=OpsForecastResponse)
def ops_forecast(req: OpsRequest):
    model = load_ops_forecast()

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

    pred = float(model.predict(X)[0])
    return OpsForecastResponse(prediction=pred)


@router.post("/ops_anomaly", response_model=OpsAnomalyResponse)
def ops_anomaly(req: OpsRequest):
    iso = load_ops_anomaly()

    # Get expected features from model
    if hasattr(iso, "feature_names_in_"):
        expected_features = list(iso.feature_names_in_)
    else:
        raise HTTPException(
            status_code=500, detail="Model does not expose feature names"
        )

    # Create DataFrame with all expected features, fill missing with 0
    feature_dict = {feat: req.features.get(feat, 0.0) for feat in expected_features}
    X = pd.DataFrame([feature_dict])

    score = float(iso.decision_function(X)[0])  # higher = more normal
    pred = int(iso.predict(X)[0])  # -1 anomaly, 1 normal
    return OpsAnomalyResponse(is_anomaly=(pred == -1), score=score)
