from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class FraudRequest(BaseModel):
    features: Dict[str, float] = Field(
        ..., description="Numeric feature dict used by fraud model"
    )


class FraudResponse(BaseModel):
    score: float
    threshold: float = 0.5
    label: int


class OpsRequest(BaseModel):
    features: Dict[str, float]


class OpsForecastResponse(BaseModel):
    prediction: float


class OpsAnomalyResponse(BaseModel):
    is_anomaly: bool
    score: float
