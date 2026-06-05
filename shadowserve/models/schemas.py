from pydantic import BaseModel, Field
from typing import Any, Optional
from datetime import datetime
import uuid


class InferenceRequest(BaseModel):
    transaction_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model_id: Optional[str] = None
    features: dict[str, Any]
    metadata: dict[str, Any] = Field(default_factory=dict)


class InferenceResponse(BaseModel):
    transaction_id: str
    prediction: Any
    probability: Optional[float] = None
    model_version: str
    latency_ms: float
    routed_to: str  # "production" | "canary" | "shadow"


class ModelPrediction(BaseModel):
    prediction: Any
    probability: Optional[float] = None
    model_version: str
    latency_ms: float
    artifact_hash: str


class DriftReport(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    ks_statistic: float
    ks_pvalue: float
    psi_score: float
    kl_divergence: float
    wasserstein_distance: float
    drift_detected: bool
    rollback_triggered: bool


class AuditEntry(BaseModel):
    transaction_id: str
    timestamp: datetime
    model_version: str
    artifact_hash: str
    input_features: dict[str, Any]
    prediction: Any
    probability: Optional[float]
    latency_ms: float
    routed_to: str
    env_metadata: dict[str, Any]


class CanaryConfig(BaseModel):
    canary_weight: float = Field(ge=0.0, le=1.0)
    shadow_enabled: bool


class HealthResponse(BaseModel):
    status: str
    version: str
    production_model: str
    challenger_model: str
    canary_weight: float
    shadow_enabled: bool
