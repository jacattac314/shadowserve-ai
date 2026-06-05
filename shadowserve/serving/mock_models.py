"""
Mock FastAPI servers simulating production (v1.2.0) and challenger (v1.3.0) models.

Production: conservative logistic-regression-style credit scorer.
Challenger: gradient-boosted challenger with slight distribution shift.
Run with:
  python -m shadowserve.serving.mock_models production --port 8001
  python -m shadowserve.serving.mock_models challenger --port 8002
"""
import asyncio
import random
import time
import sys
import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any


class PredictRequest(BaseModel):
    features: dict[str, Any]


def make_production_app() -> FastAPI:
    app = FastAPI(title="Production Model v1.2.0")

    @app.post("/predict")
    async def predict(req: PredictRequest):
        await asyncio.sleep(random.gauss(0.008, 0.002))  # ~8ms P50
        score = _logistic_score(req.features, bias=0.0)
        return {
            "prediction": int(score > 0.5),
            "probability": round(score, 4),
            "model_version": "v1.2.0",
        }

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": "v1.2.0"}

    return app


def make_challenger_app() -> FastAPI:
    app = FastAPI(title="Challenger Model v1.3.0")

    @app.post("/predict")
    async def predict(req: PredictRequest):
        await asyncio.sleep(random.gauss(0.012, 0.003))  # ~12ms P50
        score = _gbm_score(req.features, bias=0.03)
        return {
            "prediction": int(score > 0.5),
            "probability": round(score, 4),
            "model_version": "v1.3.0",
        }

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": "v1.3.0"}

    return app


_FEATURE_SCALES = {
    "credit_score":     (300.0, 850.0),
    "debt_to_income":   (0.0, 1.0),
    "annual_income":    (10_000.0, 600_000.0),
    "loan_amount":      (1_000.0, 300_000.0),
    "employment_years": (0.0, 40.0),
    "num_accounts":     (0.0, 30.0),
    "num_derogatory":   (0.0, 10.0),
}


def _normalize(features: dict) -> np.ndarray:
    """Min-max normalize known features to [0, 1]; pass unknowns through clipped."""
    vals = []
    for k, v in features.items():
        if not isinstance(v, (int, float)):
            continue
        if k in _FEATURE_SCALES:
            lo, hi = _FEATURE_SCALES[k]
            vals.append((float(v) - lo) / (hi - lo))
        else:
            vals.append(float(v))
    return np.clip(vals, 0.0, 1.0) if vals else np.array([0.5])


def _logistic_score(features: dict, bias: float = 0.0) -> float:
    """Logistic score on normalized features — credit-positive weighting."""
    normed = _normalize(features)
    # weights: credit_score(+), dti(-), income(+), loan(-), tenure(+), accounts(~), derogs(-)
    weights = np.array([2.5, -1.5, 1.0, -0.8, 0.6, 0.2, -1.2])[:len(normed)]
    z = float(np.dot(normed, weights)) + bias + random.gauss(0, 0.12) - 0.5
    return float(1 / (1 + np.exp(-z)))


def _gbm_score(features: dict, bias: float = 0.0) -> float:
    """Challenger scorer — slightly more aggressive on DTI and derogatory marks."""
    normed = _normalize(features)
    weights = np.array([2.2, -2.0, 0.9, -1.0, 0.7, 0.15, -1.6])[:len(normed)]
    z = float(np.dot(normed, weights)) + bias + random.gauss(0, 0.15) - 0.5
    return float(1 / (1 + np.exp(-z)))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "production"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else (8001 if mode == "production" else 8002)
    app = make_production_app() if mode == "production" else make_challenger_app()
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
