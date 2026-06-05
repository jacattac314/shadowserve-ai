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


def _logistic_score(features: dict, bias: float = 0.0) -> float:
    """Deterministic-ish logistic score from feature dict."""
    raw = sum(float(v) for v in features.values() if isinstance(v, (int, float)))
    z = (raw * 0.15) + bias + random.gauss(0, 0.05)
    return float(1 / (1 + np.exp(-z)))


def _gbm_score(features: dict, bias: float = 0.0) -> float:
    """Slightly more aggressive scorer mimicking gradient boosting."""
    raw = sum(float(v) for v in features.values() if isinstance(v, (int, float)))
    z = (raw * 0.18) + bias + random.gauss(0, 0.06)
    return float(1 / (1 + np.exp(-z)))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "production"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else (8001 if mode == "production" else 8002)
    app = make_production_app() if mode == "production" else make_challenger_app()
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
