import hashlib
import time
import httpx
from typing import Any
from shadowserve.models.schemas import ModelPrediction


class ModelClient:
    """Async HTTP client for a model inference endpoint."""

    def __init__(self, base_url: str, version: str, timeout: float = 5.0):
        self.base_url = base_url
        self.version = version
        self.timeout = timeout
        self._artifact_hash = hashlib.sha256(f"{base_url}:{version}".encode()).hexdigest()[:16]

    async def predict(self, features: dict[str, Any]) -> ModelPrediction:
        start = time.perf_counter()
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(f"{self.base_url}/predict", json={"features": features})
            resp.raise_for_status()
            data = resp.json()
        latency_ms = (time.perf_counter() - start) * 1000
        return ModelPrediction(
            prediction=data["prediction"],
            probability=data.get("probability"),
            model_version=self.version,
            latency_ms=latency_ms,
            artifact_hash=self._artifact_hash,
        )

    @property
    def artifact_hash(self) -> str:
        return self._artifact_hash
