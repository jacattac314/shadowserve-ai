import random
from shadowserve.serving.model_client import ModelClient
from shadowserve.models.schemas import InferenceRequest, InferenceResponse, ModelPrediction
from shadowserve.config import settings


class CanaryRouter:
    """
    Routes a configurable fraction of traffic to the challenger model.
    The remaining traffic goes to production. Both paths are synchronous
    (client waits for the routed model's response).
    """

    def __init__(self, production: ModelClient, challenger: ModelClient):
        self.production = production
        self.challenger = challenger
        self._weight = settings.canary_weight

    @property
    def weight(self) -> float:
        return self._weight

    def set_weight(self, w: float) -> None:
        assert 0.0 <= w <= 1.0
        self._weight = w

    async def route(self, request: InferenceRequest) -> tuple[InferenceResponse, str]:
        """Return (response, target) where target is 'production' or 'canary'."""
        use_challenger = random.random() < self._weight
        client = self.challenger if use_challenger else self.production
        target = "canary" if use_challenger else "production"

        pred: ModelPrediction = await client.predict(request.features)
        return (
            InferenceResponse(
                transaction_id=request.transaction_id,
                prediction=pred.prediction,
                probability=pred.probability,
                model_version=pred.model_version,
                latency_ms=pred.latency_ms,
                routed_to=target,
            ),
            target,
        )
