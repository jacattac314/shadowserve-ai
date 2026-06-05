import asyncio
import logging
from typing import Callable, Awaitable, Any
from shadowserve.serving.model_client import ModelClient
from shadowserve.models.schemas import InferenceRequest, ModelPrediction

logger = logging.getLogger(__name__)


class ShadowRouter:
    """
    Forks every request to the challenger model asynchronously.
    The caller receives the production response immediately; the shadow
    call runs in the background and its result is passed to `on_shadow_result`.
    """

    def __init__(
        self,
        challenger: ModelClient,
        on_shadow_result: Callable[[str, ModelPrediction], Awaitable[None]],
    ):
        self.challenger = challenger
        self._on_result = on_shadow_result

    async def fork(self, request: InferenceRequest) -> None:
        """Fire-and-forget shadow inference. Schedules a background task."""
        asyncio.create_task(self._shadow_call(request))

    async def _shadow_call(self, request: InferenceRequest) -> None:
        try:
            pred = await self.challenger.predict(request.features)
            await self._on_result(request.transaction_id, pred)
        except Exception as exc:
            logger.warning("Shadow call failed for %s: %s", request.transaction_id, exc)
