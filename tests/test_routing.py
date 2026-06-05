"""Tests for canary and shadow routing logic."""
import asyncio
import pytest
from unittest.mock import AsyncMock, patch
from shadowserve.models.schemas import InferenceRequest, ModelPrediction
from shadowserve.routing.canary import CanaryRouter
from shadowserve.routing.shadow import ShadowRouter


def _mock_client(prediction=1, probability=0.75, version="v1.2.0", latency=8.0):
    client = AsyncMock()
    client.predict.return_value = ModelPrediction(
        prediction=prediction,
        probability=probability,
        model_version=version,
        latency_ms=latency,
        artifact_hash="abc123",
    )
    client.artifact_hash = "abc123"
    return client


@pytest.fixture
def request_fixture():
    return InferenceRequest(features={"credit_score": 720, "debt_to_income": 0.30})


class TestCanaryRouter:
    def test_routes_to_production_by_default(self, request_fixture):
        prod = _mock_client(version="v1.2.0")
        chal = _mock_client(version="v1.3.0")
        router = CanaryRouter(prod, chal)
        router.set_weight(0.0)  # 0% to challenger

        result, target = asyncio.get_event_loop().run_until_complete(
            router.route(request_fixture)
        )
        assert target == "production"
        assert result.model_version == "v1.2.0"
        prod.predict.assert_called_once()
        chal.predict.assert_not_called()

    def test_routes_to_canary_at_100_percent(self, request_fixture):
        prod = _mock_client(version="v1.2.0")
        chal = _mock_client(version="v1.3.0")
        router = CanaryRouter(prod, chal)
        router.set_weight(1.0)  # 100% to challenger

        result, target = asyncio.get_event_loop().run_until_complete(
            router.route(request_fixture)
        )
        assert target == "canary"
        assert result.model_version == "v1.3.0"

    def test_weight_update(self):
        prod = _mock_client()
        chal = _mock_client()
        router = CanaryRouter(prod, chal)
        router.set_weight(0.25)
        assert router.weight == 0.25

    def test_weight_bounds(self):
        prod = _mock_client()
        chal = _mock_client()
        router = CanaryRouter(prod, chal)
        with pytest.raises(AssertionError):
            router.set_weight(1.5)

    def test_response_carries_transaction_id(self, request_fixture):
        prod = _mock_client()
        chal = _mock_client()
        router = CanaryRouter(prod, chal)
        router.set_weight(0.0)
        result, _ = asyncio.get_event_loop().run_until_complete(router.route(request_fixture))
        assert result.transaction_id == request_fixture.transaction_id


class TestShadowRouter:
    def test_shadow_calls_callback(self, request_fixture):
        challenger = _mock_client(version="v1.3.0", probability=0.60)
        received = []

        async def callback(tx_id, pred):
            received.append((tx_id, pred))

        router = ShadowRouter(challenger, callback)

        async def run():
            await router.fork(request_fixture)
            await asyncio.sleep(0.05)  # let background task complete

        asyncio.get_event_loop().run_until_complete(run())
        assert len(received) == 1
        assert received[0][0] == request_fixture.transaction_id
        assert received[0][1].model_version == "v1.3.0"

    def test_shadow_handles_challenger_error(self, request_fixture):
        challenger = AsyncMock()
        challenger.predict.side_effect = Exception("model unavailable")
        router = ShadowRouter(challenger, AsyncMock())

        async def run():
            await router.fork(request_fixture)
            await asyncio.sleep(0.05)

        # should not raise
        asyncio.get_event_loop().run_until_complete(run())
