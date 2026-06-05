"""Tests for the in-memory audit ledger fallback."""
import asyncio
from datetime import datetime
import pytest
from shadowserve.audit import ledger


@pytest.fixture(autouse=True)
def reset_ledger():
    ledger._memory_store.clear()
    ledger._engine = None
    yield
    ledger._memory_store.clear()


class TestAuditLedger:
    def test_write_and_fetch(self):
        async def run():
            await ledger.write_entry(
                transaction_id="tx-001",
                timestamp=datetime.utcnow(),
                model_version="v1.2.0",
                artifact_hash="abc123",
                input_features={"credit_score": 720},
                prediction=1,
                probability=0.82,
                latency_ms=9.5,
                routed_to="production",
                env_metadata={"host": "test"},
            )
            entry = await ledger.fetch_entry("tx-001")
            return entry

        entry = asyncio.get_event_loop().run_until_complete(run())
        assert entry is not None
        assert entry["transaction_id"] == "tx-001"
        assert entry["model_version"] == "v1.2.0"
        assert entry["probability"] == 0.82
        assert entry["routed_to"] == "production"

    def test_fetch_missing_returns_none(self):
        entry = asyncio.get_event_loop().run_until_complete(ledger.fetch_entry("nonexistent"))
        assert entry is None

    def test_list_recent(self):
        async def run():
            for i in range(5):
                await ledger.write_entry(
                    transaction_id=f"tx-{i:03d}",
                    timestamp=datetime.utcnow(),
                    model_version="v1.2.0",
                    artifact_hash="abc",
                    input_features={},
                    prediction=0,
                    probability=0.3,
                    latency_ms=8.0,
                    routed_to="production",
                    env_metadata={},
                )
            return await ledger.list_recent(limit=10)

        entries = asyncio.get_event_loop().run_until_complete(run())
        assert len(entries) == 5

    def test_immutability_no_update_method(self):
        """Ledger exposes no update/delete — only write and read."""
        assert not hasattr(ledger, "update_entry")
        assert not hasattr(ledger, "delete_entry")
