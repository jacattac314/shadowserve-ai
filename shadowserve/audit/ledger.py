"""
Immutable audit ledger — every inference is written once and never updated.

Uses SQLAlchemy async with PostgreSQL. For demo/dev without Postgres,
falls back to an in-memory store with a warning.
"""
import logging
import json
from datetime import datetime
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, mapped_column, Mapped
from sqlalchemy import String, Float, Boolean, DateTime, Text, select
from shadowserve.config import settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class AuditRecord(Base):
    __tablename__ = "audit_ledger"

    transaction_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    model_version: Mapped[str] = mapped_column(String(32), nullable=False)
    artifact_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    input_features: Mapped[str] = mapped_column(Text, nullable=False)  # JSON
    prediction: Mapped[str] = mapped_column(Text, nullable=False)       # JSON
    probability: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    routed_to: Mapped[str] = mapped_column(String(32), nullable=False)
    env_metadata: Mapped[str] = mapped_column(Text, nullable=False)     # JSON


# In-memory fallback
_memory_store: dict[str, dict] = {}
_engine = None
_session_factory = None


async def init_db() -> None:
    global _engine, _session_factory
    try:
        _engine = create_async_engine(settings.database_url, echo=False, pool_pre_ping=True)
        _session_factory = async_sessionmaker(_engine, expire_on_commit=False)
        async with _engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Audit ledger connected to PostgreSQL")
    except Exception as exc:
        logger.warning("PostgreSQL unavailable (%s) — using in-memory ledger", exc)
        _engine = None


async def write_entry(
    transaction_id: str,
    timestamp: datetime,
    model_version: str,
    artifact_hash: str,
    input_features: dict,
    prediction,
    probability: Optional[float],
    latency_ms: float,
    routed_to: str,
    env_metadata: dict,
) -> None:
    record = {
        "transaction_id": transaction_id,
        "timestamp": timestamp.isoformat(),
        "model_version": model_version,
        "artifact_hash": artifact_hash,
        "input_features": input_features,
        "prediction": prediction,
        "probability": probability,
        "latency_ms": latency_ms,
        "routed_to": routed_to,
        "env_metadata": env_metadata,
    }

    if _engine is None:
        _memory_store[transaction_id] = record
        return

    async with _session_factory() as session:
        row = AuditRecord(
            transaction_id=transaction_id,
            timestamp=timestamp,
            model_version=model_version,
            artifact_hash=artifact_hash,
            input_features=json.dumps(input_features),
            prediction=json.dumps(prediction),
            probability=probability,
            latency_ms=latency_ms,
            routed_to=routed_to,
            env_metadata=json.dumps(env_metadata),
        )
        session.add(row)
        await session.commit()


async def fetch_entry(transaction_id: str) -> Optional[dict]:
    if _engine is None:
        return _memory_store.get(transaction_id)

    async with _session_factory() as session:
        result = await session.execute(
            select(AuditRecord).where(AuditRecord.transaction_id == transaction_id)
        )
        row = result.scalar_one_or_none()
        if row is None:
            return None
        return {
            "transaction_id": row.transaction_id,
            "timestamp": row.timestamp.isoformat(),
            "model_version": row.model_version,
            "artifact_hash": row.artifact_hash,
            "input_features": json.loads(row.input_features),
            "prediction": json.loads(row.prediction),
            "probability": row.probability,
            "latency_ms": row.latency_ms,
            "routed_to": row.routed_to,
            "env_metadata": json.loads(row.env_metadata),
        }


async def list_recent(limit: int = 50) -> list[dict]:
    if _engine is None:
        records = list(_memory_store.values())
        return sorted(records, key=lambda r: r["timestamp"], reverse=True)[:limit]

    async with _session_factory() as session:
        result = await session.execute(
            select(AuditRecord)
            .order_by(AuditRecord.timestamp.desc())
            .limit(limit)
        )
        rows = result.scalars().all()
        return [
            {
                "transaction_id": r.transaction_id,
                "timestamp": r.timestamp.isoformat(),
                "model_version": r.model_version,
                "artifact_hash": r.artifact_hash,
                "probability": r.probability,
                "latency_ms": r.latency_ms,
                "routed_to": r.routed_to,
            }
            for r in rows
        ]
