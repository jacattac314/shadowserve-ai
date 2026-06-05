"""ShadowServe AI — main FastAPI application."""
import logging
import platform
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from jose import JWTError, jwt

from shadowserve import __version__
from shadowserve.config import settings
from shadowserve.models.schemas import (
    InferenceRequest, InferenceResponse, CanaryConfig, HealthResponse,
)
from shadowserve.serving.model_client import ModelClient
from shadowserve.routing.canary import CanaryRouter
from shadowserve.routing.shadow import ShadowRouter
from shadowserve.evaluation import drift as drift_module
from shadowserve.evaluation.metrics import (
    inference_latency, request_total, shadow_total, rollback_total,
    ks_statistic, ks_pvalue, psi_score, kl_divergence,
    wasserstein_distance, drift_detected, canary_weight_gauge,
    metrics_output,
)
from shadowserve.audit import ledger
from shadowserve.dashboard.routes import router as dashboard_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

# Module-level singletons — initialised in lifespan
production_client: Optional[ModelClient] = None
challenger_client: Optional[ModelClient] = None
canary_router: Optional[CanaryRouter] = None
shadow_router: Optional[ShadowRouter] = None


async def _on_shadow_result(transaction_id: str, pred) -> None:
    """Callback invoked when the async shadow call completes."""
    shadow_total.inc()
    if pred.probability is not None:
        drift_module.record_shadow(pred.probability)

    state = drift_module.get_state()
    ks_statistic.set(state.last_ks_statistic)
    ks_pvalue.set(state.last_ks_pvalue)
    psi_score.set(state.last_psi)
    kl_divergence.set(state.last_kl)
    wasserstein_distance.set(state.last_wasserstein)
    drift_detected.set(1 if state.drift_detected else 0)
    if state.drift_detected and state.rollback_count > 0:
        rollback_total.inc()

    await ledger.write_entry(
        transaction_id=f"shadow:{transaction_id}",
        timestamp=datetime.utcnow(),
        model_version=pred.model_version,
        artifact_hash=pred.artifact_hash,
        input_features={},  # features are already logged on the production entry
        prediction=pred.prediction,
        probability=pred.probability,
        latency_ms=pred.latency_ms,
        routed_to="shadow",
        env_metadata={"host": platform.node()},
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global production_client, challenger_client, canary_router, shadow_router

    production_client = ModelClient(
        settings.production_model_url, settings.production_model_version
    )
    challenger_client = ModelClient(
        settings.challenger_model_url, settings.challenger_model_version
    )
    canary_router = CanaryRouter(production_client, challenger_client)
    shadow_router = ShadowRouter(challenger_client, _on_shadow_result)

    await ledger.init_db()
    canary_weight_gauge.set(settings.canary_weight)
    logger.info("ShadowServe %s started", __version__)
    yield
    logger.info("ShadowServe shutting down")


app = FastAPI(
    title="ShadowServe AI",
    description="Enterprise model serving proxy with shadow routing, drift detection, and compliance auditing.",
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(dashboard_router)


# ── Auth ──────────────────────────────────────────────────────────────────────

def _verify_token(request: Request) -> dict:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing token")
    token = auth.split(" ", 1)[1]
    try:
        return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        version=__version__,
        production_model=settings.production_model_version,
        challenger_model=settings.challenger_model_version,
        canary_weight=canary_router.weight if canary_router else settings.canary_weight,
        shadow_enabled=settings.shadow_enabled,
    )


@app.post("/v1/infer", response_model=InferenceResponse)
async def infer(request: InferenceRequest):
    """
    Primary inference endpoint.

    1. Routes traffic through canary (production or challenger at configured weight).
    2. If shadow is enabled, asynchronously forks the request to the challenger.
    3. Writes an immutable audit entry.
    4. Updates Prometheus metrics.
    """
    t0 = time.perf_counter()

    response, target = await canary_router.route(request)
    request_total.labels(route_target=target).inc()
    inference_latency.labels(
        model_version=response.model_version, route_target=target
    ).observe(response.latency_ms)

    if settings.shadow_enabled and target == "production":
        await shadow_router.fork(request)

    if response.probability is not None and target == "production":
        drift_module.record_production(response.probability)

    canary_weight_gauge.set(canary_router.weight)

    env_meta = {
        "host": platform.node(),
        "python": platform.python_version(),
        "shadowserve_version": __version__,
        "canary_weight": canary_router.weight,
    }

    await ledger.write_entry(
        transaction_id=request.transaction_id,
        timestamp=datetime.utcnow(),
        model_version=response.model_version,
        artifact_hash=(
            production_client.artifact_hash if target == "production"
            else challenger_client.artifact_hash
        ),
        input_features=request.features,
        prediction=response.prediction,
        probability=response.probability,
        latency_ms=response.latency_ms,
        routed_to=target,
        env_metadata=env_meta,
    )

    return response


@app.get("/v1/audit/{transaction_id}")
async def get_audit(transaction_id: str):
    entry = await ledger.fetch_entry(transaction_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Transaction not found")
    return entry


@app.get("/v1/canary", response_model=CanaryConfig)
async def get_canary():
    return CanaryConfig(
        canary_weight=canary_router.weight,
        shadow_enabled=settings.shadow_enabled,
    )


@app.post("/v1/canary/weight")
async def update_canary_weight(config: CanaryConfig):
    canary_router.set_weight(config.canary_weight)
    settings.shadow_enabled = config.shadow_enabled
    canary_weight_gauge.set(config.canary_weight)
    logger.info("Canary weight updated to %.2f, shadow=%s", config.canary_weight, config.shadow_enabled)
    return {"ok": True, "canary_weight": config.canary_weight}


@app.get("/v1/drift")
async def get_drift():
    s = drift_module.get_state()
    return {
        "ks_statistic": s.last_ks_statistic,
        "ks_pvalue": s.last_ks_pvalue,
        "psi_score": s.last_psi,
        "kl_divergence": s.last_kl,
        "wasserstein_distance": s.last_wasserstein,
        "drift_detected": s.drift_detected,
        "drift_count": s.drift_count,
        "rollback_count": s.rollback_count,
    }


@app.get("/metrics")
async def prometheus_metrics():
    data, content_type = metrics_output()
    return Response(content=data, media_type=content_type)
