# ShadowServe AI

> Enterprise-grade, multi-tenant model serving proxy and live evaluation engine for highly regulated financial environments.

```
[ Client Apps ] ──(mTLS/JWT)──► [ ShadowServe Gateway ]
                                        │
                          ┌─────────────┴──────────────┐
                          ▼                             ▼
                   [ Canary Router ]           [ Shadow Router ]
                   (90% production /            (100% async fork
                    10% challenger)               to challenger)
                          │                             │
                          ▼                             ▼
                   [ Production ]               [ Challenger ]
                   (v1.2.0-Basel)              (v1.3.0-Experimental)
                          │                             │
                          └─────────────┬───────────────┘
                                        ▼
                             [ Real-Time Eval Engine ]
                             KS-Test · PSI · KL-Divergence
                             Wasserstein Distance · Auto-Rollback
                                        │
                             [ Compliance Audit Ledger ]
                             Immutable · Per-transaction lineage
```

## Features

| Capability | Implementation |
|---|---|
| Shadow (dark) launching | Async `BackgroundTask` fork — zero latency overhead on production path |
| Canary routing | Configurable weight split via live API (0–100%) |
| Drift detection | KS-test, PSI, KL-Divergence, Wasserstein Distance on rolling 500-sample window |
| Auto-rollback | Configurable thresholds trigger suppression of challenger traffic |
| Immutable audit ledger | PostgreSQL + in-memory fallback; per-transaction model hash + feature lineage |
| Compliance audit UI | Dashboard with transaction_id lookup — maps to SR 11-7 / Basel MRM |
| Prometheus metrics | P50/P95/P99 latency histograms, drift gauges, request counters |

## Quick Start

### Local (no Docker)

```bash
# 1. Install dependencies
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Start mock model servers
python -m shadowserve.serving.mock_models production 8001 &
python -m shadowserve.serving.mock_models challenger 8002 &

# 3. Start the gateway
uvicorn shadowserve.main:app --host 0.0.0.0 --port 8000 --reload

# 4. Open dashboard
open http://localhost:8000/dashboard/

# 5. Seed traffic
python scripts/seed_demo_traffic.py --count 200 --rate 10

# 6. Inject market anomaly (triggers drift alert)
python scripts/inject_anomaly.py --count 100
```

### Docker Compose (full stack)

```bash
docker compose up --build
# Gateway:    http://localhost:8000
# Dashboard:  http://localhost:8000/dashboard/
# Prometheus: http://localhost:9090
```

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/v1/infer` | POST | Primary inference — canary + shadow routing |
| `/v1/drift` | GET | Current drift statistics |
| `/v1/canary` | GET | Current canary configuration |
| `/v1/canary/weight` | POST | Update canary weight live |
| `/v1/audit/{tx_id}` | GET | Full audit trail for a transaction |
| `/dashboard/` | GET | Control plane UI |
| `/metrics` | GET | Prometheus metrics scrape endpoint |
| `/health` | GET | Health check |

### Inference request

```json
POST /v1/infer
{
  "features": {
    "credit_score": 720,
    "debt_to_income": 0.32,
    "annual_income": 85000,
    "loan_amount": 45000,
    "employment_years": 7,
    "num_accounts": 8,
    "num_derogatory": 0
  }
}
```

### Audit lookup

```json
GET /v1/audit/3fa85f64-5717-4562-b3fc-2c963f66afa6

{
  "transaction_id": "3fa85f64-...",
  "timestamp": "2024-01-15T14:30:22.541Z",
  "model_version": "v1.2.0",
  "artifact_hash": "a3f9c12d8e4b7f1a",
  "input_features": { "credit_score": 720, ... },
  "prediction": 1,
  "probability": 0.8312,
  "latency_ms": 8.4,
  "routed_to": "production",
  "env_metadata": { "host": "...", "shadowserve_version": "0.1.0" }
}
```

## Demo Scenarios

### 1. Shadow Deployment
Send normal traffic — watch the dashboard compare production vs. challenger predictions in real time with zero impact to production SLAs.

### 2. Drift Detection & Auto-Rollback
```bash
python scripts/inject_anomaly.py --count 100
```
Injects a high-risk credit profile cluster. Within ~30 requests the KS-test and PSI will breach thresholds, triggering the drift banner and auto-rollback on the dashboard.

### 3. Live Canary Weight Adjustment
Use the dashboard slider or:
```bash
curl -X POST "http://localhost:8000/v1/canary/weight" \
  -H "Content-Type: application/json" \
  -d '{"canary_weight": 0.50, "shadow_enabled": true}'
```

## Architecture

- **Gateway**: FastAPI async, `BackgroundTasks` for shadow forking
- **Model Clients**: Thin `httpx.AsyncClient` wrappers — swap for Triton/vLLM gRPC in production
- **Eval Engine**: SciPy `ks_2samp`, histogram-based PSI/KL, rolling deque window
- **Audit Ledger**: SQLAlchemy async + PostgreSQL; in-memory fallback for dev
- **Metrics**: `prometheus_client` with custom registry; latency histograms per `(model_version, route_target)`
- **Dashboard**: Vanilla JS + Chart.js, polls `/dashboard/api/*` every 2s

## Compliance Alignment

| Regulation | Implementation |
|---|---|
| SR 11-7 (Model Risk Management) | Artifact hash + feature vector + env metadata per inference |
| Basel III/IV | Reproducible lineage via immutable ledger; model version pinned on every record |
| RBAC | JWT bearer token verification on inference routes |

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```
