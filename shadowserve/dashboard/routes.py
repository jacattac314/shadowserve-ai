"""Dashboard API routes — serves the control plane UI and JSON data feeds."""
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse, FileResponse
from pathlib import Path
from shadowserve.evaluation import drift as drift_module
from shadowserve.audit import ledger
from shadowserve.config import settings

router = APIRouter(prefix="/dashboard", tags=["dashboard"])
STATIC = Path(__file__).parent / "static"


@router.get("/", response_class=HTMLResponse)
async def ui():
    return (STATIC / "index.html").read_text()


@router.get("/api/drift")
async def drift_status():
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
        "production_sample_size": len(s.production_scores),
        "shadow_sample_size": len(s.shadow_scores),
    }


@router.get("/api/canary")
async def canary_status():
    return {
        "canary_weight": settings.canary_weight,
        "shadow_enabled": settings.shadow_enabled,
        "production_model": settings.production_model_version,
        "challenger_model": settings.challenger_model_version,
    }


@router.post("/api/canary/weight")
async def set_canary_weight(weight: float = Query(ge=0.0, le=1.0)):
    settings.canary_weight = weight
    return {"canary_weight": weight}


@router.get("/api/audit/recent")
async def recent_entries(limit: int = Query(default=50, le=500)):
    return await ledger.list_recent(limit)


@router.get("/api/audit/{transaction_id}")
async def audit_entry(transaction_id: str):
    entry = await ledger.fetch_entry(transaction_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Transaction not found")
    return entry
