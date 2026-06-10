from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from app.app_services import get_app_services
from app.benchmark_store import get_benchmark_store
from app.settings import get_settings

router = APIRouter()


@router.get("/v1/gateway/endpoint-diagnosis")
async def endpoint_diagnosis_status(request: Request) -> dict[str, Any]:
    services = get_app_services(request)
    service = services.endpoint_diagnosis
    background = services.background_endpoint_diagnosis
    report = service.last_report
    return {
        "enabled": get_settings().auto_endpoint_diagnosis_enabled,
        "auto_maintenance_enabled": bool(background and background.apply_safe_suggestions),
        "last_auto_applied": (
            [asdict(suggestion) for suggestion in background.last_auto_applied]
            if background is not None
            else []
        ),
        "last_report": asdict(report) if report is not None else None,
    }


@router.post("/v1/gateway/endpoint-diagnosis/refresh")
async def refresh_endpoint_diagnosis(request: Request) -> dict[str, Any]:
    service = get_app_services(request).endpoint_diagnosis
    report = await service.run_once()
    return {"data": asdict(report)}


@router.get("/v1/gateway/benchmarks")
async def benchmark_status(request: Request) -> dict[str, Any]:
    settings = get_settings()
    services = get_app_services(request)
    store = get_benchmark_store(settings.benchmark_scores_path)
    snapshot = store.snapshot()
    last_report = (
        services.benchmark_research.last_report
        if services.benchmark_research is not None
        else None
    )
    return {
        "enabled": settings.benchmark_refresh_enabled,
        "max_age_seconds": settings.benchmark_refresh_max_age_seconds,
        "updated_at": snapshot.updated_at,
        "source_url": snapshot.source_url,
        "score_count": len(snapshot.scores),
        "last_refresh": asdict(last_report) if last_report is not None else None,
    }


@router.post("/v1/gateway/benchmarks/refresh")
async def refresh_benchmarks(request: Request) -> dict[str, Any]:
    services = get_app_services(request)
    if services.benchmark_research is None:
        raise HTTPException(status_code=503, detail="Benchmark refresh is disabled.")
    report = await services.benchmark_research.refresh()
    return {"data": asdict(report)}


@router.post("/v1/gateway/endpoint-diagnosis/apply")
async def apply_endpoint_diagnosis(request: Request) -> dict[str, Any]:
    service = get_app_services(request).endpoint_diagnosis
    payload = await request.json()
    suggestion_ids = payload.get("suggestion_ids") if isinstance(payload, dict) else None
    if not isinstance(suggestion_ids, list) or not all(
        isinstance(item, str) for item in suggestion_ids
    ):
        raise HTTPException(status_code=400, detail="Expected { suggestion_ids: [string, ...] }")
    applied = await service.apply_suggestions(suggestion_ids)
    return {"data": [asdict(suggestion) for suggestion in applied]}
