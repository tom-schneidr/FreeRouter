"""Typed application service container for FastAPI route handlers."""

from __future__ import annotations

from dataclasses import dataclass

import httpx
from fastapi import Request

from app.benchmark_research import BenchmarkResearchService
from app.endpoint_diagnosis import BackgroundEndpointDiagnosis, EndpointDiagnosisService
from app.live_monitor import APILiveMonitor
from app.model_catalog import ModelCatalog
from app.request_limiter import GatewayRequestLimiter
from app.router import WaterfallRouter
from app.state import StateManager

APP_SERVICES_STATE_KEY = "services"


@dataclass
class AppServices:
    gateway_state: StateManager
    model_catalog: ModelCatalog
    request_limiter: GatewayRequestLimiter
    live_monitor: APILiveMonitor
    http_client: httpx.AsyncClient
    waterfall_router: WaterfallRouter
    endpoint_diagnosis: EndpointDiagnosisService
    background_endpoint_diagnosis: BackgroundEndpointDiagnosis | None
    benchmark_research: BenchmarkResearchService | None

    async def shutdown(self) -> None:
        background = self.background_endpoint_diagnosis
        if background is not None:
            await background.stop()
        await self.http_client.aclose()


def attach_app_services(app, services: AppServices) -> None:
    setattr(app.state, APP_SERVICES_STATE_KEY, services)


def get_app_services(request: Request) -> AppServices:
    services = getattr(request.app.state, APP_SERVICES_STATE_KEY, None)
    if not isinstance(services, AppServices):
        raise RuntimeError("AppServices is not initialized on application state")
    return services
