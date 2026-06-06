"""Shared gateway construction for the HTTP server and programmatic client."""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from app.app_services import AppServices
from app.endpoint_diagnosis import (
    BackgroundEndpointDiagnosis,
    EndpointDiagnosisService,
    EndpointSupervisor,
)
from app.live_monitor import APILiveMonitor
from app.model_catalog import ModelCatalog
from app.providers import PROVIDER_QUOTAS, build_provider_adapters
from app.request_limiter import GatewayRequestLimiter
from app.router import WaterfallRouter
from app.settings import Settings, get_settings
from app.state import StateManager


@dataclass
class CoreGatewayStack:
    settings: Settings
    gateway_state: StateManager
    model_catalog: ModelCatalog
    http_client: httpx.AsyncClient
    waterfall_router: WaterfallRouter


async def build_core_gateway_stack(settings: Settings | None = None) -> CoreGatewayStack:
    settings = settings or get_settings()
    state = StateManager(
        settings.database_path,
        PROVIDER_QUOTAS,
        busy_timeout_ms=settings.sqlite_busy_timeout_ms,
    )
    await state.initialize()
    model_catalog = ModelCatalog(settings.model_catalog_path)
    model_catalog.initialize()
    limits = httpx.Limits(
        max_connections=settings.http_max_connections,
        max_keepalive_connections=settings.http_max_keepalive_connections,
        keepalive_expiry=settings.http_keepalive_expiry_seconds,
    )
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(settings.request_timeout_seconds),
        limits=limits,
    )
    providers = build_provider_adapters(settings)
    router = WaterfallRouter(
        providers,
        model_catalog,
        state,
        request_timeout_seconds=settings.request_timeout_seconds,
        http_client=http_client,
    )
    return CoreGatewayStack(
        settings=settings,
        gateway_state=state,
        model_catalog=model_catalog,
        http_client=http_client,
        waterfall_router=router,
    )


async def build_app_services(settings: Settings | None = None) -> AppServices:
    stack = await build_core_gateway_stack(settings)
    settings = stack.settings
    providers = build_provider_adapters(settings)
    limiter = GatewayRequestLimiter(
        settings.max_concurrent_requests,
        settings.request_queue_timeout_seconds,
        settings.request_queue_max_waiting_requests,
    )
    monitor = APILiveMonitor(max_events=1000)
    diagnosis = EndpointDiagnosisService(
        providers,
        stack.model_catalog,
        stack.gateway_state,
        request_timeout_seconds=settings.request_timeout_seconds,
        supervisor=EndpointSupervisor(
            enabled=settings.endpoint_diagnosis_supervisor_enabled,
            providers=providers,
            catalog=stack.model_catalog,
            state=stack.gateway_state,
            preferred_model=settings.endpoint_diagnosis_supervisor_model,
        ),
    )
    background: BackgroundEndpointDiagnosis | None = None
    if settings.auto_endpoint_diagnosis_enabled:
        background = BackgroundEndpointDiagnosis(
            diagnosis,
            interval_seconds=settings.auto_endpoint_diagnosis_interval_seconds,
            startup_delay_seconds=settings.auto_endpoint_diagnosis_startup_delay_seconds,
            apply_safe_suggestions=settings.auto_endpoint_maintenance_enabled,
        )
        background.start()
    return AppServices(
        gateway_state=stack.gateway_state,
        model_catalog=stack.model_catalog,
        request_limiter=limiter,
        live_monitor=monitor,
        http_client=stack.http_client,
        waterfall_router=stack.waterfall_router,
        endpoint_diagnosis=diagnosis,
        background_endpoint_diagnosis=background,
    )
