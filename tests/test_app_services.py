"""Tests for app factory and typed service container."""

from __future__ import annotations

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from app.app_services import AppServices, attach_app_services, get_app_services
from app.factory import build_app_services, build_core_gateway_stack
from app.settings import get_settings


@pytest.mark.asyncio
async def test_build_core_gateway_stack_initializes_router(tmp_path, monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "gateway.sqlite3"))
    monkeypatch.setenv("MODEL_CATALOG_PATH", str(tmp_path / "model_catalog.json"))
    monkeypatch.setenv("AUTO_ENDPOINT_DIAGNOSIS_ENABLED", "false")

    stack = await build_core_gateway_stack()
    try:
        assert stack.waterfall_router is not None
        assert stack.model_catalog.all_routes()
    finally:
        await stack.http_client.aclose()
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_build_app_services_includes_limiter_and_monitor(tmp_path, monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "gateway.sqlite3"))
    monkeypatch.setenv("MODEL_CATALOG_PATH", str(tmp_path / "model_catalog.json"))
    monkeypatch.setenv("AUTO_ENDPOINT_DIAGNOSIS_ENABLED", "false")

    services = await build_app_services()
    try:
        assert isinstance(services, AppServices)
        assert services.request_limiter is not None
        assert services.live_monitor is not None
    finally:
        await services.shutdown()
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_get_app_services_returns_attached_container(tmp_path, monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "gateway.sqlite3"))
    monkeypatch.setenv("MODEL_CATALOG_PATH", str(tmp_path / "model_catalog.json"))
    monkeypatch.setenv("AUTO_ENDPOINT_DIAGNOSIS_ENABLED", "false")

    services = await build_app_services()
    probe = FastAPI()
    attach_app_services(probe, services)

    @probe.get("/probe")
    async def probe_route(request: Request) -> dict[str, bool]:
        return {"same": get_app_services(request) is services}

    with TestClient(probe) as client:
        assert client.get("/probe").json()["same"] is True

    await services.shutdown()
    get_settings.cache_clear()
