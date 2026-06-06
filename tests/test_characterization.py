"""Characterization tests for externally observable retained behavior."""

from __future__ import annotations

import zipfile

import pytest
from fastapi.testclient import TestClient

from app.client import UnifiedAIClient
from app.desktop_api import DESKTOP_PROJECT_ROOT_ENV, DESKTOP_TOKEN_ENV
from app.main import app
from app.settings import get_settings


def _client(tmp_path, monkeypatch) -> TestClient:
    get_settings.cache_clear()
    settings = {
        "DATABASE_PATH": str(tmp_path / "gateway.sqlite3"),
        "MODEL_CATALOG_PATH": str(tmp_path / "model_catalog.json"),
        "AUTO_ENDPOINT_DIAGNOSIS_ENABLED": "false",
        "CEREBRAS_API_KEY": "",
        "GROQ_API_KEY": "",
        "GEMINI_API_KEY": "",
        "NVIDIA_API_KEY": "",
        "OPENROUTER_API_KEY": "",
        "SAMBANOVA_API_KEY": "",
    }
    for key, value in settings.items():
        monkeypatch.setenv(key, value)
    return TestClient(app)


def test_lifespan_exposes_gateway_services_on_app_state(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        response = client.get("/v1/gateway/health.json")
    assert response.status_code == 200
    # Indirect proof lifespan wired state, catalog, router, and diagnosis services.
    payload = response.json()
    assert payload["routes"]["enabled"] >= 1
    assert "background_endpoint_diagnosis" in payload


def test_endpoint_diagnosis_status_reports_disabled_when_auto_off(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        response = client.get("/v1/gateway/endpoint-diagnosis")
    assert response.status_code == 200
    payload = response.json()
    assert payload["enabled"] is False
    assert payload["last_report"] is None
    assert payload["last_auto_applied"] == []


def test_desktop_endpoints_require_token_on_main_app(tmp_path, monkeypatch):
    monkeypatch.setenv(DESKTOP_TOKEN_ENV, "desktop-secret")
    monkeypatch.setenv(DESKTOP_PROJECT_ROOT_ENV, str(tmp_path))
    with _client(tmp_path, monkeypatch) as client:
        denied = client.get("/v1/desktop/capabilities")
        allowed = client.get(
            "/v1/desktop/capabilities",
            headers={"X-FreeRouter-Desktop-Token": "desktop-secret"},
        )
    assert denied.status_code == 403
    assert allowed.status_code == 200
    assert allowed.json()["server"]["base_url"].endswith("/v1")


def test_desktop_backup_export_via_api(tmp_path, monkeypatch):
    monkeypatch.setenv(DESKTOP_TOKEN_ENV, "desktop-secret")
    monkeypatch.setenv(DESKTOP_PROJECT_ROOT_ENV, str(tmp_path))
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "gateway.sqlite3").write_bytes(b"sqlite")
    (data_dir / "model_catalog.json").write_text("[]\n", encoding="utf-8")

    with _client(tmp_path, monkeypatch) as client:
        response = client.post(
            "/v1/desktop/backups/export",
            headers={"X-FreeRouter-Desktop-Token": "desktop-secret"},
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    backup_path = payload["path"]
    with zipfile.ZipFile(backup_path) as archive:
        names = set(archive.namelist())
    assert "data/gateway.sqlite3" in names


def test_gateway_live_routes_are_registered(tmp_path, monkeypatch):
    """Live SSE streams are infinite; verify route registration via OpenAPI paths."""
    with _client(tmp_path, monkeypatch) as client:
        schema = client.get("/openapi.json").json()
    paths = schema.get("paths", {})
    assert "/v1/gateway/live/snapshot" in paths
    assert "/v1/gateway/live/events" in paths


@pytest.mark.asyncio
async def test_unified_client_reports_waterfall_exhausted(tmp_path, monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "gateway.sqlite3"))
    monkeypatch.setenv("MODEL_CATALOG_PATH", str(tmp_path / "model_catalog.json"))
    for key in (
        "CEREBRAS_API_KEY",
        "GROQ_API_KEY",
        "GEMINI_API_KEY",
        "NVIDIA_API_KEY",
        "OPENROUTER_API_KEY",
        "SAMBANOVA_API_KEY",
    ):
        monkeypatch.setenv(key, "")

    client = UnifiedAIClient()
    try:
        await client.initialize()
        from app.router import NoProviderAvailable

        with pytest.raises(NoProviderAvailable):
            await client.chat([{"role": "user", "content": "hello"}])
    finally:
        await client.aclose()
        get_settings.cache_clear()


def test_v1_models_includes_enabled_catalog_routes(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        models = client.get("/v1/models").json()
        catalog = client.get("/v1/gateway/models").json()
    enabled_ids = {route["route_id"] for route in catalog["data"] if route.get("enabled")}
    listed_ids = {item["id"] for item in models["data"]}
    assert "auto" in listed_ids
    assert enabled_ids.issubset(listed_ids)
