from __future__ import annotations

from app.client import UnifiedAIClient
from app.settings import get_settings


def test_unified_client_uses_configured_sqlite_busy_timeout(tmp_path, monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "gateway.sqlite3"))
    monkeypatch.setenv("MODEL_CATALOG_PATH", str(tmp_path / "model_catalog.json"))
    monkeypatch.setenv("SQLITE_BUSY_TIMEOUT_MS", "1234")

    client = UnifiedAIClient()

    assert client.state.busy_timeout_ms == 1234
    get_settings.cache_clear()
