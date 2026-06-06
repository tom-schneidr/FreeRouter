from __future__ import annotations

import pytest

from app.client import UnifiedAIClient
from app.settings import get_settings


@pytest.mark.asyncio
async def test_unified_client_chat_rejects_stream_keyword(tmp_path, monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "gateway.sqlite3"))
    monkeypatch.setenv("MODEL_CATALOG_PATH", str(tmp_path / "model_catalog.json"))

    client = UnifiedAIClient()
    with pytest.raises(ValueError, match="chat_stream"):
        await client.chat([{"role": "user", "content": "hi"}], stream=True)
    get_settings.cache_clear()


def test_unified_client_exposes_configured_state_and_catalog_before_initialize(
    tmp_path, monkeypatch
):
    get_settings.cache_clear()
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "gateway.sqlite3"))
    monkeypatch.setenv("MODEL_CATALOG_PATH", str(tmp_path / "model_catalog.json"))
    monkeypatch.setenv("SQLITE_BUSY_TIMEOUT_MS", "1234")

    client = UnifiedAIClient()

    assert client.state.busy_timeout_ms == 1234
    assert client.model_catalog.path == str(tmp_path / "model_catalog.json")
    get_settings.cache_clear()
