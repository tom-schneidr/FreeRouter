from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.runtime_paths import APP_DATA_DIR_ENV, ENV_FILE_ENV
from app.settings import Settings, get_settings


def test_settings_reject_invalid_local_operational_values():
    with pytest.raises(ValidationError):
        Settings(max_concurrent_requests=0)

    with pytest.raises(ValidationError):
        Settings(request_timeout_seconds=-1)

    with pytest.raises(ValidationError):
        Settings(request_queue_max_waiting_requests=0)


def test_settings_accept_valid_local_operational_values():
    settings = Settings(
        max_concurrent_requests=1,
        request_timeout_seconds=1,
        request_queue_max_waiting_requests=None,
        sse_chunk_replay_sleep_seconds=0,
    )

    assert settings.max_concurrent_requests == 1
    assert settings.request_queue_max_waiting_requests is None


def test_get_settings_reads_explicit_runtime_env_file(tmp_path, monkeypatch):
    env_path = tmp_path / "settings.env"
    env_path.write_text("DATABASE_PATH=custom.sqlite3\nREQUEST_TIMEOUT_SECONDS=12\n", encoding="utf-8")
    monkeypatch.setenv(ENV_FILE_ENV, str(env_path))
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.database_path == "custom.sqlite3"
    assert settings.request_timeout_seconds == 12
    get_settings.cache_clear()


def test_get_settings_uses_app_data_env_file(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("MODEL_CATALOG_PATH=models.json\n", encoding="utf-8")
    monkeypatch.delenv(ENV_FILE_ENV, raising=False)
    monkeypatch.setenv(APP_DATA_DIR_ENV, str(tmp_path))
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.model_catalog_path == "models.json"
    get_settings.cache_clear()
