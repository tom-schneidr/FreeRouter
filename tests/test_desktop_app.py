from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.desktop_api import DESKTOP_PROJECT_ROOT_ENV, DESKTOP_TOKEN_ENV, desktop_capabilities
from app.desktop_icon import _icons_equivalent
from app.desktop_settings import (
    MASKED_SECRET,
    launcher_host_port,
    settings_payload,
    write_settings,
)


def test_icon_equivalence_ignores_image_container_metadata(tmp_path):
    from PIL import Image

    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    image = Image.new("RGBA", (16, 16), (20, 40, 60, 255))
    image.save(first)
    image.save(second, optimize=True)

    assert first.read_bytes() != second.read_bytes()
    assert _icons_equivalent(first, second)


def test_icon_equivalence_detects_pixel_changes(tmp_path):
    from PIL import Image

    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGBA", (16, 16), (20, 40, 60, 255)).save(first)
    Image.new("RGBA", (16, 16), (21, 40, 60, 255)).save(second)

    assert not _icons_equivalent(first, second)


def test_desktop_settings_mask_secrets_and_preserve_existing_values(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "# local config\n"
        "GATEWAY_PORT=8000\n"
        "CEREBRAS_API_KEY=sk-existing\n"
        "MAX_CONCURRENT_REQUESTS=20\n",
        encoding="utf-8",
    )

    payload = settings_payload(tmp_path)
    secret = next(field for field in payload["fields"] if field["key"] == "CEREBRAS_API_KEY")
    assert secret["value"] == MASKED_SECRET
    assert secret["configured"] is True

    write_settings(
        tmp_path,
        {
            "GATEWAY_PORT": "8080",
            "CEREBRAS_API_KEY": MASKED_SECRET,
            "MAX_CONCURRENT_REQUESTS": "12",
        },
    )

    written = env_path.read_text(encoding="utf-8")
    assert "# local config" in written
    assert "GATEWAY_PORT=8080" in written
    assert "CEREBRAS_API_KEY=sk-existing" in written
    assert "MAX_CONCURRENT_REQUESTS=12" in written


def test_desktop_settings_append_missing_known_keys(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("GATEWAY_PORT=8000\n", encoding="utf-8")

    write_settings(tmp_path, {"DATABASE_PATH": "./data/test.sqlite3"})

    written = env_path.read_text(encoding="utf-8")
    assert "# Added by FreeRouter desktop app" in written
    assert "DATABASE_PATH=./data/test.sqlite3" in written


def test_desktop_settings_validate_numeric_values(tmp_path):
    with pytest.raises(ValueError):
        write_settings(tmp_path, {"GATEWAY_PORT": "0"})

    with pytest.raises(ValueError):
        write_settings(tmp_path, {"REQUEST_TIMEOUT_SECONDS": "-1"})


def test_launcher_host_port_reads_env_with_fallback(tmp_path):
    (tmp_path / ".env").write_text("GATEWAY_HOST=127.0.0.2\nGATEWAY_PORT=8123\n", encoding="utf-8")

    assert launcher_host_port(tmp_path) == ("127.0.0.2", 8123)

    (tmp_path / ".env").write_text("GATEWAY_PORT=not-a-number\n", encoding="utf-8")
    assert launcher_host_port(tmp_path) == ("127.0.0.1", 8000)


def test_tauri_config_points_at_canonical_react_app():
    config_path = Path("apps/desktop/src-tauri/tauri.conf.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert config["build"]["devUrl"] == "http://127.0.0.1:8000/app"
    assert config["build"]["frontendDist"] == "../../ui/dist"
    assert "../../../dist-sidecar/freerouterd" in config["bundle"]["externalBin"]


def test_sidecar_build_packages_react_dist_assets():
    script = Path("scripts/build-sidecar.ps1").read_text(encoding="utf-8")

    assert "$ReactDist = Join-Path $ProjectRoot \"apps\\ui\\dist\"" in script
    assert '"--add-data", "$ReactDist;apps\\ui\\dist"' in script


def test_desktop_api_requires_token(monkeypatch, tmp_path):
    monkeypatch.setenv(DESKTOP_TOKEN_ENV, "secret-token")
    monkeypatch.setenv(DESKTOP_PROJECT_ROOT_ENV, str(tmp_path))
    api = FastAPI()
    api.get("/v1/desktop/capabilities")(desktop_capabilities)
    client = TestClient(api)

    denied = client.get("/v1/desktop/capabilities")
    allowed = client.get(
        "/v1/desktop/capabilities",
        headers={"X-FreeRouter-Desktop-Token": "secret-token"},
    )

    assert denied.status_code == 403
    assert allowed.status_code == 200
    assert allowed.json()["desktop"] is True


def test_desktop_backup_upload_requires_token(monkeypatch, tmp_path):
    monkeypatch.setenv(DESKTOP_TOKEN_ENV, "secret-token")
    monkeypatch.setenv(DESKTOP_PROJECT_ROOT_ENV, str(tmp_path))
    from app.desktop_api import import_desktop_backup_upload

    api = FastAPI()
    api.post("/v1/desktop/backups/import-upload")(import_desktop_backup_upload)
    client = TestClient(api)

    denied = client.post("/v1/desktop/backups/import-upload")
    assert denied.status_code == 403
