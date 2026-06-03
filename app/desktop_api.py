from __future__ import annotations

import os
import tempfile
import threading
import zipfile
from pathlib import Path
from typing import Any

from fastapi import HTTPException, Request, UploadFile

from app.desktop_runtime import (
    DESKTOP_PROJECT_ROOT_ENV,
    DESKTOP_RESTART_EXIT_CODE,
    DESKTOP_TOKEN_ENV,
)
from app.desktop_settings import settings_payload, write_settings
from app.local_backup import export_backup, import_backup
from app.runtime_paths import runtime_root


def desktop_enabled() -> bool:
    return bool(os.environ.get(DESKTOP_TOKEN_ENV))


def desktop_project_root() -> Path:
    raw_root = os.environ.get(DESKTOP_PROJECT_ROOT_ENV)
    if raw_root:
        return Path(raw_root)
    return runtime_root()


def require_desktop_request(request: Request) -> None:
    expected = os.environ.get(DESKTOP_TOKEN_ENV)
    supplied = request.headers.get("x-freerouter-desktop-token")
    if not expected or supplied != expected:
        raise HTTPException(status_code=403, detail="Desktop app controls are not enabled.")


def desktop_capabilities(request: Request) -> dict[str, Any]:
    require_desktop_request(request)
    root = desktop_project_root()
    request_base = str(request.base_url).rstrip("/")
    host = request.url.hostname or "127.0.0.1"
    port = request.url.port or 80
    base_url = f"{request_base}/v1"
    return {
        "desktop": True,
        "project_root": str(root),
        "server": {
            "status": "running",
            "host": host,
            "port": port,
            "base_url": base_url,
            "app_url": f"{request_base}/app-next",
        },
    }


def desktop_settings_payload(request: Request) -> dict[str, Any]:
    require_desktop_request(request)
    return settings_payload(desktop_project_root())


async def save_desktop_settings(request: Request) -> dict[str, Any]:
    require_desktop_request(request)
    payload = await request.json()
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Expected a JSON object.")
    try:
        settings = write_settings(desktop_project_root(), payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"settings": settings, "restart_required": True}


def export_desktop_backup(request: Request) -> dict[str, Any]:
    require_desktop_request(request)
    target = export_backup()
    return {"ok": True, "path": str(target)}


async def import_desktop_backup(request: Request) -> dict[str, Any]:
    require_desktop_request(request)
    payload = await request.json()
    if not isinstance(payload, dict) or not payload.get("path"):
        raise HTTPException(status_code=400, detail="Expected { path: string }.")
    restored = import_backup(Path(payload["path"]), overwrite=bool(payload.get("overwrite")))
    return {"ok": True, "restored": [str(item) for item in restored]}


async def import_desktop_backup_upload(request: Request) -> dict[str, Any]:
    require_desktop_request(request)
    form = await request.form()
    upload = form.get("file")
    if not isinstance(upload, UploadFile):
        raise HTTPException(status_code=400, detail="Expected multipart file field 'file'.")
    overwrite = str(form.get("overwrite", "")).lower() in {"1", "true", "yes", "on"}
    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
        tmp.write(await upload.read())
        tmp_path = Path(tmp.name)
    try:
        restored = import_backup(tmp_path, overwrite=overwrite)
    except (FileExistsError, ValueError, zipfile.BadZipFile) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)
    return {"ok": True, "restored": [str(item) for item in restored]}


def desktop_logs(request: Request) -> dict[str, Any]:
    require_desktop_request(request)
    log_path = desktop_project_root() / "data" / "desktop-app.log"
    if not log_path.exists():
        return {"lines": []}
    return {"lines": log_path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)[-600:]}


def request_desktop_restart(request: Request) -> dict[str, Any]:
    require_desktop_request(request)

    def exit_soon() -> None:
        os._exit(DESKTOP_RESTART_EXIT_CODE)

    threading.Timer(0.35, exit_soon).start()
    return {"ok": True, "status": "restarting"}
