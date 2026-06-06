from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from app.desktop_api import (
    desktop_capabilities,
    desktop_logs,
    desktop_settings_payload,
    export_desktop_backup,
    import_desktop_backup,
    import_desktop_backup_upload,
    request_desktop_restart,
    save_desktop_settings,
)

router = APIRouter()


@router.get("/v1/desktop/capabilities")
async def desktop_capabilities_endpoint(request: Request) -> dict[str, Any]:
    return desktop_capabilities(request)


@router.post("/v1/desktop/restart")
async def desktop_restart_endpoint(request: Request) -> dict[str, Any]:
    return request_desktop_restart(request)


@router.get("/v1/desktop/settings")
async def desktop_settings_endpoint(request: Request) -> dict[str, Any]:
    return desktop_settings_payload(request)


@router.post("/v1/desktop/settings")
async def save_desktop_settings_endpoint(request: Request) -> dict[str, Any]:
    return await save_desktop_settings(request)


@router.post("/v1/desktop/backups/export")
async def export_desktop_backup_endpoint(request: Request) -> dict[str, Any]:
    return export_desktop_backup(request)


@router.post("/v1/desktop/backups/import")
async def import_desktop_backup_endpoint(request: Request) -> dict[str, Any]:
    return await import_desktop_backup(request)


@router.post("/v1/desktop/backups/import-upload")
async def import_desktop_backup_upload_endpoint(request: Request) -> dict[str, Any]:
    return await import_desktop_backup_upload(request)


@router.get("/v1/desktop/logs")
async def desktop_logs_endpoint(request: Request) -> dict[str, Any]:
    return desktop_logs(request)
