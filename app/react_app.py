from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles


REACT_APP_ROUTE = "/app"
REACT_APP_COMPAT_ROUTE = "/app-next"


def react_dist_path(project_root: Path | None = None) -> Path:
    roots: list[Path] = []
    if project_root is not None:
        roots.append(project_root)
    if getattr(sys, "frozen", False):
        roots.append(Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent)))
    roots.append(Path(__file__).resolve().parent.parent)
    roots.append(Path.cwd())

    for root in roots:
        candidate = root / "apps" / "ui" / "dist"
        if (candidate / "index.html").exists():
            return candidate

    return roots[0] / "apps" / "ui" / "dist"


def mount_react_app(app: FastAPI, *, project_root: Path | None = None) -> None:
    dist = react_dist_path(project_root)
    assets = dist / "assets"

    if assets.exists():
        app.mount(f"{REACT_APP_ROUTE}/assets", StaticFiles(directory=assets), name="react-assets")

    @app.get(REACT_APP_ROUTE, include_in_schema=False)
    async def react_app_index(request: Request) -> Response:
        return _react_index_response(dist, request)

    @app.get(f"{REACT_APP_ROUTE}/{{path:path}}", include_in_schema=False)
    async def react_app_fallback(path: str, request: Request) -> Response:
        return _react_index_response(dist, request)

    @app.get(REACT_APP_COMPAT_ROUTE, include_in_schema=False)
    async def react_app_compat_index(request: Request) -> Response:
        return _react_index_response(dist, request)

    @app.get(f"{REACT_APP_COMPAT_ROUTE}/{{path:path}}", include_in_schema=False)
    async def react_app_compat_fallback(path: str, request: Request) -> Response:
        return _react_index_response(dist, request)


def _react_index_response(dist: Path, request: Request) -> Response:
    index_path = dist / "index.html"
    if index_path.exists():
        return HTMLResponse(index_path.read_text(encoding="utf-8"))
    return HTMLResponse(
        "<!doctype html><title>FreeRouter UI Missing</title>"
        "<main><h1>FreeRouter UI build missing</h1>"
        "<p>Run npm run build:web, then restart FreeRouter.</p></main>",
        status_code=503,
    )
