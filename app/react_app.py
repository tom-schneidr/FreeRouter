from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles


REACT_APP_ROUTE = "/app-next"


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

    @app.get(REACT_APP_ROUTE, response_class=HTMLResponse, include_in_schema=False)
    async def react_app_index() -> str:
        return _react_index_html(dist)

    @app.get(f"{REACT_APP_ROUTE}/{{path:path}}", response_class=HTMLResponse, include_in_schema=False)
    async def react_app_fallback(path: str) -> str:
        return _react_index_html(dist)


def _react_index_html(dist: Path) -> str:
    index_path = dist / "index.html"
    if index_path.exists():
        return index_path.read_text(encoding="utf-8")
    return """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>FreeRouter React App</title>
    <style>
      body { margin: 0; min-height: 100vh; display: grid; place-items: center; background: #07111f; color: #e5edf8; font-family: Segoe UI, system-ui, sans-serif; }
      main { width: min(680px, calc(100vw - 48px)); border: 1px solid #24354d; border-radius: 8px; background: #101b2e; padding: 28px; box-shadow: 0 20px 60px rgba(0, 0, 0, .28); }
      h1 { margin: 0 0 10px; font-size: 22px; }
      p { margin: 0; color: #91a4bd; line-height: 1.6; }
      code { color: #bfdbfe; }
    </style>
  </head>
  <body>
    <main>
      <h1>React app is not built yet</h1>
      <p>Run <code>npm install</code> and <code>npm run build:web</code>, then reload this page.</p>
    </main>
  </body>
</html>"""
