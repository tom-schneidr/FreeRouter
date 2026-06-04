from __future__ import annotations

import argparse
import datetime as dt
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from app.desktop_bridge import DesktopBridge
from app.desktop_icon import build_icon_image
from app.desktop_runtime import APP_NAME, DEFAULT_HOST, DEFAULT_PORT, DesktopServerController
from app.desktop_screen import primary_work_area
from app.desktop_settings import launcher_host_port
from app.ui.brand import inline_icon_html


class FreeRouterDesktopApp:
    def __init__(self, host: str, port: int, reload: bool = False) -> None:
        self.controller = DesktopServerController(host=host, port=port, reload=reload)
        self.bridge = DesktopBridge(self.controller)
        self.window = None
        self.tray_icon: Any | None = None
        self.exiting = False
        self.log_path = self.controller.project_root / "data" / "desktop-app.log"

    def run(self) -> None:
        import webview

        self._log("Desktop launcher starting.")
        bounds = primary_work_area()
        self.window = webview.create_window(
            APP_NAME,
            html=self._startup_html(),
            x=bounds.x,
            y=bounds.y,
            width=bounds.width,
            height=bounds.height,
            min_size=(720, 520),
            maximized=True,
            background_color="#07111f",
            text_select=True,
        )
        self.bridge.bind_window(self.window)
        self.window.events.closing += self._on_closing
        try:
            webview.start(
                self._after_webview_start,
                gui="edgechromium",
                private_mode=False,
                storage_path=str(self.controller.project_root / "data" / "webview"),
            )
        finally:
            self._log("Desktop launcher stopping.")
            self._stop_tray()
            self.controller.stop()

    def _after_webview_start(self) -> None:
        self._log("WebView event loop is ready.")
        threading.Thread(target=self._start_server_and_load_app, daemon=True).start()

    def _start_server_and_load_app(self) -> None:
        try:
            if self.window is None:
                return
            server_status = self.controller.start()
            self._log(f"Server start status: {server_status}")
            if server_status["status"] == "port_conflict":
                self.window.load_html(
                    self._error_html(
                        server_status["detail"] or "The configured port is already in use."
                    )
                )
                return

            deadline = time.monotonic() + 18
            while time.monotonic() < deadline:
                if self.controller.refresh()["status"] == "running":
                    app_url = self._desktop_app_url()
                    self._log(f"Loading app URL: {app_url}")
                    self.window.load_url(app_url)
                    self._start_tray()
                    return
                time.sleep(0.25)

            self.window.load_html(
                self._error_html(
                    self.controller.status_detail
                    or "The server process started, but the health check did not become ready."
                )
            )
        except Exception:
            self._log("Fatal startup error:\n" + traceback.format_exc())
            if self.window is not None:
                self.window.load_html(self._error_html("Desktop startup failed. See data/desktop-app.log."))

    def _on_closing(self) -> bool:
        if self.exiting or self.window is None:
            return True
        self.window.hide()
        return False

    def _start_tray(self) -> None:
        if self.tray_icon is not None:
            return
        import pystray

        menu = pystray.Menu(
            pystray.MenuItem("Show FreeRouter", self._tray_show, default=True),
            pystray.MenuItem("Hide to tray", self._tray_hide),
            pystray.MenuItem("Chat", lambda: self._tray_open("/app#chat")),
            pystray.MenuItem("Models", lambda: self._tray_open("/app#models")),
            pystray.MenuItem("Copy base URL", lambda: self.bridge.copy_base_url()),
            pystray.MenuItem("Restart server", lambda: self._tray_restart()),
            pystray.MenuItem("Quit", lambda: self._tray_quit()),
        )
        self.tray_icon = pystray.Icon(APP_NAME, build_icon_image(64), APP_NAME, menu)
        threading.Thread(target=self.tray_icon.run, daemon=True).start()
        self._log("Tray icon started.")

    def _stop_tray(self) -> None:
        if self.tray_icon is not None:
            self.tray_icon.stop()
            self.tray_icon = None

    def _tray_show(self) -> None:
        if self.window is None:
            return
        self.window.restore()
        self.window.show()

    def _tray_hide(self) -> None:
        if self.window is None:
            return
        self.window.hide()

    def _tray_open(self, path: str) -> None:
        if self.window is None:
            return
        self.window.load_url(self._desktop_app_url(path))
        self._tray_show()

    def _tray_restart(self) -> None:
        self.controller.restart()
        if self.window is not None:
            self.window.load_url(self._desktop_app_url())
            self._tray_show()

    def _tray_quit(self) -> None:
        self.exiting = True
        self._stop_tray()
        if self.window is not None:
            self.window.destroy()

    def _desktop_app_url(self, path: str = "/app") -> str:
        separator = "&" if "?" in path else "?"
        return (
            f"http://{self.controller.host}:{self.controller.port}{path}"
            f"{separator}desktop_token={self.controller.desktop_token}"
        )

    def _log(self, message: str) -> None:
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            timestamp = dt.datetime.now(tz=dt.UTC).isoformat(timespec="seconds")
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"[{timestamp}] {message}\n")
        except OSError:
            pass

    def _startup_html(self) -> str:
        return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{APP_NAME}</title>
    <style>
      body {{ margin:0; min-height:100vh; display:grid; place-items:center; background:#07111f; color:#e2e8f0; font-family:Segoe UI, system-ui, sans-serif; }}
      main {{ width:min(560px, calc(100vw - 48px)); border:1px solid #263449; background:#0f172a; border-radius:8px; padding:28px; box-shadow:0 24px 80px rgba(0,0,0,.35); }}
      h1 {{ margin:0 0 10px; font-size:22px; }}
      p {{ margin:0; color:#94a3b8; line-height:1.6; }}
      .bar {{ height:4px; overflow:hidden; border-radius:999px; background:#1e293b; margin-top:22px; }}
      .bar::before {{ content:""; display:block; width:40%; height:100%; background:#38bdf8; animation:load 1.1s ease-in-out infinite; }}
      @keyframes load {{ 0% {{ transform:translateX(-110%); }} 100% {{ transform:translateX(260%); }} }}
    </style>
  </head>
  <body>
    <main>
      {inline_icon_html(size=56)}
      <h1>Starting FreeRouter</h1>
      <p>Opening the local gateway and desktop controls...</p>
      <div class="bar" aria-hidden="true"></div>
    </main>
  </body>
</html>"""

    def _error_html(self, message: str) -> str:
        return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{APP_NAME}</title>
    <style>
      body {{ margin:0; min-height:100vh; display:grid; place-items:center; background:#07111f; color:#e2e8f0; font-family:Segoe UI, system-ui, sans-serif; }}
      main {{ width:min(720px, calc(100vw - 48px)); border:1px solid #263449; background:#0f172a; border-radius:8px; padding:28px; box-shadow:0 24px 80px rgba(0,0,0,.35); }}
      h1 {{ margin:0 0 12px; font-size:22px; }}
      p {{ color:#94a3b8; line-height:1.6; }}
      code {{ color:#93c5fd; }}
    </style>
  </head>
  <body>
    <main>
      {inline_icon_html(size=56)}
      <h1>FreeRouter could not start</h1>
      <p>{message}</p>
      <p>Free <code>{self.controller.host}:{self.controller.port}</code> or update <code>GATEWAY_PORT</code> in <code>.env</code>, then reopen FreeRouter.</p>
    </main>
  </body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FreeRouter as a local desktop app.")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    env_host, env_port = launcher_host_port(root)
    host = args.host or env_host or DEFAULT_HOST
    port = args.port or env_port or DEFAULT_PORT
    FreeRouterDesktopApp(host, port, args.reload).run()


if __name__ == "__main__":
    main()
