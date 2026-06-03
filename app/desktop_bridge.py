from __future__ import annotations

from pathlib import Path
from typing import Any

from app.desktop_runtime import DesktopServerController, build_desktop_launch_command
from app.desktop_settings import launcher_host_port, settings_payload, write_settings
from app.local_backup import export_backup, import_backup


class DesktopBridge:
    def __init__(self, controller: DesktopServerController) -> None:
        self.controller = controller
        self.window: Any = None

    def bind_window(self, window: Any) -> None:
        self.window = window

    def get_capabilities(self) -> dict[str, Any]:
        command = build_desktop_launch_command(self.controller.project_root)
        return {
            "desktop": True,
            "project_root": str(self.controller.project_root),
            "launch_command": command.__dict__,
            "server": self.controller.refresh(),
        }

    def get_server_status(self) -> dict[str, Any]:
        return self.controller.refresh()

    def restart_server(self) -> dict[str, Any]:
        return self.controller.restart()

    def get_settings(self) -> dict[str, Any]:
        return settings_payload(self.controller.project_root)

    def save_settings(self, values: dict[str, Any]) -> dict[str, Any]:
        updated = write_settings(self.controller.project_root, values)
        host, port = launcher_host_port(self.controller.project_root)
        self.controller.host = host
        self.controller.port = port
        return {"settings": updated, "server": self.controller.refresh(), "restart_required": True}

    def export_backup(self) -> dict[str, Any]:
        target = export_backup()
        return {"ok": True, "path": str(target)}

    def import_backup(self, path: str, overwrite: bool = False) -> dict[str, Any]:
        restored = import_backup(Path(path), overwrite=overwrite)
        return {"ok": True, "restored": [str(item) for item in restored]}

    def choose_backup_file(self) -> dict[str, Any]:
        if self.window is None:
            return {"ok": False, "error": "No desktop window is available."}
        try:
            import webview

            result = self.window.create_file_dialog(
                webview.OPEN_DIALOG,
                allow_multiple=False,
                file_types=("FreeRouter backups (*.zip)", "All files (*.*)"),
            )
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        if not result:
            return {"ok": False, "cancelled": True}
        return {"ok": True, "path": result[0]}

    def get_logs(self, tail: int = 400) -> dict[str, Any]:
        return {"lines": self.controller.logs(tail)}

    def copy_base_url(self) -> dict[str, Any]:
        text = self.controller.base_url
        try:
            import tkinter as tk

            root = tk.Tk()
            root.withdraw()
            root.clipboard_clear()
            root.clipboard_append(text)
            root.update()
            root.destroy()
        except Exception as exc:
            return {"ok": False, "error": str(exc), "text": text}
        return {"ok": True, "text": text}
