from __future__ import annotations

import json
import os
import queue
import secrets
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

APP_NAME = "FreeRouter"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
MAX_LOG_LINES = 2000
PORT_FALLBACK_SCAN_LIMIT = 20
DESKTOP_TOKEN_ENV = "FREEROUTER_DESKTOP_TOKEN"
DESKTOP_PROJECT_ROOT_ENV = "FREEROUTER_DESKTOP_PROJECT_ROOT"
DESKTOP_RESTART_EXIT_CODE = 42


@dataclass(frozen=True)
class DesktopLaunchCommand:
    target: str
    arguments: str
    working_directory: str
    icon_path: str | None = None


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def venv_python(project_root_path: Path) -> Path:
    return project_root_path / ".venv" / "Scripts" / "python.exe"


def venv_pythonw(project_root_path: Path) -> Path:
    return project_root_path / ".venv" / "Scripts" / "pythonw.exe"


def build_desktop_launch_command(project_root_path: Path | None = None) -> DesktopLaunchCommand:
    root = project_root_path or project_root()
    target = venv_pythonw(root)
    icon_path = root / "data" / "freerouter.ico"
    return DesktopLaunchCommand(
        target=str(target),
        arguments="-m app.desktop_app",
        working_directory=str(root),
        icon_path=str(icon_path) if icon_path.exists() else None,
    )


def server_python() -> str:
    executable = Path(sys.executable)
    if executable.name.lower() == "pythonw.exe":
        sibling = executable.with_name("python.exe")
        if sibling.exists():
            return str(sibling)
    return str(executable)


def probe_freerouter(host: str, port: int, timeout: float = 0.8) -> bool:
    url = f"http://{host}:{port}/v1/gateway/health.json"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, TimeoutError, ValueError, urllib.error.URLError):
        return False
    return isinstance(payload, dict) and payload.get("service") == "freerouter"


def is_port_open(host: str, port: int, timeout: float = 0.4) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def find_available_port(host: str, preferred_port: int, *, limit: int = PORT_FALLBACK_SCAN_LIMIT) -> int | None:
    for offset in range(1, limit + 1):
        candidate = preferred_port + offset
        if not is_port_open(host, candidate):
            return candidate
    return None


class DesktopServerController:
    def __init__(
        self,
        *,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        project_root_path: Path | None = None,
        reload: bool = False,
    ) -> None:
        self.host = host
        self.port = port
        self.project_root = project_root_path or project_root()
        self.reload = reload
        self.process: subprocess.Popen[str] | None = None
        self.attached_existing = False
        self.status = "stopped"
        self.status_detail: str | None = None
        self._log_queue: queue.Queue[str | None] = queue.Queue()
        self._logs: list[str] = []
        self._lock = threading.RLock()
        self.desktop_token = secrets.token_urlsafe(32)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    @property
    def app_url(self) -> str:
        return f"http://{self.host}:{self.port}/app"

    def start(self) -> dict[str, Any]:
        with self._lock:
            if self.process is not None and self.process.poll() is None:
                self.status = "running"
                return self.snapshot()
            if probe_freerouter(self.host, self.port):
                self.attached_existing = True
                self.status = "running"
                self.status_detail = "Attached to an already running FreeRouter server."
                self._append_log("Attached to existing FreeRouter server.\n")
                return self.snapshot()
            if is_port_open(self.host, self.port):
                original_port = self.port
                fallback_port = find_available_port(self.host, self.port)
                if fallback_port is None:
                    self.status = "port_conflict"
                    self.status_detail = (
                        f"{self.host}:{self.port} is already in use and no nearby free port was found."
                    )
                    self._append_log(f"Port conflict: {self.status_detail}\n")
                    return self.snapshot()
                self.port = fallback_port
                self.status_detail = (
                    f"{self.host}:{original_port} is in use by another process; "
                    f"FreeRouter started on {self.host}:{self.port}."
                )
                self._append_log(self.status_detail + "\n")

            self.status = "starting"
            startup_detail = self.status_detail
            args = [
                server_python(),
                "-m",
                "uvicorn",
                "app.main:app",
                "--host",
                self.host,
                "--port",
                str(self.port),
            ]
            if self.reload:
                args.append("--reload")

            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env[DESKTOP_TOKEN_ENV] = self.desktop_token
            env[DESKTOP_PROJECT_ROOT_ENV] = str(self.project_root)
            startupinfo = None
            creationflags = 0
            if os.name == "nt":
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

            self.process = subprocess.Popen(
                args,
                cwd=self.project_root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                startupinfo=startupinfo,
                creationflags=creationflags,
            )
            self.attached_existing = False
            self._append_log(f"Starting FreeRouter at {self.base_url}\n")
            threading.Thread(target=self._read_process_output, daemon=True).start()
            threading.Thread(target=self._watch_process_exit, daemon=True).start()
            deadline = time.monotonic() + 12
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    self.status = "stopped"
                    self.status_detail = f"Server exited with code {self.process.returncode}."
                    return self.snapshot()
                if probe_freerouter(self.host, self.port, timeout=0.4):
                    self.status = "running"
                    self.status_detail = startup_detail
                    return self.snapshot()
                time.sleep(0.25)
            self.status = "starting"
            self.status_detail = "Server process started; waiting for health check."
            return self.snapshot()

    def stop(self) -> dict[str, Any]:
        with self._lock:
            if self.attached_existing:
                self.attached_existing = False
                self.status = "stopped"
                self.status_detail = "Detached from existing server."
                return self.snapshot()
            if self.process is None or self.process.poll() is not None:
                self.status = "stopped"
                return self.snapshot()
            self.status = "stopping"
            self.process.terminate()
            try:
                self.process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self._append_log("Server did not stop cleanly; forcing shutdown.\n")
                self.process.kill()
                self.process.wait(timeout=3)
            self.status = "stopped"
            self.status_detail = None
            return self.snapshot()

    def restart(self) -> dict[str, Any]:
        with self._lock:
            self._append_log("\nRestarting FreeRouter server...\n")
            self.stop()
            return self.start()

    def refresh(self) -> dict[str, Any]:
        self._drain_logs()
        with self._lock:
            if self.process is not None and self.process.poll() is not None:
                self.status = "stopped"
                self.status_detail = f"Server exited with code {self.process.returncode}."
            elif self.status in {"starting", "running"} and probe_freerouter(self.host, self.port, timeout=0.25):
                self.status = "running"
                self.status_detail = None
            return self.snapshot()

    def logs(self, tail: int = 400) -> list[str]:
        self._drain_logs()
        return self._logs[-tail:]

    def snapshot(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "detail": self.status_detail,
            "host": self.host,
            "port": self.port,
            "base_url": self.base_url,
            "app_url": self.app_url,
            "attached_existing": self.attached_existing,
            "pid": self.process.pid if self.process is not None and self.process.poll() is None else None,
        }

    def _read_process_output(self) -> None:
        if self.process is None or self.process.stdout is None:
            return
        for line in self.process.stdout:
            self._log_queue.put(line)
        self._log_queue.put(None)

    def _drain_logs(self) -> None:
        while True:
            try:
                line = self._log_queue.get_nowait()
            except queue.Empty:
                break
            if line is None:
                self._append_log("\nFreeRouter server stopped.\n")
            else:
                self._append_log(line)

    def _watch_process_exit(self) -> None:
        process = self.process
        if process is None:
            return
        returncode = process.wait()
        if returncode != DESKTOP_RESTART_EXIT_CODE:
            return
        self._append_log("\nDesktop restart requested by local app controls.\n")
        with self._lock:
            if self.process is process:
                self.process = None
                self.status = "stopped"
                self.status_detail = "Restart requested by desktop controls."
        self.start()

    def _append_log(self, text: str) -> None:
        self._logs.extend(text.splitlines(keepends=True))
        if len(self._logs) > MAX_LOG_LINES:
            self._logs = self._logs[-MAX_LOG_LINES:]
