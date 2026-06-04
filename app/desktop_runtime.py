from __future__ import annotations

import json
import socket
import sys
import urllib.error
import urllib.request
from pathlib import Path

APP_NAME = "FreeRouter"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
PORT_FALLBACK_SCAN_LIMIT = 20
DESKTOP_TOKEN_ENV = "FREEROUTER_DESKTOP_TOKEN"
DESKTOP_PROJECT_ROOT_ENV = "FREEROUTER_DESKTOP_PROJECT_ROOT"
DESKTOP_RESTART_EXIT_CODE = 42


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


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
