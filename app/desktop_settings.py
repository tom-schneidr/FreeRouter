from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

MASKED_SECRET = "********"
SECRET_KEY_PARTS = ("API_KEY", "TOKEN", "SECRET", "PASSWORD")


@dataclass(frozen=True)
class DesktopSetting:
    key: str
    label: str
    group: str
    kind: str = "text"
    default: str = ""
    secret: bool = False


DESKTOP_SETTINGS: tuple[DesktopSetting, ...] = (
    DesktopSetting("GATEWAY_HOST", "Host", "Runtime", default="127.0.0.1"),
    DesktopSetting("GATEWAY_PORT", "Port", "Runtime", kind="int", default="8000"),
    DesktopSetting("CEREBRAS_API_KEY", "Cerebras API key", "Provider Keys", secret=True),
    DesktopSetting("GROQ_API_KEY", "Groq API key", "Provider Keys", secret=True),
    DesktopSetting("GEMINI_API_KEY", "Gemini API key", "Provider Keys", secret=True),
    DesktopSetting("NVIDIA_API_KEY", "NVIDIA API key", "Provider Keys", secret=True),
    DesktopSetting("OPENROUTER_API_KEY", "OpenRouter API key", "Provider Keys", secret=True),
    DesktopSetting("SAMBANOVA_API_KEY", "SambaNova API key", "Provider Keys", secret=True),
    DesktopSetting("DATABASE_PATH", "SQLite database path", "Storage", default="./data/gateway.sqlite3"),
    DesktopSetting("MODEL_CATALOG_PATH", "Model catalog path", "Storage", default="./data/model_catalog.json"),
    DesktopSetting("REQUEST_TIMEOUT_SECONDS", "Provider timeout seconds", "Request Limits", kind="float", default="90"),
    DesktopSetting("MAX_CONCURRENT_REQUESTS", "Max concurrent requests", "Request Limits", kind="int", default="20"),
    DesktopSetting(
        "STREAMING_RELEASE_SLOT_AFTER_ROUTE_SELECTED",
        "Release request slot after stream route selected",
        "Request Limits",
        kind="bool",
        default="true",
    ),
    DesktopSetting(
        "REQUEST_QUEUE_TIMEOUT_SECONDS",
        "Request queue timeout seconds",
        "Request Limits",
        kind="float",
        default="30",
    ),
    DesktopSetting(
        "REQUEST_QUEUE_MAX_WAITING_REQUESTS",
        "Max queued requests",
        "Request Limits",
        kind="optional_int",
        default="200",
    ),
    DesktopSetting("SQLITE_BUSY_TIMEOUT_MS", "SQLite busy timeout ms", "Storage", kind="int", default="5000"),
    DesktopSetting("GATEWAY_MODEL_NAME", "Gateway model name", "Runtime", default="auto"),
    DesktopSetting(
        "AUTO_ENDPOINT_DIAGNOSIS_ENABLED",
        "Automatic endpoint diagnosis",
        "Endpoint Maintenance",
        kind="bool",
        default="true",
    ),
    DesktopSetting(
        "AUTO_ENDPOINT_MAINTENANCE_ENABLED",
        "Automatic safe endpoint maintenance",
        "Endpoint Maintenance",
        kind="bool",
        default="true",
    ),
    DesktopSetting(
        "AUTO_ENDPOINT_DIAGNOSIS_INTERVAL_SECONDS",
        "Endpoint diagnosis interval seconds",
        "Endpoint Maintenance",
        kind="int",
        default="21600",
    ),
    DesktopSetting(
        "AUTO_ENDPOINT_DIAGNOSIS_STARTUP_DELAY_SECONDS",
        "Endpoint diagnosis startup delay seconds",
        "Endpoint Maintenance",
        kind="int",
        default="10",
    ),
    DesktopSetting(
        "ENDPOINT_DIAGNOSIS_SUPERVISOR_ENABLED",
        "Endpoint diagnosis supervisor",
        "Endpoint Maintenance",
        kind="bool",
        default="false",
    ),
    DesktopSetting(
        "ENDPOINT_DIAGNOSIS_SUPERVISOR_MODEL",
        "Preferred supervisor route/model",
        "Endpoint Maintenance",
    ),
)

SETTINGS_BY_KEY = {setting.key: setting for setting in DESKTOP_SETTINGS}


def project_env_path(project_root: Path) -> Path:
    return project_root / ".env"


def read_env_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def launcher_host_port(project_root: Path) -> tuple[str, int]:
    values = read_env_values(project_env_path(project_root))
    host = values.get("GATEWAY_HOST") or "127.0.0.1"
    raw_port = values.get("GATEWAY_PORT") or "8000"
    try:
        port = int(raw_port)
    except ValueError:
        port = 8000
    return host, port


def settings_payload(project_root: Path) -> dict[str, Any]:
    path = project_env_path(project_root)
    values = read_env_values(path)
    fields: list[dict[str, Any]] = []
    for setting in DESKTOP_SETTINGS:
        raw_value = values.get(setting.key, setting.default)
        fields.append(
            {
                "key": setting.key,
                "label": setting.label,
                "group": setting.group,
                "kind": setting.kind,
                "secret": setting.secret,
                "value": MASKED_SECRET if setting.secret and raw_value else raw_value,
                "configured": bool(raw_value) if setting.secret else None,
            }
        )
    return {
        "env_path": str(path),
        "groups": sorted({setting.group for setting in DESKTOP_SETTINGS}),
        "fields": fields,
    }


def write_settings(project_root: Path, incoming_values: dict[str, Any]) -> dict[str, Any]:
    path = project_env_path(project_root)
    existing = read_env_values(path)
    updates: dict[str, str] = {}
    for key, raw_value in incoming_values.items():
        setting = SETTINGS_BY_KEY.get(key)
        if setting is None:
            continue
        if setting.secret and raw_value in {"", None, MASKED_SECRET}:
            continue
        updates[key] = _normalize_setting_value(setting, raw_value)

    lines = _updated_env_lines(path, updates, existing)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return settings_payload(project_root)


def _updated_env_lines(path: Path, updates: dict[str, str], existing: dict[str, str]) -> list[str]:
    seen: set[str] = set()
    lines: list[str] = []
    if path.exists():
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                lines.append(raw_line)
                continue
            key, _ = stripped.split("=", 1)
            key = key.strip()
            if key in updates:
                lines.append(f"{key}={updates[key]}")
                seen.add(key)
            else:
                lines.append(raw_line)

    append_keys = [
        setting.key
        for setting in DESKTOP_SETTINGS
        if setting.key in updates and setting.key not in seen and setting.key not in existing
    ]
    if append_keys:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append("# Added by FreeRouter desktop app")
        for key in append_keys:
            lines.append(f"{key}={updates[key]}")
    return lines


def _normalize_setting_value(setting: DesktopSetting, raw_value: Any) -> str:
    value = "" if raw_value is None else str(raw_value).strip()
    if setting.kind == "bool":
        lowered = value.lower()
        if lowered not in {"true", "false"}:
            raise ValueError(f"{setting.key} must be true or false")
        return lowered
    if setting.kind == "int":
        parsed = int(value)
        if parsed <= 0 and setting.key != "AUTO_ENDPOINT_DIAGNOSIS_STARTUP_DELAY_SECONDS":
            raise ValueError(f"{setting.key} must be greater than 0")
        if setting.key == "AUTO_ENDPOINT_DIAGNOSIS_STARTUP_DELAY_SECONDS" and parsed < 0:
            raise ValueError(f"{setting.key} must be greater than or equal to 0")
        return str(parsed)
    if setting.kind == "optional_int":
        if value == "":
            return ""
        parsed = int(value)
        if parsed <= 0:
            raise ValueError(f"{setting.key} must be greater than 0 when set")
        return str(parsed)
    if setting.kind == "float":
        parsed = float(value)
        if parsed <= 0:
            raise ValueError(f"{setting.key} must be greater than 0")
        return str(parsed).rstrip("0").rstrip(".") if "." in str(parsed) else str(parsed)
    return value
