from __future__ import annotations

import os
from pathlib import Path

APP_DATA_DIR_ENV = "FREEROUTER_APP_DATA_DIR"
DESKTOP_PROJECT_ROOT_ENV = "FREEROUTER_DESKTOP_PROJECT_ROOT"
ENV_FILE_ENV = "FREEROUTER_ENV_PATH"


def source_root() -> Path:
    return Path(__file__).resolve().parent.parent


def runtime_root() -> Path:
    raw_root = os.environ.get(APP_DATA_DIR_ENV) or os.environ.get(DESKTOP_PROJECT_ROOT_ENV)
    return Path(raw_root) if raw_root else source_root()


def runtime_env_path() -> Path:
    raw_path = os.environ.get(ENV_FILE_ENV)
    return Path(raw_path) if raw_path else runtime_root() / ".env"


def runtime_data_dir() -> Path:
    return runtime_root() / "data"


def runtime_backup_dir() -> Path:
    return runtime_root() / "backups"


def configure_desktop_runtime(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "data").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(APP_DATA_DIR_ENV, str(root))
    os.environ.setdefault(ENV_FILE_ENV, str(root / ".env"))
    os.environ.setdefault("DATABASE_PATH", str(root / "data" / "gateway.sqlite3"))
    os.environ.setdefault("MODEL_CATALOG_PATH", str(root / "data" / "model_catalog.json"))


def legacy_env_candidates(root: Path) -> list[Path]:
    candidates: list[Path] = []
    raw_source_env = os.environ.get("FREEROUTER_SOURCE_ENV")
    if raw_source_env:
        candidates.append(Path(raw_source_env))
    for base in (source_root(), Path.cwd(), Path(os.getcwd())):
        candidates.append(base / ".env")
        candidates.extend(parent / ".env" for parent in list(base.parents)[:4])
    try:
        import sys

        candidates.append(Path(sys.executable).resolve().parent / ".env")
    except OSError:
        pass
    seen: set[Path] = set()
    unique: list[Path] = []
    target = root / ".env"
    for candidate in candidates:
        resolved = candidate.resolve() if candidate.exists() else candidate.absolute()
        if resolved == target.resolve() or resolved in seen:
            continue
        seen.add(resolved)
        unique.append(candidate)
    return unique
