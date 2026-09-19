"""JSON file persistence for the model catalog."""

from __future__ import annotations

import json
import os
from typing import Any

CATALOG_SCHEMA_VERSION = 2


def ensure_catalog_directory(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def load_catalog_json(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    if isinstance(raw, list):
        # Version 1 catalogs were bare arrays.  Keep them readable so an
        # upgrade never requires a destructive migration step.
        return raw
    if isinstance(raw, dict) and isinstance(raw.get("routes"), list):
        version = raw.get("schema_version", 1)
        if not isinstance(version, int) or version < 1:
            raise ValueError(f"Invalid model catalog schema version: {path}")
        return raw["routes"]
    raise ValueError(f"Model catalog must be a route array or versioned object: {path}")


def save_catalog_json(path: str, rows: list[dict[str, Any]]) -> None:
    ensure_catalog_directory(path)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "schema_version": CATALOG_SCHEMA_VERSION,
                "routes": rows,
            },
            handle,
            indent=2,
        )
        handle.write("\n")
