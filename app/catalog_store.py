"""JSON file persistence for the model catalog."""

from __future__ import annotations

import json
import os
from typing import Any


def ensure_catalog_directory(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def load_catalog_json(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, list):
        raise ValueError(f"Model catalog must be a JSON array: {path}")
    return raw


def save_catalog_json(path: str, rows: list[dict[str, Any]]) -> None:
    ensure_catalog_directory(path)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
        handle.write("\n")
