#!/usr/bin/env python3
"""Fail when core modules import desktop or UI packages (baseline-enforced)."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = ROOT / "app"

FORBIDDEN_PREFIXES = (
    "app.ui",
    "app.desktop_api",
    "app.desktop_runtime",
    "app.desktop_settings",
    "app.tray_launcher",
    "app.react_app",
)

SCAN_FILES = [
    CORE_ROOT / "router.py",
    CORE_ROOT / "state.py",
    CORE_ROOT / "model_catalog.py",
    CORE_ROOT / "factory.py",
    CORE_ROOT / "app_services.py",
    CORE_ROOT / "routing_policy.py",
    CORE_ROOT / "state_rules.py",
    CORE_ROOT / "openai_stream_routing.py",
    CORE_ROOT / "stream_route.py",
    CORE_ROOT / "codex_compat.py",
    CORE_ROOT / "client.py",
]


def forbidden_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(FORBIDDEN_PREFIXES):
                    hits.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith(FORBIDDEN_PREFIXES):
                hits.append(node.module)
    return hits


def main() -> int:
    violations: list[str] = []
    for path in SCAN_FILES:
        if not path.exists():
            continue
        for hit in forbidden_imports(path):
            violations.append(f"{path.relative_to(ROOT)} imports forbidden {hit}")
    if violations:
        print("Import boundary violations:")
        for item in violations:
            print(f"  - {item}")
        return 1
    print("Import boundary check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
