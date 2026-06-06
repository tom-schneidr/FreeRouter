#!/usr/bin/env python3
"""Report module sizes; warn when soft targets are exceeded."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = {
    "app/api": 400,
    "app": 600,
}


def line_count(path: Path) -> int:
    return sum(1 for _ in path.open(encoding="utf-8"))


def main() -> int:
    warnings: list[str] = []
    for path in sorted((ROOT / "app").rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        lines = line_count(path)
        if rel.startswith("app/api/") and lines > TARGETS["app/api"]:
            warnings.append(f"{rel}: {lines} lines (api target {TARGETS['app/api']})")
        elif rel.startswith("app/") and not rel.startswith("app/api/") and lines > TARGETS["app"]:
            if rel in {"app/model_catalog.py", "app/state.py", "app/endpoint_diagnosis.py"}:
                continue  # documented baseline debt
            warnings.append(f"{rel}: {lines} lines (core target {TARGETS['app']})")
    if warnings:
        print("Module size warnings:")
        for item in warnings:
            print(f"  - {item}")
    else:
        print("Module size check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
