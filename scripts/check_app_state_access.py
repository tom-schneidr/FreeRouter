#!/usr/bin/env python3
"""Disallow new direct request.app.state.* access outside AppServices."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERN = re.compile(r"request\.app\.state\.")
ALLOWLIST = {
    ROOT / "app" / "app_services.py",
}


def main() -> int:
    violations: list[str] = []
    for path in (ROOT / "app").rglob("*.py"):
        if path in ALLOWLIST:
            continue
        text = path.read_text(encoding="utf-8")
        if PATTERN.search(text):
            violations.append(path.relative_to(ROOT).as_posix())
    if violations:
        print("Direct request.app.state access found:")
        for item in violations:
            print(f"  - {item}")
        return 1
    print("App state access check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
