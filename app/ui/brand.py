from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BRAND_DIR = PROJECT_ROOT / "apps" / "ui" / "src" / "assets" / "brand"
FAVICON_PATH = BRAND_DIR / "favicon.png"
LOGO_PATH = BRAND_DIR / "logo.png"

FAVICON_LINK = '<link rel="icon" type="image/png" href="/favicon.ico">'
