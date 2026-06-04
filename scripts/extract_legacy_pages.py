"""Extract legacy embed HTML pages from git into app/legacy_pages/."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "app" / "legacy_pages"
COMMIT = "f9e533a"


def git_show(path: str) -> str:
    raw = subprocess.check_output(["git", "show", f"{COMMIT}:{path}"])
    return raw.decode("utf-8")


def main() -> None:
    main_py = git_show("app/main.py")
    embed_py = git_show("app/ui/embed.py")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "__init__.py").write_text('"""Legacy HTML pages served with ?embed=1 for React iframes."""\n', encoding="utf-8")
    (ROOT / "app" / "ui" / "embed.py").write_text(embed_py, encoding="utf-8")

    health_m = re.search(
        r'^ROUTE_HEALTH_HTML = inject_legacy_nav_branding\("""(.*?)"""\)\s*$',
        main_py,
        re.M | re.S,
    )
    live_m = re.search(
        r'^LIVE_API_HTML = inject_legacy_nav_branding\(r"""(.*?)"""\)\s*$',
        main_py,
        re.M | re.S,
    )
    usage_m = re.search(
        r"async def provider_status_page.*?return HTMLResponse\(with_embed_support\(inject_legacy_nav_branding\(r\"\"\"(.*?)\"\"\"\)\)\)",
        main_py,
        re.S,
    )
    if not health_m or not live_m or not usage_m:
        raise SystemExit("Could not extract one or more legacy HTML blocks")

    (OUT / "health_page.py").write_text(
        'from app.ui.brand import inject_legacy_nav_branding\n'
        'from app.ui.embed import with_embed_support\n\n'
        "ROUTE_HEALTH_HTML = inject_legacy_nav_branding(" + repr(health_m.group(1)) + ")\n",
        encoding="utf-8",
    )
    (OUT / "live_page.py").write_text(
        'from app.ui.brand import inject_legacy_nav_branding\n'
        'from app.ui.embed import with_embed_support\n\n'
        "LIVE_API_HTML = inject_legacy_nav_branding(" + repr(live_m.group(1)) + ")\n",
        encoding="utf-8",
    )
    (OUT / "usage_page.py").write_text(
        'from app.ui.brand import inject_legacy_nav_branding\n'
        'from app.ui.embed import with_embed_support\n\n'
        "USAGE_STATS_HTML = inject_legacy_nav_branding(" + repr(usage_m.group(1)) + ")\n",
        encoding="utf-8",
    )
    print(f"Wrote legacy pages to {OUT}")


if __name__ == "__main__":
    main()
