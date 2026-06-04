from __future__ import annotations

import base64
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BRAND_DIR = PROJECT_ROOT / "apps" / "ui" / "src" / "assets" / "brand"
FAVICON_PATH = BRAND_DIR / "favicon.png"
LOGO_PATH = BRAND_DIR / "logo.png"

FAVICON_LINK = '<link rel="icon" type="image/png" href="/favicon.ico">'

NAV_BRAND_CSS = """
      .nav-brand { display: inline-flex; align-items: center; gap: 0.65rem; text-decoration: none; flex-shrink: 0; }
      .nav-brand img { display: block; height: 1.35rem; width: auto; max-width: 9rem; }
"""

NAV_BRAND_CSS_INDENT_4 = """
    .nav-brand { display: inline-flex; align-items: center; gap: 0.65rem; text-decoration: none; flex-shrink: 0; }
    .nav-brand img { display: block; height: 1.35rem; width: auto; max-width: 9rem; }
"""

SIDEBAR_BRAND_CSS = """
      .brand {
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        gap: 6px;
        padding: 20px 18px 16px;
        border-bottom: 1px solid var(--line-soft);
      }
      .brand-logo {
        display: block;
        width: min(176px, 100%);
        height: auto;
      }
      .brand-icon {
        display: none;
        width: 42px;
        height: 42px;
      }
      .brand-subtitle {
        display: block;
        color: var(--muted);
        font-size: 12px;
        padding-left: 2px;
      }
"""


def nav_brand_html(*, href: str = "/") -> str:
    return (
        f'<a class="nav-brand" href="{href}">'
        f'<img src="/brand/logo.png" alt="FreeRouter">'
        f"</a>"
    )


def sidebar_brand_html() -> str:
    return """
        <div class="brand">
          <img class="brand-icon" src="/brand/favicon.png" alt="FreeRouter">
          <img class="brand-logo" src="/brand/logo.png" alt="FreeRouter">
          <span class="brand-subtitle">Local AI gateway</span>
        </div>""".strip()


def inline_logo_html(*, width: int = 160, alt: str = "FreeRouter") -> str:
    encoded = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
    return (
        f'<img src="data:image/png;base64,{encoded}" alt="{alt}" '
        f'style="display:block;width:{width}px;max-width:100%;height:auto;margin:0 auto 16px;">'
    )


def inline_icon_html(*, size: int = 48, alt: str = "FreeRouter") -> str:
    encoded = base64.b64encode(FAVICON_PATH.read_bytes()).decode("ascii")
    return (
        f'<img src="data:image/png;base64,{encoded}" alt="{alt}" '
        f'style="display:block;width:{size}px;height:{size}px;margin:0 auto 16px;">'
    )


LEGACY_NAV_H1_CSS = (
    "      nav h1 { font-size: 1rem; font-weight: 700; background: linear-gradient(135deg, #60a5fa, #a78bfa); "
    "-webkit-background-clip: text; -webkit-text-fill-color: transparent; }\n"
)
LEGACY_NAV_H1_CSS_MULTILINE = (
    "    nav h1 { font-size: 1rem; font-weight: 700; background: linear-gradient(135deg, #60a5fa, #a78bfa);\n"
    "      -webkit-background-clip: text; -webkit-text-fill-color: transparent; }\n"
)


def inject_legacy_nav_branding(html: str) -> str:
    if FAVICON_LINK not in html:
        html = html.replace(
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n    ' + FAVICON_LINK,
            1,
        )
    html = html.replace(LEGACY_NAV_H1_CSS, NAV_BRAND_CSS)
    html = html.replace(LEGACY_NAV_H1_CSS_MULTILINE, NAV_BRAND_CSS_INDENT_4)
    html = html.replace(
        '<h1>FreeRouter</h1><span class="nav-spacer"></span>',
        nav_brand_html() + '<span class="nav-spacer"></span>',
    )
    html = html.replace(
        "<h1>FreeRouter</h1>\n      <span class=\"nav-spacer\"></span>",
        nav_brand_html() + '\n      <span class="nav-spacer"></span>',
    )
    html = html.replace(
        "    <h1>FreeRouter</h1>\n    <span class=\"model-badge\"",
        "    " + nav_brand_html() + '\n    <span class="model-badge"',
    )
    return html
