from __future__ import annotations

import re

THEME_STORAGE_KEY = "freerouter.theme"

THEME_BOOT_SCRIPT = """
    <script id="fr-theme-boot">
      (function () {
        var key = "freerouter.theme";
        var theme = "dark";
        try {
          var saved = localStorage.getItem(key);
          if (saved === "light" || saved === "dark") theme = saved;
          else if (window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches) theme = "light";
        } catch (e) {}
        document.documentElement.setAttribute("data-theme", theme);
      })();
    </script>
"""

THEME_TOGGLE_CSS = """
      .theme-toggle {
        border: 1px solid var(--border, var(--line, #24354d));
        border-radius: 7px;
        background: var(--bg-tertiary, var(--surface-2, #142238));
        color: var(--text, var(--text, #e5edf8));
        min-height: 34px;
        padding: 0 12px;
        font: inherit;
        cursor: pointer;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        white-space: nowrap;
      }
      .theme-toggle:hover {
        border-color: var(--accent, #4f8cff);
        color: var(--text, #e5edf8);
      }
      nav .theme-toggle {
        min-height: 32px;
        padding: 0 10px;
        font-size: 0.82rem;
        background: var(--bg-tertiary, #1e293b);
      }
"""

THEME_RUNTIME_SCRIPT = """
    <script id="fr-theme-runtime">
      (function () {
        var KEY = "freerouter.theme";
        function labelFor(theme) {
          return theme === "dark" ? "Light mode" : "Dark mode";
        }
        function broadcast(theme) {
          document.querySelectorAll("iframe.embed-frame").forEach(function (frame) {
            try {
              frame.contentWindow.postMessage({ type: "freerouter-theme", theme: theme }, "*");
            } catch (e) {}
          });
          if (window.parent !== window) {
            try {
              window.parent.postMessage({ type: "freerouter-theme-broadcast", theme: theme }, "*");
            } catch (e) {}
          }
        }
        function applyTheme(theme) {
          if (theme !== "light" && theme !== "dark") return;
          document.documentElement.setAttribute("data-theme", theme);
          try { localStorage.setItem(KEY, theme); } catch (e) {}
          document.querySelectorAll("[data-theme-label]").forEach(function (el) {
            el.textContent = labelFor(theme);
          });
          broadcast(theme);
        }
        window.FreeRouterTheme = {
          apply: applyTheme,
          toggle: function () {
            var current = document.documentElement.getAttribute("data-theme") || "dark";
            applyTheme(current === "dark" ? "light" : "dark");
          }
        };
        document.addEventListener("click", function (event) {
          var button = event.target.closest("[data-theme-toggle]");
          if (!button) return;
          event.preventDefault();
          window.FreeRouterTheme.toggle();
        });
        window.addEventListener("storage", function (event) {
          if (event.key === KEY && event.newValue) applyTheme(event.newValue);
        });
        window.addEventListener("message", function (event) {
          if (!event.data) return;
          if (event.data.type === "freerouter-theme") applyTheme(event.data.theme);
          if (event.data.type === "freerouter-theme-broadcast") applyTheme(event.data.theme);
        });
        document.addEventListener("DOMContentLoaded", function () {
          var theme = document.documentElement.getAttribute("data-theme") || "dark";
          document.querySelectorAll("[data-theme-label]").forEach(function (el) {
            el.textContent = labelFor(theme);
          });
        });
      })();
    </script>
"""

THEME_TOGGLE_BUTTON = (
    '<button type="button" class="theme-toggle" data-theme-toggle '
    'aria-label="Toggle color theme"><span data-theme-label>Light mode</span></button>'
)

CLASSIC_THEME_TOKENS_CSS = """
    <style id="fr-theme-tokens-classic">
      html[data-theme="dark"] {
        color-scheme: dark;
        --bg-primary: #0a0e1a;
        --bg-secondary: #111827;
        --bg-tertiary: #1e293b;
        --border: #2d3a4f;
        --text: #e2e8f0;
        --text-strong: #f8fafc;
        --text-muted: #94a3b8;
        --accent: #3b82f6;
        --accent-glow: rgba(59, 130, 246, 0.15);
        --green: #22c55e;
        --red: #ef4444;
        --amber: #f59e0b;
        --purple: #a78bfa;
        --link: #93c5fd;
        --code-bg: #0a0e1a;
        --code-text: #93c5fd;
        --ok: #22c55e;
        --warn: #f59e0b;
        --bad: #ef4444;
        --panel-shadow: rgba(0, 0, 0, 0.28);
        --row-hover: rgba(59, 130, 246, 0.08);
      }
      html[data-theme="light"] {
        color-scheme: light;
        --bg-primary: #f8fafc;
        --bg-secondary: #ffffff;
        --bg-tertiary: #f1f5f9;
        --border: #cbd5e1;
        --text: #0f172a;
        --text-strong: #020617;
        --text-muted: #64748b;
        --accent: #2563eb;
        --accent-glow: rgba(37, 99, 235, 0.12);
        --green: #15803d;
        --red: #dc2626;
        --amber: #d97706;
        --purple: #7c3aed;
        --link: #1d4ed8;
        --code-bg: #eef2ff;
        --code-text: #1e3a8a;
        --ok: #15803d;
        --warn: #b45309;
        --bad: #dc2626;
        --panel-shadow: rgba(15, 23, 42, 0.08);
        --row-hover: rgba(37, 99, 235, 0.06);
      }
    </style>
"""

DESKTOP_THEME_TOKENS_CSS = """
    <style id="fr-theme-tokens-desktop">
      html[data-theme="dark"] {
        color-scheme: dark;
        --bg: #07111f;
        --bg-soft: #0b1424;
        --surface: #101b2e;
        --surface-2: #142238;
        --sidebar-bg: #091321;
        --line: #24354d;
        --line-soft: rgba(148, 163, 184, 0.16);
        --text: #e5edf8;
        --muted: #91a4bd;
        --subtle: #667892;
        --accent: #4f8cff;
        --accent-2: #22c55e;
        --warn: #f59e0b;
        --danger: #ef4444;
        --input-bg: #08111f;
        --table-head-bg: #0d1728;
        --iframe-bg: #0a0e1a;
        --log-bg: #050b15;
        --log-text: #c7d2fe;
        --main-glow: rgba(59, 130, 246, 0.12);
        --shadow: 0 20px 60px rgba(0, 0, 0, 0.28);
        --topbar-bg: rgba(7, 17, 31, 0.86);
        --empty-bg: rgba(8, 17, 31, 0.74);
        --route-item-bg: rgba(8, 17, 31, 0.5);
        --brand-shadow: rgba(37, 99, 235, 0.3);
      }
      html[data-theme="light"] {
        color-scheme: light;
        --bg: #eef2f7;
        --bg-soft: #ffffff;
        --surface: #ffffff;
        --surface-2: #f8fafc;
        --sidebar-bg: #e2e8f0;
        --line: #cbd5e1;
        --line-soft: rgba(100, 116, 139, 0.18);
        --text: #0f172a;
        --muted: #64748b;
        --subtle: #94a3b8;
        --accent: #2563eb;
        --accent-2: #15803d;
        --warn: #b45309;
        --danger: #dc2626;
        --input-bg: #ffffff;
        --table-head-bg: #f1f5f9;
        --iframe-bg: #f8fafc;
        --log-bg: #f8fafc;
        --log-text: #334155;
        --main-glow: rgba(37, 99, 235, 0.08);
        --shadow: 0 16px 40px rgba(15, 23, 42, 0.08);
        --topbar-bg: rgba(255, 255, 255, 0.92);
        --empty-bg: rgba(248, 250, 252, 0.96);
        --route-item-bg: rgba(241, 245, 249, 0.92);
        --brand-shadow: rgba(37, 99, 235, 0.18);
      }
    </style>
"""

DOCS_THEME_TOKENS_CSS = """
    <style id="fr-theme-tokens-docs">
      html[data-theme="dark"] {
        color-scheme: dark;
        --fr-bg: #07111f;
        --fr-bg-soft: #0b1424;
        --fr-surface: #101b2e;
        --fr-surface-2: #142238;
        --fr-line: #24354d;
        --fr-text: #e5edf8;
        --fr-muted: #91a4bd;
        --fr-accent: #4f8cff;
        --fr-link: #93c5fd;
        --fr-code-bg: #050b15;
        --fr-code-text: #c7d2fe;
        --fr-info-chip: rgba(79, 140, 255, 0.16);
        --fr-info-chip-text: #bfdbfe;
        --fr-version-chip: rgba(34, 197, 94, 0.16);
        --fr-version-chip-text: #bbf7d0;
      }
      html[data-theme="light"] {
        color-scheme: light;
        --fr-bg: #f1f5f9;
        --fr-bg-soft: #ffffff;
        --fr-surface: #ffffff;
        --fr-surface-2: #f8fafc;
        --fr-line: #cbd5e1;
        --fr-text: #0f172a;
        --fr-muted: #64748b;
        --fr-accent: #2563eb;
        --fr-link: #1d4ed8;
        --fr-code-bg: #f8fafc;
        --fr-code-text: #334155;
        --fr-info-chip: rgba(37, 99, 235, 0.12);
        --fr-info-chip-text: #1d4ed8;
        --fr-version-chip: rgba(21, 128, 61, 0.12);
        --fr-version-chip-text: #15803d;
      }
    </style>
"""

ROOT_THEME_BLOCK_RE = re.compile(
    r"\s*:root\s*\{[^}]*--bg-primary[^}]*\}\s*",
    re.DOTALL,
)
DESKTOP_ROOT_BLOCK_RE = re.compile(
    r"\s*:root\s*\{[^}]*--bg:\s*#07111f[^}]*\}\s*",
    re.DOTALL,
)
DOCS_ROOT_BLOCK_RE = re.compile(
    r"\s*:root\s*\{[^}]*--fr-bg[^}]*\}\s*",
    re.DOTALL,
)


def _inject_before_head_close(html: str, fragment: str) -> str:
    if fragment.strip() in html:
        return html
    if "</head>" in html:
        return html.replace("</head>", f"{fragment}\n  </head>", 1)
    return html


def _inject_before_body_close(html: str, fragment: str) -> str:
    if fragment.strip() in html:
        return html
    if "</body>" in html:
        return html.replace("</body>", f"{fragment}\n  </body>", 1)
    return html


def with_web_page(html: str, *, embed: bool = False) -> str:
    from app.ui.embed import with_embed_support

    _ = embed
    if "fr-theme-tokens-classic" not in html:
        html = ROOT_THEME_BLOCK_RE.sub("\n", html)
        html = html.replace("color: #fff;", "color: var(--text-strong);")
        html = html.replace("color: #93c5fd", "color: var(--link)")
        html = html.replace("background: rgba(15, 23, 42, 0.75);", "background: var(--bg-tertiary);")
        html = _inject_before_head_close(
            html,
            CLASSIC_THEME_TOKENS_CSS + THEME_TOGGLE_CSS + THEME_BOOT_SCRIPT,
        )
        if '<nav' in html and 'data-theme-toggle' not in html:
            html = html.replace(
                '<span class="nav-spacer"></span>',
                f'<span class="nav-spacer"></span>{THEME_TOGGLE_BUTTON}',
                1,
            )
        html = _inject_before_body_close(html, THEME_RUNTIME_SCRIPT)
    html = with_embed_support(html)
    return html


def inject_desktop_shell_theme(html: str) -> str:
    if "fr-theme-tokens-desktop" in html:
        return html
    html = DESKTOP_ROOT_BLOCK_RE.sub("\n", html)
    html = html.replace("background: #091321;", "background: var(--sidebar-bg);")
    html = html.replace("background: radial-gradient(circle at 76% -10%, rgba(59,130,246,.12), transparent 34%), var(--bg);",
                        "background: radial-gradient(circle at 76% -10%, var(--main-glow), transparent 34%), var(--bg);")
    html = html.replace("background: rgba(7, 17, 31, .86);", "background: var(--topbar-bg);")
    html = html.replace("background: #08111f;", "background: var(--input-bg);")
    html = html.replace("background: #0d1728;", "background: var(--table-head-bg);")
    html = html.replace("background: #0a0e1a;", "background: var(--iframe-bg);")
    html = html.replace("background: #050b15;", "background: var(--log-bg);")
    html = html.replace("color: #c7d2fe;", "color: var(--log-text);")
    html = html.replace("color: #d9e4f2;", "color: var(--text);")
    html = html.replace("background: rgba(8,17,31,.74);", "background: var(--empty-bg);")
    html = html.replace("background: rgba(8,17,31,.5);", "background: var(--route-item-bg);")
    html = html.replace("box-shadow: 0 10px 28px rgba(37,99,235,.3);", "box-shadow: 0 10px 28px var(--brand-shadow);")
    html = html.replace("box-shadow: var(--shadow);", "box-shadow: var(--shadow);")
    html = _inject_before_head_close(
        html,
        DESKTOP_THEME_TOKENS_CSS + THEME_TOGGLE_CSS + THEME_BOOT_SCRIPT,
    )
    html = html.replace(
        '<button id="openDocs" type="button">API Docs</button>',
        f'{THEME_TOGGLE_BUTTON}\n            <button id="openDocs" type="button">API Docs</button>',
        1,
    )
    html = _inject_before_body_close(html, THEME_RUNTIME_SCRIPT)
    return html


def inject_docs_theme(html: str) -> str:
    html = DOCS_ROOT_BLOCK_RE.sub("\n", html)
    html = html.replace("color: #bfdbfe;", "color: var(--fr-info-chip-text);")
    html = html.replace("color: #bbf7d0;", "color: var(--fr-version-chip-text);")
    html = html.replace("color: #93c5fd;", "color: var(--fr-link);")
    html = html.replace("background: rgba(79, 140, 255, 0.16);", "background: var(--fr-info-chip);")
    html = html.replace("background: rgba(34, 197, 94, 0.16);", "background: var(--fr-version-chip);")
    html = html.replace("background: #050b15 !important;", "background: var(--fr-code-bg) !important;")
    html = html.replace("color: #c7d2fe !important;", "color: var(--fr-code-text) !important;")
    html = _inject_before_head_close(html, DOCS_THEME_TOKENS_CSS + THEME_BOOT_SCRIPT)
    html = _inject_before_body_close(html, THEME_RUNTIME_SCRIPT)
    return html
