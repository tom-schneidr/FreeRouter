from __future__ import annotations

THEME_STORAGE_KEY = "freerouter.theme"

THEME_TOGGLE_BUTTON = """
          <button class="fr-theme-toggle" type="button" data-fr-theme-toggle aria-label="Switch theme">
            <span class="fr-theme-track" aria-hidden="true"><span class="fr-theme-knob"></span></span>
            <span class="fr-theme-label">Theme</span>
          </button>
"""

THEME_FLOATING_BUTTON = f"""
    <div class="fr-theme-floating" aria-label="Theme controls">
{THEME_TOGGLE_BUTTON.rstrip()}
    </div>
"""

THEME_HEAD_FRAGMENT = f"""
    <script id="fr-theme-boot">
      (function () {{
        var key = "{THEME_STORAGE_KEY}";
        var preference = "system";
        var theme = "dark";
        try {{
          preference = localStorage.getItem(key) || "system";
          theme = preference === "light" || preference === "dark"
            ? preference
            : (window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
        }} catch (_) {{}}
        theme = theme === "light" ? "light" : "dark";
        document.documentElement.dataset.theme = theme;
        document.documentElement.dataset.themePreference = preference === "light" || preference === "dark" ? preference : "system";
        document.documentElement.style.colorScheme = theme;
      }})();
    </script>
    <style id="fr-theme-styles">
      :root,
      html[data-theme="dark"] {{
        color-scheme: dark;
        --bg-primary: #0a0e1a;
        --bg-secondary: #111827;
        --bg-tertiary: #1e293b;
        --border: #2d3a4f;
        --text: #e2e8f0;
        --text-muted: #94a3b8;
        --accent: #3b82f6;
        --accent-glow: rgba(59, 130, 246, 0.15);
        --green: #22c55e;
        --red: #ef4444;
        --amber: #f59e0b;
        --purple: #a78bfa;
        --ok: #22c55e;
        --warn: #f59e0b;
        --bad: #ef4444;

        --bg: #07111f;
        --bg-soft: #0b1424;
        --surface: #101b2e;
        --surface-2: #142238;
        --line: #24354d;
        --line-soft: rgba(148, 163, 184, 0.16);
        --muted: #91a4bd;
        --subtle: #667892;
        --accent-2: #22c55e;
        --danger: #ef4444;
        --shadow: 0 20px 60px rgba(0, 0, 0, 0.28);

        --fr-bg: #07111f;
        --fr-bg-soft: #0b1424;
        --fr-surface: #101b2e;
        --fr-surface-2: #142238;
        --fr-line: #24354d;
        --fr-text: #e5edf8;
        --fr-muted: #91a4bd;
        --fr-accent: #4f8cff;
        --fr-ok: #22c55e;
        --fr-warn: #f59e0b;
        --fr-danger: #ef4444;

        --heading: #f8fafc;
        --link: #93c5fd;
        --link-strong: #bfdbfe;
        --on-accent: #ffffff;
        --input-bg: #08111f;
        --sidebar-bg: #091321;
        --table-head-bg: rgba(15, 23, 42, 0.75);
        --table-head-solid: #0d1728;
        --table-row-hover: rgba(59, 130, 246, 0.08);
        --surface-alpha: rgba(16, 27, 46, 0.9);
        --empty-bg: rgba(8, 17, 31, 0.74);
        --code-bg: #050b15;
        --code-inline-bg: var(--bg-tertiary);
        --code-text: #c7d2fe;
        --success-text: #bbf7d0;
        --success-bg: rgba(34, 197, 94, 0.12);
        --success-border: rgba(34, 197, 94, 0.45);
        --warning-text: #fcd34d;
        --warning-bg: rgba(245, 158, 11, 0.12);
        --warning-border: rgba(245, 158, 11, 0.5);
        --danger-text: #fecaca;
        --danger-bg: rgba(239, 68, 68, 0.12);
        --danger-border: rgba(239, 68, 68, 0.5);
        --purple-text: #ddd6fe;
        --purple-bg: rgba(167, 139, 250, 0.14);
        --purple-border: rgba(167, 139, 250, 0.5);
        --modal-backdrop: rgba(2, 6, 23, 0.72);
        --button-hover-bg: #1a2c47;
        --button-hover-border: #3a5578;
      }}

      html[data-theme="light"] {{
        color-scheme: light;
        --bg-primary: #f6f8fb;
        --bg-secondary: #ffffff;
        --bg-tertiary: #edf3fa;
        --border: #c9d6e8;
        --text: #142033;
        --text-muted: #5d7089;
        --accent: #2563eb;
        --accent-glow: rgba(37, 99, 235, 0.1);
        --green: #15803d;
        --red: #dc2626;
        --amber: #b45309;
        --purple: #6d28d9;
        --ok: #15803d;
        --warn: #b45309;
        --bad: #dc2626;

        --bg: #f6f8fb;
        --bg-soft: #edf3fa;
        --surface: #ffffff;
        --surface-2: #edf3fa;
        --line: #c9d6e8;
        --line-soft: rgba(15, 23, 42, 0.08);
        --muted: #5d7089;
        --subtle: #7b8da5;
        --accent-2: #16a34a;
        --warn: #b45309;
        --danger: #dc2626;
        --shadow: 0 14px 34px rgba(15, 23, 42, 0.1);

        --fr-bg: #f6f8fb;
        --fr-bg-soft: #edf3fa;
        --fr-surface: #ffffff;
        --fr-surface-2: #edf3fa;
        --fr-line: #c9d6e8;
        --fr-text: #142033;
        --fr-muted: #5d7089;
        --fr-accent: #2563eb;
        --fr-ok: #15803d;
        --fr-warn: #b45309;
        --fr-danger: #dc2626;

        --heading: #0f172a;
        --link: #1d4ed8;
        --link-strong: #1e40af;
        --on-accent: #ffffff;
        --input-bg: #ffffff;
        --sidebar-bg: #f2f6fb;
        --table-head-bg: #edf3fa;
        --table-head-solid: #edf3fa;
        --table-row-hover: rgba(37, 99, 235, 0.08);
        --surface-alpha: rgba(255, 255, 255, 0.92);
        --empty-bg: rgba(255, 255, 255, 0.74);
        --code-bg: #eef3fb;
        --code-inline-bg: #eef3fb;
        --code-text: #1e40af;
        --success-text: #166534;
        --success-bg: rgba(22, 163, 74, 0.1);
        --success-border: rgba(22, 101, 52, 0.28);
        --warning-text: #92400e;
        --warning-bg: rgba(217, 119, 6, 0.12);
        --warning-border: rgba(180, 83, 9, 0.28);
        --danger-text: #b91c1c;
        --danger-bg: rgba(220, 38, 38, 0.1);
        --danger-border: rgba(185, 28, 28, 0.28);
        --purple-text: #6d28d9;
        --purple-bg: rgba(124, 58, 237, 0.1);
        --purple-border: rgba(109, 40, 217, 0.28);
        --modal-backdrop: rgba(15, 23, 42, 0.36);
        --button-hover-bg: #dbe7f7;
        --button-hover-border: #9db5d3;
      }}

      html {{
        background: var(--bg-primary, var(--bg));
      }}
      body {{
        background: var(--bg-primary, var(--bg));
        color: var(--text);
      }}
      a {{
        color: var(--link);
      }}
      h1, h2, h3, h4, h5, h6, strong {{
        color: inherit;
      }}
      input, textarea, select {{
        background: var(--input-bg);
        color: var(--text);
      }}
      code {{
        color: var(--code-text);
      }}
      th {{
        background: var(--table-head-bg);
      }}
      tr:hover {{
        background: var(--table-row-hover);
      }}
      ::selection {{
        background: var(--accent-glow);
        color: var(--text);
      }}
      .fr-theme-toggle {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 0.45rem;
        min-height: 2rem;
        padding: 0.2rem 0.6rem;
        border: 1px solid var(--border, var(--line));
        border-radius: 999px;
        background: var(--bg-tertiary, var(--surface-2));
        color: var(--text);
        font: inherit;
        font-size: 0.8rem;
        font-weight: 600;
        line-height: 1;
        white-space: nowrap;
        cursor: pointer;
        box-shadow: none;
      }}
      .fr-theme-toggle:hover {{
        border-color: var(--accent);
        background: var(--accent-glow);
        color: var(--text);
        transform: none;
      }}
      .fr-theme-choice {{
        display: grid;
        gap: 0.75rem;
      }}
      .fr-theme-choice-head {{
        display: flex;
        align-items: start;
        justify-content: space-between;
        gap: 1rem;
      }}
      .fr-theme-choice-title {{
        display: grid;
        gap: 0.25rem;
      }}
      .fr-theme-choice-title strong {{
        font-size: 0.95rem;
      }}
      .fr-theme-choice-title span {{
        color: var(--muted, var(--text-muted));
        font-size: 0.82rem;
        line-height: 1.45;
      }}
      .fr-theme-segmented {{
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.35rem;
        padding: 0.3rem;
        border: 1px solid var(--border, var(--line));
        border-radius: 9px;
        background: var(--bg-primary, var(--bg));
      }}
      .fr-theme-option {{
        min-height: 2.35rem;
        border: 1px solid transparent;
        border-radius: 7px;
        background: transparent;
        color: var(--muted, var(--text-muted));
        font: inherit;
        font-size: 0.84rem;
        font-weight: 650;
        cursor: pointer;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 0.4rem;
        padding: 0 0.65rem;
      }}
      .fr-theme-option:hover {{
        color: var(--text);
        background: var(--accent-glow);
      }}
      .fr-theme-option.active {{
        border-color: rgba(79, 140, 255, 0.42);
        background: var(--surface-2, var(--bg-tertiary));
        color: var(--text);
        box-shadow: var(--shadow);
      }}
      .fr-theme-option-dot {{
        width: 0.72rem;
        height: 0.72rem;
        border-radius: 999px;
        border: 1px solid currentColor;
        flex: 0 0 auto;
      }}
      .fr-theme-option[data-fr-theme-option="system"] .fr-theme-option-dot {{
        background: linear-gradient(90deg, var(--code-bg) 0 50%, #f8fafc 50% 100%);
      }}
      .fr-theme-option[data-fr-theme-option="light"] .fr-theme-option-dot {{
        background: #f8fafc;
      }}
      .fr-theme-option[data-fr-theme-option="dark"] .fr-theme-option-dot {{
        background: #0a0e1a;
      }}
      .fr-theme-track {{
        position: relative;
        width: 2rem;
        height: 1.05rem;
        border-radius: 999px;
        background: var(--bg-primary, var(--bg));
        border: 1px solid var(--border, var(--line));
        flex: 0 0 auto;
      }}
      html[data-theme-preference="system"] .fr-theme-track::after {{
        content: "";
        position: absolute;
        inset: 3px 9px 3px 9px;
        border-left: 1px solid var(--muted, var(--text-muted));
        border-right: 1px solid var(--muted, var(--text-muted));
        opacity: 0.65;
      }}
      .fr-theme-knob {{
        position: absolute;
        top: 2px;
        left: 2px;
        width: calc(1.05rem - 6px);
        height: calc(1.05rem - 6px);
        border-radius: 50%;
        background: var(--accent);
        transition: transform 0.18s ease, background 0.18s ease;
      }}
      html[data-theme="light"] .fr-theme-knob {{
        transform: translateX(0.92rem);
        background: var(--amber);
      }}
      .fr-theme-floating {{
        position: fixed;
        top: 0.75rem;
        right: 1rem;
        z-index: 1000;
      }}
      html.embed-mode .fr-theme-floating,
      html.embed-mode .fr-theme-toggle {{
        display: none !important;
      }}
      @media (max-width: 720px) {{
        .fr-theme-toggle {{
          padding-inline: 0.45rem;
        }}
        .fr-theme-label {{
          display: none;
        }}
      }}
    </style>
    <script id="fr-theme-script">
      (function () {{
        var key = "{THEME_STORAGE_KEY}";
        var media = null;
        try {{
          media = window.matchMedia ? window.matchMedia("(prefers-color-scheme: light)") : null;
        }} catch (_) {{}}
        function normalizePreference(preference) {{
          return preference === "light" || preference === "dark" ? preference : "system";
        }}
        function resolveTheme(preference) {{
          preference = normalizePreference(preference);
          if (preference === "light" || preference === "dark") return preference;
          return media && media.matches ? "light" : "dark";
        }}
        function storedPreference() {{
          try {{ return localStorage.getItem(key); }} catch (_) {{ return null; }}
        }}
        function currentPreference() {{
          return normalizePreference(document.documentElement.dataset.themePreference || storedPreference() || "system");
        }}
        function currentTheme() {{
          return resolveTheme(currentPreference());
        }}
        function persist(preference) {{
          try {{ localStorage.setItem(key, preference); }} catch (_) {{}}
        }}
        function nextPreference(preference) {{
          if (preference === "system") return "light";
          if (preference === "light") return "dark";
          return "system";
        }}
        function updateToggles(preference, theme) {{
          var next = nextPreference(preference);
          document.querySelectorAll("[data-fr-theme-toggle]").forEach(function (button) {{
            button.setAttribute("aria-label", "Theme: " + preference + ". Switch to " + next + " mode.");
            button.setAttribute("title", "Theme: " + preference + ". Click for " + next + ".");
            var label = button.querySelector(".fr-theme-label");
            if (label) {{
              label.textContent = preference === "system" ? "System" : (theme === "dark" ? "Dark" : "Light");
            }}
          }});
          document.querySelectorAll("[data-fr-theme-option]").forEach(function (button) {{
            var active = button.getAttribute("data-fr-theme-option") === preference;
            button.classList.toggle("active", active);
            button.setAttribute("aria-pressed", active ? "true" : "false");
          }});
        }}
        function postToFrames(preference, theme) {{
          document.querySelectorAll("iframe").forEach(function (frame) {{
            try {{
              if (frame.contentWindow) {{
                frame.contentWindow.postMessage({{
                  source: "freerouter",
                  type: "theme",
                  preference: preference,
                  theme: theme
                }}, "*");
              }}
            }} catch (_) {{}}
          }});
        }}
        function apply(preference, options) {{
          options = options || {{}};
          preference = normalizePreference(preference);
          var theme = resolveTheme(preference);
          document.documentElement.dataset.theme = theme;
          document.documentElement.dataset.themePreference = preference;
          document.documentElement.style.colorScheme = theme;
          updateToggles(preference, theme);
          if (options.persist !== false) persist(preference);
          if (options.broadcast) {{
            postToFrames(preference, theme);
            try {{
              if (window.parent && window.parent !== window) {{
                window.parent.postMessage({{
                  source: "freerouter",
                  type: "theme",
                  preference: preference,
                  theme: theme
                }}, "*");
              }}
            }} catch (_) {{}}
          }}
        }}
        window.FreeRouterTheme = {{
          getTheme: currentTheme,
          getPreference: currentPreference,
          setTheme: function (preference, options) {{ apply(preference, options || {{ persist: true, broadcast: true }}); }},
          toggle: function () {{
            apply(nextPreference(currentPreference()), {{ persist: true, broadcast: true }});
          }}
        }};
        window.addEventListener("message", function (event) {{
          var data = event.data || {{}};
          if (data.source === "freerouter" && data.type === "theme") {{
            apply(data.preference || data.theme, {{ persist: true, broadcast: false }});
          }}
        }});
        window.addEventListener("storage", function (event) {{
          if (event.key === key && event.newValue) {{
            apply(event.newValue, {{ persist: false, broadcast: true }});
          }}
        }});
        if (media) {{
          var onSystemThemeChange = function () {{
            if (currentPreference() === "system") {{
              apply("system", {{ persist: false, broadcast: true }});
            }}
          }};
          if (media.addEventListener) media.addEventListener("change", onSystemThemeChange);
          else if (media.addListener) media.addListener(onSystemThemeChange);
        }}
        document.addEventListener("DOMContentLoaded", function () {{
          document.querySelectorAll("[data-fr-theme-toggle]").forEach(function (button) {{
            button.addEventListener("click", function () {{ window.FreeRouterTheme.toggle(); }});
          }});
          document.querySelectorAll("[data-fr-theme-option]").forEach(function (button) {{
            button.addEventListener("click", function () {{
              window.FreeRouterTheme.setTheme(button.getAttribute("data-fr-theme-option"));
            }});
          }});
          document.querySelectorAll("iframe").forEach(function (frame) {{
            frame.addEventListener("load", function () {{ postToFrames(currentPreference(), currentTheme()); }});
          }});
          apply(currentPreference(), {{ persist: false, broadcast: true }});
        }});
      }})();
    </script>
"""

_THEME_CONTROL_STYLE_MARKER = "      .fr-theme-toggle {"
_THEME_SYNC_SCRIPT = f"""
    <script id="fr-theme-script">
      (function () {{
        var key = "{THEME_STORAGE_KEY}";
        var media = null;
        try {{
          media = window.matchMedia ? window.matchMedia("(prefers-color-scheme: light)") : null;
        }} catch (_) {{}}
        function normalizePreference(preference) {{
          return preference === "light" || preference === "dark" ? preference : "system";
        }}
        function resolveTheme(preference) {{
          preference = normalizePreference(preference);
          if (preference === "light" || preference === "dark") return preference;
          return media && media.matches ? "light" : "dark";
        }}
        function storedPreference() {{
          try {{ return localStorage.getItem(key); }} catch (_) {{ return null; }}
        }}
        function currentPreference() {{
          return normalizePreference(document.documentElement.dataset.themePreference || storedPreference() || "system");
        }}
        function persist(preference) {{
          try {{ localStorage.setItem(key, preference); }} catch (_) {{}}
        }}
        function apply(preference, options) {{
          options = options || {{}};
          preference = normalizePreference(preference);
          var theme = resolveTheme(preference);
          document.documentElement.dataset.theme = theme;
          document.documentElement.dataset.themePreference = preference;
          document.documentElement.style.colorScheme = theme;
          if (options.persist !== false) persist(preference);
        }}
        window.FreeRouterTheme = {{
          getTheme: function () {{ return resolveTheme(currentPreference()); }},
          getPreference: currentPreference,
          setTheme: function (preference, options) {{ apply(preference, options || {{ persist: true, broadcast: false }}); }}
        }};
        window.addEventListener("message", function (event) {{
          var data = event.data || {{}};
          if (data.source === "freerouter" && data.type === "theme") {{
            apply(data.preference || data.theme, {{ persist: true, broadcast: false }});
          }}
        }});
        window.addEventListener("storage", function (event) {{
          if (event.key === key && event.newValue) {{
            apply(event.newValue, {{ persist: false, broadcast: false }});
          }}
        }});
        if (media) {{
          var onSystemThemeChange = function () {{
            if (currentPreference() === "system") {{
              apply("system", {{ persist: false, broadcast: false }});
            }}
          }};
          if (media.addEventListener) media.addEventListener("change", onSystemThemeChange);
          else if (media.addListener) media.addListener(onSystemThemeChange);
        }}
        document.addEventListener("DOMContentLoaded", function () {{
          apply(currentPreference(), {{ persist: false, broadcast: false }});
        }});
      }})();
    </script>
"""

THEME_SYNC_HEAD_FRAGMENT = (
    THEME_HEAD_FRAGMENT[: THEME_HEAD_FRAGMENT.index(_THEME_CONTROL_STYLE_MARKER)]
    + "    </style>\n"
    + _THEME_SYNC_SCRIPT
)


def with_theme_sync(html: str) -> str:
    """Inject shared theme preference sync without any in-page theme controls."""

    if "fr-theme-styles" not in html and "</head>" in html:
        html = html.replace("</head>", f"{THEME_SYNC_HEAD_FRAGMENT}\n  </head>", 1)
    return html


def with_theme_support(html: str, *, nav: bool = True, floating: bool = False, controls: bool = True) -> str:
    """Inject shared FreeRouter theme support into an HTML document."""

    has_toggle = '<button class="fr-theme-toggle"' in html
    if "fr-theme-styles" not in html and "</head>" in html:
        fragment = THEME_HEAD_FRAGMENT if controls else THEME_SYNC_HEAD_FRAGMENT
        html = html.replace("</head>", f"{fragment}\n  </head>", 1)
    if not controls:
        return html
    if nav and not has_toggle and "</nav>" in html:
        html = html.replace("</nav>", f"{THEME_TOGGLE_BUTTON.rstrip()}\n        </nav>", 1)
        has_toggle = True
    if floating and not has_toggle:
        if "</body>" in html:
            html = html.replace("</body>", f"{THEME_FLOATING_BUTTON.rstrip()}\n  </body>", 1)
        elif "<body>" in html:
            html = html.replace("<body>", f"<body>\n{THEME_FLOATING_BUTTON.rstrip()}", 1)
    return html
