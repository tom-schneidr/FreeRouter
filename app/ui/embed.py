from __future__ import annotations

from app.ui.theme import with_theme_support

EMBED_HEAD_FRAGMENT = """
    <style id="fr-embed-styles">
      html.embed-mode, html.embed-mode body {
        height: var(--embed-height, 100%);
        max-height: var(--embed-height, 100%);
        margin: 0;
        overflow: hidden;
      }
      html.embed-mode body {
        display: flex;
        flex-direction: column;
      }
      html.embed-mode nav { display: none !important; }
      html.embed-mode main {
        flex: 1;
        min-height: 0;
        overflow: auto;
        max-width: none;
        margin: 0;
        padding: 0.75rem 1rem;
      }
      html.embed-mode .app {
        flex: 1;
        min-height: 0;
        height: auto;
        display: flex;
        overflow: hidden;
      }
      html.embed-mode .chat-panel { min-height: 0; }
      html.embed-mode .route-panel {
        height: auto;
        max-height: none;
        min-height: 0;
      }
      html.embed-mode main {
        overflow-x: hidden;
      }
      html.embed-mode .table-wrap:not(.usage-wrap) {
        max-width: 100%;
      }
      html.embed-mode table:not(.usage-table) {
        min-width: 0;
        width: 100%;
      }
      html.embed-mode table:not(.usage-table) th,
      html.embed-mode table:not(.usage-table) td {
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        max-width: 12rem;
      }
    </style>
    <script>
      (function () {
        if (new URLSearchParams(location.search).get('embed') !== '1') return;
        document.documentElement.classList.add('embed-mode');
        function syncEmbedHeight() {
          document.documentElement.style.setProperty('--embed-height', window.innerHeight + 'px');
        }
        syncEmbedHeight();
        window.addEventListener('resize', syncEmbedHeight);
      })();
    </script>
"""


def with_embed_support(html: str) -> str:
    html = with_theme_support(html)
    if "fr-embed-styles" in html:
        return html
    if "</head>" in html:
        return html.replace("</head>", f"{EMBED_HEAD_FRAGMENT}\n  </head>", 1)
    return html
