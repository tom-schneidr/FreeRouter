from __future__ import annotations

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
      html.embed-mode .table-wrap { max-width: 100%; }
      html.embed-mode table { min-width: 0; width: 100%; }
      html.embed-mode th, html.embed-mode td {
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        max-width: 12rem;
      }
      html.embed-mode th:last-child, html.embed-mode td:last-child,
      html.embed-mode th:nth-last-child(2), html.embed-mode td:nth-last-child(2) {
        max-width: 6rem;
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
    if "fr-embed-styles" in html:
        return html
    if "</head>" in html:
        return html.replace("</head>", f"{EMBED_HEAD_FRAGMENT}\n  </head>", 1)
    return html
