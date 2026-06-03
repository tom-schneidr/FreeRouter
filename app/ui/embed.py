from __future__ import annotations

EMBED_HEAD_FRAGMENT = """
    <style id="fr-embed-styles">
      html.embed-mode nav { display: none !important; }
      html.embed-mode body { margin: 0; overflow: auto; }
      html.embed-mode main { max-width: none; padding: 1rem 1.25rem; }
      html.embed-mode .app { height: calc(100vh - 82px); overflow: hidden; }
      html.embed-mode .route-panel { height: calc(100vh - 132px); }
      html.embed-mode .chat-panel { min-height: 0; }
    </style>
    <script>
      (function () {
        if (new URLSearchParams(location.search).get('embed') !== '1') return;
        document.documentElement.classList.add('embed-mode');
      })();
    </script>
"""


def with_embed_support(html: str) -> str:
    if "fr-embed-styles" in html:
        return html
    if "</head>" in html:
        return html.replace("</head>", f"{EMBED_HEAD_FRAGMENT}\n  </head>", 1)
    return html
