from __future__ import annotations

from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse

DOCS_THEME_STYLE = """
    <style id="fr-docs-theme">
      :root {
        color-scheme: dark;
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
      }
      html, body {
        margin: 0;
        background: var(--fr-bg);
        color: var(--fr-text);
        font-family: "Segoe UI", Inter, system-ui, sans-serif;
      }
      .swagger-ui {
        color: var(--fr-text);
      }
      .swagger-ui .topbar {
        background: var(--fr-surface);
        border-bottom: 1px solid var(--fr-line);
        padding: 10px 0;
      }
      .swagger-ui .topbar .download-url-wrapper input[type="text"],
      .swagger-ui .topbar .download-url-wrapper .select-label select {
        border: 1px solid var(--fr-line);
        background: var(--fr-bg-soft);
        color: var(--fr-text);
        border-radius: 7px;
      }
      .swagger-ui .topbar .download-url-wrapper .btn,
      .swagger-ui .btn {
        background: var(--fr-accent);
        border-color: var(--fr-accent);
        color: #fff;
        border-radius: 7px;
        box-shadow: none;
      }
      .swagger-ui .btn.cancel {
        background: var(--fr-surface-2);
        border-color: var(--fr-line);
        color: var(--fr-text);
      }
      .swagger-ui .info .title,
      .swagger-ui .info h1, .swagger-ui .info h2, .swagger-ui .info h3,
      .swagger-ui .info h4, .swagger-ui .info h5,
      .swagger-ui .info li, .swagger-ui .info p, .swagger-ui .info table,
      .swagger-ui .info a, .swagger-ui .info .base-url,
      .swagger-ui section h3, .swagger-ui .tab li,
      .swagger-ui label, .swagger-ui .parameter__name,
      .swagger-ui .parameter__type, .swagger-ui .response-col_status,
      .swagger-ui .response-col_links, .swagger-ui table thead tr td,
      .swagger-ui table thead tr th, .swagger-ui .model-title,
      .swagger-ui .prop-type, .swagger-ui .prop-format,
      .swagger-ui .model .property.primitive {
        color: var(--fr-text);
      }
      .swagger-ui .info .title small {
        background: rgba(79, 140, 255, 0.16);
        color: #bfdbfe;
      }
      .swagger-ui .info .title small.version-stamp {
        background: rgba(34, 197, 94, 0.16);
        color: #bbf7d0;
      }
      .swagger-ui .info a, .swagger-ui .info .link, .swagger-ui a {
        color: #93c5fd;
      }
      .swagger-ui .info .description, .swagger-ui .markdown p,
      .swagger-ui .markdown li, .swagger-ui .renderedMarkdown p {
        color: var(--fr-muted);
      }
      .swagger-ui .scheme-container {
        background: var(--fr-surface);
        border: 1px solid var(--fr-line);
        border-radius: 8px;
        box-shadow: none;
        margin: 0 0 18px;
        padding: 12px 16px;
      }
      .swagger-ui .scheme-container .schemes > label {
        color: var(--fr-muted);
      }
      .swagger-ui select {
        background: var(--fr-bg-soft);
        border: 1px solid var(--fr-line);
        color: var(--fr-text);
        border-radius: 7px;
      }
      .swagger-ui .opblock-tag {
        color: var(--fr-text);
        border-bottom: 1px solid var(--fr-line);
      }
      .swagger-ui .opblock-tag small {
        color: var(--fr-muted);
      }
      .swagger-ui .opblock {
        border: 1px solid var(--fr-line);
        border-radius: 8px;
        background: var(--fr-surface);
        box-shadow: none;
        margin-bottom: 10px;
      }
      .swagger-ui .opblock .opblock-summary {
        border-color: var(--fr-line);
      }
      .swagger-ui .opblock .opblock-summary-method {
        border-radius: 6px;
        font-weight: 700;
      }
      .swagger-ui .opblock.opblock-get .opblock-summary {
        border-color: rgba(79, 140, 255, 0.45);
        background: rgba(79, 140, 255, 0.08);
      }
      .swagger-ui .opblock.opblock-post .opblock-summary {
        border-color: rgba(34, 197, 94, 0.45);
        background: rgba(34, 197, 94, 0.08);
      }
      .swagger-ui .opblock.opblock-put .opblock-summary {
        border-color: rgba(245, 158, 11, 0.45);
        background: rgba(245, 158, 11, 0.08);
      }
      .swagger-ui .opblock.opblock-delete .opblock-summary {
        border-color: rgba(239, 68, 68, 0.45);
        background: rgba(239, 68, 68, 0.08);
      }
      .swagger-ui .opblock.opblock-patch .opblock-summary {
        border-color: rgba(167, 139, 250, 0.45);
        background: rgba(167, 139, 250, 0.08);
      }
      .swagger-ui .opblock-body,
      .swagger-ui .opblock-section,
      .swagger-ui .table-container,
      .swagger-ui .responses-wrapper,
      .swagger-ui .parameters-container {
        background: var(--fr-bg-soft);
      }
      .swagger-ui .opblock-description-wrapper p,
      .swagger-ui .opblock-external-docs-wrapper p,
      .swagger-ui .opblock-title_normal,
      .swagger-ui .response-col_description__inner div.markdown,
      .swagger-ui .parameter__extension, .swagger-ui .parameter__in {
        color: var(--fr-muted);
      }
      .swagger-ui table thead tr td, .swagger-ui table thead tr th {
        border-color: var(--fr-line);
        color: var(--fr-muted);
      }
      .swagger-ui table tbody tr td {
        border-color: var(--fr-line);
        color: var(--fr-text);
      }
      .swagger-ui input[type="text"], .swagger-ui input[type="password"],
      .swagger-ui input[type="search"], .swagger-ui input[type="email"],
      .swagger-ui input[type="file"], .swagger-ui textarea {
        background: var(--fr-bg-soft);
        border: 1px solid var(--fr-line);
        color: var(--fr-text);
        border-radius: 7px;
      }
      .swagger-ui .model-box, .swagger-ui .model, .swagger-ui section.models {
        background: var(--fr-surface);
      }
      .swagger-ui section.models {
        border: 1px solid var(--fr-line);
        border-radius: 8px;
      }
      .swagger-ui section.models.is-open h4 {
        border-color: var(--fr-line);
      }
      .swagger-ui .model-container {
        background: var(--fr-bg-soft);
      }
      .swagger-ui .prop-row .prop-name, .swagger-ui .prop-row .prop-format {
        color: #bfdbfe;
      }
      .swagger-ui .response-control-media-type__accept-message {
        color: var(--fr-muted);
      }
      .swagger-ui .dialog-ux .modal-ux {
        background: var(--fr-surface);
        border: 1px solid var(--fr-line);
        color: var(--fr-text);
      }
      .swagger-ui .dialog-ux .modal-ux-header {
        border-bottom: 1px solid var(--fr-line);
      }
      .swagger-ui .dialog-ux .modal-ux-content p,
      .swagger-ui .dialog-ux .modal-ux-content label,
      .swagger-ui .dialog-ux .modal-ux-content h4 {
        color: var(--fr-text);
      }
      .swagger-ui .auth-wrapper, .swagger-ui .auth-container {
        border-color: var(--fr-line);
      }
      .swagger-ui .loading-container .loading::before {
        border-color: rgba(79, 140, 255, 0.25);
        border-top-color: var(--fr-accent);
      }
      .swagger-ui .copy-to-clipboard {
        background: var(--fr-surface-2);
      }
      .swagger-ui .copy-to-clipboard button {
        background: var(--fr-surface-2);
      }
      .swagger-ui .highlight-code > .microlight,
      .swagger-ui .microlight, .swagger-ui .highlight-code {
        background: #050b15 !important;
        color: #c7d2fe !important;
        border-radius: 8px;
      }
      .swagger-ui .opblock-body pre.microlight,
      .swagger-ui .responses-inner pre.microlight,
      .swagger-ui .model-box pre {
        background: #050b15 !important;
        color: #c7d2fe !important;
      }
      .swagger-ui svg.arrow {
        fill: var(--fr-muted);
      }
      .swagger-ui .wrapper {
        padding: 0 16px 24px;
      }
    </style>
"""


def swagger_docs_html(*, openapi_url: str, title: str) -> HTMLResponse:
    response = get_swagger_ui_html(
        openapi_url=openapi_url,
        title=title,
        swagger_ui_parameters={"syntaxHighlight.theme": "agate"},
    )
    html = response.body.decode("utf-8")
    if "fr-docs-theme" not in html:
        html = html.replace("</head>", f"{DOCS_THEME_STYLE}\n</head>", 1)
    return HTMLResponse(html)
