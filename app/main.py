from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import asdict
from time import time
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

from app.chat_page import CHAT_HTML
from app.model_catalog import ModelCatalog
from app.providers import PROVIDER_QUOTAS, ProviderError, ProviderRateLimited, build_provider_adapters
from app.router import NoProviderAvailable, WaterfallRouter, validate_chat_completion_payload
from app.settings import get_settings
from app.state import StateManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    state = StateManager(settings.database_path, PROVIDER_QUOTAS)
    await state.initialize()
    model_catalog = ModelCatalog(settings.model_catalog_path)
    model_catalog.initialize()

    app.state.gateway_state = state
    app.state.model_catalog = model_catalog
    app.state.waterfall_router = WaterfallRouter(
        build_provider_adapters(settings),
        model_catalog,
        state,
        request_timeout_seconds=settings.request_timeout_seconds,
    )
    yield


app = FastAPI(
    title="FreeRouter",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return """
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>FreeRouter - API Gateway</title>
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
        <style>
          *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
          :root {
            --bg-primary: #0a0e1a; --bg-secondary: #111827; --bg-tertiary: #1e293b;
            --border: #2d3a4f; --text: #e2e8f0; --text-muted: #94a3b8;
            --accent: #3b82f6; --accent-glow: rgba(59,130,246,0.15);
            --font: 'Inter', system-ui, sans-serif;
          }
          html, body { height: 100%; }
          body { font-family: var(--font); background: var(--bg-primary); color: var(--text); display: flex; flex-direction: column; }
          nav { display: flex; align-items: center; gap: 1rem; padding: 0.75rem 1.5rem; background: var(--bg-secondary); border-bottom: 1px solid var(--border); flex-shrink: 0; }
          nav h1 { font-size: 1rem; font-weight: 700; background: linear-gradient(135deg, #60a5fa, #a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
          nav a { color: var(--text-muted); text-decoration: none; font-size: 0.85rem; transition: color 0.2s; }
          nav a:hover { color: var(--text); }
          .nav-spacer { flex: 1; }
          main { max-width: 800px; margin: 4rem auto; padding: 2rem; background: var(--bg-secondary); border: 1px solid var(--border); border-radius: 12px; }
          h2 { font-size: 1.5rem; font-weight: 600; margin-bottom: 1.5rem; color: #fff; }
          p { color: var(--text-muted); line-height: 1.6; margin-bottom: 1.5rem; font-size: 0.95rem; }
          code { background: var(--bg-primary); border: 1px solid var(--border); padding: 0.4rem 0.6rem; border-radius: 6px; font-family: monospace; color: #93c5fd; }
          .links { display: grid; gap: 0.75rem; }
          .link-card { display: flex; align-items: center; gap: 1rem; padding: 1rem 1.25rem; background: var(--bg-tertiary); border: 1px solid var(--border); border-radius: 8px; text-decoration: none; color: var(--text); transition: all 0.2s; }
          .link-card:hover { border-color: var(--accent); background: var(--accent-glow); transform: translateY(-1px); }
          .link-icon { font-size: 1.25rem; }
        </style>
      </head>
      <body>
        <nav>
          <h1>FreeRouter</h1>
          <span class="nav-spacer"></span>
          <a href="/">Home</a>
          <a href="/chat">Chat</a>
          <a href="/models">Models</a>
        </nav>
        <main>
          <h2>Welcome to FreeRouter</h2>
          <p>The gateway is active. Use the following as your OpenAI-compatible base URL in any client:</p>
          <p><code>http://127.0.0.1:8000/v1</code></p>
          <div class="links">
            <a href="/chat" class="link-card"><span class="link-icon">💬</span> Chat Playground</a>
            <a href="/models" class="link-card"><span class="link-icon">📊</span> Model Catalog & Ranking</a>
            <a href="/docs" class="link-card"><span class="link-icon">📖</span> API Documentation</a>
            <a href="/v1/providers/status" class="link-card"><span class="link-icon">📈</span> Provider Quota Status</a>
          </div>
        </main>
      </body>
    </html>
    """


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Response:
    return Response(status_code=204)


@app.get("/chat", response_class=HTMLResponse)
async def chat_page() -> str:
    return CHAT_HTML


@app.get("/models", response_class=HTMLResponse)
async def model_catalog_page() -> str:
    return MODEL_CATALOG_HTML


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/models")
async def models(request: Request) -> dict[str, Any]:
    settings = get_settings()
    catalog: ModelCatalog = request.app.state.model_catalog
    created = int(time())
    return {
        "object": "list",
        "data": [
            {
                "id": settings.gateway_model_name,
                "object": "model",
                "created": created,
                "owned_by": "freerouter",
            }
        ]
        + [
            {
                "id": route.route_id,
                "object": "model",
                "created": created,
                "owned_by": route.provider_name,
            }
            for route in catalog.enabled_routes()
        ],
    }


@app.get("/v1/gateway/models")
async def gateway_models(request: Request) -> dict[str, Any]:
    catalog: ModelCatalog = request.app.state.model_catalog
    return catalog.to_payload()


@app.put("/v1/gateway/models")
async def update_gateway_models(request: Request) -> dict[str, Any]:
    catalog: ModelCatalog = request.app.state.model_catalog
    payload = await request.json()
    routes = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(routes, list):
        raise HTTPException(status_code=400, detail="Expected a JSON array or { data: [...] }")
    try:
        catalog.replace_routes(routes)
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return catalog.to_payload()


@app.get("/v1/providers/status")
async def provider_status(request: Request) -> dict[str, Any]:
    router: WaterfallRouter = request.app.state.waterfall_router
    state: StateManager = request.app.state.gateway_state
    catalog: ModelCatalog = request.app.state.model_catalog

    providers = []
    for provider in router.providers:
        provider_state = await state.get_state(provider.name)
        availability = await state.check_available(provider.name)
        providers.append(
            {
                "name": provider.name,
                "configured": provider.is_configured,
                "available": availability.available,
                "unavailable_reason": availability.reason,
                "retry_after_seconds": availability.retry_after_seconds,
                "tokens_used_today": provider_state.tokens_used_today,
                "requests_today": provider_state.requests_today,
                "requests_this_minute": provider_state.requests_this_minute,
                "cooldown_until": provider_state.cooldown_until,
                "max_context_tokens": provider.max_context_tokens,
                "models": [
                    asdict(route)
                    for route in catalog.all_routes()
                    if route.provider_name == provider.name
                ],
            }
        )

    return {"object": "list", "data": providers}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> JSONResponse:
    payload: dict[str, Any] = await request.json()
    router: WaterfallRouter = request.app.state.waterfall_router

    try:
        result = await router.route_chat_completion(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except NoProviderAvailable as exc:
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "message": "No configured provider is currently available",
                    "type": "provider_unavailable",
                    "code": "waterfall_exhausted",
                    "attempts": [asdict(attempt) for attempt in exc.attempts],
                }
            },
        )
    except ProviderError as exc:
        return JSONResponse(
            status_code=exc.status_code or 502,
            content={
                "error": {
                    "message": str(exc),
                    "type": "provider_error",
                    "code": exc.status_code,
                    "body": exc.body,
                }
            },
        )

    return JSONResponse(
        content=result.body,
        headers={
            "X-Gateway-Provider": result.provider_name,
            "X-Gateway-Route": result.route_id,
            "X-Gateway-Model": result.model_id,
            "X-Gateway-Attempts": json.dumps([asdict(attempt) for attempt in result.attempts]),
        },
    )


async def _sse_route_chat(
    payload: dict[str, Any],
    router: WaterfallRouter,
) -> AsyncGenerator[str, None]:
    """Generator that emits SSE events as the waterfall tries each route."""

    def evt(data: dict[str, Any]) -> str:
        return f"data: {json.dumps(data)}\n\n"

    try:
        validate_chat_completion_payload(payload)
    except ValueError as exc:
        yield evt({"type": "error", "message": str(exc)})
        return
    if payload.get("stream"):
        yield evt({"type": "error", "message": "Streaming not supported via this endpoint."})
        return

    requested_model = payload.get("model", "auto")
    routes = router.model_catalog.enabled_routes(requested_model)
    estimated_prompt_tokens = sum(
        len(str(m.get("content", ""))) // 4 for m in payload.get("messages", [])
    )
    max_new = payload.get("max_tokens") or 4096
    estimated_total = estimated_prompt_tokens + max_new

    async with httpx.AsyncClient(timeout=router.request_timeout_seconds) as client:
        for route in routes:
            provider = router.provider_by_name.get(route.provider_name)
            if not provider or not provider.is_configured:
                yield evt({"type": "route_skip", "provider": route.provider_name, "model_id": route.model_id, "reason": "not_configured"})
                continue


            ctx = route.context_window or provider.max_context_tokens
            if ctx and estimated_prompt_tokens > ctx:
                yield evt({"type": "route_skip", "provider": route.provider_name, "model_id": route.model_id, "reason": "context_exceeded"})
                continue

            avail = await router.state.try_reserve_request(route.provider_name, estimated_total)
            if not avail.available:
                yield evt({"type": "route_skip", "provider": route.provider_name, "model_id": route.model_id, "reason": avail.reason or "quota_exhausted"})
                continue

            yield evt({"type": "route_trying", "provider": route.provider_name, "model_id": route.model_id})
            await asyncio.sleep(0)

            try:
                response = await provider.chat_completion(client, payload, route.model_id)
            except ProviderRateLimited as exc:
                await router.state.mark_exhausted(route.provider_name, headers=exc.headers, status_code=exc.status_code)
                yield evt({"type": "route_fail", "provider": route.provider_name, "model_id": route.model_id, "reason": "rate_limited_429"})
                continue
            except httpx.TimeoutException:
                yield evt({"type": "route_fail", "provider": route.provider_name, "model_id": route.model_id, "reason": "timeout"})
                continue
            except httpx.RequestError as exc:
                yield evt({"type": "route_fail", "provider": route.provider_name, "model_id": route.model_id, "reason": exc.__class__.__name__})
                continue
            except ProviderError as exc:
                if exc.status_code and (500 <= exc.status_code < 600 or exc.status_code in {401, 403, 404}):
                    reason = "server_error" if exc.status_code >= 500 else ("auth_error" if exc.status_code in {401, 403} else "model_not_found")
                    yield evt({"type": "route_fail", "provider": route.provider_name, "model_id": route.model_id, "reason": reason})
                    continue
                yield evt({"type": "error", "message": str(exc)})
                return

            # Success
            usage = response.body.get("usage")
            await router.state.record_success(
                route.provider_name,
                usage=usage if isinstance(usage, dict) else None,
                headers=response.headers,
                status_code=response.status_code,
            )
            yield evt({"type": "route_selected", "provider": route.provider_name, "model_id": route.model_id})

            content = ""
            try:
                content = response.body["choices"][0]["message"]["content"] or ""
            except (KeyError, IndexError):
                pass
            chunk_size = 12
            for i in range(0, len(content), chunk_size):
                yield evt({"type": "content", "text": content[i:i + chunk_size]})
                await asyncio.sleep(0.015)

            yield evt({"type": "done", "content": content, "provider": route.provider_name, "model_id": route.model_id})
            return

    yield evt({"type": "error", "message": "All providers exhausted. No model could serve this request."})


@app.post("/v1/chat/completions/stream-route")
async def chat_completions_stream_route(request: Request) -> StreamingResponse:
    payload: dict[str, Any] = await request.json()
    router: WaterfallRouter = request.app.state.waterfall_router
    return StreamingResponse(
        _sse_route_chat(payload, router),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


MODEL_CATALOG_HTML = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Models - FreeRouter</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
      *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
      :root {
        --bg-primary: #0a0e1a; --bg-secondary: #111827; --bg-tertiary: #1e293b;
        --border: #2d3a4f; --text: #e2e8f0; --text-muted: #94a3b8;
        --accent: #3b82f6; --accent-glow: rgba(59,130,246,0.15);
        --green: #22c55e; --red: #ef4444; --amber: #f59e0b; --purple: #a78bfa;
        --font: 'Inter', system-ui, sans-serif;
      }
      body { font-family: var(--font); background: var(--bg-primary); color: var(--text); }
      nav { display: flex; align-items: center; gap: 1rem; padding: 0.75rem 1.5rem; background: var(--bg-secondary); border-bottom: 1px solid var(--border); flex-shrink: 0; }
      nav h1 { font-size: 1rem; font-weight: 700; background: linear-gradient(135deg, #60a5fa, #a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
      nav a { color: var(--text-muted); text-decoration: none; font-size: 0.85rem; transition: color 0.2s; }
      nav a:hover { color: var(--text); }
      .nav-spacer { flex: 1; }
      main { padding: 2rem; max-width: 1200px; margin: auto; }
      .toolbar { display: flex; gap: 0.75rem; flex-wrap: wrap; margin-bottom: 1.5rem; align-items: center; }
      input, select, button, textarea {
        border: 1px solid var(--border); border-radius: 8px; background: var(--bg-primary); color: var(--text);
        padding: 0.55rem 0.75rem; font: inherit; font-size: 0.9rem; transition: border-color 0.2s;
      }
      input:focus, select:focus, textarea:focus { border-color: var(--accent); outline: none; }
      button { cursor: pointer; background: var(--accent); border: none; color: #fff; font-weight: 500; transition: background 0.2s; }
      button:hover { background: #2563eb; }
      button.secondary { background: var(--bg-tertiary); border: 1px solid var(--border); color: var(--text); }
      button.secondary:hover { background: var(--border); }
      .grid { display: grid; gap: 0.75rem; }
      details { background: var(--bg-secondary); border: 1px solid var(--border); border-radius: 12px; overflow: hidden; transition: transform 0.2s; }
      summary { cursor: pointer; padding: 1rem 1.25rem; display: grid; grid-template-columns: 4rem 1fr auto; gap: 1rem; align-items: center; }
      .rank { font-size: 1.25rem; font-weight: 700; color: var(--accent); }
      .provider { text-transform: uppercase; color: var(--green); font-size: 0.75rem; letter-spacing: 0.05em; font-weight: 600; }
      .muted { color: var(--text-muted); }
      .pill { display: inline-block; padding: 0.15rem 0.5rem; border-radius: 999px; background: var(--bg-tertiary); margin: 0.1rem; font-size: 0.75rem; border: 1px solid var(--border); }
      .body { border-top: 1px solid var(--border); padding: 1.25rem; display: grid; gap: 1rem; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); background: var(--bg-primary); }
      label { display: grid; gap: 0.4rem; color: var(--text-muted); font-size: 0.85rem; font-weight: 500; }
      textarea { min-height: 5rem; resize: vertical; }
      .status { min-height: 1.5rem; color: var(--green); font-size: 0.85rem; margin-bottom: 1rem; }
      .disabled { opacity: 0.6; filter: grayscale(1); }
      code { background: var(--bg-tertiary); border: 1px solid var(--border); padding: 0.1rem 0.35rem; border-radius: 0.35rem; color: #93c5fd; }
      details.dragging { opacity: 0.4; transform: scale(0.98); }
      details.drag-over { border-top: 2px solid var(--accent); }
      details[draggable="true"] summary { cursor: grab; }
      details[draggable="true"] summary:active { cursor: grabbing; }
    </style>
  </head>
  <body>
    <nav>
      <h1>FreeRouter</h1>
      <span class="nav-spacer"></span>
      <a href="/">Home</a>
      <a href="/chat">Chat</a>
      <a href="/models">Models</a>
      <a href="/v1/providers/status">Quota Status</a>
    </nav>
    <main>
      <div class="toolbar">
        <input id="search" placeholder="Search models/providers/tags..." style="flex: 1; min-width: 250px;">
        <select id="provider"><option value="">All providers</option></select>
        <button id="save">Save Ranking</button>
        <button id="reload" class="secondary">Reload</button>
      </div>
      <p class="status" id="status"></p>
      <div id="models" class="grid"></div>
    </main>
    <script>
      let routes = [];

      const $ = (id) => document.getElementById(id);
      const normalize = (value) => String(value || '').toLowerCase();

      async function load() {
        $('status').textContent = 'Loading model catalog...';
        const response = await fetch('/v1/gateway/models');
        const payload = await response.json();
        routes = payload.data;
        populateProviders();
        render();
        $('status').textContent = `Loaded ${routes.length} model routes from ${payload.catalog_path}`;
      }

      function populateProviders() {
        const providerSelect = $('provider');
        const selected = providerSelect.value;
        const providers = [...new Set(routes.map((route) => route.provider_name))].sort();
        providerSelect.innerHTML = '<option value="">All providers</option>' + providers.map(
          (provider) => `<option value="${provider}">${provider}</option>`
        ).join('');
        providerSelect.value = selected;
      }

      function render() {
        const query = normalize($('search').value);
        const provider = $('provider').value;
        const visible = routes
          .slice()
          .sort((a, b) => a.rank - b.rank || a.provider_name.localeCompare(b.provider_name))
          .filter((route) => !provider || route.provider_name === provider)
          .filter((route) => {
            const haystack = normalize([
              route.route_id, route.provider_name, route.model_id, route.display_name,
              route.quality, route.speed, route.tags?.join(' '), route.notes
            ].join(' '));
            return !query || haystack.includes(query);
          });

        $('models').innerHTML = visible.map((route) => card(route)).join('');
      }

      let draggedId = null;

      function card(route) {
        const tags = (route.tags || []).map((tag) => `<span class="pill">${escapeHtml(tag)}</span>`).join('');
        return `
          <details class="${route.enabled ? '' : 'disabled'}" draggable="true"
            ondragstart="dragStart(event, '${route.route_id}')"
            ondragover="dragOver(event)"
            ondrop="drop(event, '${route.route_id}')"
            ondragenter="event.currentTarget.classList.add('drag-over')"
            ondragleave="event.currentTarget.classList.remove('drag-over')"
            ondragend="event.target.classList.remove('dragging')">
            <summary>
              <span class="rank">#${route.rank}</span>
              <span>
                <strong>${escapeHtml(route.display_name)}</strong>
                <span class="muted"> ${escapeHtml(route.model_id)}</span><br>
                <span class="provider">${escapeHtml(route.provider_name)}</span>
                ${tags}
              </span>
              <span>${route.enabled ? 'Enabled' : 'Disabled'}</span>
            </summary>
            <div class="body">
              ${field(route, 'rank', 'Rank', 'number')}
              ${field(route, 'enabled', 'Enabled', 'checkbox')}
              ${field(route, 'display_name', 'Display name')}
              ${field(route, 'provider_name', 'Provider')}
              ${field(route, 'model_id', 'Provider model ID')}
              ${field(route, 'context_window', 'Context window', 'number')}
              ${field(route, 'quality', 'Quality')}
              ${field(route, 'speed', 'Speed')}
              ${field(route, 'cost', 'Cost')}
              ${field(route, 'tags', 'Tags, comma-separated')}
              ${field(route, 'source_url', 'Source URL')}
              <label style="grid-column: 1 / -1">Notes
                <textarea data-id="${route.route_id}" data-key="notes">${escapeHtml(route.notes || '')}</textarea>
              </label>
            </div>
          </details>
        `;
      }

      function field(route, key, label, type = 'text') {
        const value = Array.isArray(route[key]) ? route[key].join(', ') : (route[key] ?? '');
        if (type === 'checkbox') {
          return `<label>${label}<input type="checkbox" data-id="${route.route_id}" data-key="${key}" ${route[key] ? 'checked' : ''}></label>`;
        }
        return `<label>${label}<input type="${type}" data-id="${route.route_id}" data-key="${key}" value="${escapeHtml(value)}"></label>`;
      }

      function collectEdits() {
        document.querySelectorAll('[data-id][data-key]').forEach((input) => {
          const route = routes.find((item) => item.route_id === input.dataset.id);
          const key = input.dataset.key;
          if (!route) return;
          if (key === 'enabled') route[key] = input.checked;
          else if (key === 'rank' || key === 'context_window') route[key] = input.value === '' ? null : Number(input.value);
          else if (key === 'tags') route[key] = input.value.split(',').map((tag) => tag.trim()).filter(Boolean);
          else route[key] = input.value;
        });
      }

      async function save() {
        collectEdits();
        $('status').textContent = 'Saving...';
        const response = await fetch('/v1/gateway/models', {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ data: routes }),
        });
        const payload = await response.json();
        if (!response.ok) {
          $('status').textContent = payload.detail || 'Save failed';
          return;
        }
        routes = payload.data;
        render();
        $('status').textContent = 'Saved. New requests will use the updated ranking immediately.';
      }

      function escapeHtml(value) {
        return String(value ?? '').replace(/[&<>"']/g, (char) => ({
          '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
        }[char]));
      }

      function dragStart(e, id) {
        draggedId = id;
        e.dataTransfer.effectAllowed = 'move';
        setTimeout(() => e.target.classList.add('dragging'), 0);
      }

      function dragOver(e) {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'move';
      }

      function drop(e, dropId) {
        e.preventDefault();
        e.currentTarget.classList.remove('drag-over');
        if (!draggedId || draggedId === dropId) return;

        routes.sort((a, b) => a.rank - b.rank || a.provider_name.localeCompare(b.provider_name));
        const fromIndex = routes.findIndex(r => r.route_id === draggedId);
        const toIndex = routes.findIndex(r => r.route_id === dropId);
        if (fromIndex < 0 || toIndex < 0) return;

        const [moved] = routes.splice(fromIndex, 1);
        routes.splice(toIndex, 0, moved);

        routes.forEach((r, i) => r.rank = i + 1);
        render();
        save();
      }

      $('search').addEventListener('input', render);
      $('provider').addEventListener('change', render);
      $('save').addEventListener('click', save);
      $('reload').addEventListener('click', load);
      document.addEventListener('input', (event) => {
        if (event.target.matches('[data-id][data-key]')) collectEdits();
      });
      load();
    </script>
  </body>
</html>
"""
