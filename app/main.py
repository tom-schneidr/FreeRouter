from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from time import perf_counter, time
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

from app.chat_page import CHAT_HTML
from app.endpoint_diagnosis import (
    BackgroundEndpointDiagnosis,
    EndpointDiagnosisService,
    EndpointSupervisor,
)
from app.live_monitor import APILiveMonitor
from app.model_catalog import ModelCatalog
from app.providers import PROVIDER_QUOTAS, ProviderError, build_provider_adapters
from app.request_limiter import GatewayRequestLimiter
from app.router import (
    NoProviderAvailable,
    RouteStreamDiag,
    WaterfallRouter,
    validate_chat_completion_payload,
)
from app.settings import get_settings
from app.state import StateManager
from app.stream_route import stream_route_chat

WEB_SEARCH_TOOL = {"type": "web_search_preview"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    gateway_http_client: httpx.AsyncClient | None = None
    state = StateManager(
        settings.database_path,
        PROVIDER_QUOTAS,
        busy_timeout_ms=settings.sqlite_busy_timeout_ms,
    )
    await state.initialize()
    model_catalog = ModelCatalog(settings.model_catalog_path)
    model_catalog.initialize()
    providers = build_provider_adapters(settings)

    app.state.gateway_state = state
    app.state.model_catalog = model_catalog
    app.state.request_limiter = GatewayRequestLimiter(
        settings.max_concurrent_requests,
        settings.request_queue_timeout_seconds,
        settings.request_queue_max_waiting_requests,
    )
    app.state.live_monitor = APILiveMonitor(max_events=1000)
    limits = httpx.Limits(
        max_connections=settings.http_max_connections,
        max_keepalive_connections=settings.http_max_keepalive_connections,
        keepalive_expiry=settings.http_keepalive_expiry_seconds,
    )
    gateway_http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(settings.request_timeout_seconds),
        limits=limits,
    )
    app.state.http_client = gateway_http_client
    app.state.waterfall_router = WaterfallRouter(
        providers,
        model_catalog,
        state,
        request_timeout_seconds=settings.request_timeout_seconds,
        http_client=gateway_http_client,
    )
    app.state.endpoint_diagnosis = EndpointDiagnosisService(
        providers,
        model_catalog,
        state,
        request_timeout_seconds=settings.request_timeout_seconds,
        supervisor=EndpointSupervisor(
            enabled=settings.endpoint_diagnosis_supervisor_enabled,
            providers=providers,
            catalog=model_catalog,
            state=state,
            preferred_model=settings.endpoint_diagnosis_supervisor_model,
        ),
    )
    app.state.background_endpoint_diagnosis = None
    if settings.auto_endpoint_diagnosis_enabled:
        background = BackgroundEndpointDiagnosis(
            app.state.endpoint_diagnosis,
            interval_seconds=settings.auto_endpoint_diagnosis_interval_seconds,
            startup_delay_seconds=settings.auto_endpoint_diagnosis_startup_delay_seconds,
            apply_safe_suggestions=settings.auto_endpoint_maintenance_enabled,
        )
        app.state.background_endpoint_diagnosis = background
        background.start()
    try:
        yield
    finally:
        background = app.state.background_endpoint_diagnosis
        if background is not None:
            await background.stop()
        if gateway_http_client is not None:
            await gateway_http_client.aclose()


app = FastAPI(
    title="FreeRouter",
    version="0.1.0",
    lifespan=lifespan,
)


def _monitor_trim(
    value: Any, *, max_string: int = 1200, max_items: int = 30, depth: int = 0
) -> Any:
    if depth >= 4:
        return "<truncated-depth>"
    if isinstance(value, str):
        if len(value) <= max_string:
            return value
        return value[:max_string] + f"... <truncated {len(value) - max_string} chars>"
    if isinstance(value, list):
        items = [
            _monitor_trim(item, max_string=max_string, max_items=max_items, depth=depth + 1)
            for item in value[:max_items]
        ]
        if len(value) > max_items:
            items.append(f"<truncated {len(value) - max_items} items>")
        return items
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for idx, (k, v) in enumerate(value.items()):
            if idx >= max_items:
                out["__truncated__"] = f"{len(value) - max_items} keys omitted"
                break
            out[str(k)] = _monitor_trim(
                v, max_string=max_string, max_items=max_items, depth=depth + 1
            )
        return out
    return value


def _assistant_text_from_response_body(body: Any) -> str:
    if not isinstance(body, dict):
        return ""
    direct_content = body.get("content")
    if isinstance(direct_content, str):
        return direct_content
    message = body.get("message")
    if isinstance(message, dict):
        message_content = message.get("content")
        if isinstance(message_content, str):
            return message_content
        if isinstance(message_content, list):
            return _content_parts_to_text(message_content)
    choices = body.get("choices")
    if isinstance(choices, list) and choices:
        first_choice = choices[0]
        if isinstance(first_choice, dict):
            choice_message = first_choice.get("message")
            if isinstance(choice_message, dict):
                message_content = choice_message.get("content")
                if isinstance(message_content, str):
                    return message_content
                if isinstance(message_content, list):
                    return _content_parts_to_text(message_content)
            choice_text = first_choice.get("text")
            if isinstance(choice_text, str):
                return choice_text
    return ""


def _content_parts_to_text(parts: list[Any]) -> str:
    text_parts: list[str] = []
    for part in parts:
        if isinstance(part, str):
            text_parts.append(part)
        elif isinstance(part, dict):
            text = part.get("text") or part.get("content")
            if isinstance(text, str):
                text_parts.append(text)
    return "\n".join(text_parts)


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
          <a href="/health">Health</a>
          <a href="/status">Provider Usage</a>
          <a href="/live">Live Traffic</a>
        </nav>
        <main>
          <h2>Welcome to FreeRouter</h2>
          <p>The gateway is active. Use the following as your OpenAI-compatible base URL in any client:</p>
          <p><code>http://127.0.0.1:8000/v1</code></p>
          <div class="links">
            <a href="/chat" class="link-card"><span class="link-icon">💬</span> Chat Playground</a>
            <a href="/models" class="link-card"><span class="link-icon">📊</span> Model Catalog & Ranking</a>
            <a href="/health" class="link-card"><span class="link-icon">🩺</span> Route Health</a>
            <a href="/docs" class="link-card"><span class="link-icon">📖</span> API Documentation</a>
            <a href="/status" class="link-card"><span class="link-icon">📈</span> Provider Usage</a>
            <a href="/live" class="link-card"><span class="link-icon">🛰️</span> Live API Traffic</a>
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


@app.get("/health", response_class=HTMLResponse)
async def route_health_page() -> str:
    return ROUTE_HEALTH_HTML


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
    state: StateManager = request.app.state.gateway_state
    return await _catalog_payload_with_health(catalog, state)


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
    state: StateManager = request.app.state.gateway_state
    return await _catalog_payload_with_health(catalog, state)


@app.post("/v1/gateway/models/reset")
async def reset_gateway_models(request: Request) -> dict[str, Any]:
    catalog: ModelCatalog = request.app.state.model_catalog
    state: StateManager = request.app.state.gateway_state
    catalog.reset_to_defaults()
    return await _catalog_payload_with_health(catalog, state)


@app.post("/v1/gateway/models/auto-rank")
async def auto_rank_gateway_models(request: Request) -> dict[str, Any]:
    catalog: ModelCatalog = request.app.state.model_catalog
    state: StateManager = request.app.state.gateway_state
    catalog.auto_rank_routes()
    catalog.save()
    return await _catalog_payload_with_health(catalog, state)


@app.post("/v1/gateway/models/{route_id}/disable")
async def disable_gateway_model(route_id: str, request: Request) -> dict[str, Any]:
    catalog: ModelCatalog = request.app.state.model_catalog
    state: StateManager = request.app.state.gateway_state
    try:
        route = catalog.set_route_enabled(route_id, False)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    route_state = await state.get_route_state(route.route_id, route.provider_name, route.model_id)
    return {"data": {**asdict(route), "health": asdict(route_state)}}


@app.post("/v1/gateway/models/{route_id}/enable")
async def enable_gateway_model(route_id: str, request: Request) -> dict[str, Any]:
    catalog: ModelCatalog = request.app.state.model_catalog
    state: StateManager = request.app.state.gateway_state
    try:
        route = catalog.set_route_enabled(route_id, True)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    route_state = await state.get_route_state(route.route_id, route.provider_name, route.model_id)
    return {"data": {**asdict(route), "health": asdict(route_state)}}


@app.post("/v1/gateway/models/{route_id}/health/reset")
async def reset_gateway_model_health(route_id: str, request: Request) -> dict[str, Any]:
    catalog: ModelCatalog = request.app.state.model_catalog
    state: StateManager = request.app.state.gateway_state
    route = next((route for route in catalog.all_routes() if route.route_id == route_id), None)
    if route is None:
        raise HTTPException(status_code=404, detail=f"Unknown route_id: {route_id}")
    route_state = await state.clear_route_health(
        route.route_id, route.provider_name, route.model_id
    )
    return {"data": {**asdict(route), "health": asdict(route_state)}}


@app.get("/v1/gateway/endpoint-diagnosis")
async def endpoint_diagnosis_status(request: Request) -> dict[str, Any]:
    service: EndpointDiagnosisService = request.app.state.endpoint_diagnosis
    background: BackgroundEndpointDiagnosis | None = request.app.state.background_endpoint_diagnosis
    report = service.last_report
    return {
        "enabled": get_settings().auto_endpoint_diagnosis_enabled,
        "auto_maintenance_enabled": bool(background and background.apply_safe_suggestions),
        "last_auto_applied": (
            [asdict(suggestion) for suggestion in background.last_auto_applied]
            if background is not None
            else []
        ),
        "last_report": asdict(report) if report is not None else None,
    }


@app.post("/v1/gateway/endpoint-diagnosis/refresh")
async def refresh_endpoint_diagnosis(request: Request) -> dict[str, Any]:
    service: EndpointDiagnosisService = request.app.state.endpoint_diagnosis
    report = await service.run_once()
    return {"data": asdict(report)}


@app.post("/v1/gateway/endpoint-diagnosis/apply")
async def apply_endpoint_diagnosis(request: Request) -> dict[str, Any]:
    service: EndpointDiagnosisService = request.app.state.endpoint_diagnosis
    payload = await request.json()
    suggestion_ids = payload.get("suggestion_ids") if isinstance(payload, dict) else None
    if not isinstance(suggestion_ids, list) or not all(
        isinstance(item, str) for item in suggestion_ids
    ):
        raise HTTPException(status_code=400, detail="Expected { suggestion_ids: [string, ...] }")
    applied = await service.apply_suggestions(suggestion_ids)
    return {"data": [asdict(suggestion) for suggestion in applied]}


async def _catalog_payload_with_health(
    catalog: ModelCatalog, state: StateManager
) -> dict[str, Any]:
    payload = catalog.to_payload()
    route_rows = [
        (
            route_payload["route_id"],
            route_payload["provider_name"],
            route_payload["model_id"],
        )
        for route_payload in payload["data"]
    ]
    health_map = await state.get_route_states_batch(route_rows)
    for route_payload in payload["data"]:
        route_state = health_map[route_payload["route_id"]]
        route_payload["health"] = asdict(route_state)
    return payload


@app.get("/status", response_class=HTMLResponse)
async def provider_status_page() -> HTMLResponse:
    return HTMLResponse(r"""
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Usage Stats - FreeRouter</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
      *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
      :root { --bg-primary: #0a0e1a; --bg-secondary: #111827; --bg-tertiary: #1e293b; --border: #2d3a4f; --text: #e2e8f0; --text-muted: #94a3b8; --accent: #3b82f6; --accent-glow: rgba(59,130,246,0.15); --green: #22c55e; --red: #ef4444; --amber: #f59e0b; --font: 'Inter', system-ui, sans-serif; }
      body { font-family: var(--font); background: var(--bg-primary); color: var(--text); }
      nav { display: flex; align-items: center; gap: 1rem; padding: 0.75rem 1.5rem; background: var(--bg-secondary); border-bottom: 1px solid var(--border); }
      nav h1 { font-size: 1rem; font-weight: 700; background: linear-gradient(135deg, #60a5fa, #a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
      nav a { color: var(--text-muted); text-decoration: none; font-size: 0.85rem; }
      nav a:hover { color: var(--text); }
      .nav-spacer { flex: 1; }
      main { max-width: 1280px; margin: auto; padding: 2rem; }
      h2 { margin-bottom: 0.5rem; }
      .muted { color: var(--text-muted); }
      .toolbar { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 1rem; margin: 1.5rem 0; }
      input, select, button { border: 1px solid var(--border); border-radius: 8px; background: var(--bg-primary); color: var(--text); padding: 0.55rem 0.75rem; font: inherit; font-size: 0.9rem; }
      button { border: none; background: var(--accent); color: white; cursor: pointer; font-weight: 600; }
      button:hover { background: #2563eb; }
      .filters { display: flex; flex-wrap: wrap; gap: 0.75rem; }
      .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 0.75rem; margin: 1.5rem 0; }
      .summary-card { padding: 1rem; background: var(--bg-secondary); border: 1px solid var(--border); border-radius: 12px; }
      .summary-card .label { color: var(--text-muted); font-size: 0.74rem; margin-bottom: 0.35rem; text-transform: uppercase; letter-spacing: 0.04em; }
      .summary-card .value { font-size: 1.45rem; font-weight: 700; }
      .table-wrap { overflow: auto; border: 1px solid var(--border); border-radius: 12px; background: var(--bg-secondary); }
      table { width: 100%; border-collapse: collapse; min-width: 980px; }
      th, td { padding: 0.75rem 0.85rem; border-bottom: 1px solid var(--border); text-align: left; font-size: 0.86rem; vertical-align: middle; }
      th { color: var(--text-muted); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.04em; background: rgba(15, 23, 42, 0.75); position: sticky; top: 0; z-index: 1; }
      tbody tr:hover { background: rgba(59,130,246,0.08); }
      .model { font-weight: 700; color: #fff; }
      .route { margin-top: 0.2rem; color: var(--text-muted); font-size: 0.76rem; }
      .provider { color: var(--green); text-transform: uppercase; letter-spacing: 0.05em; font-size: 0.72rem; font-weight: 700; }
      .pill { display: inline-flex; align-items: center; padding: 0.22rem 0.5rem; border-radius: 999px; border: 1px solid rgba(34,197,94,0.45); color: #bbf7d0; background: rgba(34,197,94,0.12); font-size: 0.75rem; white-space: nowrap; }
      .pill.warning { border-color: rgba(245,158,11,0.5); color: #fcd34d; background: rgba(245,158,11,0.12); }
      .pill.error { border-color: rgba(239,68,68,0.5); color: #fecaca; background: rgba(239,68,68,0.12); }
      .pill.neutral { border-color: var(--border); color: var(--text-muted); background: var(--bg-tertiary); }
      .details-row td { background: var(--bg-primary); padding: 0; }
      .details { display: grid; grid-template-columns: repeat(auto-fit, minmax(155px, 1fr)); gap: 0.7rem; padding: 1rem; }
      .stat { padding: 0.75rem; border: 1px solid var(--border); border-radius: 10px; background: var(--bg-secondary); }
      .stat .label { color: var(--text-muted); font-size: 0.72rem; margin-bottom: 0.3rem; text-transform: uppercase; letter-spacing: 0.04em; }
      .stat .value { font-size: 0.95rem; font-weight: 650; word-break: break-word; }
      .expand { background: var(--bg-tertiary); border: 1px solid var(--border); color: var(--text); padding: 0.35rem 0.6rem; }
      .expand:hover { background: var(--border); }
      .empty { padding: 2rem; text-align: center; color: var(--text-muted); background: var(--bg-secondary); border: 1px solid var(--border); border-radius: 12px; }
      .right { text-align: right; }
      @media (max-width: 720px) { main { padding: 1rem; } nav { flex-wrap: wrap; } .filters { width: 100%; } input, select { flex: 1; min-width: 12rem; } }
    </style>
  </head>
  <body>
    <nav>
      <h1>FreeRouter</h1><span class="nav-spacer"></span>
      <a href="/">Home</a><a href="/chat">Chat</a><a href="/models">Models</a><a href="/health">Health</a><a href="/status">Provider Usage</a><a href="/live">Live Traffic</a>
    </nav>
    <main>
      <h2>Usage Stats</h2>
      <p class="muted">Provider quotas, health state, and per-model usage tracked locally by FreeRouter.</p>
      <section id="summaryCards" class="summary-grid"></section>
      <div class="toolbar">
        <div class="filters">
          <input id="search" type="search" placeholder="Search models or providers">
          <select id="providerFilter"><option value="">All providers</option></select>
          <select id="healthFilter">
            <option value="">All health states</option>
            <option value="active">Active</option>
            <option value="rate_limited">Rate limited</option>
            <option value="too_slow">Too slow</option>
            <option value="potentially_outdated">Potentially outdated</option>
          </select>
        </div>
        <span id="summary" class="muted">Loading...</span>
        <button id="reload">Reload</button>
      </div>
      <div id="tableRoot"></div>
    </main>
    <script>
      const tableRoot = document.getElementById('tableRoot');
      const cardsEl = document.getElementById('summaryCards');
      const summaryEl = document.getElementById('summary');
      const searchEl = document.getElementById('search');
      const providerFilterEl = document.getElementById('providerFilter');
      const healthFilterEl = document.getElementById('healthFilter');
      let allModels = [];
      let providers = [];
      let expanded = new Set();
      const escapeHtml = (value) => String(value ?? '').replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
      const fmt = (value) => value == null ? 'Unknown' : Number(value).toLocaleString();
      const fmtDate = (value) => value ? new Date(Number(value) * 1000).toLocaleString() : 'Never';
      const healthLabel = (value) => String(value || 'active').replace(/_/g, ' ');
      function healthPill(status) {
        if (status === 'active') return '<span class="pill">Active</span>';
        if (status === 'rate_limited') return '<span class="pill warning">Rate limited</span>';
        return `<span class="pill error">${escapeHtml(healthLabel(status))}</span>`;
      }
      function providerStatus(provider) {
        if (!provider?.configured) return 'Not configured';
        if (!provider.available) return provider.unavailable_reason || 'Limited';
        return 'Available';
      }
      function stat(label, value) {
        return `<div class="stat"><div class="label">${escapeHtml(label)}</div><div class="value">${escapeHtml(value)}</div></div>`;
      }
      function renderCards(models) {
        const totalTokens = models.reduce((sum, model) => sum + Number(model.usage?.total_tokens || 0), 0);
        const totalSuccesses = models.reduce((sum, model) => sum + Number(model.usage?.successes || 0), 0);
        const totalFailures = models.reduce((sum, model) => sum + Number(model.usage?.failures || 0), 0);
        const limitedRoutes = models.filter((model) => model.health?.status && model.health.status !== 'active').length;
        const providerRequests = providers.reduce((sum, provider) => sum + Number(provider.requests_today || 0), 0);
        const providerTokens = providers.reduce((sum, provider) => sum + Number(provider.tokens_used_today || 0), 0);
        cardsEl.innerHTML = `
          <div class="summary-card"><div class="label">Requests today</div><div class="value">${fmt(providerRequests)}</div></div>
          <div class="summary-card"><div class="label">Provider tokens today</div><div class="value">${fmt(providerTokens)}</div></div>
          <div class="summary-card"><div class="label">Model successes</div><div class="value">${fmt(totalSuccesses)}</div></div>
          <div class="summary-card"><div class="label">Model failures</div><div class="value">${fmt(totalFailures)}</div></div>
          <div class="summary-card"><div class="label">Route tokens recorded</div><div class="value">${fmt(totalTokens)}</div></div>
          <div class="summary-card"><div class="label">Automatically limited</div><div class="value">${fmt(limitedRoutes)}</div></div>
        `;
      }
      function filteredModels() {
        const query = searchEl.value.trim().toLowerCase();
        const provider = providerFilterEl.value;
        const health = healthFilterEl.value;
        return allModels.filter((model) => {
          const haystack = `${model.display_name} ${model.model_id} ${model.provider_name}`.toLowerCase();
          return (!query || haystack.includes(query))
            && (!provider || model.provider_name === provider)
            && (!health || (model.health?.status || 'active') === health);
        });
      }
      function detailRow(model) {
        const usage = model.usage || {};
        const health = model.health || {};
        const provider = providers.find((item) => item.name === model.provider_name);
        return `
          <tr class="details-row">
            <td colspan="10">
              <div class="details">
                ${stat('Route ID', model.route_id)}
                ${stat('Enabled', model.enabled ? 'Yes' : 'No')}
                ${stat('Context window', model.context_window ? fmt(model.context_window) : 'Unknown')}
                ${stat('Prompt tokens', fmt(usage.prompt_tokens || 0))}
                ${stat('Completion tokens', fmt(usage.completion_tokens || 0))}
                ${stat('Rate limits', fmt(usage.rate_limits || 0))}
                ${stat('Timeouts', fmt(usage.timeouts || 0))}
                ${stat('Not found errors', fmt(usage.not_found || 0))}
                ${stat('Consecutive failures', fmt(health.consecutive_failures || 0))}
                ${stat('Last event', fmtDate(usage.last_used_at))}
                ${stat('Last status code', usage.last_status_code ?? 'None')}
                ${stat('Provider state', providerStatus(provider))}
                ${stat('Provider requests today', fmt(provider?.requests_today || 0))}
                ${stat('Provider cooldown until', fmtDate(provider?.cooldown_until))}
              </div>
            </td>
          </tr>
        `;
      }
      function modelRow(model) {
        const usage = model.usage || {};
        const health = model.health || {};
        const isExpanded = expanded.has(model.route_id);
        return `
          <tr>
            <td>
              <div class="model">${escapeHtml(model.display_name || model.model_id)}</div>
              <div class="route">${escapeHtml(model.model_id)}</div>
            </td>
            <td><span class="provider">${escapeHtml(model.provider_name)}</span></td>
            <td>${healthPill(health.status || 'active')}</td>
            <td class="right">${fmt(usage.successes || 0)}</td>
            <td class="right">${fmt(usage.failures || 0)}</td>
            <td class="right">${fmt(usage.total_tokens || 0)}</td>
            <td class="right">${fmt(usage.prompt_tokens || 0)}</td>
            <td class="right">${fmt(usage.completion_tokens || 0)}</td>
            <td>${fmtDate(usage.last_used_at)}</td>
            <td><button class="expand" data-route-id="${escapeHtml(model.route_id)}">${isExpanded ? 'Hide' : 'View'} stats</button></td>
          </tr>
          ${isExpanded ? detailRow(model) : ''}
        `;
      }
      function render() {
        const models = filteredModels();
        renderCards(allModels);
        summaryEl.textContent = `${models.length} of ${allModels.length} model route${allModels.length === 1 ? '' : 's'} shown`;
        if (!models.length) {
          tableRoot.innerHTML = '<div class="empty">No model usage matches the current filters.</div>';
          return;
        }
        tableRoot.innerHTML = `
          <div class="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Provider</th>
                  <th>Health</th>
                  <th class="right">Successes</th>
                  <th class="right">Failures</th>
                  <th class="right">Tokens</th>
                  <th class="right">Prompt</th>
                  <th class="right">Completion</th>
                  <th>Last Used</th>
                  <th>Details</th>
                </tr>
              </thead>
              <tbody>${models.map(modelRow).join('')}</tbody>
            </table>
          </div>
        `;
        tableRoot.querySelectorAll('.expand').forEach((button) => {
          button.addEventListener('click', () => {
            const routeId = button.dataset.routeId;
            if (expanded.has(routeId)) expanded.delete(routeId);
            else expanded.add(routeId);
            render();
          });
        });
      }
      async function load() {
        summaryEl.textContent = 'Loading...';
        tableRoot.innerHTML = '';
        const response = await fetch('/v1/providers/status');
        if (!response.ok) {
          tableRoot.innerHTML = '<div class="empty">Could not load usage stats.</div>';
          summaryEl.textContent = 'Load failed';
          return;
        }
        const payload = await response.json();
        providers = payload.data || [];
        allModels = providers.flatMap((provider) => provider.models || []).sort((a, b) => {
          const tokenDelta = Number(b.usage?.total_tokens || 0) - Number(a.usage?.total_tokens || 0);
          return tokenDelta || Number(a.rank || 9999) - Number(b.rank || 9999);
        });
        providerFilterEl.innerHTML = '<option value="">All providers</option>' + providers
          .map((provider) => `<option value="${escapeHtml(provider.name)}">${escapeHtml(provider.name)}</option>`)
          .join('');
        render();
      }
      document.getElementById('reload').addEventListener('click', load);
      searchEl.addEventListener('input', render);
      providerFilterEl.addEventListener('change', render);
      healthFilterEl.addEventListener('change', render);
      load();
    </script>
  </body>
</html>
""")


@app.get("/v1/providers/status")
async def provider_status(request: Request) -> dict[str, Any]:
    router: WaterfallRouter = request.app.state.waterfall_router
    state: StateManager = request.app.state.gateway_state
    catalog: ModelCatalog = request.app.state.model_catalog

    all_routes = catalog.all_routes()
    route_usage = await state.get_route_usage_stats([route.route_id for route in all_routes])
    health_map = await state.get_route_states_batch(
        [(r.route_id, r.provider_name, r.model_id) for r in all_routes]
    )
    providers_snap = await state.snapshot_providers_usage(
        [provider.name for provider in router.providers]
    )
    providers = []
    for provider in router.providers:
        pair = providers_snap.get(provider.name)
        if pair is None:
            provider_state = await state.get_state(provider.name)
            availability = await state.check_available(provider.name)
        else:
            provider_state, availability = pair
        models = []
        for route in all_routes:
            if route.provider_name != provider.name:
                continue
            route_state = health_map[route.route_id]
            route_payload = asdict(route)
            route_payload["health"] = asdict(route_state)
            route_payload["usage"] = route_usage[route.route_id]
            models.append(route_payload)
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
                "models": models,
            }
        )

    return {"object": "list", "data": providers}


@app.get("/live", response_class=HTMLResponse)
async def live_api_page() -> HTMLResponse:
    return HTMLResponse(LIVE_API_HTML)


@app.get("/v1/gateway/live/snapshot")
async def live_api_snapshot(request: Request) -> dict[str, Any]:
    monitor: APILiveMonitor = request.app.state.live_monitor
    return {"object": "list", "data": await monitor.snapshot()}


@app.get("/v1/gateway/live/events")
async def live_api_events(request: Request) -> StreamingResponse:
    monitor: APILiveMonitor = request.app.state.live_monitor

    async def event_stream():
        queue = await monitor.subscribe()
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=1.0)
                    payload = APILiveMonitor.event_to_payload(event)
                    yield f"data: {json.dumps(payload)}\n\n"
                except TimeoutError:
                    yield ": heartbeat\n\n"
                except asyncio.CancelledError:
                    break
        finally:
            await monitor.unsubscribe(queue)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _payload_with_required_web_search(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object")
    prepared = dict(payload)
    tools = prepared.get("tools")
    if not isinstance(tools, list):
        tools = []
    elif not _has_web_search_tool(tools):
        tools = list(tools)
    if not _has_web_search_tool(tools):
        tools.append(dict(WEB_SEARCH_TOOL))
    prepared["tools"] = tools
    prepared["tool_choice"] = dict(WEB_SEARCH_TOOL)
    return prepared


def _has_web_search_tool(tools: list[Any]) -> bool:
    return any(
        isinstance(tool, dict) and tool.get("type") == WEB_SEARCH_TOOL["type"] for tool in tools
    )


async def _publish_route_stream_diag(
    monitor: APILiveMonitor,
    request_id: str,
    diag: RouteStreamDiag,
) -> None:
    if diag.event_type == "usage_summary":
        if diag.usage:
            await monitor.publish(
                event_type="usage_update",
                request_id=request_id,
                payload={"usage": _monitor_trim(dict(diag.usage))},
            )
        return
    if diag.event_type == "route_trying":
        await monitor.publish(
            event_type="route_attempt",
            request_id=request_id,
            payload={
                "route_event": {
                    "type": "route_trying",
                    "provider_name": diag.provider_name,
                    "route_id": diag.route_id,
                    "model_id": diag.model_id,
                }
            },
        )
        return
    if diag.event_type == "route_skipped":
        await monitor.publish(
            event_type="route_attempt",
            request_id=request_id,
            payload={
                "route_event": {
                    "type": "route_skip",
                    "provider_name": diag.provider_name,
                    "route_id": diag.route_id,
                    "model_id": diag.model_id,
                    "reason": diag.reason,
                }
            },
        )
        return
    if diag.event_type == "route_failed":
        await monitor.publish(
            event_type="route_attempt",
            request_id=request_id,
            payload={
                "route_event": {
                    "type": "route_fail",
                    "provider_name": diag.provider_name,
                    "route_id": diag.route_id,
                    "model_id": diag.model_id,
                    "reason": diag.reason,
                }
            },
        )
        return
    if diag.event_type == "route_flagged":
        await monitor.publish(
            event_type="route_attempt",
            request_id=request_id,
            payload={
                "route_event": {
                    "type": "route_flagged",
                    "provider_name": diag.provider_name,
                    "route_id": diag.route_id,
                    "model_id": diag.model_id,
                    "reason": diag.reason,
                }
            },
        )
        return
    if diag.event_type == "route_selected":
        await monitor.publish(
            event_type="route_selected",
            request_id=request_id,
            payload={
                "provider_name": diag.provider_name,
                "route_id": diag.route_id,
                "model_id": diag.model_id,
            },
        )
        return


async def _route_chat_completion_stream_request(
    request: Request,
    *,
    payload: dict[str, Any],
    path: str,
    required_tag: str | None = None,
) -> Response:
    router: WaterfallRouter = request.app.state.waterfall_router
    limiter: GatewayRequestLimiter = request.app.state.request_limiter
    monitor: APILiveMonitor = request.app.state.live_monitor
    request_id = uuid.uuid4().hex[:12]
    started_at = perf_counter()
    client_ip = request.client.host if request.client else None

    try:
        validate_chat_completion_payload(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    await monitor.publish(
        event_type="request_started",
        request_id=request_id,
        payload={
            "path": path,
            "stream": True,
            "model": payload.get("model"),
            "client_ip": client_ip,
            "request_payload": _monitor_trim(payload),
        },
    )

    if not await limiter.acquire():
        await monitor.publish(
            event_type="request_rejected",
            request_id=request_id,
            payload={"status_code": 429, "reason": "request_queue_timeout"},
        )
        return JSONResponse(
            status_code=429,
            content={
                "error": {
                    "message": "Gateway is busy; retry shortly",
                    "type": "gateway_overloaded",
                    "code": "request_queue_timeout",
                }
            },
            headers={"Retry-After": "1"},
        )

    settings = get_settings()
    limiter_slot_held = True
    selected_provider = ""
    selected_route = ""
    selected_model = ""

    async def openai_sse_stream():
        nonlocal selected_provider, selected_route, selected_model, limiter_slot_held
        completed = False
        try:
            async for part in router.iter_chat_completion_openai_stream(
                payload,
                required_tag=required_tag,
                require_assistant_content=required_tag == "web-search",
            ):
                if isinstance(part, RouteStreamDiag):
                    await _publish_route_stream_diag(monitor, request_id, part)
                    if part.event_type == "route_selected":
                        selected_provider = part.provider_name or ""
                        selected_route = part.route_id or ""
                        selected_model = part.model_id or ""
                        if (
                            settings.streaming_release_slot_after_route_selected
                            and limiter_slot_held
                        ):
                            limiter.release()
                            limiter_slot_held = False
                else:
                    yield part
            completed = True
        except NoProviderAvailable as exc:
            await monitor.publish(
                event_type="request_failed",
                request_id=request_id,
                payload={
                    "status_code": 503,
                    "reason": "waterfall_exhausted",
                    "attempts": len(exc.attempts),
                    "attempts_detail": [asdict(attempt) for attempt in exc.attempts],
                    "latency_ms": round((perf_counter() - started_at) * 1000),
                },
            )
            yield (
                "data: "
                + json.dumps(
                    {
                        "error": {
                            "message": "No configured provider is currently available",
                            "type": "provider_unavailable",
                            "code": "waterfall_exhausted",
                            "attempts": [asdict(attempt) for attempt in exc.attempts],
                        }
                    }
                )
                + "\n\n"
            )
        except ValueError as exc:
            await monitor.publish(
                event_type="request_failed",
                request_id=request_id,
                payload={
                    "status_code": 400,
                    "reason": "validation_error",
                    "message": str(exc),
                    "latency_ms": round((perf_counter() - started_at) * 1000),
                },
            )
            yield "data: " + json.dumps({"error": {"message": str(exc), "type": "invalid_request"}}) + "\n\n"
        finally:
            if limiter_slot_held:
                limiter.release()
                limiter_slot_held = False
            if completed:
                await monitor.publish(
                    event_type="request_completed",
                    request_id=request_id,
                    payload={
                        "status_code": 200,
                        "provider_name": selected_provider,
                        "route_id": selected_route,
                        "model_id": selected_model,
                        "latency_ms": round((perf_counter() - started_at) * 1000),
                    },
                )

    return StreamingResponse(
        openai_sse_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _route_chat_completion_request(
    request: Request,
    *,
    payload: dict[str, Any],
    path: str,
    required_tag: str | None = None,
) -> JSONResponse:
    router: WaterfallRouter = request.app.state.waterfall_router
    limiter: GatewayRequestLimiter = request.app.state.request_limiter
    monitor: APILiveMonitor = request.app.state.live_monitor
    request_id = uuid.uuid4().hex[:12]
    started_at = perf_counter()
    client_ip = request.client.host if request.client else None
    await monitor.publish(
        event_type="request_started",
        request_id=request_id,
        payload={
            "path": path,
            "stream": bool(payload.get("stream")),
            "model": payload.get("model"),
            "client_ip": client_ip,
            "request_payload": _monitor_trim(payload),
        },
    )

    if not await limiter.acquire():
        await monitor.publish(
            event_type="request_rejected",
            request_id=request_id,
            payload={"status_code": 429, "reason": "request_queue_timeout"},
        )
        return JSONResponse(
            status_code=429,
            content={
                "error": {
                    "message": "Gateway is busy; retry shortly",
                    "type": "gateway_overloaded",
                    "code": "request_queue_timeout",
                }
            },
            headers={"Retry-After": "1"},
        )

    try:
        result = await router.route_chat_completion(
            payload,
            required_tag=required_tag,
            require_assistant_content=required_tag == "web-search",
        )
    except ValueError as exc:
        await monitor.publish(
            event_type="request_failed",
            request_id=request_id,
            payload={
                "status_code": 400,
                "reason": "validation_error",
                "message": str(exc),
                "response_body": {"detail": str(exc)},
            },
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except NoProviderAvailable as exc:
        await monitor.publish(
            event_type="request_failed",
            request_id=request_id,
            payload={
                "status_code": 503,
                "reason": "waterfall_exhausted",
                "attempts": len(exc.attempts),
                "attempts_detail": [asdict(attempt) for attempt in exc.attempts],
                "latency_ms": round((perf_counter() - started_at) * 1000),
            },
        )
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
        await monitor.publish(
            event_type="request_failed",
            request_id=request_id,
            payload={
                "status_code": exc.status_code or 502,
                "reason": "provider_error",
                "message": str(exc),
                "response_body": _monitor_trim(
                    {
                        "error": {
                            "message": str(exc),
                            "status_code": exc.status_code,
                            "body": exc.body,
                        }
                    }
                ),
                "latency_ms": round((perf_counter() - started_at) * 1000),
            },
        )
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
    finally:
        limiter.release()

    await monitor.publish(
        event_type="request_completed",
        request_id=request_id,
        payload={
            "status_code": 200,
            "provider_name": result.provider_name,
            "route_id": result.route_id,
            "model_id": result.model_id,
            "attempts": len(result.attempts),
            "usage": _monitor_trim(
                result.body.get("usage") if isinstance(result.body, dict) else None
            ),
            "attempts_detail": [asdict(attempt) for attempt in result.attempts],
            "assistant_text": _monitor_trim(
                _assistant_text_from_response_body(result.body),
                max_string=8000,
            ),
            "response_body": _monitor_trim(result.body),
            "latency_ms": round((perf_counter() - started_at) * 1000),
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


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    payload: dict[str, Any] = await request.json()
    if payload.get("stream"):
        return await _route_chat_completion_stream_request(
            request,
            payload=payload,
            path="/v1/chat/completions",
        )
    return await _route_chat_completion_request(
        request,
        payload=payload,
        path="/v1/chat/completions",
    )


@app.post("/v1/chat/completions/web-search")
async def chat_completions_web_search(request: Request) -> Response:
    payload: Any = await request.json()
    try:
        prepared_payload = _payload_with_required_web_search(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if prepared_payload.get("stream"):
        return await _route_chat_completion_stream_request(
            request,
            payload=prepared_payload,
            path="/v1/chat/completions/web-search",
            required_tag="web-search",
        )
    return await _route_chat_completion_request(
        request,
        payload=prepared_payload,
        path="/v1/chat/completions/web-search",
        required_tag="web-search",
    )


@app.post("/v1/chat/completions/stream-route")
async def chat_completions_stream_route(request: Request) -> Response:
    payload: dict[str, Any] = await request.json()
    router: WaterfallRouter = request.app.state.waterfall_router
    limiter: GatewayRequestLimiter = request.app.state.request_limiter
    monitor: APILiveMonitor = request.app.state.live_monitor
    request_id = uuid.uuid4().hex[:12]
    started_at = perf_counter()
    client_ip = request.client.host if request.client else None
    done_published = False
    await monitor.publish(
        event_type="request_started",
        request_id=request_id,
        payload={
            "path": "/v1/chat/completions/stream-route",
            "stream": True,
            "model": payload.get("model"),
            "client_ip": client_ip,
            "request_payload": _monitor_trim(payload),
        },
    )
    if not await limiter.acquire():
        await monitor.publish(
            event_type="request_rejected",
            request_id=request_id,
            payload={"status_code": 429, "reason": "request_queue_timeout"},
        )
        return JSONResponse(
            status_code=429,
            content={
                "error": {
                    "message": "Gateway is busy; retry shortly",
                    "type": "gateway_overloaded",
                    "code": "request_queue_timeout",
                }
            },
            headers={"Retry-After": "1"},
        )

    stream_settings = get_settings()
    limiter_slot_held = True

    async def on_stream_event(event_payload: dict[str, Any]) -> None:
        nonlocal done_published, limiter_slot_held
        event_type = event_payload.get("type")
        if event_type == "usage":
            usage = event_payload.get("usage")
            if isinstance(usage, dict) and usage:
                await monitor.publish(
                    event_type="usage_update",
                    request_id=request_id,
                    payload={"usage": _monitor_trim(usage)},
                )
            return
        if event_type == "route_selected":
            await monitor.publish(
                event_type="route_selected",
                request_id=request_id,
                payload={
                    "provider_name": event_payload.get("provider"),
                    "route_id": event_payload.get("route_id"),
                    "model_id": event_payload.get("model_id"),
                    "stream_event": _monitor_trim(event_payload),
                },
            )
            if (
                stream_settings.streaming_release_slot_after_route_selected
                and limiter_slot_held
            ):
                limiter.release()
                limiter_slot_held = False
            return
        if event_type in {"route_trying", "route_skip", "route_fail", "route_flagged"}:
            await monitor.publish(
                event_type="route_attempt",
                request_id=request_id,
                payload={
                    "route_event": {
                        "type": event_type,
                        "provider_name": event_payload.get("provider"),
                        "route_id": event_payload.get("route_id"),
                        "model_id": event_payload.get("model_id"),
                        "reason": event_payload.get("reason"),
                    }
                },
            )
            return
        if event_type == "content":
            await monitor.publish(
                event_type="response_content",
                request_id=request_id,
                payload={
                    "stream_event": _monitor_trim(event_payload),
                    "content": event_payload.get("text") or "",
                },
            )
            return
        if event_type == "error":
            await monitor.publish(
                event_type="request_failed",
                request_id=request_id,
                payload={
                    "status_code": 502,
                    "reason": "stream_error",
                    "message": event_payload.get("message"),
                    "response_body": _monitor_trim(event_payload),
                    "latency_ms": round((perf_counter() - started_at) * 1000),
                },
            )
            done_published = True
            return
        if event_type == "done":
            await monitor.publish(
                event_type="request_completed",
                request_id=request_id,
                payload={
                    "status_code": 200,
                    "provider_name": event_payload.get("provider"),
                    "route_id": event_payload.get("route_id"),
                    "model_id": event_payload.get("model_id"),
                    "route_event": {
                        "type": "success",
                        "provider_name": event_payload.get("provider"),
                        "route_id": event_payload.get("route_id"),
                        "model_id": event_payload.get("model_id"),
                    },
                    "assistant_text": _monitor_trim(
                        event_payload.get("content") or "",
                        max_string=8000,
                    ),
                    "response_body": _monitor_trim(event_payload),
                    "latency_ms": round((perf_counter() - started_at) * 1000),
                },
            )
            done_published = True

    async def limited_stream():
        nonlocal done_published, limiter_slot_held
        try:
            async for chunk in stream_route_chat(
                payload,
                router,
                chunk_replay_sleep_seconds=get_settings().sse_chunk_replay_sleep_seconds,
                on_emit=on_stream_event,
            ):
                yield chunk
        finally:
            if limiter_slot_held:
                limiter.release()
                limiter_slot_held = False
            if not done_published:
                await monitor.publish(
                    event_type="request_closed",
                    request_id=request_id,
                    payload={
                        "status_code": 499,
                        "reason": "client_closed_or_stream_interrupted",
                        "latency_ms": round((perf_counter() - started_at) * 1000),
                    },
                )

    return StreamingResponse(
        limited_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


ROUTE_HEALTH_HTML = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Route Health - FreeRouter</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
      *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
      :root { --bg-primary: #0a0e1a; --bg-secondary: #111827; --bg-tertiary: #1e293b; --border: #2d3a4f; --text: #e2e8f0; --text-muted: #94a3b8; --accent: #3b82f6; --green: #22c55e; --red: #ef4444; --amber: #f59e0b; --font: 'Inter', system-ui, sans-serif; }
      body { font-family: var(--font); background: var(--bg-primary); color: var(--text); }
      nav { display: flex; align-items: center; gap: 1rem; padding: 0.75rem 1.5rem; background: var(--bg-secondary); border-bottom: 1px solid var(--border); }
      nav h1 { font-size: 1rem; font-weight: 700; background: linear-gradient(135deg, #60a5fa, #a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
      nav a { color: var(--text-muted); text-decoration: none; font-size: 0.85rem; }
      nav a:hover { color: var(--text); }
      .nav-spacer { flex: 1; }
      main { max-width: 1100px; margin: auto; padding: 2rem; }
      h2 { margin-bottom: 0.5rem; }
      .muted { color: var(--text-muted); }
      .toolbar { display: flex; justify-content: space-between; align-items: center; gap: 1rem; margin: 1.5rem 0; }
      button { border: none; border-radius: 8px; background: var(--accent); color: white; padding: 0.55rem 0.8rem; font: inherit; cursor: pointer; }
      button.secondary { background: var(--bg-tertiary); border: 1px solid var(--border); color: var(--text); }
      .grid { display: grid; gap: 0.75rem; }
      .card { display: grid; grid-template-columns: 1fr auto; gap: 1rem; padding: 1rem; background: var(--bg-secondary); border: 1px solid var(--border); border-radius: 12px; }
      .provider { color: var(--green); text-transform: uppercase; letter-spacing: 0.05em; font-size: 0.75rem; font-weight: 700; }
      .model { margin-top: 0.25rem; color: var(--text-muted); font-size: 0.85rem; }
      .pill { align-self: start; padding: 0.25rem 0.55rem; border-radius: 999px; border: 1px solid var(--border); color: var(--text-muted); font-size: 0.78rem; }
      .pill.rate_limited { border-color: rgba(245,158,11,0.5); color: #fcd34d; background: rgba(245,158,11,0.12); }
      .pill.potentially_outdated, .pill.too_slow { border-color: rgba(239,68,68,0.5); color: #fecaca; background: rgba(239,68,68,0.12); }
      .empty { padding: 2rem; text-align: center; color: var(--text-muted); background: var(--bg-secondary); border: 1px solid var(--border); border-radius: 12px; }
    </style>
  </head>
  <body>
    <nav>
      <h1>FreeRouter</h1><span class="nav-spacer"></span>
      <a href="/">Home</a><a href="/chat">Chat</a><a href="/models">Models</a><a href="/health">Health</a><a href="/status">Provider Usage</a><a href="/live">Live Traffic</a>
    </nav>
    <main>
      <h2>Route Health</h2>
      <p class="muted">Routes automatically limited by FreeRouter are listed here. Active routes are hidden.</p>
      <div class="toolbar"><span id="summary" class="muted">Loading...</span><button id="reload">Reload</button></div>
      <div id="routes" class="grid"></div>
    </main>
    <script>
      const routesEl = document.getElementById('routes');
      const summaryEl = document.getElementById('summary');
      const esc = (value) => String(value ?? '').replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));
      const formatTime = (value) => value ? new Date(value * 1000).toLocaleString() : 'not scheduled';
      async function load() {
        const response = await fetch('/v1/gateway/models');
        const payload = await response.json();
        const limited = payload.data.filter((route) => route.health && route.health.status !== 'active');
        summaryEl.textContent = `${limited.length} automatically limited route${limited.length === 1 ? '' : 's'}`;
        routesEl.innerHTML = limited.length ? limited.map((route) => `
          <div class="card">
            <div>
              <div class="provider">${esc(route.provider_name)}</div>
              <strong>${esc(route.display_name)}</strong>
              <div class="model">${esc(route.model_id)}</div>
              <div class="model">Reason: ${esc(route.health.status_reason || route.health.status)} · failures: ${route.health.consecutive_failures}</div>
              <div class="model">Next probe: ${esc(formatTime(route.health.next_probe_at))}</div>
              <div class="model"><button class="secondary" data-route-id="${esc(route.route_id)}">Clear flag</button></div>
            </div>
            <span class="pill ${esc(route.health.status)}">${esc(route.health.status.replace(/_/g, ' '))}</span>
          </div>
        `).join('') : '<div class="empty">No routes are currently automatically limited.</div>';
        routesEl.querySelectorAll('[data-route-id]').forEach((button) => {
          button.addEventListener('click', async () => {
            button.disabled = true;
            button.textContent = 'Clearing...';
            const response = await fetch(`/v1/gateway/models/${encodeURIComponent(button.dataset.routeId)}/health/reset`, { method: 'POST' });
            if (response.ok) load();
            else button.textContent = 'Failed';
          });
        });
      }
      document.getElementById('reload').addEventListener('click', load);
      load();
    </script>
  </body>
</html>
"""


LIVE_API_HTML = r"""
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Live API Traffic - FreeRouter</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
      *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
      :root { --bg-primary: #0a0e1a; --bg-secondary: #111827; --bg-tertiary: #1e293b; --border: #2d3a4f; --text: #e2e8f0; --text-muted: #94a3b8; --accent: #3b82f6; --font: 'Inter', system-ui, sans-serif; --ok: #22c55e; --warn: #f59e0b; --bad: #ef4444; }
      body { font-family: var(--font); background: var(--bg-primary); color: var(--text); }
      nav { display: flex; align-items: center; gap: 1rem; padding: 0.75rem 1.5rem; background: var(--bg-secondary); border-bottom: 1px solid var(--border); }
      nav h1 { font-size: 1rem; font-weight: 700; background: linear-gradient(135deg, #60a5fa, #a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
      nav a { color: var(--text-muted); text-decoration: none; font-size: 0.85rem; }
      nav a:hover { color: var(--text); }
      .nav-spacer { flex: 1; }
      main { max-width: 1280px; margin: auto; padding: 1.5rem; }
      .toolbar { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; gap: 1rem; flex-wrap: wrap; }
      .badge { border: 1px solid var(--border); border-radius: 999px; padding: 0.2rem 0.6rem; color: var(--text-muted); background: var(--bg-tertiary); font-size: 0.78rem; }
      .badge.live { border-color: rgba(34, 197, 94, 0.45); color: #bbf7d0; background: rgba(34, 197, 94, 0.14); }
      .table-wrap { border: 1px solid var(--border); border-radius: 12px; overflow: auto; background: var(--bg-secondary); }
      table { width: 100%; border-collapse: collapse; min-width: 980px; }
      th, td { padding: 0.6rem 0.75rem; border-bottom: 1px solid var(--border); text-align: left; font-size: 0.82rem; vertical-align: top; }
      th { color: var(--text-muted); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.04em; background: rgba(15, 23, 42, 0.75); position: sticky; top: 0; z-index: 1; }
      tr:hover { background: rgba(59, 130, 246, 0.08); }
      .ok { color: var(--ok); } .warn { color: var(--warn); } .bad { color: var(--bad); }
      code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; color: #93c5fd; }
      .muted { color: var(--text-muted); }
      .expand-btn { border: 1px solid var(--border); border-radius: 6px; background: var(--bg-tertiary); color: var(--text); font-size: 0.75rem; padding: 0.2rem 0.45rem; cursor: pointer; }
      .expand-btn:hover { background: var(--border); }
      .details-row td { background: var(--bg-primary); }
      .details-wrap { display: grid; gap: 0.75rem; }
      .details-wrap .md-body {
        border: 1px solid var(--border); background: var(--bg-secondary); border-radius: 8px; padding: 0.65rem 0.85rem;
        font-size: 0.8rem; line-height: 1.45; color: var(--text); max-height: min(28rem, 70vh); overflow: auto;
      }
      .details-wrap .md-body p { margin: 0.35rem 0; }
      .details-wrap .md-body p:first-child { margin-top: 0; }
      .details-wrap .md-body p:last-child { margin-bottom: 0; }
      .details-wrap .md-body h1, .details-wrap .md-body h2, .details-wrap .md-body h3,
      .details-wrap .md-body h4, .details-wrap .md-body h5, .details-wrap .md-body h6 {
        margin: 0.6rem 0 0.35rem; font-weight: 600; color: #f8fafc;
      }
      .details-wrap .md-body h1 { font-size: 1.05rem; }
      .details-wrap .md-body h2 { font-size: 0.98rem; }
      .details-wrap .md-body h3 { font-size: 0.92rem; }
      .details-wrap .md-body ul, .details-wrap .md-body ol { margin: 0.35rem 0 0.35rem 1.15rem; padding: 0; }
      .details-wrap .md-body li { margin: 0.15rem 0; }
      .details-wrap .md-body a { color: #93c5fd; text-decoration: underline; text-underline-offset: 2px; }
      .details-wrap .md-body pre { margin: 0.45rem 0; white-space: pre-wrap; word-break: break-word; border: 1px solid var(--border); background: var(--bg-primary); border-radius: 6px; padding: 0.55rem 0.65rem; font-size: 0.74rem; color: var(--text-muted); }
      .details-wrap .md-body .md-table-wrap { overflow-x: auto; max-width: 100%; margin: 0.35rem 0; }
      .details-wrap .md-body table.md-table { border-collapse: collapse; font-size: 0.85em; width: max-content; max-width: 100%; }
      .details-wrap .md-body table.md-table th,
      .details-wrap .md-body table.md-table td { border: 1px solid var(--border); padding: 0.3rem 0.45rem; vertical-align: top; }
      .details-wrap .md-body table.md-table th { background: rgba(30, 41, 59, 0.65); font-weight: 600; color: var(--text); }
      .details-wrap pre { margin: 0; white-space: pre-wrap; word-break: break-word; border: 1px solid var(--border); background: var(--bg-secondary); border-radius: 8px; padding: 0.65rem; font-size: 0.75rem; color: var(--text-muted); }
      .attempts { display: grid; gap: 0.45rem; position: relative; }
      .attempt { position: relative; display: grid; grid-template-columns: 1.9rem 1fr auto; gap: 0.68rem; align-items: center; border: 1px solid rgba(148,163,184,0.16); background: linear-gradient(180deg, rgba(30,41,59,0.5), rgba(15,23,42,0.55)); border-radius: 10px; padding: 0.58rem 0.7rem; }
      .attempt::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px; border-radius: 10px 0 0 10px; background: rgba(100,116,139,0.45); }
      .attempt.ok::before { background: rgba(34,197,94,0.7); }
      .attempt.warn::before { background: rgba(245,158,11,0.75); }
      .attempt.bad::before { background: rgba(239,68,68,0.75); }
      .attempt-marker { width: 1.25rem; height: 1.25rem; border-radius: 999px; border: 1px solid var(--border); display: inline-flex; align-items: center; justify-content: center; font-size: 0.66rem; color: var(--text-muted); background: var(--bg-primary); }
      .attempt-main { display: grid; gap: 0.16rem; min-width: 0; }
      .attempt-title { color: var(--text); font-size: 0.81rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
      .attempt-sub { color: var(--text-muted); font-size: 0.73rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
      .attempt-sub .label { color: #93c5fd; opacity: 0.9; }
      .attempt-meta { display: inline-flex; gap: 0.35rem; align-items: center; }
      .attempt-status { font-size: 0.67rem; border: 1px solid var(--border); border-radius: 999px; padding: 0.12rem 0.44rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-muted); background: rgba(15,23,42,0.55); white-space: nowrap; font-weight: 600; }
      .attempt-status.selected, .attempt-status.done { color: #bbf7d0; border-color: rgba(34,197,94,0.45); background: rgba(34,197,94,0.12); }
      .attempt-status.failed, .attempt-status.route_fail { color: #fecaca; border-color: rgba(239,68,68,0.45); background: rgba(239,68,68,0.12); }
      .attempt-status.skipped, .attempt-status.route_skip { color: #fcd34d; border-color: rgba(245,158,11,0.45); background: rgba(245,158,11,0.12); }
      .attempt-status.flagged, .attempt-status.route_flagged { color: #ddd6fe; border-color: rgba(167,139,250,0.5); background: rgba(167,139,250,0.14); }
      .attempt-http { color: var(--text-muted); font-size: 0.7rem; border: 1px solid var(--border); border-radius: 999px; padding: 0.1rem 0.38rem; white-space: nowrap; }
      .attempt-empty { color: var(--text-muted); font-size: 0.78rem; border: 1px dashed var(--border); border-radius: 8px; padding: 0.6rem; }
    </style>
  </head>
  <body>
    <nav>
      <h1>FreeRouter</h1><span class="nav-spacer"></span>
      <a href="/">Home</a><a href="/chat">Chat</a><a href="/models">Models</a><a href="/health">Health</a><a href="/status">Provider Usage</a><a href="/live">Live Traffic</a>
    </nav>
    <main>
      <div class="toolbar">
        <div>
          <h2>Live API Traffic</h2>
          <p class="muted">Real-time API requests coming through this gateway.</p>
        </div>
        <div>
          <span id="liveBadge" class="badge">Connecting...</span>
          <span id="countBadge" class="badge">0 events</span>
        </div>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr><th>Time</th><th>Phase</th><th>Request</th><th title="Usage fields from upstream JSON when provided (stream often sends only at end of SSE)">Upstream usage</th><th>Model</th><th>Status</th><th>Provider/Route</th><th>Latency</th><th>Details</th><th>Expand</th></tr>
          </thead>
          <tbody id="rows"></tbody>
        </table>
      </div>
    </main>
    <script>
      const rowsEl = document.getElementById('rows');
      const liveBadge = document.getElementById('liveBadge');
      const countBadge = document.getElementById('countBadge');
      const maxRows = 250;
      const requests = new Map();
      const seenEventIds = new Set();
      const expanded = new Set();
      function formatProviderUsage(u) {
        if (!u || typeof u !== 'object') return '';
        const num = (x) => (x == null || x === '') ? NaN : Number(x);
        const total = num(u.total_tokens);
        if (!Number.isNaN(total)) return String(total);
        const prompt = num(u.prompt_tokens);
        const completion = num(u.completion_tokens);
        const parts = [];
        if (!Number.isNaN(prompt)) parts.push(prompt);
        if (!Number.isNaN(completion)) parts.push(completion);
        return parts.length ? parts.join(' + ') : '';
      }
      const esc = (v) => String(v ?? '').replace(/[&<>"']/g, (c) => ({ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;' }[c]));
      function escapeHtmlDom(s) {
        const d = document.createElement('div');
        d.textContent = s;
        return d.innerHTML;
      }
      function renderInlineMarkdown(text) {
        return escapeHtmlDom(text)
          .replace(/`([^`]+)`/g, '<code>$1</code>')
          .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
          .replace(/\*([^*]+)\*/g, '<em>$1</em>')
          .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>');
      }
      function parseTableRow(line) {
        const t = line.trim();
        if (!t.includes('|')) return null;
        const parts = t.split('|');
        if (parts.length < 3) return null;
        return parts.slice(1, -1).map((c) => c.trim());
      }
      function isSeparatorRow(cells) {
        return cells.length > 0 && cells.every((c) => /^:?-{3,}:?$/.test(c.trim()));
      }
      function cellAlign(cell) {
        const c = cell.trim();
        if (/^:-+$/.test(c)) return 'left';
        if (/^-+:$/.test(c)) return 'right';
        if (/^:-+:$/.test(c)) return 'center';
        return 'left';
      }
      function renderTableHtml(header, align, bodyRows) {
        const ths = header.map((h, idx) => {
          const a = align[idx] || 'left';
          return `<th style="text-align:${a}">${renderInlineMarkdown(h)}</th>`;
        });
        const trs = bodyRows.map((row) =>
          `<tr>${row.map((cell, idx) => {
            const a = align[idx] || 'left';
            return `<td style="text-align:${a}">${renderInlineMarkdown(cell)}</td>`;
          }).join('')}</tr>`
        );
        return `<div class="md-table-wrap"><table class="md-table"><thead><tr>${ths.join('')}</tr></thead><tbody>${trs.join('')}</tbody></table></div>`;
      }
      function detectGFMTable(lines, start) {
        if (start + 1 >= lines.length) return null;
        const row0 = lines[start].trim();
        const row1 = lines[start + 1].trim();
        if (!row0 || !row1) return null;
        const header = parseTableRow(row0);
        const sepCells = parseTableRow(row1);
        if (!header || !sepCells || header.length !== sepCells.length) return null;
        if (!isSeparatorRow(sepCells)) return null;
        const align = sepCells.map(cellAlign);
        const body = [];
        let j = start + 2;
        while (j < lines.length) {
          const tr = lines[j].trim();
          if (!tr) break;
          const row = parseTableRow(tr);
          if (!row || row.length !== header.length) break;
          body.push(row);
          j++;
        }
        return { html: renderTableHtml(header, align, body), nextIndex: j };
      }
      function renderMarkdown(markdown) {
        const codeBlocks = [];
        const protectedText = String(markdown || '').replace(/```(\w+)?\n?([\s\S]*?)```/g, (_, lang, code) => {
          const token = `\u0000CODE${codeBlocks.length}\u0000`;
          codeBlocks.push(`<pre><code>${escapeHtmlDom(code.replace(/\n$/, ''))}</code></pre>`);
          return token;
        });
        const lines = protectedText.split('\n');
        const blocks = [];
        let paragraph = [];
        let listItems = [];
        let orderedItems = [];
        function flushParagraph() {
          if (!paragraph.length) return;
          blocks.push(`<p>${paragraph.map(renderInlineMarkdown).join('<br>')}</p>`);
          paragraph = [];
        }
        function flushLists() {
          if (listItems.length) {
            blocks.push(`<ul>${listItems.map((item) => `<li>${renderInlineMarkdown(item)}</li>`).join('')}</ul>`);
            listItems = [];
          }
          if (orderedItems.length) {
            blocks.push(`<ol>${orderedItems.map((item) => `<li>${renderInlineMarkdown(item)}</li>`).join('')}</ol>`);
            orderedItems = [];
          }
        }
        function flushAll() {
          flushParagraph();
          flushLists();
        }
        let i = 0;
        while (i < lines.length) {
          const rawLine = lines[i];
          const line = rawLine.trimEnd();
          const trimmed = line.trim();
          const codeMatch = trimmed.match(/^\u0000CODE(\d+)\u0000$/);
          if (codeMatch) {
            flushAll();
            blocks.push(codeBlocks[Number(codeMatch[1])]);
            i++;
            continue;
          }
          if (!trimmed) {
            flushAll();
            i++;
            continue;
          }
          const tbl = detectGFMTable(lines, i);
          if (tbl) {
            flushAll();
            blocks.push(tbl.html);
            i = tbl.nextIndex;
            continue;
          }
          const heading = trimmed.match(/^(#{1,6})\s+(.+)$/);
          if (heading) {
            flushAll();
            const level = Math.min(6, heading[1].length);
            blocks.push(`<h${level}>${renderInlineMarkdown(heading[2])}</h${level}>`);
            i++;
            continue;
          }
          const bullet = trimmed.match(/^[-*]\s+(.+)$/);
          if (bullet) {
            flushParagraph();
            orderedItems = [];
            listItems.push(bullet[1]);
            i++;
            continue;
          }
          const ordered = trimmed.match(/^\d+\.\s+(.+)$/);
          if (ordered) {
            flushParagraph();
            listItems = [];
            orderedItems.push(ordered[1]);
            i++;
            continue;
          }
          flushLists();
          paragraph.push(trimmed);
          i++;
        }
        flushAll();
        return blocks.join('');
      }
      const when = (ts) => ts ? new Date(Number(ts) * 1000).toLocaleTimeString() : '';
      const isUsefulText = (value) => typeof value === 'string' && value.trim() && value.trim() !== '<truncated-depth>';
      function statusClass(status) {
        if (status == null) return '';
        if (Number(status) >= 500 || Number(status) === 499) return 'bad';
        if (Number(status) >= 400) return 'warn';
        return 'ok';
      }
      function normalizeEvent(evt) {
        if (!evt || !evt.request_id) return null;
        if (evt.event_id != null) {
          if (seenEventIds.has(evt.event_id)) return null;
          seenEventIds.add(evt.event_id);
        }
        const payload = evt.payload || {};
        const current = requests.get(evt.request_id) || {
          request_id: evt.request_id,
          first_ts: evt.timestamp,
          last_ts: evt.timestamp,
          path: payload.path || '',
          model: payload.model || payload.model_id || '',
          tokens: '',
          status_code: null,
          provider_name: '',
          route_id: '',
          latency_ms: null,
          phase: 'started',
          reason: '',
          request_payload: null,
          response_body: null,
          stream_event: null,
          assistant_text: '',
          route_attempts: [],
        };
        current.last_ts = evt.timestamp || current.last_ts;
        current.path = payload.path || current.path;
        current.model = payload.model || payload.model_id || current.model;
        if (payload.usage && typeof payload.usage === 'object') {
          const formatted = formatProviderUsage(payload.usage);
          current.tokens = formatted || 'Unknown';
        } else if (payload.request_payload?.max_tokens || payload.request_payload?.max_completion_tokens) {
          current.tokens = 'Unknown';
        }
        current.provider_name = payload.provider_name || current.provider_name;
        current.route_id = payload.route_id || current.route_id;
        if (payload.status_code != null) current.status_code = payload.status_code;
        if (payload.latency_ms != null) current.latency_ms = payload.latency_ms;
        current.reason = payload.reason || payload.message || current.reason;
        if (payload.request_payload != null) current.request_payload = payload.request_payload;
        if (isUsefulText(payload.assistant_text)) {
          current.assistant_text = payload.assistant_text;
        }
        if (payload.response_body != null) {
          current.response_body = payload.response_body;
          const extracted = aiResponse(payload.response_body);
          if (isUsefulText(extracted)) current.assistant_text = extracted;
        }
        if (payload.stream_event != null) current.stream_event = payload.stream_event;
        if (evt.event_type === 'response_content' && typeof payload.content === 'string') {
          current.assistant_text = (current.assistant_text || '') + payload.content;
        }
        if (Array.isArray(payload.attempts_detail)) {
          current.route_attempts = payload.attempts_detail.map((attempt) => ({
            status: attempt.status || '',
            provider_name: attempt.provider_name || '',
            route_id: attempt.route_id || '',
            model_id: attempt.model_id || '',
            reason: attempt.reason || '',
            status_code: attempt.status_code ?? null,
          }));
        }
        if (payload.route_event) {
          current.route_attempts.push({
            status: payload.route_event.type || '',
            provider_name: payload.route_event.provider_name || '',
            route_id: payload.route_event.route_id || '',
            model_id: payload.route_event.model_id || '',
            reason: payload.route_event.reason || '',
            status_code: payload.route_event.status_code ?? null,
          });
          if (current.route_attempts.length > 40) {
            current.route_attempts = current.route_attempts.slice(-40);
          }
        }

        if (evt.event_type === 'request_started') current.phase = 'in_progress';
        else if (evt.event_type === 'route_selected' && current.phase !== 'completed') current.phase = 'routing';
        else if (evt.event_type === 'request_completed') current.phase = 'completed';
        else if (evt.event_type === 'request_failed') current.phase = 'failed';
        else if (evt.event_type === 'request_rejected') current.phase = 'rejected';
        else if (evt.event_type === 'request_closed') current.phase = 'closed';

        requests.set(evt.request_id, current);
        if (requests.size > maxRows) {
          const oldest = [...requests.values()].sort((a, b) => (a.last_ts || 0) - (b.last_ts || 0))[0];
          if (oldest) requests.delete(oldest.request_id);
        }
        return current;
      }
      function phaseLabel(phase) {
        if (phase === 'completed') return 'Done';
        if (phase === 'failed') return 'Failed';
        if (phase === 'rejected') return 'Rejected';
        if (phase === 'closed') return 'Closed';
        if (phase === 'routing') return 'Routing';
        return 'In progress';
      }
      function phaseClass(phase, statusCode) {
        if (phase === 'completed') return 'ok';
        if (phase === 'failed' || phase === 'rejected' || phase === 'closed') return 'bad';
        return statusClass(statusCode);
      }
      function addEvent(evt) {
        if (!normalizeEvent(evt)) return;
        render();
      }
      function toggleExpand(requestId) {
        if (expanded.has(requestId)) expanded.delete(requestId);
        else expanded.add(requestId);
        render();
      }
      function pretty(value) {
        try { return JSON.stringify(value ?? {}, null, 2); } catch { return String(value); }
      }
      function userMessage(payload) {
        const messages = payload?.messages;
        if (!Array.isArray(messages)) return '';
        const lastUser = [...messages].reverse().find((message) => message?.role === 'user');
        const content = lastUser?.content ?? '';
        if (typeof content === 'string') return content;
        if (Array.isArray(content)) {
          return content.map((item) => typeof item === 'string' ? item : (item?.text || item?.content || '')).filter(Boolean).join('\\n');
        }
        return String(content || '');
      }
      function aiResponse(body) {
        if (!body) return '';
        if (body === '<truncated-depth>') return '';
        if (isUsefulText(body.content)) return body.content;
        if (isUsefulText(body.text)) return body.text;
        if (isUsefulText(body.message?.content)) return body.message.content;
        const message = body.choices?.[0]?.message?.content;
        if (isUsefulText(message)) return message;
        if (Array.isArray(message)) {
          return message
            .map((item) => typeof item === 'string' ? item : (item?.text || item?.content || ''))
            .filter(isUsefulText)
            .join('\\n');
        }
        return '';
      }
      function statusLabel(status) {
        const raw = String(status || '').trim();
        if (!raw) return 'Attempt';
        return raw.replace(/_/g, ' ');
      }
      function renderAttempts(attempts) {
        if (!attempts || !attempts.length) {
          return '<div class="attempt-empty">No route attempts captured for this request.</div>';
        }
        const canonicalStatus = (raw) => {
          const status = String(raw || '').toLowerCase();
          if (status === 'route_trying' || status === 'trying') return 'trying';
          if (status.includes('selected') || status.includes('done') || status === 'success') return 'success';
          if (status.includes('flag')) return 'flagged';
          if (status.includes('fail') || status.includes('error') || status === 'timeout') return 'failed';
          if (status.includes('skip')) return 'skipped';
          return status || 'unknown';
        };
        const priority = (status) => {
          if (status === 'success') return 5;
          if (status === 'flagged') return 4;
          if (status === 'failed') return 3;
          if (status === 'skipped') return 2;
          return 1;
        };
        const grouped = new Map();
        for (let index = 0; index < attempts.length; index += 1) {
          const item = attempts[index];
          const status = canonicalStatus(item.status);
          if (status === 'trying') continue;
          const key = [item.provider_name || '', item.model_id || '', item.route_id || ''].join('|');
          const existing = grouped.get(key);
          if (!existing) {
            grouped.set(key, { ...item, status, _order: index });
            continue;
          }
          if (priority(status) >= priority(existing.status)) {
            grouped.set(key, { ...existing, ...item, status, _order: existing._order });
          }
        }
        const consolidated = [...grouped.values()].sort((a, b) => {
          const successDelta = Number(a.status === 'success') - Number(b.status === 'success');
          if (successDelta !== 0) return successDelta;
          return (a._order || 0) - (b._order || 0);
        });
        if (!consolidated.length) {
          return '<div class="attempt-empty">No completed route attempts yet.</div>';
        }
        return `<div class="attempts">${consolidated.map((item, idx) => {
          const status = canonicalStatus(item.status);
          const provider = item.provider_name || 'unknown';
          const model = item.model_id || 'unknown-model';
          const route = item.route_id || 'n/a';
          const reason = item.reason || '';
          const code = item.status_code != null ? `HTTP ${item.status_code}` : '';
          const marker = status === 'success' ? '✓' : (status === 'failed' ? 'x' : (status === 'flagged' ? '⚑' : '!'));
          const tone = status === 'success' ? 'ok' : (status === 'failed' ? 'bad' : 'warn');
          const reasonText = reason ? reason.replace(/_/g, ' ') : '';
          return `<div class="attempt ${tone}">
            <span class="attempt-marker">${esc(marker)}</span>
            <div class="attempt-main">
              <div class="attempt-title">${idx + 1}. ${esc(provider)} / <code>${esc(model)}</code></div>
              <div class="attempt-sub"><span class="label">Route:</span> <code>${esc(route)}</code>${reasonText ? ` · <span class="label">Reason:</span> ${esc(reasonText)}` : ''}</div>
            </div>
            <div class="attempt-meta">
              <span class="attempt-status ${esc(status)}">${esc(statusLabel(status))}</span>
              ${code ? `<span class="attempt-http">${esc(code)}</span>` : ''}
            </div>
          </div>`;
        }).join('')}</div>`;
      }
      function render() {
        const rows = [...requests.values()].sort((a, b) => (b.last_ts || 0) - (a.last_ts || 0));
        const activeCount = rows.filter((row) => row.phase === 'in_progress' || row.phase === 'routing').length;
        countBadge.textContent = `${rows.length} requests`;
        liveBadge.textContent = activeCount > 0 ? `Live (${activeCount} active)` : 'Live';
        rowsEl.innerHTML = rows.map((row) => {
          const status = row.status_code ?? '';
          const open = expanded.has(row.request_id);
          const main = `<tr>
            <td>${esc(when(row.last_ts))}</td>
            <td class="${phaseClass(row.phase, status)}">${esc(phaseLabel(row.phase))}</td>
            <td><code>${esc(row.request_id)}</code></td>
            <td>${esc(row.tokens || '')}</td>
            <td>${esc(row.model || '')}</td>
            <td class="${statusClass(status)}">${esc(status)}</td>
            <td>${esc(row.provider_name || '')}<div class="muted"><code>${esc(row.route_id || '')}</code></div></td>
            <td>${esc(row.latency_ms != null ? `${row.latency_ms}ms` : '')}</td>
            <td class="muted">${esc(row.reason || '')}</td>
            <td><button class="expand-btn" data-request-id="${esc(row.request_id)}">${open ? 'Hide' : 'Show'}</button></td>
          </tr>`;
          if (!open) return main;
          const assistantText = isUsefulText(row.assistant_text)
            ? row.assistant_text
            : aiResponse(row.response_body || row.stream_event);
          const userPlain = userMessage(row.request_payload);
          const userHtml = userPlain.trim() ? renderMarkdown(userPlain) : '<p class="muted">(empty)</p>';
          const aiPlain = (assistantText || '').trim();
          const aiHtml = aiPlain ? renderMarkdown(assistantText) : '<p class="muted">(empty)</p>';
          const rawBody = pretty(row.response_body || row.stream_event || {});
          const details = `<tr class="details-row"><td colspan="10"><div class="details-wrap">
            <div><strong>User message</strong><div class="md-body">${userHtml}</div></div>
            <div><strong>AI response</strong><div class="md-body">${aiHtml}</div></div>
            <details><summary>Raw response payload</summary><pre>${esc(rawBody)}</pre></details>
            <div><strong>Route attempts</strong>${renderAttempts(row.route_attempts || [])}</div>
          </div></td></tr>`;
          return main + details;
        }).join('');
        rowsEl.querySelectorAll('.expand-btn').forEach((button) => {
          button.addEventListener('click', () => toggleExpand(button.dataset.requestId));
        });
      }
      async function loadSnapshot() {
        const response = await fetch('/v1/gateway/live/snapshot');
        if (!response.ok) return;
        const payload = await response.json();
        for (const evt of (payload.data || [])) addEvent(evt);
      }
      function connect() {
        if (window.liveEventSource) {
          window.liveEventSource.close();
        }
        const es = new EventSource('/v1/gateway/live/events');
        window.liveEventSource = es;
        es.onopen = () => {
          liveBadge.className = 'badge live';
        };
        es.onmessage = (msg) => {
          try { addEvent(JSON.parse(msg.data)); } catch {}
        };
        es.onerror = () => {
          liveBadge.textContent = 'Reconnecting...';
          liveBadge.className = 'badge';
        };
      }
      window.addEventListener('pagehide', () => {
        if (window.liveEventSource) {
          window.liveEventSource.close();
          window.liveEventSource = null;
        }
      });
      loadSnapshot();
      connect();
    </script>
  </body>
</html>
"""


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
      .pill.warning { border-color: rgba(245, 158, 11, 0.45); color: #fcd34d; background: rgba(245, 158, 11, 0.12); }
      .body { border-top: 1px solid var(--border); padding: 1.25rem; display: grid; gap: 1rem; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); background: var(--bg-primary); }
      label { display: grid; gap: 0.4rem; color: var(--text-muted); font-size: 0.85rem; font-weight: 500; }
      .meta { display: grid; gap: 0.35rem; color: var(--text-muted); font-size: 0.85rem; font-weight: 500; }
      .meta span { color: var(--text); font-weight: 400; word-break: break-word; }
      .filter-help { color: var(--text-muted); font-size: 0.85rem; margin-top: -0.75rem; margin-bottom: 1rem; }
      .filter-menu { position: relative; overflow: visible; }
      .filter-menu summary { display: flex; align-items: center; gap: 0.5rem; padding: 0.55rem 0.75rem; border-radius: 8px; border: 1px solid var(--border); background: var(--bg-tertiary); cursor: pointer; user-select: none; }
      .filter-menu[open] summary { border-color: var(--accent); }
      .filter-badge { display: none; min-width: 1.35rem; height: 1.35rem; padding: 0 0.35rem; border-radius: 999px; background: var(--accent); color: white; font-size: 0.75rem; align-items: center; justify-content: center; }
      .filter-badge.active { display: inline-flex; }
      .filter-panel { position: fixed; z-index: 5; top: 7rem; left: 50%; transform: translateX(-50%); width: min(44rem, calc(100vw - 2rem)); max-height: calc(100vh - 8rem); overflow: auto; display: grid; gap: 1rem; padding: 1rem; background: var(--bg-secondary); border: 1px solid var(--border); border-radius: 12px; box-shadow: 0 24px 60px rgba(0, 0, 0, 0.35); }
      .filter-section { display: grid; gap: 0.5rem; }
      .filter-section h3 { font-size: 0.8rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; }
      .filter-options { display: flex; gap: 0.5rem; flex-wrap: wrap; }
      .filter-option { display: inline-flex; align-items: center; gap: 0.35rem; padding: 0.35rem 0.55rem; border: 1px solid var(--border); border-radius: 999px; color: var(--text); background: var(--bg-primary); cursor: pointer; }
      .filter-option input { width: auto; padding: 0; accent-color: var(--accent); }
      .filter-option:has(input:checked) { border-color: var(--accent); background: var(--accent-glow); color: #bfdbfe; }
      .active-filters { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-top: -0.5rem; margin-bottom: 1rem; }
      .active-filter { display: inline-flex; align-items: center; gap: 0.35rem; padding: 0.25rem 0.55rem; border-radius: 999px; border: 1px solid var(--border); background: var(--bg-tertiary); color: var(--text); font-size: 0.8rem; }
      .active-filter button { padding: 0; border: none; background: transparent; color: var(--text-muted); font-weight: 700; }
      .active-filter button:hover { background: transparent; color: var(--text); }
      .summary-actions { display: inline-flex; gap: 0.5rem; align-items: center; justify-content: flex-end; }
      .state-label { font-size: 0.8rem; color: var(--text-muted); }
      .toggle { min-width: 4.75rem; padding: 0.28rem 0.55rem; border-radius: 999px; font-size: 0.78rem; background: rgba(34, 197, 94, 0.12); border: 1px solid rgba(34, 197, 94, 0.35); color: #bbf7d0; }
      .toggle:hover { background: rgba(34, 197, 94, 0.2); }
      .toggle.disable { background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.35); color: #fecaca; }
      .toggle.disable:hover { background: rgba(239, 68, 68, 0.18); }
      @media (max-width: 700px) {
        summary { grid-template-columns: 3rem 1fr; }
        .summary-actions { grid-column: 1 / -1; justify-content: flex-start; }
        .filter-panel { top: 5rem; width: calc(100vw - 2rem); max-height: calc(100vh - 6rem); }
      }
      textarea { min-height: 5rem; resize: vertical; }
      .status { min-height: 1.5rem; color: var(--green); font-size: 0.85rem; margin-bottom: 1rem; }
      details.disabled { border-color: rgba(148, 163, 184, 0.25); }
      details.disabled summary > span:not(.summary-actions), details.disabled .body { opacity: 0.6; }
      code { background: var(--bg-tertiary); border: 1px solid var(--border); padding: 0.1rem 0.35rem; border-radius: 0.35rem; color: #93c5fd; }
      details.dragging { opacity: 0.4; transform: scale(0.98); }
      details.drag-over { border-top: 2px solid var(--accent); }
      details[draggable="true"] summary { cursor: grab; }
      details[draggable="true"] summary:active { cursor: grabbing; }
      .update-count { display: none; margin-left: 0.35rem; min-width: 1.25rem; height: 1.25rem; padding: 0 0.35rem; border-radius: 999px; background: var(--amber); color: #111827; font-size: 0.72rem; align-items: center; justify-content: center; }
      .update-count.active { display: inline-flex; }
      .modal-backdrop { position: fixed; inset: 0; z-index: 20; display: none; align-items: center; justify-content: center; padding: 1rem; background: rgba(2, 6, 23, 0.72); }
      .modal-backdrop.open { display: flex; }
      .modal { width: min(46rem, 100%); max-height: min(42rem, calc(100vh - 2rem)); overflow: auto; background: var(--bg-secondary); border: 1px solid var(--border); border-radius: 12px; box-shadow: 0 24px 70px rgba(0, 0, 0, 0.45); }
      .modal-header, .modal-actions { display: flex; align-items: center; justify-content: space-between; gap: 1rem; padding: 1rem; border-bottom: 1px solid var(--border); }
      .modal-actions { border-top: 1px solid var(--border); border-bottom: none; justify-content: flex-end; }
      .modal-body { display: grid; gap: 0.75rem; padding: 1rem; }
      .suggestion { display: grid; grid-template-columns: auto 1fr; gap: 0.75rem; align-items: start; padding: 0.85rem; border: 1px solid var(--border); border-radius: 8px; background: var(--bg-primary); }
      .suggestion input { margin-top: 0.2rem; accent-color: var(--accent); }
      .suggestion-title { font-weight: 700; }
      .suggestion-meta { margin-top: 0.25rem; color: var(--text-muted); font-size: 0.8rem; word-break: break-word; }
      .empty-updates { color: var(--text-muted); padding: 0.5rem 0; }
    </style>
  </head>
  <body>
    <nav>
      <h1>FreeRouter</h1>
      <span class="nav-spacer"></span>
      <a href="/">Home</a>
      <a href="/chat">Chat</a>
      <a href="/models">Models</a>
      <a href="/health">Health</a>
      <a href="/status">Provider Usage</a>
      <a href="/live">Live Traffic</a>
    </nav>
    <main>
      <div class="toolbar">
        <input id="search" placeholder="Search models/providers/tags..." style="flex: 1; min-width: 250px;">
        <details class="filter-menu" id="filterMenu">
          <summary>Filters <span id="filterCount" class="filter-badge"></span></summary>
          <div class="filter-panel">
            <section class="filter-section">
              <h3>Capabilities</h3>
              <div id="capabilityOptions" class="filter-options"></div>
            </section>
            <section class="filter-section">
              <h3>Provider</h3>
              <div id="providerOptions" class="filter-options"></div>
            </section>
            <section class="filter-section">
              <h3>Minimum context</h3>
              <div class="filter-options">
                <label class="filter-option"><input type="radio" name="contextFilter" value="" checked> Any</label>
                <label class="filter-option"><input type="radio" name="contextFilter" value="32000"> 32K+</label>
                <label class="filter-option"><input type="radio" name="contextFilter" value="128000"> 128K+</label>
                <label class="filter-option"><input type="radio" name="contextFilter" value="256000"> 256K+</label>
                <label class="filter-option"><input type="radio" name="contextFilter" value="1000000"> 1M+</label>
              </div>
            </section>
            <section class="filter-section">
              <h3>Routeability</h3>
              <div class="filter-options">
                <label class="filter-option"><input type="radio" name="routeability" value=""> All catalog entries</label>
                <label class="filter-option"><input type="radio" name="routeability" value="routeable"> Enabled only</label>
                <label class="filter-option"><input type="radio" name="routeability" value="disabled"> Disabled/specialized</label>
              </div>
            </section>
            <button id="clearFilters" class="secondary" type="button">Clear filters</button>
          </div>
        </details>
        <button id="save">Save Ranking</button>
        <button id="autoRank" class="secondary" type="button" title="Reorder text-capable routes by automatic quality score. Routes that cannot handle a normal text exchange are removed; disabled routes stay disabled.">Auto-rank</button>
        <button id="reload" class="secondary">Reload</button>
        <button id="updates" class="secondary" type="button">Updates <span id="updateCount" class="update-count"></span></button>
        <button id="reset" class="secondary" style="margin-left: auto; color: var(--red); border-color: rgba(239, 68, 68, 0.3);">Reset to Defaults</button>
      </div>
      <p class="filter-help">Tags are normalized to capabilities. Use the filter menu to combine capabilities, provider, context, and routeability.</p>
      <div id="activeFilters" class="active-filters"></div>
      <p class="status" id="status"></p>
      <div id="models" class="grid"></div>
    </main>
    <div id="updateModal" class="modal-backdrop" role="dialog" aria-modal="true" aria-labelledby="updateTitle">
      <div class="modal">
        <div class="modal-header">
          <div>
            <h2 id="updateTitle">Endpoint Updates</h2>
            <p class="muted" id="updateSummary">Loading suggestions...</p>
          </div>
          <button id="closeUpdates" class="secondary" type="button">Close</button>
        </div>
        <div id="updateList" class="modal-body"></div>
        <div class="modal-actions">
          <button id="refreshUpdates" class="secondary" type="button">Check now</button>
          <button id="selectAllUpdates" class="secondary" type="button">Select all</button>
          <button id="applyUpdates" type="button">Apply selected</button>
        </div>
      </div>
    </div>
    <script>
      let routes = [];
      let endpointSuggestions = [];

      const $ = (id) => document.getElementById(id);
      const normalize = (value) => String(value || '').toLowerCase();
      const capabilityLabels = {
        text: 'Text',
        reasoning: 'Reasoning',
        coding: 'Coding',
        'tool-use': 'Tool use',
        'web-search': 'Web search',
        vision: 'Vision',
        audio: 'Audio',
        safety: 'Safety',
        moderation: 'Moderation',
        translation: 'Translation',
        classification: 'Classification',
        rag: 'RAG',
      };

      async function load() {
        $('status').textContent = 'Loading model catalog...';
        const response = await fetch('/v1/gateway/models');
        const payload = await response.json();
        routes = payload.data;
        populateFilters();
        render();
        $('status').textContent = `Loaded ${routes.length} model routes from ${payload.catalog_path}`;
        checkEndpointSuggestions(false);
      }

      function populateCheckboxes(containerId, name, options) {
        const selected = new Set(getChecked(name));
        $(containerId).innerHTML = options.map((option) => `
          <label class="filter-option">
            <input type="checkbox" name="${name}" value="${escapeHtml(option.value)}" ${selected.has(option.value) ? 'checked' : ''}>
            ${escapeHtml(option.label)}
          </label>
        `).join('');
      }

      function populateFilters() {
        const capabilities = [...new Set(routes.flatMap((route) => route.tags || []))]
          .sort((a, b) => (capabilityLabels[a] || a).localeCompare(capabilityLabels[b] || b))
          .map((tag) => ({ value: tag, label: capabilityLabels[tag] || tag }));
        populateCheckboxes('capabilityOptions', 'capabilityFilter', capabilities);
        populateCheckboxes(
          'providerOptions',
          'providerFilter',
          [...new Set(routes.map((route) => route.provider_name))].sort().map((provider) => ({ value: provider, label: provider }))
        );
      }

      function getChecked(name) {
        return [...document.querySelectorAll(`input[name="${name}"]:checked`)].map((input) => input.value);
      }

      function getRadio(name) {
        return document.querySelector(`input[name="${name}"]:checked`)?.value || '';
      }

      function setRadio(name, value) {
        document.querySelectorAll(`input[name="${name}"]`).forEach((input) => {
          input.checked = input.value === value;
        });
      }

      function render() {
        const query = normalize($('search').value);
        const capabilities = getChecked('capabilityFilter');
        const providers = getChecked('providerFilter');
        const minContext = Number(getRadio('contextFilter') || 0);
        const routeability = getRadio('routeability');
        const visible = routes
          .slice()
          .sort((a, b) => a.rank - b.rank || a.provider_name.localeCompare(b.provider_name))
          .filter((route) => !capabilities.length || capabilities.every((tag) => (route.tags || []).includes(tag)))
          .filter((route) => !providers.length || providers.includes(route.provider_name))
          .filter((route) => !minContext || Number(route.context_window || 0) >= minContext)
          .filter((route) => {
            if (routeability === 'routeable') return route.enabled;
            if (routeability === 'disabled') return !route.enabled;
            return true;
          })
          .filter((route) => {
            const haystack = normalize([
              route.route_id, route.provider_name, route.model_id, route.display_name,
              route.quality, route.speed, route.tags?.join(' '), route.notes
            ].join(' '));
            return !query || haystack.includes(query);
          });

        $('models').innerHTML = visible.map((route) => card(route)).join('');
        renderActiveFilters(capabilities, providers, minContext, routeability);
        $('status').textContent = `Showing ${visible.length} of ${routes.length} model routes.`;
      }

      function contextLabel(value) {
        return {
          32000: '32K+ context',
          128000: '128K+ context',
          256000: '256K+ context',
          1000000: '1M+ context',
        }[value] || '';
      }

      function renderActiveFilters(capabilities, providers, minContext, routeability) {
        const count = capabilities.length + providers.length + (minContext ? 1 : 0) + (routeability ? 1 : 0);
        $('filterCount').textContent = count || '';
        $('filterCount').classList.toggle('active', count > 0);

        const chips = [
          ...capabilities.map((tag) => ({ kind: 'capabilityFilter', value: tag, label: capabilityLabels[tag] || tag })),
          ...providers.map((provider) => ({ kind: 'providerFilter', value: provider, label: provider })),
        ];
        if (minContext) chips.push({ kind: 'contextFilter', value: '', label: contextLabel(minContext) });
        if (routeability === 'routeable') chips.push({ kind: 'routeability', value: '', label: 'Enabled only' });
        if (routeability === 'disabled') chips.push({ kind: 'routeability', value: '', label: 'Disabled/specialized' });

        $('activeFilters').innerHTML = chips.map((chip) => `
          <span class="active-filter">
            ${escapeHtml(chip.label)}
            <button type="button" aria-label="Remove ${escapeHtml(chip.label)} filter" onclick="removeFilter('${escapeHtml(chip.kind)}', '${escapeHtml(chip.value)}')">x</button>
          </span>
        `).join('');
      }

      function removeFilter(kind, value) {
        if (kind === 'contextFilter' || kind === 'routeability') {
          setRadio(kind, value);
        } else {
          document.querySelectorAll(`input[name="${kind}"][value="${CSS.escape(value)}"]`).forEach((input) => {
            input.checked = false;
          });
        }
        render();
      }

      let draggedId = null;

      function card(route) {
        const tags = (route.tags || []).map((tag) => `<span class="pill">${escapeHtml(tag)}</span>`).join('');
        const health = route.health?.status && route.health.status !== 'active'
          ? `<span class="pill warning">${escapeHtml(route.health.status.replace(/_/g, ' '))}</span>`
          : '';
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
                ${health}
              </span>
              <span class="summary-actions">
                <span class="state-label">${route.enabled ? 'Enabled' : 'Disabled'}</span>
                <button type="button" class="toggle ${route.enabled ? 'disable' : ''}" onclick="toggleRouteEnabled(event, '${route.route_id}')">
                  ${route.enabled ? 'Disable' : 'Enable'}
                </button>
              </span>
            </summary>
            <div class="body">
              ${field(route, 'rank', 'Rank', 'number')}
              ${meta('Display name', route.display_name)}
              ${meta('Provider', route.provider_name)}
              ${meta('Provider model ID', route.model_id)}
              ${meta('Context window', route.context_window ? route.context_window.toLocaleString() : 'Unknown')}
              ${meta('Quality', route.quality)}
              ${meta('Rank score', route.rank_score ?? 'Unknown')}
              ${meta('Rank source', route.rank_source || 'heuristic')}
              ${meta('Rank reason', route.rank_reason || 'N/A')}
              ${meta('Speed', route.speed)}
              ${meta('Cost', route.cost)}
              ${meta('Tags', tags || 'None', true)}
              ${meta('Source URL', route.source_url ? `<a href="${escapeHtml(route.source_url)}" target="_blank" rel="noreferrer">${escapeHtml(route.source_url)}</a>` : 'None', true)}
              ${meta('Notes', route.notes || 'None')}
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

      function meta(label, value, html = false) {
        return `<div class="meta">${label}<span>${html ? value : escapeHtml(value)}</span></div>`;
      }

      function toggleRouteEnabled(event, id) {
        event.preventDefault();
        event.stopPropagation();
        const route = routes.find((item) => item.route_id === id);
        if (!route) return;
        route.enabled = !route.enabled;
        render();
        save();
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

      function summarizeAutoRank(before, after) {
        const beforeById = new Map(before.map((route) => [route.route_id, route]));
        const afterIds = new Set(after.map((route) => route.route_id));
        const removed = before.filter((route) => !afterIds.has(route.route_id)).length;
        const moved = after.filter((route) => beforeById.has(route.route_id) && beforeById.get(route.route_id).rank !== route.rank).length;
        const disabled = after.filter((route) => !route.enabled).length;
        return `Auto-ranked ${after.length} text-capable route${after.length === 1 ? '' : 's'}: ${moved} moved, ${removed} non-text/specialized removed, ${disabled} disabled kept disabled.`;
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
        populateFilters();
        render();
        $('status').textContent = 'Saved. New requests will use the updated ranking immediately.';
      }

      function escapeHtml(value) {
        return String(value ?? '').replace(/[&<>"']/g, (char) => ({
          '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
        }[char]));
      }

      async function checkEndpointSuggestions(openWhenFound = true) {
        const response = await fetch('/v1/gateway/endpoint-diagnosis');
        if (!response.ok) return;
        const payload = await response.json();
        endpointSuggestions = payload.last_report?.suggestions || [];
        $('updateCount').textContent = endpointSuggestions.length || '';
        $('updateCount').classList.toggle('active', endpointSuggestions.length > 0);
        if (openWhenFound && endpointSuggestions.length) openUpdateModal();
      }

      function openUpdateModal() {
        $('updateModal').classList.add('open');
        renderEndpointSuggestions();
      }

      function closeUpdateModal() {
        $('updateModal').classList.remove('open');
      }

      function suggestionActionLabel(action) {
        return {
          add_route: 'New route',
          remove_route: 'Remove route',
          clear_stale: 'Recovered route',
        }[action] || action;
      }

      function renderEndpointSuggestions() {
        $('updateSummary').textContent = endpointSuggestions.length
          ? `${endpointSuggestions.length} suggested update${endpointSuggestions.length === 1 ? '' : 's'} found. Nothing changes until you apply them.`
          : 'No pending endpoint updates.';
        $('updateList').innerHTML = endpointSuggestions.length ? endpointSuggestions.map((item) => `
          <label class="suggestion">
            <input type="checkbox" name="endpointSuggestion" value="${escapeHtml(item.suggestion_id)}">
            <span>
              <span class="suggestion-title">${escapeHtml(item.title)}</span>
              <span class="pill">${escapeHtml(suggestionActionLabel(item.action))}</span>
              <div class="suggestion-meta">${escapeHtml(item.provider_name)} / ${escapeHtml(item.model_id)}</div>
              <div class="suggestion-meta">${escapeHtml(item.details)}</div>
            </span>
          </label>
        `).join('') : '<div class="empty-updates">No pending endpoint updates.</div>';
      }

      async function refreshEndpointSuggestions() {
        $('updateSummary').textContent = 'Checking provider catalogs...';
        const response = await fetch('/v1/gateway/endpoint-diagnosis/refresh', { method: 'POST' });
        const payload = await response.json();
        endpointSuggestions = payload.data?.suggestions || [];
        $('updateCount').textContent = endpointSuggestions.length || '';
        $('updateCount').classList.toggle('active', endpointSuggestions.length > 0);
        renderEndpointSuggestions();
      }

      async function applyEndpointSuggestions() {
        const suggestionIds = [...document.querySelectorAll('input[name="endpointSuggestion"]:checked')]
          .map((input) => input.value);
        if (!suggestionIds.length) {
          $('updateSummary').textContent = 'Choose at least one suggestion to apply.';
          return;
        }
        $('updateSummary').textContent = 'Applying selected updates...';
        const response = await fetch('/v1/gateway/endpoint-diagnosis/apply', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ suggestion_ids: suggestionIds }),
        });
        if (!response.ok) {
          $('updateSummary').textContent = 'Could not apply selected updates.';
          return;
        }
        await checkEndpointSuggestions(false);
        await load();
        renderEndpointSuggestions();
        $('updateSummary').textContent = 'Applied selected updates.';
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
      $('filterMenu').addEventListener('change', render);
      $('clearFilters').addEventListener('click', () => {
        document.querySelectorAll('input[name="capabilityFilter"], input[name="providerFilter"]').forEach((input) => {
          input.checked = false;
        });
        setRadio('contextFilter', '');
        setRadio('routeability', '');
        render();
      });
      document.addEventListener('click', (event) => {
        if (!$('filterMenu').contains(event.target)) $('filterMenu').open = false;
      });
      document.addEventListener('keydown', (event) => {
        if (event.key === 'Escape') {
          $('filterMenu').open = false;
          closeUpdateModal();
        }
      });
      $('save').addEventListener('click', save);
      $('autoRank').addEventListener('click', async () => {
        const before = routes.map((route) => ({ ...route }));
        $('status').textContent = 'Auto-ranking text-capable routes by quality score; disabled routes will stay disabled...';
        $('autoRank').disabled = true;
        const response = await fetch('/v1/gateway/models/auto-rank', { method: 'POST' });
        const payload = await response.json();
        $('autoRank').disabled = false;
        if (!response.ok) {
          $('status').textContent = payload.detail || 'Auto-rank failed';
          return;
        }
        routes = payload.data;
        populateFilters();
        render();
        $('status').textContent = summarizeAutoRank(before, routes);
      });
      $('reload').addEventListener('click', load);
      $('updates').addEventListener('click', openUpdateModal);
      $('closeUpdates').addEventListener('click', closeUpdateModal);
      $('refreshUpdates').addEventListener('click', refreshEndpointSuggestions);
      $('selectAllUpdates').addEventListener('click', () => {
        document.querySelectorAll('input[name="endpointSuggestion"]').forEach((input) => {
          input.checked = true;
        });
      });
      $('applyUpdates').addEventListener('click', applyEndpointSuggestions);
      $('updateModal').addEventListener('click', (event) => {
        if (event.target === $('updateModal')) closeUpdateModal();
      });
      $('reset').addEventListener('click', async () => {
        if (!confirm('Are you sure you want to restore the default model rankings? This will overwrite your custom order.')) return;
        $('status').textContent = 'Resetting catalog...';
        const response = await fetch('/v1/gateway/models/reset', { method: 'POST' });
        const payload = await response.json();
        routes = payload.data;
        populateFilters();
        render();
        $('status').textContent = 'Restored default model rankings.';
      });
      document.addEventListener('input', (event) => {
        if (event.target.matches('[data-id][data-key]')) collectEdits();
      });
      load();
      setTimeout(() => checkEndpointSuggestions(true), 1500);
      setInterval(() => checkEndpointSuggestions(true), 60000);
    </script>
  </body>
</html>
"""
