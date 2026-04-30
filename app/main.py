from __future__ import annotations

import json
from contextlib import asynccontextmanager
from dataclasses import asdict
from time import time
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

from app.chat_page import CHAT_HTML
from app.endpoint_diagnosis import (
    BackgroundEndpointDiagnosis,
    EndpointDiagnosisService,
    EndpointSupervisor,
)
from app.model_catalog import ModelCatalog
from app.providers import PROVIDER_QUOTAS, ProviderError, build_provider_adapters
from app.request_limiter import GatewayRequestLimiter
from app.router import (
    NoProviderAvailable,
    WaterfallRouter,
)
from app.settings import get_settings
from app.state import StateManager
from app.stream_route import stream_route_chat


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
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
    )
    app.state.waterfall_router = WaterfallRouter(
        providers,
        model_catalog,
        state,
        request_timeout_seconds=settings.request_timeout_seconds,
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
          <a href="/health">Health</a>
          <a href="/status">Provider Usage</a>
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
    for route_payload in payload["data"]:
        route_state = await state.get_route_state(
            route_payload["route_id"],
            route_payload["provider_name"],
            route_payload["model_id"],
        )
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
      <a href="/">Home</a><a href="/chat">Chat</a><a href="/models">Models</a><a href="/health">Health</a><a href="/status">Provider Usage</a>
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
    providers = []
    for provider in router.providers:
        provider_state = await state.get_state(provider.name)
        availability = await state.check_available(provider.name)
        models = []
        for route in all_routes:
            if route.provider_name != provider.name:
                continue
            route_state = await state.get_route_state(
                route.route_id, route.provider_name, route.model_id
            )
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


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> JSONResponse:
    payload: dict[str, Any] = await request.json()
    router: WaterfallRouter = request.app.state.waterfall_router
    limiter: GatewayRequestLimiter = request.app.state.request_limiter

    if not await limiter.acquire():
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
    finally:
        limiter.release()

    return JSONResponse(
        content=result.body,
        headers={
            "X-Gateway-Provider": result.provider_name,
            "X-Gateway-Route": result.route_id,
            "X-Gateway-Model": result.model_id,
            "X-Gateway-Attempts": json.dumps([asdict(attempt) for attempt in result.attempts]),
        },
    )


@app.post("/v1/chat/completions/stream-route")
async def chat_completions_stream_route(request: Request) -> Response:
    payload: dict[str, Any] = await request.json()
    router: WaterfallRouter = request.app.state.waterfall_router
    limiter: GatewayRequestLimiter = request.app.state.request_limiter
    if not await limiter.acquire():
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

    async def limited_stream():
        try:
            async for chunk in stream_route_chat(payload, router):
                yield chunk
        finally:
            limiter.release()

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
      <a href="/">Home</a><a href="/chat">Chat</a><a href="/models">Models</a><a href="/health">Health</a><a href="/status">Provider Usage</a>
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
