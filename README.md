# FreeRouter

FreeRouter is a local AI routing layer that turns a pile of free-tier provider accounts into one
reliable, low-maintenance API for agent systems, scripts, and apps.

Instead of manually choosing and reordering dozens of models, FreeRouter keeps a ranked catalog of
text-capable models, tries the strongest available route first, tracks provider quotas locally, and
falls back automatically when a route is rate-limited, slow, stale, or unavailable. It is designed to
sit underneath workloads like multi-agent simulations where many independent agents may ask for AI
work at the same time.

It also exposes an OpenAI-style API surface, so many existing clients can use it by changing only
their base URL. That compatibility is the adapter layer; the main value is automatic routing,
fallback, quota awareness, endpoint health, and low-touch maintenance across multiple providers.

## Why Use It?

- **One local AI gateway:** Point your projects at FreeRouter instead of wiring each provider into
  every app.
- **Automatic model ordering:** Routes are ranked by model capability and provider usefulness rather
  than by discovery order.
- **Graceful fallback:** If a provider is out of quota, too slow, missing a model, or temporarily
  unhealthy, the request rolls to the next best route.
- **Burst-friendly for agent workloads:** A local request limiter and SQLite-backed quota reservation
  help absorb sudden bursts from cron jobs or many collaborating agents.
- **Set-and-forget maintenance:** Background diagnosis can remove confirmed dead routes and clear
  recovered health flags while leaving newly found models disabled until you accept them.
- **Transparent control:** The Models and Status pages show ranking, route health, usage, and
  pending endpoint updates.

## Integration Guide For Other Projects

Prefer integrating FreeRouter over HTTP. Run FreeRouter as a local service, then point your app or
agent framework at:

```text
http://localhost:8000/v1
```

Use the default model name:

```text
auto
```

Send chat requests to:

```text
POST /v1/chat/completions
```

Example request:

```json
{
  "model": "auto",
  "messages": [
    { "role": "system", "content": "You are a helpful assistant." },
    { "role": "user", "content": "Summarize today's company status." }
  ],
  "temperature": 0.7,
  "max_tokens": 800
}
```

Example Python HTTP integration:

```python
import httpx

response = httpx.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "auto",
        "messages": [{"role": "user", "content": "Write a short update."}],
    },
    timeout=120,
)
response.raise_for_status()
print(response.json()["choices"][0]["message"]["content"])
```

### Routing transparency

When FreeRouter successfully routes a request, the response includes opt-in HTTP headers that
identify which route answered. Standard OpenAI and Anthropic clients ignore these headers.
Response bodies always echo your requested `model` (typically `auto`); assistant content and
usage still come from the routed provider.

| Header | Meaning |
|--------|---------|
| `X-Gateway-Provider` | Provider that answered (for example `groq`) |
| `X-Gateway-Route` | Catalog route id (for example `groq-llama-3-3-70b`) |
| `X-Gateway-Model` | Provider model id actually called |

These headers are returned on all inference endpoints (`/v1/chat/completions`,
`/v1/chat/completions/web-search`, `/v1/responses`, `/v1/messages`, and
`/v1/chat/completions/stream-route`), including streaming requests. On streams, headers are sent
once a route is selected and before the first response bytes.

To inspect the routed model explicitly:

```python
response.raise_for_status()
print(response.headers.get("X-Gateway-Route"))
print(response.headers.get("X-Gateway-Model"))
```

The response body `model` field always echoes the value you requested (typically `auto`)
across chat completions, responses, and messages. Use the headers to see which route
actually answered.

For requests that must use web search, call the dedicated web-search endpoint:

```text
POST /v1/chat/completions/web-search
```

This endpoint accepts the same chat-completion payload shape, but FreeRouter prepares it for a
web-enabled model before routing:

- It adds the web search tool if the caller did not include one.
- It forces `tool_choice` to the web search tool so the upstream model must call web search.
- It routes only through enabled catalog routes tagged `web-search`.
- It translates that intent for provider-specific APIs. For Groq Compound routes, FreeRouter sends
  `compound_custom.tools.enabled_tools = ["web_search"]` instead of OpenAI's
  `web_search_preview` tool object.

Example request:

```json
{
  "model": "auto",
  "messages": [
    {
      "role": "user",
      "content": "Search the web and summarize the latest release notes for Python."
    }
  ]
}
```

The upstream request sent by FreeRouter includes:

```json
{
  "tools": [{ "type": "web_search_preview" }],
  "tool_choice": { "type": "web_search_preview" }
}
```

Example Python web-search integration:

```python
import httpx

response = httpx.post(
    "http://localhost:8000/v1/chat/completions/web-search",
    json={
        "model": "auto",
        "messages": [
            {
                "role": "user",
                "content": "Use web search to find the latest LTS Node.js version.",
            }
        ],
    },
    timeout=120,
)
response.raise_for_status()
print(response.json()["choices"][0]["message"]["content"])
```

Most OpenAI-compatible SDKs can also use FreeRouter by setting:

```text
base_url = http://localhost:8000/v1
model = auto
```

### Codex CLI

Codex's current custom-provider config expects the Responses wire API. FreeRouter supports that
through `POST /v1/responses`, which adapts Codex's Responses payloads to the chat-completions router
internally:

```toml
# ~/.codex/config.toml
model = "auto"
model_provider = "freerouter"

[model_providers.freerouter]
name = "FreeRouter"
base_url = "http://127.0.0.1:8000/v1"
env_key = "FREEROUTER_API_KEY"
wire_api = "responses"
```

Then start Codex with any non-empty local key value:

```powershell
$env:FREEROUTER_API_KEY = "sk-local"
codex
```

Do not reuse a built-in provider key such as `openai` or `ollama` for this block; use a distinct
provider key like `freerouter` so Codex applies the custom `base_url`.

The Responses adapter covers text output, function tools, function-call outputs, and streamed text
or function-call items. FreeRouter still sends the actual model work through OpenAI-compatible
`/v1/chat/completions` upstream providers, so provider quality depends on how well the selected
free-tier route follows tool-calling instructions.

For Python projects running in the same environment, there is also a small programmatic wrapper:

```python
from app.client import ask_ai

result = await ask_ai([
    {"role": "user", "content": "Draft a weekly CEO briefing."}
])
print(result.body)
```

HTTP is usually the cleaner boundary between projects. The Python wrapper is useful for scripts or
tightly coupled local tooling.

### Integration Rules For Agentic Coding AIs

If another coding agent is integrating with this repository, give it these rules:

- Treat FreeRouter as the AI gateway. Do not call provider APIs directly unless explicitly asked.
- Use `http://localhost:8000/v1` as the base URL and `auto` as the model.
- Send normal chat-completion payloads with `messages`.
- Use `/v1/chat/completions/web-search` when the task requires current web information. Do not rely
  on the normal chat route to enable web search.
- Let FreeRouter choose models, handle fallback, track quotas, and manage route health.
- Expect `429` with `code=request_queue_timeout` when the local gateway is overloaded; retry with
  backoff instead of spawning more parallel calls.
- New discovered models are intentionally disabled until accepted in the Models UI.
- Runtime state lives in `data/`; do not commit SQLite databases or generated model catalogs.

## File Structure

```text
.
|-- apps/
|   |-- ui/                     # React control plane (built to apps/ui/dist, served at /app)
|   `-- desktop/                # Tauri desktop shell
|-- app/
|   |-- client.py               # Programmatic client wrapper around router
|   |-- react_app.py            # Mounts built React UI at /app
|   |-- endpoint_diagnosis.py   # Catalog diagnosis and reviewable update suggestions
|   |-- local_backup.py         # Local state export/import CLI
|   |-- main.py                 # FastAPI app entrypoint and router wiring
|   |-- stream_route.py         # Chat UI streaming API (/v1/chat/completions/stream-route)
|   |-- model_catalog.py        # Editable ranked model catalog defaults
|   |-- model_discovery.py      # Structured /models payload discovery helpers
|   |-- provider_errors.py      # Provider error classification helpers
|   |-- router.py               # Ranked model waterfall routing engine
|   |-- settings.py             # .env-backed configuration
|   |-- state.py                # SQLite quota/cooldown/RPM tracker
|   |-- tray_launcher.py        # Local tray console
|   `-- providers/
|       |-- base.py             # OpenAI-compatible provider adapter
|       `-- registry.py         # Provider order, quotas, endpoints, models
|-- data/                       # Runtime SQLite DB and editable model catalog
|-- tests/                      # Router/state/catalog/discovery/provider tests
|-- .env.example                # API key template
|-- backup-local-state.ps1      # Export local catalog/database backup
|-- restore-local-state.ps1     # Restore local catalog/database backup
|-- run.bat                     # Execution-policy-safe Windows launcher
|-- run.ps1                     # PowerShell bootstrap-and-run script
|-- run.sh                      # Linux/macOS bootstrap-and-run script
|-- validate.ps1                # Run local tests and lint
`-- pyproject.toml              # Python dependencies
```

## Run Locally

```powershell
.\run.bat
```

Linux/macOS:

```bash
bash ./run.sh
```

The launcher creates `.venv` if needed, installs dependencies, creates `.env` from `.env.example`
if missing, and starts the API at `http://127.0.0.1:8000/v1`. Use `run.bat` if PowerShell blocks
local scripts with an execution policy error.

NVIDIA API keys come from the NVIDIA model catalog at `https://build.nvidia.com/models`; the
OpenAI-compatible API base URL used by the gateway is `https://integrate.api.nvidia.com/v1`.

Useful options:

```powershell
.\run.bat -InstallOnly
.\run.bat -Port 8080
.\run.bat -NoReload
.\run.bat -RuntimeOnly
```

```bash
bash ./run.sh --install-only
bash ./run.sh --port 8080
bash ./run.sh --no-reload
bash ./run.sh --runtime-only
```

By default `run.ps1` installs the project with dev extras (`.[dev]`) so `pytest` and `ruff`
are available in the local virtual environment. Use `-RuntimeOnly` when you only want
runtime dependencies.

Point OpenAI-compatible clients at:

```text
http://localhost:8000/v1
```

Open the control plane UI at `http://localhost:8000/app`.

## Desktop App

Day-to-day development:

```powershell
npm run desktop:dev
```

This builds the React UI, starts a source backend on `http://127.0.0.1:8000`, and launches the Tauri shell against `http://127.0.0.1:8000/app`.

Build the packaged desktop executable:

```powershell
npm run build:desktop
```

Create Start Menu / Desktop shortcuts after a build:

```powershell
.\install-desktop.ps1
```

Stop stray local gateway processes:

```powershell
npm run stop
```

See `desktop-help.html` for tray behavior and shortcut details.

## Model Ranking

Open the model catalog in the control plane at `http://127.0.0.1:8000/app#models`, or use the API:

Each enabled model route has a `rank`; lower numbers are attempted first. The route points at a
provider and model ID, so the gateway can use multiple models on the same platform while still
tracking that platform's quota correctly.

You can also edit the catalog directly:

```text
data/model_catalog.json
```

The launcher keeps `.env` in sync with new config keys but does not overwrite existing API keys.

## Automatic Endpoint Diagnosis

FreeRouter can check provider model availability in the background. When enabled, it calls each
configured provider's OpenAI-compatible `/models` endpoint and creates suggestions for confirmed
dead-route removals, recovered routes, and newly discovered models. With automatic maintenance
enabled, safe cleanup is applied in the background: confirmed dead routes are removed and recovered
route health flags are cleared. New route additions remain reviewable suggestions and are disabled
by default until you accept and enable them.

Configure the cadence in `.env`:

```text
AUTO_ENDPOINT_DIAGNOSIS_ENABLED=true
AUTO_ENDPOINT_MAINTENANCE_ENABLED=true
AUTO_ENDPOINT_DIAGNOSIS_INTERVAL_SECONDS=21600
AUTO_ENDPOINT_DIAGNOSIS_STARTUP_DELAY_SECONDS=10
ENDPOINT_DIAGNOSIS_SUPERVISOR_ENABLED=false
ENDPOINT_DIAGNOSIS_SUPERVISOR_MODEL=
```

For bursty local agent workloads, tune:

```text
MAX_CONCURRENT_REQUESTS=20
REQUEST_QUEUE_TIMEOUT_SECONDS=30
SQLITE_BUSY_TIMEOUT_MS=5000
```

When `ENDPOINT_DIAGNOSIS_SUPERVISOR_ENABLED=true`, missing provider pricing data can be checked by
an enabled free route from the local catalog. Set `ENDPOINT_DIAGNOSIS_SUPERVISOR_MODEL` to a route
ID, provider model ID, or display name to prefer one; otherwise FreeRouter picks an enabled text route
and prioritizes routes tagged `web-search`. The supervisor only allows add-route suggestions when it
returns a high-confidence free-tier chat-model verdict; otherwise discovery fails closed.

You can inspect or trigger the refresh manually:

```text
GET  /v1/gateway/endpoint-diagnosis
POST /v1/gateway/endpoint-diagnosis/refresh
POST /v1/gateway/endpoint-diagnosis/apply
```

## Endpoints

```text
GET  /
GET  /app
GET  /docs
GET  /v1/models
POST /v1/responses
GET  /v1/gateway/health.json
GET  /v1/gateway/models
PUT  /v1/gateway/models
POST /v1/gateway/models/reset
POST /v1/gateway/models/auto-rank
POST /v1/gateway/models/{route_id}/disable
POST /v1/gateway/models/{route_id}/enable
POST /v1/gateway/models/{route_id}/health/reset
GET  /v1/gateway/endpoint-diagnosis
POST /v1/gateway/endpoint-diagnosis/refresh
POST /v1/gateway/endpoint-diagnosis/apply
GET  /v1/providers/status
POST /v1/chat/completions
POST /v1/chat/completions/web-search
POST /v1/chat/completions/stream-route
POST /v1/messages
```

`/v1/chat/completions` returns provider completion content in the OpenAI shape, but the
response `model` field always echoes your request (typically `auto`). When routing
succeeds, gateway transparency headers are returned as `X-Gateway-Provider`,
`X-Gateway-Route`, and `X-Gateway-Model` (including on streaming responses). The same
body/header rules apply across all inference endpoints listed above.

`/v1/chat/completions/web-search` is the same gateway response surface, but it always injects and
requires web-search intent and only considers routes tagged `web-search`. Provider adapters may
translate that intent to the provider's native web-search request format.

## Validate

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
```

Or run both checks through the local validation wrapper:

```powershell
.\validate.ps1
```

Linux/macOS:

```bash
./validate.sh
```

## Local State Backup

Export the editable model catalog, SQLite state, and non-secret local settings:

```powershell
.\backup-local-state.ps1
```

Restore a backup:

```powershell
.\restore-local-state.ps1 .\backups\freerouter-local-state-YYYYMMDD-HHMMSS.zip -Overwrite
```

Linux/macOS:

```bash
./backup-local-state.sh
./restore-local-state.sh ./backups/freerouter-local-state-YYYYMMDD-HHMMSS.zip --overwrite
```
