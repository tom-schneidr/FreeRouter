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

Most OpenAI-compatible SDKs can also use FreeRouter by setting:

```text
base_url = http://localhost:8000/v1
model = auto
```

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
- Let FreeRouter choose models, handle fallback, track quotas, and manage route health.
- Expect `429` with `code=request_queue_timeout` when the local gateway is overloaded; retry with
  backoff instead of spawning more parallel calls.
- New discovered models are intentionally disabled until accepted in the Models UI.
- Runtime state lives in `data/`; do not commit SQLite databases or generated model catalogs.

## File Structure

```text
.
├── app/
│   ├── chat_page.py            # Chat playground HTML template
│   ├── client.py               # Programmatic client wrapper around router
│   ├── endpoint_diagnosis.py   # Catalog diagnosis and reviewable update suggestions
│   ├── main.py                 # FastAPI app and OpenAI-compatible endpoints
│   ├── model_catalog.py        # Editable ranked model catalog defaults
│   ├── model_discovery.py      # Structured /models payload discovery helpers
│   ├── provider_errors.py      # Provider error classification helpers
│   ├── router.py               # Ranked model waterfall routing engine
│   ├── settings.py             # .env-backed configuration
│   ├── state.py                # SQLite quota/cooldown/RPM tracker
│   └── providers/
│       ├── base.py             # OpenAI-compatible provider adapter
│       └── registry.py         # Provider order, quotas, endpoints, models
├── data/                       # Runtime SQLite DB and editable model catalog
├── tests/                      # Router/state/catalog/discovery/provider tests
├── .env.example                # API key template
├── run.bat                     # Execution-policy-safe Windows launcher
├── run.ps1                     # PowerShell bootstrap-and-run script
├── run.sh                      # Linux/macOS bootstrap-and-run script
└── pyproject.toml              # Python dependencies
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

## Model Ranking

Open the model catalog UI:

```text
http://127.0.0.1:8000/models
```

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
GET  /chat
GET  /health
GET  /models
GET  /status
GET  /v1/models
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
POST /v1/chat/completions/stream-route
```

`/v1/chat/completions` returns the upstream provider JSON unchanged. Gateway diagnostics are
returned as `X-Gateway-Provider`, `X-Gateway-Route`, `X-Gateway-Model`, and
`X-Gateway-Attempts` headers.

## Validate

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
```
