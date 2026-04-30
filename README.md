# FreeRouter

Local OpenAI-compatible gateway with a state-aware waterfall router for free-tier AI providers.

Default model-strength waterfall:

The router ranks all available free-tier models based on their raw benchmark strength and capabilities (DeepSeek V4 Pro, GPT OSS 120B, Gemini Pro, etc.). It queries the most powerful models first. If the top model is rate-limited or unavailable, it gracefully falls back to the next strongest model in the list. If two providers offer the exact same model, the router prioritizes the provider with the most generous free-tier limits.

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
└── pyproject.toml              # Python dependencies
```

## Run Locally

```powershell
.\run.bat
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
configured provider's OpenAI-compatible `/models` endpoint and creates reviewable suggestions for
confirmed dead-route removals, recovered routes, and newly discovered models. The gateway does
not apply those changes automatically; open the Models page and use the Updates popup to choose what
to accept.

Configure the cadence in `.env`:

```text
AUTO_ENDPOINT_DIAGNOSIS_ENABLED=true
AUTO_ENDPOINT_DIAGNOSIS_INTERVAL_SECONDS=21600
AUTO_ENDPOINT_DIAGNOSIS_STARTUP_DELAY_SECONDS=10
ENDPOINT_DIAGNOSIS_SUPERVISOR_ENABLED=false
ENDPOINT_DIAGNOSIS_SUPERVISOR_MODEL=
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
GET  /health
GET  /models
GET  /v1/models
GET  /v1/gateway/models
PUT  /v1/gateway/models
GET  /v1/gateway/endpoint-diagnosis
POST /v1/gateway/endpoint-diagnosis/refresh
POST /v1/gateway/endpoint-diagnosis/apply
GET  /v1/providers/status
POST /v1/chat/completions
```

`/v1/chat/completions` returns the upstream provider JSON unchanged. Gateway diagnostics are
returned as `X-Gateway-Provider`, `X-Gateway-Route`, `X-Gateway-Model`, and
`X-Gateway-Attempts` headers.

## Validate

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
```
