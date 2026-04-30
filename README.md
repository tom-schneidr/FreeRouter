# FreeRouter

Local OpenAI-compatible gateway with a state-aware waterfall router for free-tier AI providers.

Default model-strength waterfall:

The router ranks all available free-tier models based on their raw benchmark strength and capabilities (DeepSeek V4 Pro, GPT OSS 120B, Gemini Pro, etc.). It queries the most powerful models first. If the top model is rate-limited or unavailable, it gracefully falls back to the next strongest model in the list. If two providers offer the exact same model, the router prioritizes the provider with the most generous free-tier limits.

## File Structure

```text
.
├── app/
│   ├── main.py                 # FastAPI app and OpenAI-compatible endpoints
│   ├── model_catalog.py        # Editable ranked model catalog defaults
│   ├── router.py               # Ranked model waterfall routing engine
│   ├── settings.py             # .env-backed configuration
│   ├── state.py                # SQLite quota/cooldown/RPM tracker
│   └── providers/
│       ├── base.py             # OpenAI-compatible provider adapter
│       └── registry.py         # Provider order, quotas, endpoints, models
├── data/                       # Runtime SQLite DB and editable model catalog
├── tests/                      # State and router behavior tests
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
```

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

## Endpoints

```text
GET  /health
GET  /models
GET  /v1/models
GET  /v1/gateway/models
PUT  /v1/gateway/models
GET  /v1/providers/status
POST /v1/chat/completions
```

`/v1/chat/completions` returns the upstream provider JSON unchanged. Gateway diagnostics are
returned as `X-Gateway-Provider`, `X-Gateway-Route`, `X-Gateway-Model`, and
`X-Gateway-Attempts` headers.

## Validate

```powershell
python -m pytest
python -m ruff check .
```
