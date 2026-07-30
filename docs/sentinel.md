# FreeRouter Sentinel

Sentinel turns a model route into an evidence-backed agent route. It runs four short, deterministic
checks through the selected provider, stores the results in FreeRouter's existing SQLite database,
and exposes virtual model profiles that fail closed when no route qualifies.

## What Sentinel checks

| Check | Passing evidence | Why it matters |
|---|---|---|
| Tool-call conformance | One native function call with exact schema-valid arguments | Coding agents must invoke tools instead of describing an invocation in prose. |
| Structured JSON | Exact output matching a strict JSON schema | Agent loops need machine-readable results without repair heuristics. |
| Streaming readiness | Valid OpenAI SSE content plus a `[DONE]` terminator | Interactive tools need predictable incremental output. |
| Canary privacy | A synthetic system canary is not repeated after an injection attempt | Catches basic instruction-exfiltration failures without using real secrets. |

The canary is generated from the public route ID. It is synthetic, contains no user data, and is
never stored in evidence. Sentinel stores only the pass/fail result, score, latency, safe summary,
remediation, and timestamps.

## Run the workflow

1. Start FreeRouter.

   ```powershell
   .\run.bat
   ```

2. Open [http://127.0.0.1:8000/app#sentinel](http://127.0.0.1:8000/app#sentinel).

3. Add a provider API key under **Settings**, then choose an enabled free-tier route.

4. Select **Evaluate**. A run sends four short prompts and consumes provider request/token quota,
   but the hard `$0` guard refuses any route not explicitly labelled `free-tier`.

5. Confirm that `safe-coding` or `fast-coding` shows at least one qualified route.

6. Use the profile as an OpenAI-compatible model:

   ```powershell
   $body = @{
     model = "safe-coding"
     messages = @(@{ role = "user"; content = "Return the current working directory." })
   } | ConvertTo-Json -Depth 5

   Invoke-RestMethod `
     -Uri http://127.0.0.1:8000/v1/chat/completions `
     -Method Post `
     -ContentType application/json `
     -Body $body
   ```

If no route qualifies, the inference API returns a structured `no_qualifying_route` error with the
profile, active `$0` guard, required checks, and a remediation. It never silently falls back to an
untested or potentially paid route.

## Profiles

### `safe-coding`

Requires a current score of at least 80 and passing results for tool calls, JSON, streaming, and
canary privacy. This is the recommended default for tool-using coding agents.

### `fast-coding`

Requires a current score of at least 70 with passing tool-call, streaming, and canary checks. Among
qualifying routes, models marked `fast` or `very-fast` are preferred. The `$0` guard is still hard.

Evidence is current for seven days. After that, profiles exclude the route until it is evaluated
again.

## OpenCode

The Sentinel screen generates a copyable `opencode.json` using OpenCode's generic
`@ai-sdk/openai-compatible` provider:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "freerouter/safe-coding",
  "provider": {
    "freerouter": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "FreeRouter Sentinel",
      "options": {
        "baseURL": "http://127.0.0.1:8000/v1",
        "apiKey": "sk-local"
      },
      "models": {
        "safe-coding": { "name": "Sentinel · Safe coding" },
        "fast-coding": { "name": "Sentinel · Fast coding" }
      }
    }
  }
}
```

Save it as `opencode.json` in the project you will open with OpenCode. No FreeRouter-specific fork
or patch is required. The configuration follows OpenCode's official
[custom provider documentation](https://opencode.ai/docs/providers/#custom-provider).

Run the local doctor before launching OpenCode:

```powershell
Invoke-RestMethod `
  "http://127.0.0.1:8000/v1/gateway/sentinel/doctor?profile=safe-coding"
```

The doctor checks the hard `$0` policy, provider configuration, current evidence, and whether at
least one route qualifies.

## API

```text
GET  /v1/gateway/sentinel
GET  /v1/gateway/sentinel/doctor?profile=safe-coding
POST /v1/gateway/sentinel/routes/{route_id}/evaluate
```

`GET /v1/models` also advertises `safe-coding` and `fast-coding` as virtual models.

## Local visual demo

To preview all readiness states without provider credentials, use explicit temporary paths:

```powershell
python scripts/seed_sentinel_demo.py `
  --database C:\tmp\freerouter-sentinel-demo.sqlite3 `
  --catalog C:\tmp\freerouter-sentinel-demo-catalog.json

$env:DATABASE_PATH = "C:\tmp\freerouter-sentinel-demo.sqlite3"
$env:MODEL_CATALOG_PATH = "C:\tmp\freerouter-sentinel-demo-catalog.json"
$env:AUTO_ENDPOINT_DIAGNOSIS_ENABLED = "false"
$env:BENCHMARK_REFRESH_ENABLED = "false"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

This demo writes only to the paths supplied on the command line. It never changes the normal
`data/` database or catalog.
