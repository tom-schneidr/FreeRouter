# Sentinel Round 2 approval demo

## Start (about 2 minutes)

```powershell
.\.venv\Scripts\python.exe scripts\seed_sentinel_demo.py `
  --database .venv\sentinel-demo\gateway.sqlite3 `
  --catalog .venv\sentinel-demo\catalog.json

$env:DATABASE_PATH = (Resolve-Path .venv\sentinel-demo\gateway.sqlite3)
$env:MODEL_CATALOG_PATH = (Resolve-Path .venv\sentinel-demo\catalog.json)
$env:AUTO_ENDPOINT_DIAGNOSIS_ENABLED = "false"
$env:BENCHMARK_REFRESH_ENABLED = "false"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8765
```

In a second terminal:

```powershell
$env:FREEROUTER_API_TARGET = "http://127.0.0.1:8765"
npm --workspace @freerouter/ui run dev
```

Open [http://127.0.0.1:5173/app/#sentinel](http://127.0.0.1:5173/app/#sentinel).

## Approval checklist

- [ ] Hero reads “One trusted runtime. Two protected products.” and shows the fail-closed $0 guard.
- [ ] SemesterOS uses `safe-study`, reports no tool execution, and copies a config without an API key.
- [ ] AgentRange uses `safe-security`, reports proposal-only tools, and keeps the key server-side.
- [ ] Run both preflight buttons; each returns an actionable healthy/degraded/blocked result.
- [ ] Recent trusted runs show healthy, fallback-used, and blocked receipts with run IDs and no content.
- [ ] Select each profile and confirm its qualified count and remediation.
- [ ] Open route evidence and confirm all four deterministic probes are inspectable.
- [ ] Filter Agent ready / Needs evidence / Blocked and search by provider or model.
- [ ] Copy a consumer config and the OpenCode config; a visible confirmation appears.
- [ ] At 390px width, cards stack, buttons remain reachable, and the page has no horizontal overflow.
- [ ] Browser console contains no errors or warnings.

## API spot checks

```powershell
Invoke-RestMethod "http://127.0.0.1:8765/v1/gateway/sentinel/doctor?profile=safe-study"
Invoke-RestMethod "http://127.0.0.1:8765/v1/gateway/sentinel/preflight?profile=safe-security&chat=true&json=true&stream=true&tools=true"
```

Demo data contains synthetic route evidence and content-free request receipts only. It uses dummy provider configuration for presentation; do not click route evaluation unless real provider credentials are configured server-side.
