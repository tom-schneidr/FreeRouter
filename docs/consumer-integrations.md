# Sentinel consumer integration contracts

FreeRouter Sentinel is the shared OpenAI-compatible AI runtime for SemesterOS and AgentRange. The gateway keeps legacy response bodies stable and adds evidence-backed virtual models, preflight health, and versioned response receipts.

## Profiles

| Consumer | Primary model | Tool policy | Timeout | Retries | Explicit fallback |
| --- | --- | --- | ---: | ---: | --- |
| StudyShell / SemesterOS | `safe-study` | none; return a structured plan only | 60s | 1 | `auto` |
| ClusterLab / AgentRange | `safe-security` | proposal-only; the consumer approves execution | 90s | 2 | `auto` |

Both profiles require current Sentinel evidence and an explicitly free-tier route. A profile never silently falls through to `auto`: the consumer must make that downgrade and retain the preflight or request failure reason in its own trace.

## Preflight and doctor

Run preflight before a workflow that depends on a capability:

```text
GET /v1/gateway/sentinel/preflight?profile=safe-study&chat=true&json=true&stream=true&tools=false
GET /v1/gateway/sentinel/preflight?profile=safe-security&chat=true&json=true&stream=true&tools=true
```

The response status is `healthy`, `degraded`, or `blocked`. `degraded` is runnable but lacks route redundancy. `blocked` includes an actionable `next_action`. For a compact startup check use the existing endpoint:

```text
GET /v1/gateway/sentinel/doctor?profile=safe-study
GET /v1/gateway/sentinel/doctor?profile=safe-security
```

## Stable response receipt

Successful OpenAI-compatible responses preserve the existing `X-Gateway-*` headers and add contract v1 headers:

| Header | Meaning |
| --- | --- |
| `X-FreeRouter-Contract-Version` | Receipt schema version (currently `1`) |
| `X-FreeRouter-Run-Id` | Correlation ID used by the Sentinel receipt ledger |
| `X-Gateway-Provider` / `X-FreeRouter-Provider` | Selected provider |
| `X-Gateway-Route` / `X-FreeRouter-Route` | Selected FreeRouter route ID |
| `X-Gateway-Model` / `X-FreeRouter-Model` | Concrete provider model |
| `X-FreeRouter-Profile` | Requested Sentinel profile, if any |
| `X-FreeRouter-Policy-Verdict` | `allowed`, `blocked`, `pending`, or `not-evaluated` |
| `X-FreeRouter-Capabilities` | Capabilities inferred from the request |
| `X-FreeRouter-Tool-Policy` | `none`, `proposal-only`, or `declared` |
| `X-FreeRouter-Latency-Ms` | End-to-end gateway latency for non-streaming responses |
| `X-FreeRouter-Attempts` | Waterfall attempts |
| `X-FreeRouter-Fallback` | `direct` or `fallback` |
| `X-FreeRouter-Fallback-Reason` | First actionable downgrade reason |

Streaming headers are sent immediately to preserve client compatibility. They contain the run ID, profile, requested capabilities, and a `pending` verdict. Read the completed trace from the Sentinel dashboard by run ID for final route and latency metadata.

## Structured output contract

Send an OpenAI `response_format` with `type: json_schema`; do not rely on “return JSON” prompt text. Validate the returned object in the consumer. Retry only on transport failure, timeout, invalid JSON, or a retryable provider response, bounded by the profile's `max_retries`. Never retry a policy block with the same profile.

## StudyShell / SemesterOS

StudyShell currently reads `FREEROUTER_BASE_URL` and keeps `FREEROUTER_MODEL = "auto"` in `src/utils/freerouter.ts`. Its Tauri Rust client sends requests server-side and currently discards response headers.

1. Keep `FREEROUTER_BASE_URL=http://127.0.0.1:8000/v1` in `.env`. Do not add a browser API key.
2. Change the primary model constant to `safe-study` and add an `auto` fallback constant.
3. Before a study workflow, call the `safe-study` preflight URL. Use `safe-study` for healthy/degraded; use `auto` only when blocked and display the reason.
4. In `src-tauri/src/ai_client.rs`, capture `X-FreeRouter-Run-Id` and the existing `X-Gateway-*` headers before consuming `resp.text()`.
5. Keep execution outside the model. `safe-study` rejects any request containing tools.

## ClusterLab / AgentRange

ClusterLab already reads `FREEROUTER_BASE_URL`, `FREEROUTER_API_KEY`, and `FREEROUTER_MODEL` in its Python backend.

```dotenv
FREEROUTER_BASE_URL=http://127.0.0.1:8000/v1
FREEROUTER_MODEL=safe-security
FREEROUTER_FALLBACK_MODEL=auto
FREEROUTER_API_KEY=<server-side-only>
```

1. Preflight `safe-security` with `tools=true` before the graph run.
2. In `clusterlab/llm.py`, read the receipt headers from `urlopen(...)` before the body and attach the run ID/route/provider/model to `LlmResult`.
3. Tool calls are proposals. ClusterLab remains responsible for authorization and execution.
4. If preflight is blocked or the profile returns `no_qualifying_route`, retry the configured `auto` model and mark the result degraded with the original reason.
5. Keep `FREEROUTER_API_KEY` in the backend process environment only; never return it through the API or UI.

A runnable standard-library example is provided at [examples/sentinel_consumer.py](../examples/sentinel_consumer.py).
