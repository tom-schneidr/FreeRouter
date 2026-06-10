# Capability Tags

Capability tags describe what a **route** can handle. They are routing constraints, not marketing labels.

## Hard vs soft tags

| Tag | Routing | Verified by |
|-----|---------|-------------|
| `text` | Hard (baseline) | Any successful chat probe |
| `tool-use` | Hard | Dual live probes **and** runtime traffic only (`source: probe` / `runtime`) |
| `web-search` | Hard | `/web-search` path or `web_search_preview` in payload |
| `vision` | Hard | Probe with tiny image part |
| `json-schema` | Hard | Probe with trivial `response_format` schema |
| `reasoning` | Soft (ranking/UI unless request sets reasoning config) | Provider-specific probe |
| `coding` | Soft (informational only) | Heuristic / metadata |

**Fail closed:** For hard tags, `unknown` or `inconclusive` does **not** satisfy routing. A route must have `status: supported` for every required hard capability.

## Provenance

Every capability claim records:

- `source`: `provider_metadata` | `registry` | `probe` | `runtime` | `manual`
- `confidence`: `high` | `medium` | `low`
- `status`: `unknown` | `supported` | `unsupported` | `inconclusive`

Priority (highest wins unless locked): `manual` > `probe` > `runtime` > `registry` > `provider_metadata`.

User **tag locks** prevent automation from overwriting manual pins.

## Separate concerns

| Concept | Example |
|---------|---------|
| Route capability tag | `web-search` on a `ModelRoute` |
| HTTP endpoint | `POST /v1/chat/completions/web-search` |
| Agent tool name | `web_search_preview` in the request payload |

`web_search_preview` tools do **not** imply `tool-use`. Only `type: "function"` tools require the `tool-use` tag.

## Discovery pipeline

```text
1. Parse provider /models metadata  → medium-confidence claims
2. Apply capability registry          → high-confidence overrides
3. Respect manual tag locks
4. Run probes during endpoint diagnosis refresh (see **Probe schedule** below)

## Probe schedule

Each endpoint diagnosis refresh probes up to **12 enabled routes per configured provider**
(`PROBE_BUDGET_PER_PROVIDER`). Routes are ranked by:

```text
priority = 0.55 × staleness + 0.45 × rank_score
```

- **Staleness** — time since the latest `source: probe` claim, scaled to `0..1` over **7 days**
  (`PROBE_RECHECK_SECONDS`). Never-probed routes use staleness **1.25** so they are verified
  before recently checked low-priority routes.
- **Rank score** — `(max_rank - rank + 1) / max_rank` so rank 1 is 1.0 and tail routes approach 0.

Higher priority routes are probed first. With budget `B` and `N` enabled routes per provider,
a cold catalog is fully touched in about `ceil(N / B)` refreshes; after that each route is
re-probed at least every 7 days, while top-ranked routes are checked more often when equally stale.
5. Derive tags[] from supported capabilities
```

OpenRouter free models receive `text` by default. `tool-use` appears in `tags[]` only after successful **dual probes** (`echo` + `add` function tools) or validated **runtime** traffic. The registry and provider metadata may record hints in `capabilities`, but those alone never publish the `tool-use` tag.

**Autonomous correction:** When a client sends function tools and the model replies with prose/JSON instead of structured `tool_calls`, FreeRouter demotes the route, removes the tag, and waterfall-fails over to the next candidate.

## Routing match

```python
required_capabilities.issubset(route.tags)
```

Live monitor events include `required_capabilities` on `request_started` and `route_tags` on `route_selected` so operators can see required vs matched capabilities.
