# Capability Tags

Capability tags describe what a **route** can handle. They are routing constraints, not marketing labels.

## Hard vs soft tags

| Tag | Routing | Verified by |
|-----|---------|-------------|
| `text` | Hard (baseline) | Any successful chat probe |
| `tool-use` | Hard | Exact function-call probe **and** runtime traffic only (`source: probe` / `runtime`) |
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
- `last_attempted_at`, `next_probe_at`, and `reason` when a probe was transiently unavailable

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
5. Derive tags[] from supported capabilities
```

Tool-use claims also retain a profile under these evidence-only keys:

- `tool-use.required-exact-call`
- `tool-use.auto-selection`
- `tool-use.tool-result-continuation`
- `tool-use.multi-turn-stability`

The public `tool-use` tag remains the transport-compatibility gate. The profile
feeds automatic ordering for agentic requests, so a model that can emit one
well-formed call but repeatedly fails continuation is still discoverable while
being ranked below a model with equivalent intelligence and stronger evidence.

## Probe schedule

Each endpoint diagnosis refresh probes up to **12 enabled routes per configured provider**
(`PROBE_BUDGET_PER_PROVIDER`). Routes are ranked by:

```text
priority = 0.55 × staleness + 0.45 × rank_score
```

- **Staleness** — time since the latest successful or explicit unsupported verification, scaled to
  `0..1` over **7 days** (`PROBE_RECHECK_SECONDS`). A rate limit, timeout, or temporary provider
  failure is not verification; it gets a bounded retry time in `next_probe_at` instead of making
  the route look fresh for a week.
- **Retryability** — routes whose transient retry window has elapsed return to the probe pool,
  while routes still inside that window are skipped for this refresh.
- **Rank score** — `(max_rank - rank + 1) / max_rank` so rank 1 is 1.0 and tail routes approach 0.

Higher priority routes are probed first. With budget `B` and `N` enabled routes per provider,
a cold catalog is fully touched in about `ceil(N / B)` refreshes; after that each route is
re-probed at least every 7 days, while top-ranked routes are checked more often when equally stale.

OpenRouter free models receive `text` by default. `tool-use` appears in `tags[]` only after a
successful exact-call probe or validated runtime traffic. Provider metadata and registry entries
remain discovery hints and probe candidates; they do not publish a confirmed `tool-use` tag by
themselves. Discovery also stores a small provenance record (`discovery_source`, timestamp,
pricing evidence, modalities, and the provider's explicit tool claim) so automatic decisions can
be inspected later.

Catalog files written by current versions use a versioned object with `schema_version: 2` and a
`routes` array. Older bare-array catalogs remain readable and are migrated on the next normal save;
route-level fields added by newer versions have defaults so existing user catalogs remain usable.

**Autonomous correction:** When a client sends function tools and the model replies with prose/JSON
instead of structured `tool_calls`, FreeRouter records the typed failure and waterfall-fails over
to the next candidate. A confirmed transport capability is not erased by one malformed response;
only explicit upstream rejection changes the transport classification. Repeated valid calls or
repeated user turns are recorded separately as loop-prone behaviour so automatic reliability can
lower the route's preference without confusing model behaviour with protocol support.

## Routing match

```python
required_capabilities.issubset(route.tags)
```

Live monitor events include `required_capabilities` on `request_started` and `route_tags` on `route_selected` so operators can see required vs matched capabilities.
