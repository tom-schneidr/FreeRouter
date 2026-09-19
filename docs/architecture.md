# Architecture

FreeRouter has three layers with a deliberately narrow boundary between them. The product goal is to
pool independent free-tier quotas; the architecture keeps that goal separate from the protocol a
provider happens to expose.

1. **Client API adapters** in `app/api/` translate OpenAI, Responses, and Anthropic requests into the
   gateway's internal request shape. They also add route-transparency headers and stream responses.
2. **The waterfall router** in `app/router.py` filters the editable catalog by requested
   capabilities, ranks eligible routes, reserves local quota, and retries recoverable failures. The
   router records structured attempts instead of hiding fallback behaviour.
3. **Provider adapters and local state** in `app/providers/` and `app/state.py` isolate upstream
   protocol differences from quota, cooldown, and concurrency bookkeeping. The current registry uses
   upstream OpenAI-compatible endpoints, while `ProviderAdapter` is the extension point for a native
   provider protocol later. SQLite is used for durable local state; the React UI reads the same service
   APIs.

Sentinel is an evidence gate beside the router. It runs four small deterministic probes, stores
timestamped results, and exposes virtual profiles such as `safe-coding`. A profile can enter the
router only when its required checks are fresh, passing, and explicitly marked free-tier. If no route
qualifies, the request fails closed with a remediation message.

The repository also contains a React control plane and a Tauri shell. They are presentation and
operations surfaces over the same FastAPI service, rather than a second routing implementation.

## Request path

```text
client -> API adapter -> capability filter -> ranked waterfall
       -> provider adapter -> response/stream validation
       -> transparency headers + local state/receipt
```

Provider quality, quotas, pricing, and availability are external inputs. Local tests use fakes and
deterministic fixtures; a live provider key is required to verify an upstream route end to end.
