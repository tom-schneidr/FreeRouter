# OpenClaw integration and reliability

FreeRouter is best connected to OpenClaw through the OpenAI Chat Completions wire API. The `auto`
model selects a provider/model route for each turn and gives tool-bearing requests a separate,
empirically learned order.

```json5
{
  agents: {
    defaults: {
      model: { primary: "freerouter/auto" },
    },
  },
  models: {
    mode: "merge",
    providers: {
      freerouter: {
        baseUrl: "http://127.0.0.1:8000/v1",
        apiKey: "sk-local",
        api: "openai-completions",
        models: [
          {
            id: "auto",
            name: "FreeRouter Auto",
            reasoning: false,
            input: ["text"],
            contextWindow: 128000,
            maxTokens: 32000,
            compat: {
              supportsTools: true,
              strictMessageKeys: false,
            },
          },
        ],
      },
    },
  },
}
```

Use a non-empty local key; FreeRouter does not currently validate this value. `GET /v1/models`
publishes `capabilities.tools` and `supported_parameters` extensions so setup tools can confirm the
gateway surface.

## What FreeRouter verifies

Before a streamed tool response is exposed, FreeRouter assembles every fragmented call and verifies:

- the function was declared in the request;
- arguments are valid JSON and satisfy the supplied JSON Schema;
- call IDs are stable and unique (missing IDs are deterministically repaired);
- named, required, disabled, and parallel tool-choice rules are respected;
- the stream reached a real terminal event.

Invalid pre-commit responses fall through to the next route. A provider failure after ordinary text
has already been exposed is returned as an explicit stream error and is not counted as success.

Tool capability probes cover an exact forced call, autonomous selection among distractors, and a
tool-result continuation. Unknown discovered text models receive an exploratory probe unless their
provider metadata explicitly says tools are unsupported.

## Adaptive ordering

FreeRouter keeps generic model quality and probe results as cold-start priors. For tool requests,
candidates are ordered by time-decayed valid-call versus typed-failure evidence, with a conservative
posterior lower bound. Evidence is conditioned on the tool schema when matching history exists,
recent evidence weighs more heavily, and one lucky success cannot outrank a stable route.

The Models page and `GET /v1/gateway/models` expose each route's score and weighted observations.
Failures are separated into categories such as unknown tool, schema mismatch, action promise,
truncated stream, and provider stream error.

## Operational settings

```text
ROUTING_REJECT_INITIAL_ACTION_PROMISE=true
ROUTING_ALLOW_UNCONFIRMED_TOOL_USE_FALLBACK=true
```

The action-promise guard rejects initial answers such as "I will write it now" when no structured
tool call follows. Unconfirmed registry or metadata hints begin as last-resort candidates, but strong
matching empirical evidence can move them ahead of a repeatedly failing confirmed route. Set the
fallback option to `false` for a strict probe-confirmed-only environment.

FreeRouter's Responses adapter intentionally rejects `previous_response_id`: it does not pretend to
provide server-side Responses state over stateless Chat Completions providers. OpenClaw should use
`api: "openai-completions"` unless a future stateful Responses store is enabled.

For production validation, pin the OpenClaw version and run repeated sandboxed tasks that assert real
effects (file contents, test exit codes, browser state), not merely the final assistant text.
