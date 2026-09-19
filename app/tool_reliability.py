from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from math import sqrt
from typing import TYPE_CHECKING, Any, Literal

from app.model_ranking import tool_use_behavior_score
from app.tool_use_validation import function_tool_calls_from_body, parse_function_tool_arguments

if TYPE_CHECKING:
    from app.model_catalog import ModelRoute


ToolOutcomeCategory = Literal[
    "valid_call",
    "valid_text",
    "provider_rejected",
    "malformed_call",
    "missing_call_id",
    "duplicate_call_id",
    "unknown_tool",
    "schema_invalid",
    "wrong_named_choice",
    "parallel_not_allowed",
    "action_promise",
    "truncated_stream",
    "stream_error",
    "continuation_success",
    "continuation_failure",
    "repeated_call",
    "repeated_request",
]


SUCCESS_CATEGORIES = frozenset(
    {
        "valid_call",
        "continuation_success",
    }
)


def tool_request_fingerprint(payload: dict[str, Any]) -> str | None:
    """Identify a stable tool schema/choice workload without storing request content."""

    tools = payload.get("tools")
    if not isinstance(tools, list) or not tools:
        return None
    material = {
        "tools": tools,
        "tool_choice": payload.get("tool_choice"),
        "parallel_tool_calls": payload.get("parallel_tool_calls"),
    }
    encoded = json.dumps(
        material,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def tool_failure_outcome_category(category: str) -> ToolOutcomeCategory:
    """Map validation detail onto the bounded reliability failure taxonomy."""

    if category == "undeclared_function":
        return "unknown_tool"
    if category == "arguments_schema_mismatch":
        return "schema_invalid"
    if category == "invalid_call_id":
        return "missing_call_id"
    if category == "duplicate_call_id":
        return "duplicate_call_id"
    if category == "tool_choice_named_mismatch":
        return "wrong_named_choice"
    if category == "parallel_tool_calls_disabled":
        return "parallel_not_allowed"
    return "malformed_call"


def _tool_call_signature(call: dict[str, Any]) -> str | None:
    function = call.get("function")
    if not isinstance(function, dict) or not isinstance(function.get("name"), str):
        return None
    arguments = parse_function_tool_arguments(function.get("arguments"))
    if arguments is None:
        return None
    return json.dumps(
        {"name": function["name"], "arguments": arguments},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def repeated_tool_call_in_request(payload: dict[str, Any], body: dict[str, Any]) -> bool:
    """Detect a valid tool call that repeats an earlier action in the payload.

    This is a bounded diagnostic signal.  It does not reject the response or
    assume that every repeated command is wrong; it simply gives automatic
    reliability ranking evidence about loop-prone model behaviour.
    """
    current = {_tool_call_signature(call) for call in function_tool_calls_from_body(body)}
    current.discard(None)
    if not current:
        return False
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return False
    previous: set[str] = set()
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        calls = message.get("tool_calls")
        if not isinstance(calls, list):
            continue
        for call in calls:
            if isinstance(call, dict):
                signature = _tool_call_signature(call)
                if signature:
                    previous.add(signature)
    return bool(current.intersection(previous))


def repeated_user_request_in_payload(payload: dict[str, Any]) -> bool:
    """Detect an identical user turn after a tool result or assistant action."""
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return False
    user_texts: list[str] = []
    saw_tool_history = False
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") in {"tool", "assistant"} and (
            message.get("role") == "tool" or isinstance(message.get("tool_calls"), list)
        ):
            saw_tool_history = True
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                user_texts.append(" ".join(content.split()).lower())
    return saw_tool_history and len(user_texts) >= 2 and user_texts[-1] == user_texts[-2]


def runtime_tool_outcome_category(
    payload: dict[str, Any],
    body: dict[str, Any],
    default: ToolOutcomeCategory,
) -> ToolOutcomeCategory:
    if repeated_user_request_in_payload(payload):
        return "repeated_request"
    if repeated_tool_call_in_request(payload, body):
        return "repeated_call"
    return default


@dataclass(frozen=True)
class ToolReliabilitySnapshot:
    route_id: str
    successes: int = 0
    failures: int = 0
    last_observed_at: int | None = None

    @property
    def observations(self) -> int:
        return self.successes + self.failures


def route_tool_reliability_score(
    route: ModelRoute,
    snapshot: ToolReliabilitySnapshot | None,
) -> float:
    """Return a conservative route-specific probability of a usable tool turn.

    Catalog rank is intentionally only a cold-start prior. Once real observations
    accumulate, their posterior lower confidence estimate controls tool routing.
    """

    confirmed = "tool-use" in route.tags
    prior_mean = _tool_probe_prior(route) if confirmed else 0.62
    prior_strength = 4.0 if confirmed else 2.0
    successes = snapshot.successes if snapshot is not None else 0
    failures = snapshot.failures if snapshot is not None else 0
    alpha = prior_mean * prior_strength + successes
    beta = (1.0 - prior_mean) * prior_strength + failures
    total = alpha + beta
    mean = alpha / total
    variance = (alpha * beta) / ((total * total) * (total + 1.0))
    # Approximate one-sided 90% posterior lower bound. This deliberately rewards
    # stable routes over a route with one lucky success.
    return max(0.0, mean - 1.282 * sqrt(max(0.0, variance)))


def _tool_probe_prior(route: ModelRoute) -> float:
    claim = route.capabilities.get("tool-use")
    evidence = claim.evidence.lower() if claim is not None else ""
    if "openclaw tool profile" not in evidence:
        return 0.90
    def profile_status(name: str) -> str:
        structured = route.capabilities.get(f"tool-use.{name}")
        if structured is not None:
            return structured.status
        marker = f"{name.replace('-', '_')}="
        for part in evidence.split(";"):
            item = part.strip()
            if item.startswith(marker):
                return item.removeprefix(marker).strip()
        return "unknown"

    auto_status = profile_status("auto-selection")
    continuation_status = profile_status("tool-result-continuation")
    stability_status = profile_status("multi-turn-stability")
    auto_ok = auto_status == "supported"
    continuation_ok = continuation_status == "supported"
    stability_ok = stability_status == "supported"
    behavior_failed = (
        auto_status == "unsupported"
        or continuation_status == "unsupported"
        or stability_status == "unsupported"
    )
    if auto_ok and continuation_ok and stability_ok:
        return 0.97
    if auto_ok and continuation_ok and stability_status != "unsupported":
        return 0.95
    if behavior_failed:
        return 0.70
    return 0.86


def tool_route_sort_key(
    route: ModelRoute,
    snapshot: ToolReliabilitySnapshot | None,
) -> tuple[float, int, int, int, str, str]:
    """Order tool routes by empirical reliability, confirmation, then catalog rank."""

    return (
        route_tool_reliability_score(route, snapshot),
        tool_use_behavior_score(route),
        1 if "tool-use" in route.tags else 0,
        -route.rank,
        route.provider_name,
        route.model_id,
    )
