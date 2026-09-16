from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from math import sqrt
from typing import TYPE_CHECKING, Any, Literal

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
    auto_ok = "auto_selection=supported" in evidence
    continuation_ok = "tool_result_continuation=supported" in evidence
    behavior_failed = (
        "auto_selection=unsupported" in evidence
        or "tool_result_continuation=unsupported" in evidence
    )
    if auto_ok and continuation_ok:
        return 0.95
    if behavior_failed:
        return 0.76
    return 0.86


def tool_route_sort_key(
    route: ModelRoute,
    snapshot: ToolReliabilitySnapshot | None,
) -> tuple[float, int, int, str, str]:
    """Order tool routes by empirical reliability, confirmation, then catalog rank."""

    return (
        route_tool_reliability_score(route, snapshot),
        1 if "tool-use" in route.tags else 0,
        -route.rank,
        route.provider_name,
        route.model_id,
    )
