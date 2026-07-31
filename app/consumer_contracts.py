"""Stable downstream consumer contracts and safe receipt metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from app.agent_profiles import AGENT_PROFILES, AgentProfile, is_agent_profile

CONTRACT_VERSION = "1"


@dataclass(frozen=True)
class ConsumerDefinition:
    consumer_id: str
    name: str
    product: str
    profile_id: str
    description: str
    repository: str
    accent: str


CONSUMERS = {
    "semesteros": ConsumerDefinition(
        consumer_id="semesteros",
        name="SemesterOS",
        product="StudyShell",
        profile_id="safe-study",
        description="Structured study plans and summaries with no model-selected tool execution.",
        repository="StudyShell / SemesterOS",
        accent="violet",
    ),
    "agentrange": ConsumerDefinition(
        consumer_id="agentrange",
        name="AgentRange",
        product="ClusterLab",
        profile_id="safe-security",
        description="Security-lab planning with proposal-only tools and downstream approval.",
        repository="ClusterLab / AgentRange",
        accent="amber",
    ),
}


def requested_profile(model: Any) -> AgentProfile | None:
    if not is_agent_profile(model):
        return None
    return AGENT_PROFILES[model]


def policy_verdict(model: Any, *, allowed: bool = True) -> str:
    if not is_agent_profile(model):
        return "not-evaluated"
    return "allowed" if allowed else "blocked"


def fallback_details(attempts: list[Any]) -> tuple[bool, str]:
    """Return a deterministic fallback flag and first actionable downgrade reason."""
    normalized: list[dict[str, Any]] = []
    for attempt in attempts:
        if hasattr(attempt, "__dataclass_fields__"):
            normalized.append(asdict(attempt))
        elif isinstance(attempt, dict):
            normalized.append(dict(attempt))
    selected_index = next(
        (index for index, item in enumerate(normalized) if item.get("status") == "selected"),
        len(normalized) - 1,
    )
    prior = normalized[: max(0, selected_index)]
    meaningful = [
        item
        for item in prior
        if item.get("status") in {"failed", "flagged", "rate_limited", "cooldown"}
    ]
    if not meaningful:
        return False, ""
    first = meaningful[0]
    reason = str(first.get("reason") or first.get("status") or "route_unavailable")
    route = str(first.get("route_id") or first.get("provider_name") or "earlier route")
    return True, f"{route}: {reason}"


def profile_contract(profile: AgentProfile) -> dict[str, Any]:
    return {
        "version": CONTRACT_VERSION,
        "profile_id": profile.profile_id,
        "consumer_id": profile.consumer_id,
        "zero_cost_only": profile.zero_cost_only,
        "tool_policy": profile.tool_policy,
        "timeout_seconds": profile.timeout_seconds,
        "max_retries": profile.max_retries,
        "required_checks": list(profile.required_checks),
        "fallback_model": "auto",
        "fallback_mode": "consumer-controlled",
    }


def consumer_connection(consumer: ConsumerDefinition, base_url: str) -> dict[str, Any]:
    root = base_url.rstrip("/")
    if not root.endswith("/v1"):
        root = f"{root}/v1"
    profile = AGENT_PROFILES[consumer.profile_id]
    env_lines = [
        f"FREEROUTER_BASE_URL={root}",
        f"FREEROUTER_MODEL={profile.profile_id}",
        "FREEROUTER_FALLBACK_MODEL=auto",
    ]
    if consumer.consumer_id == "agentrange":
        env_lines.append("FREEROUTER_API_KEY=<server-side-only>")
    else:
        env_lines.append("# No API key in browser or frontend storage")
    env = "\n".join(env_lines)
    return {
        **asdict(consumer),
        "contract": profile_contract(profile),
        "env": env,
        "preflight_url": (
            f"{root}/gateway/sentinel/preflight?profile={profile.profile_id}"
            "&chat=true&json=true&stream=true"
            f"&tools={'true' if profile.tool_policy != 'none' else 'false'}"
        ),
        "doctor_url": f"{root}/gateway/sentinel/doctor?profile={profile.profile_id}",
    }


def safe_receipt(
    *,
    event_type: str,
    request_id: str,
    timestamp: int,
    context: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    """Reduce monitor data to non-content metadata safe for persistence and UI display."""
    if event_type not in {
        "request_completed",
        "request_failed",
        "request_rejected",
        "request_closed",
    }:
        return None
    requested_model = context.get("model")
    profile = requested_profile(requested_model)
    attempts_detail = payload.get("attempts_detail")
    attempts = attempts_detail if isinstance(attempts_detail, list) else []
    fallback_used, fallback_reason = fallback_details(attempts)
    status = {
        "request_completed": "healthy" if not fallback_used else "degraded",
        "request_failed": "blocked",
        "request_rejected": "blocked",
        "request_closed": "blocked",
    }[event_type]
    reason = str(payload.get("reason") or fallback_reason or "")
    capabilities = context.get("required_capabilities")
    return {
        "run_id": request_id,
        "created_at": timestamp,
        "consumer_id": profile.consumer_id if profile else None,
        "profile_id": profile.profile_id if profile else None,
        "status": status,
        "policy_verdict": policy_verdict(requested_model, allowed=event_type == "request_completed"),
        "provider_name": str(payload.get("provider_name") or ""),
        "route_id": str(payload.get("route_id") or ""),
        "model_id": str(payload.get("model_id") or ""),
        "latency_ms": int(payload.get("latency_ms") or 0),
        "attempts": int(payload.get("attempts") or len(attempts)),
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason or reason,
        "stream": bool(context.get("stream")),
        "capabilities": list(capabilities) if isinstance(capabilities, list) else [],
        "tool_policy": profile.tool_policy if profile else "unmanaged",
        "request_path": str(context.get("path") or ""),
    }
