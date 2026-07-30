"""Zero-cost virtual routing profiles backed by Sentinel evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import time
from typing import Any, TypeGuard

from app.model_catalog import ModelRoute
from app.sentinel_types import SentinelEvaluation

FREE_COST_LABELS = frozenset({"free", "free-tier", "zero", "$0", "0"})
EVIDENCE_MAX_AGE_SECONDS = 7 * 24 * 60 * 60


@dataclass(frozen=True)
class AgentProfile:
    profile_id: str
    name: str
    description: str
    minimum_score: int
    required_checks: tuple[str, ...]
    zero_cost_only: bool = True
    preferred_speed: tuple[str, ...] = ()


AGENT_PROFILES = {
    "safe-coding": AgentProfile(
        profile_id="safe-coding",
        name="Safe coding",
        description="Verified tools, JSON, streaming, and canary privacy for higher-trust agents.",
        minimum_score=80,
        required_checks=("tool_call", "structured_json", "streaming", "canary"),
    ),
    "fast-coding": AgentProfile(
        profile_id="fast-coding",
        name="Fast coding",
        description="Streaming and tool-ready routes, preferring fast free-tier models.",
        minimum_score=70,
        required_checks=("tool_call", "streaming", "canary"),
        preferred_speed=("fast", "very-fast"),
    ),
}


def is_agent_profile(model: Any) -> TypeGuard[str]:
    return isinstance(model, str) and model in AGENT_PROFILES


def route_is_zero_cost(route: ModelRoute) -> bool:
    return route.cost.strip().lower() in FREE_COST_LABELS


def evaluation_is_fresh(
    evaluation: SentinelEvaluation, *, now: int | None = None
) -> bool:
    return (now or int(time())) - evaluation.completed_at <= EVIDENCE_MAX_AGE_SECONDS


def route_qualifies(
    route: ModelRoute,
    evaluation: SentinelEvaluation | None,
    profile: AgentProfile,
    *,
    now: int | None = None,
) -> tuple[bool, str]:
    if profile.zero_cost_only and not route_is_zero_cost(route):
        return False, "Route is not explicitly marked free-tier; the $0 guard rejected it."
    if evaluation is None:
        return False, "No Sentinel evidence yet."
    if not evaluation_is_fresh(evaluation, now=now):
        return False, "Sentinel evidence is older than 7 days."
    if evaluation.score < profile.minimum_score:
        return False, f"Score {evaluation.score} is below the required {profile.minimum_score}."
    probes = {probe.check_id: probe for probe in evaluation.probes}
    failed = [
        check_id
        for check_id in profile.required_checks
        if check_id not in probes or probes[check_id].status != "pass"
    ]
    if failed:
        return False, f"Required checks need attention: {', '.join(failed)}."
    return True, "Qualified by current Sentinel evidence."


def filter_routes_for_profile(
    routes: list[ModelRoute],
    evaluations: dict[str, SentinelEvaluation],
    profile_id: str,
) -> list[ModelRoute]:
    profile = AGENT_PROFILES[profile_id]
    qualified = [
        route
        for route in routes
        if route_qualifies(route, evaluations.get(route.route_id), profile)[0]
    ]
    if profile.preferred_speed:
        speed_order = {speed: index for index, speed in enumerate(profile.preferred_speed)}
        qualified.sort(
            key=lambda route: (
                speed_order.get(route.speed.lower(), len(speed_order)),
                route.rank,
            )
        )
    return qualified


def profile_diagnostic(
    profile: AgentProfile,
    routes: list[ModelRoute],
    evaluations: dict[str, SentinelEvaluation],
    configured_providers: set[str],
) -> dict[str, Any]:
    enabled = [route for route in routes if route.enabled]
    free = [route for route in enabled if route_is_zero_cost(route)]
    configured = [route for route in free if route.provider_name in configured_providers]
    evaluated = [route for route in configured if route.route_id in evaluations]
    qualified = filter_routes_for_profile(configured, evaluations, profile.profile_id)
    if qualified:
        status = "ready"
        message = f"{len(qualified)} verified $0 route{'s' if len(qualified) != 1 else ''} available."
        remediation = "Use this profile as the model in any OpenAI-compatible client."
    elif not configured:
        status = "blocked"
        message = "No configured free-tier route can satisfy this profile."
        remediation = "Add a provider API key, then run Sentinel on an enabled free-tier route."
    elif not evaluated:
        status = "untested"
        message = "Free-tier routes exist, but none have current evidence."
        remediation = "Run Sentinel on a configured route. Each run sends four short test prompts."
    else:
        status = "blocked"
        message = "No route meets every hard constraint."
        remediation = "Open a route card, fix its failed checks, and run the evaluation again."
    return {
        **asdict(profile),
        "status": status,
        "message": message,
        "remediation": remediation,
        "counts": {
            "enabled": len(enabled),
            "free": len(free),
            "configured": len(configured),
            "evaluated": len(evaluated),
            "qualified": len(qualified),
        },
        "qualified_route_ids": [route.route_id for route in qualified],
    }
