"""Sentinel orchestration, profile diagnostics, and OpenCode setup."""

from __future__ import annotations

import json
from typing import Any

import httpx

from app.agent_profiles import (
    AGENT_PROFILES,
    filter_routes_for_profile,
    profile_diagnostic,
    route_is_zero_cost,
)
from app.consumer_contracts import (
    CONSUMERS,
    CONTRACT_VERSION,
    consumer_connection,
    profile_contract,
)
from app.model_catalog import ModelCatalog, ModelRoute
from app.providers.base import ProviderAdapter
from app.sentinel import SentinelEvaluator
from app.sentinel_store import SentinelStore


class SentinelService:
    def __init__(
        self,
        providers: list[ProviderAdapter],
        catalog: ModelCatalog,
        client: httpx.AsyncClient,
        store: SentinelStore,
        *,
        timeout_seconds: float = 45.0,
    ) -> None:
        self.providers = providers
        self.provider_by_name = {provider.name: provider for provider in providers}
        self.catalog = catalog
        self.client = client
        self.store = store
        self.evaluator = SentinelEvaluator(store, timeout_seconds=timeout_seconds)

    async def evaluate_route(self, route_id: str):
        route = self._route(route_id)
        provider = self.provider_by_name.get(route.provider_name)
        if provider is None:
            raise LookupError(f"Provider {route.provider_name} is not registered.")
        if not provider.is_configured:
            raise ValueError(
                f"{route.provider_name} needs an API key before Sentinel can evaluate this route."
            )
        if not route_is_zero_cost(route):
            raise ValueError(
                "Sentinel refused this route because it is not explicitly marked free-tier."
            )
        return await self.evaluator.evaluate(provider, self.client, route)

    async def snapshot(self, *, base_url: str) -> dict[str, Any]:
        routes = self.catalog.all_routes()
        evaluations = await self.store.latest_for_routes([route.route_id for route in routes])
        configured = {
            provider.name for provider in self.providers if provider.is_configured
        }
        route_rows = [
            self._route_payload(route, evaluations.get(route.route_id), configured)
            for route in routes
        ]
        profiles = [
            profile_diagnostic(profile, routes, evaluations, configured)
            for profile in AGENT_PROFILES.values()
        ]
        receipts = await self.store.recent_receipts(limit=12)
        consumers = []
        for consumer in CONSUMERS.values():
            profile = AGENT_PROFILES[consumer.profile_id]
            diagnostic = next(
                row for row in profiles if row["profile_id"] == profile.profile_id
            )
            consumers.append(
                {
                    **consumer_connection(consumer, base_url),
                    "readiness": diagnostic,
                }
            )
        return {
            "object": "sentinel.snapshot",
            "contract_version": CONTRACT_VERSION,
            "zero_cost_guard": {
                "enabled": True,
                "message": (
                    "Profiles only route to models explicitly marked free-tier. "
                    "Unknown or paid cost labels fail closed."
                ),
            },
            "summary": {
                "routes": len(routes),
                "configured": sum(1 for row in route_rows if row["configured"]),
                "evaluated": sum(1 for row in route_rows if row["evaluation"] is not None),
                "ready": sum(
                    1
                    for row in route_rows
                    if row["evaluation"] and row["evaluation"]["readiness"] == "ready"
                ),
                "blocked": sum(
                    1
                    for row in route_rows
                    if row["evaluation"] and row["evaluation"]["readiness"] == "blocked"
                ),
            },
            "profiles": profiles,
            "consumers": consumers,
            "receipts": receipts,
            "routes": route_rows,
            "opencode": opencode_setup(base_url),
        }

    async def preflight(
        self,
        profile_id: str,
        *,
        chat: bool = True,
        json_output: bool = True,
        stream: bool = True,
        tools: bool = False,
    ) -> dict[str, Any]:
        profile = AGENT_PROFILES.get(profile_id)
        if profile is None:
            raise KeyError(f"Unknown Sentinel profile: {profile_id}")
        routes = self.catalog.all_routes()
        evaluations = await self.store.latest_for_routes([route.route_id for route in routes])
        configured = {
            provider.name for provider in self.providers if provider.is_configured
        }
        configured_routes = [
            route
            for route in routes
            if route.enabled and route.provider_name in configured
        ]
        qualified = filter_routes_for_profile(
            configured_routes, evaluations, profile.profile_id
        )

        def capability_ready(check_id: str) -> bool:
            for route in qualified:
                evaluation = evaluations.get(route.route_id)
                if evaluation is None:
                    continue
                probes = {probe.check_id: probe for probe in evaluation.probes}
                if check_id in probes and probes[check_id].status == "pass":
                    return True
            return False

        checks: list[dict[str, Any]] = []
        if chat:
            checks.append(
                {
                    "id": "chat",
                    "status": "pass" if qualified else "fail",
                    "message": (
                        f"{len(qualified)} evidence-backed route(s) support chat."
                        if qualified
                        else "No configured route currently satisfies the profile contract."
                    ),
                    "action": "Run Sentinel on a configured free-tier route." if not qualified else "",
                }
            )
        desired = (
            ("structured_json", json_output, "Structured JSON"),
            ("streaming", stream, "Streaming"),
        )
        for check_id, requested, label in desired:
            if not requested:
                continue
            ready = capability_ready(check_id)
            checks.append(
                {
                    "id": check_id,
                    "status": "pass" if ready else "fail",
                    "message": f"{label} evidence is current." if ready else f"{label} is not verified on a qualified route.",
                    "action": "Run the route evaluation and resolve the failed probe." if not ready else "",
                }
            )
        if tools:
            allowed = profile.tool_policy != "none"
            ready = allowed and capability_ready("tool_call")
            checks.append(
                {
                    "id": "tool_call",
                    "status": "pass" if ready else "fail",
                    "message": (
                        f"Tool calls are {profile.tool_policy} and verified."
                        if ready
                        else (
                            "This profile deliberately blocks tool calls."
                            if not allowed
                            else "Tool-call conformance is not verified."
                        )
                    ),
                    "action": (
                        "Use structured output for a plan, or select safe-security."
                        if not allowed
                        else ("Run Sentinel and resolve the tool-call probe." if not ready else "")
                    ),
                }
            )
        failures = [check for check in checks if check["status"] == "fail"]
        if failures:
            status = "blocked"
            reason = failures[0]["message"]
            action = failures[0]["action"]
        elif len(qualified) == 1:
            status = "degraded"
            reason = "Contract is healthy, but only one route currently qualifies."
            action = "Evaluate another configured free-tier route for fallback resilience."
            checks.append(
                {
                    "id": "route_resilience",
                    "status": "warn",
                    "message": reason,
                    "action": action,
                }
            )
        else:
            status = "healthy"
            reason = f"{len(qualified)} routes satisfy every requested capability."
            action = "No action required."
        return {
            "ok": status != "blocked",
            "status": status,
            "profile": profile_contract(profile),
            "requested": {
                "chat": chat,
                "structured_json": json_output,
                "streaming": stream,
                "tool_calling": tools,
            },
            "checks": checks,
            "qualified_route_ids": [route.route_id for route in qualified],
            "reason": reason,
            "next_action": action,
            "fallback": {
                "model": "auto",
                "mode": "consumer-controlled",
                "message": (
                    "Profiles never silently bypass policy. A consumer may explicitly retry "
                    "with its configured auto model and must retain the blocked/degraded reason."
                ),
            },
        }

    async def doctor(self, profile_id: str) -> dict[str, Any]:
        profile = AGENT_PROFILES.get(profile_id)
        if profile is None:
            raise KeyError(f"Unknown Sentinel profile: {profile_id}")
        routes = self.catalog.all_routes()
        evaluations = await self.store.latest_for_routes([route.route_id for route in routes])
        configured = {
            provider.name for provider in self.providers if provider.is_configured
        }
        diagnostic = profile_diagnostic(profile, routes, evaluations, configured)
        checks = [
            {
                "id": "zero_cost_guard",
                "status": "pass",
                "message": "Hard $0 guard is enabled and fails closed.",
            },
            {
                "id": "configured_route",
                "status": "pass" if diagnostic["counts"]["configured"] else "fail",
                "message": (
                    f"{diagnostic['counts']['configured']} configured free-tier route(s)."
                    if diagnostic["counts"]["configured"]
                    else "No free-tier provider route has an API key."
                ),
            },
            {
                "id": "current_evidence",
                "status": "pass" if diagnostic["counts"]["evaluated"] else "fail",
                "message": (
                    f"{diagnostic['counts']['evaluated']} route(s) have current evidence."
                    if diagnostic["counts"]["evaluated"]
                    else "No configured route has Sentinel evidence."
                ),
            },
            {
                "id": "qualified_route",
                "status": "pass" if diagnostic["counts"]["qualified"] else "fail",
                "message": diagnostic["message"],
            },
        ]
        return {
            "ok": diagnostic["status"] == "ready",
            "profile": diagnostic,
            "checks": checks,
            "next_action": diagnostic["remediation"],
        }

    def _route(self, route_id: str) -> ModelRoute:
        for route in self.catalog.all_routes():
            if route.route_id == route_id:
                return route
        raise KeyError(f"Unknown route_id: {route_id}")

    def _route_payload(
        self,
        route: ModelRoute,
        evaluation,
        configured_providers: set[str],
    ) -> dict[str, Any]:
        return {
            "route_id": route.route_id,
            "provider_name": route.provider_name,
            "model_id": route.model_id,
            "display_name": route.display_name,
            "rank": route.rank,
            "enabled": route.enabled,
            "configured": route.provider_name in configured_providers,
            "cost": route.cost,
            "zero_cost": route_is_zero_cost(route),
            "speed": route.speed,
            "quality": route.quality,
            "evaluation": evaluation.to_dict() if evaluation else None,
        }


def opencode_setup(base_url: str) -> dict[str, Any]:
    root = base_url.rstrip("/")
    if not root.endswith("/v1"):
        root = f"{root}/v1"
    config = {
        "$schema": "https://opencode.ai/config.json",
        "model": "freerouter/safe-coding",
        "provider": {
            "freerouter": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "FreeRouter Sentinel",
                "options": {
                    "baseURL": root,
                    "apiKey": "sk-local",
                },
                "models": {
                    profile_id: {"name": f"Sentinel · {profile.name}"}
                    for profile_id, profile in AGENT_PROFILES.items()
                },
            }
        },
    }
    return {
        "config": config,
        "config_json": json.dumps(config, indent=2),
        "steps": [
            "Run the Sentinel doctor and evaluate routes until your profile is ready.",
            "Copy this JSON into opencode.json in your project.",
            "Start OpenCode and select any evidence-backed Sentinel profile.",
        ],
        "doctor_url": f"{root}/gateway/sentinel/doctor?profile=safe-coding",
    }
