from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.agent_profiles import AGENT_PROFILES, is_agent_profile
from app.consumer_contracts import CONTRACT_VERSION, fallback_details, policy_verdict


@dataclass(frozen=True)
class GatewayRouteInfo:
    provider_name: str
    route_id: str
    model_id: str
    request_class: str = "normal"
    required_capabilities: frozenset[str] = frozenset()
    run_id: str = ""
    requested_model: str | None = None
    latency_ms: int | None = None
    attempts: int | None = None
    fallback_used: bool = False
    fallback_reason: str = ""


def gateway_contract_headers(
    *,
    run_id: str,
    requested_model: Any,
    policy_allowed: bool | None = True,
    required_capabilities: frozenset[str] = frozenset(),
    latency_ms: int | None = None,
    attempts: int | None = None,
    fallback_used: bool | None = None,
    fallback_reason: str = "",
) -> dict[str, str]:
    profile_id = requested_model if is_agent_profile(requested_model) else ""
    headers = {
        "X-FreeRouter-Contract-Version": CONTRACT_VERSION,
        "X-FreeRouter-Run-Id": run_id,
        "X-FreeRouter-Profile": profile_id,
        "X-FreeRouter-Policy-Verdict": (
            "pending"
            if policy_allowed is None and profile_id
            else policy_verdict(requested_model, allowed=policy_allowed is not False)
        ),
        "X-FreeRouter-Capabilities": ", ".join(
            sorted(cap for cap in required_capabilities if cap != "text")
        ),
    }
    if profile_id:
        headers["X-FreeRouter-Tool-Policy"] = AGENT_PROFILES[profile_id].tool_policy
    if latency_ms is not None:
        headers["X-FreeRouter-Latency-Ms"] = str(max(0, latency_ms))
    if attempts is not None:
        headers["X-FreeRouter-Attempts"] = str(max(0, attempts))
    if fallback_used is not None:
        headers["X-FreeRouter-Fallback"] = "fallback" if fallback_used else "direct"
        headers["X-FreeRouter-Fallback-Reason"] = fallback_reason
    return headers


def gateway_route_headers(info: GatewayRouteInfo) -> dict[str, str]:
    headers = {
        "X-Gateway-Provider": info.provider_name,
        "X-Gateway-Route": info.route_id,
        "X-Gateway-Model": info.model_id,
        "X-Gateway-Request-Class": info.request_class,
        "X-FreeRouter-Provider": info.provider_name,
        "X-FreeRouter-Route": info.route_id,
        "X-FreeRouter-Model": info.model_id,
    }
    display_caps = sorted(cap for cap in info.required_capabilities if cap != "text")
    headers["X-Gateway-Required-Capabilities"] = ", ".join(display_caps)
    headers.update(
        gateway_contract_headers(
            run_id=info.run_id,
            requested_model=info.requested_model,
            required_capabilities=info.required_capabilities,
            latency_ms=info.latency_ms,
            attempts=info.attempts,
            fallback_used=info.fallback_used,
            fallback_reason=info.fallback_reason,
        )
    )
    return headers


def gateway_request_headers(
    *,
    request_class: str,
    required_capabilities: frozenset[str],
    run_id: str = "",
    requested_model: Any = None,
) -> dict[str, str]:
    display_caps = sorted(cap for cap in required_capabilities if cap != "text")
    headers = {
        "X-Gateway-Request-Class": request_class,
        "X-Gateway-Required-Capabilities": ", ".join(display_caps),
    }
    headers.update(
        gateway_contract_headers(
            run_id=run_id,
            requested_model=requested_model,
            policy_allowed=None,
            required_capabilities=required_capabilities,
        )
    )
    return headers


def route_fallback_headers(attempts: list[Any]) -> tuple[bool, str]:
    return fallback_details(attempts)


class GatewayRoutingContext:
    """Mutable routing selection shared between a stream handler and the HTTP response."""

    def __init__(self) -> None:
        self._info: GatewayRouteInfo | None = None
        self.request_class = "normal"
        self.required_capabilities: frozenset[str] = frozenset()
        self.run_id = ""
        self.requested_model: Any = None

    @property
    def ready(self) -> bool:
        return self._info is not None

    @property
    def info(self) -> GatewayRouteInfo | None:
        return self._info

    def configure_request(
        self,
        *,
        request_class: str,
        required_capabilities: frozenset[str],
        run_id: str = "",
        requested_model: Any = None,
    ) -> None:
        self.request_class = request_class
        self.required_capabilities = required_capabilities
        self.run_id = run_id
        self.requested_model = requested_model

    def request_headers(self) -> dict[str, str]:
        return gateway_request_headers(
            request_class=self.request_class,
            required_capabilities=self.required_capabilities,
            run_id=self.run_id,
            requested_model=self.requested_model,
        )

    def set(self, provider_name: str, route_id: str, model_id: str) -> None:
        self._info = GatewayRouteInfo(
            provider_name=provider_name,
            route_id=route_id,
            model_id=model_id,
            request_class=self.request_class,
            required_capabilities=self.required_capabilities,
            run_id=self.run_id,
            requested_model=self.requested_model,
        )
