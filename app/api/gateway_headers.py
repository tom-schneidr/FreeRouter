from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GatewayRouteInfo:
    provider_name: str
    route_id: str
    model_id: str
    request_class: str = "normal"
    required_capabilities: frozenset[str] = frozenset()


def gateway_route_headers(info: GatewayRouteInfo) -> dict[str, str]:
    headers = {
        "X-Gateway-Provider": info.provider_name,
        "X-Gateway-Route": info.route_id,
        "X-Gateway-Model": info.model_id,
        "X-Gateway-Request-Class": info.request_class,
    }
    display_caps = sorted(cap for cap in info.required_capabilities if cap != "text")
    headers["X-Gateway-Required-Capabilities"] = ", ".join(display_caps)
    return headers


def gateway_request_headers(
    *,
    request_class: str,
    required_capabilities: frozenset[str],
) -> dict[str, str]:
    display_caps = sorted(cap for cap in required_capabilities if cap != "text")
    return {
        "X-Gateway-Request-Class": request_class,
        "X-Gateway-Required-Capabilities": ", ".join(display_caps),
    }


class GatewayRoutingContext:
    """Mutable routing selection shared between a stream handler and the HTTP response."""

    def __init__(self) -> None:
        self._info: GatewayRouteInfo | None = None
        self.request_class = "normal"
        self.required_capabilities: frozenset[str] = frozenset()

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
    ) -> None:
        self.request_class = request_class
        self.required_capabilities = required_capabilities

    def request_headers(self) -> dict[str, str]:
        return gateway_request_headers(
            request_class=self.request_class,
            required_capabilities=self.required_capabilities,
        )

    def set(self, provider_name: str, route_id: str, model_id: str) -> None:
        self._info = GatewayRouteInfo(
            provider_name=provider_name,
            route_id=route_id,
            model_id=model_id,
            request_class=self.request_class,
            required_capabilities=self.required_capabilities,
        )
