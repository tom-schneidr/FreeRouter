from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GatewayRouteInfo:
    provider_name: str
    route_id: str
    model_id: str


def gateway_route_headers(info: GatewayRouteInfo) -> dict[str, str]:
    return {
        "X-Gateway-Provider": info.provider_name,
        "X-Gateway-Route": info.route_id,
        "X-Gateway-Model": info.model_id,
    }


class GatewayRoutingContext:
    """Mutable routing selection shared between a stream handler and the HTTP response."""

    def __init__(self) -> None:
        self._info: GatewayRouteInfo | None = None

    @property
    def ready(self) -> bool:
        return self._info is not None

    @property
    def info(self) -> GatewayRouteInfo | None:
        return self._info

    def set(self, provider_name: str, route_id: str, model_id: str) -> None:
        self._info = GatewayRouteInfo(
            provider_name=provider_name,
            route_id=route_id,
            model_id=model_id,
        )
