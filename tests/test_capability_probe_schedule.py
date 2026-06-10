from __future__ import annotations

from app.capability_probe_schedule import (
    PROBE_RECHECK_SECONDS,
    capability_probe_priority,
    select_routes_for_capability_probe,
)
from app.capability_tags import CapabilityClaim, tags_to_capabilities
from app.model_catalog import ModelRoute, route_id_for


def _route(rank: int, *, probed_at: int | None = None, route_id: str = "r") -> ModelRoute:
    capabilities = tags_to_capabilities(["text", "tool-use"], source="manual")
    if probed_at is not None:
        capabilities["text"] = CapabilityClaim(
            tag="text",
            status="supported",
            source="probe",
            confidence="high",
            checked_at=probed_at,
        )
        capabilities["tool-use"] = CapabilityClaim(
            tag="tool-use",
            status="supported",
            source="probe",
            confidence="high",
            checked_at=probed_at,
        )
    return ModelRoute(
        route_id=route_id_for("groq", route_id),
        provider_name="groq",
        model_id=route_id,
        display_name=route_id,
        rank=rank,
        enabled=True,
        tags=["text", "tool-use"],
        capabilities=capabilities,
    )


def test_never_probed_routes_outrank_fresh_top_routes():
    now = 1_700_000_000
    routes = [
        _route(1, probed_at=now - 60, route_id="fresh-top"),
        _route(50, probed_at=None, route_id="stale-low"),
    ]
    selected = select_routes_for_capability_probe(
        routes, provider_name="groq", budget=1, now=now
    )
    assert [route.model_id for route in selected] == ["stale-low"]


def test_higher_rank_wins_when_staleness_is_equal():
    now = 1_700_000_000
    probed_at = now - PROBE_RECHECK_SECONDS
    routes = [
        _route(20, probed_at=probed_at, route_id="low"),
        _route(2, probed_at=probed_at, route_id="high"),
    ]
    selected = select_routes_for_capability_probe(
        routes, provider_name="groq", budget=1, now=now
    )
    assert selected[0].model_id == "high"


def test_catalog_rotates_to_unprobed_routes_after_top_ranks_are_fresh():
    now = 1_700_000_000
    routes = [_route(rank, probed_at=now - 1, route_id=f"m{rank}") for rank in range(1, 7)]
    first = select_routes_for_capability_probe(
        routes, provider_name="groq", budget=2, now=now
    )
    assert {route.rank for route in first} == {1, 2}

    routes = [
        _route(
            rank,
            probed_at=now if rank <= 2 else None,
            route_id=f"m{rank}",
        )
        for rank in range(1, 7)
    ]
    second = select_routes_for_capability_probe(
        routes, provider_name="groq", budget=2, now=now
    )
    assert {route.rank for route in second} == {3, 4}


def test_priority_is_monotonic_in_rank_for_fresh_routes():
    now = 1_700_000_000
    fresh = _route(1, probed_at=now, route_id="a")
    other = _route(5, probed_at=now, route_id="b")
    assert capability_probe_priority(fresh, now=now, max_rank=5) > capability_probe_priority(
        other, now=now, max_rank=5
    )
