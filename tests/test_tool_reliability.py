from __future__ import annotations

from dataclasses import replace

from app.capability_tags import tags_to_capabilities
from app.model_catalog import ModelRoute
from app.state import StateManager
from app.state_types import ProviderQuota
from app.tool_reliability import (
    ToolReliabilitySnapshot,
    route_tool_reliability_score,
    tool_route_sort_key,
)


def _route(route_id: str, *, rank: int, confirmed: bool = True) -> ModelRoute:
    tags = ["text", "tool-use"] if confirmed else ["text"]
    source = "runtime" if confirmed else "registry"
    capabilities = tags_to_capabilities(tags, source=source)
    if not confirmed:
        capabilities["tool-use"] = tags_to_capabilities(
            ["tool-use"], source="registry", confidence="medium"
        )["tool-use"]
    return ModelRoute(
        route_id=route_id,
        provider_name="test",
        model_id=route_id,
        display_name=route_id,
        rank=rank,
        tags=tags,
        capabilities=capabilities,
    )


def test_tool_reliability_prefers_stable_route_over_higher_generic_rank():
    high_rank = _route("high-rank", rank=1)
    reliable = _route("reliable", rank=10)
    noisy = ToolReliabilitySnapshot("high-rank", successes=17, failures=3)
    stable = ToolReliabilitySnapshot("reliable", successes=99, failures=1)

    assert tool_route_sort_key(reliable, stable) > tool_route_sort_key(high_rank, noisy)
    assert route_tool_reliability_score(reliable, stable) > route_tool_reliability_score(
        high_rank, noisy
    )


def test_strong_empirical_reliability_can_beat_confirmation_prior():
    confirmed = _route("confirmed", rank=20, confirmed=True)
    hinted = _route("hinted", rank=1, confirmed=False)
    bad_history = ToolReliabilitySnapshot("confirmed", successes=1, failures=12)
    good_history = ToolReliabilitySnapshot("hinted", successes=50, failures=0)

    assert tool_route_sort_key(hinted, good_history) > tool_route_sort_key(confirmed, bad_history)


def test_openclaw_probe_behavior_seeds_cold_start_ranking():
    strong = _route("strong", rank=10)
    weak = _route("weak", rank=1)
    strong.capabilities["tool-use"] = replace(
        strong.capabilities["tool-use"],
        evidence=(
            "OpenClaw tool profile: required_exact_call=supported; "
            "auto_selection=supported; tool_result_continuation=supported"
        ),
    )
    weak.capabilities["tool-use"] = replace(
        weak.capabilities["tool-use"],
        evidence=(
            "OpenClaw tool profile: required_exact_call=supported; "
            "auto_selection=unsupported; tool_result_continuation=inconclusive"
        ),
    )

    assert tool_route_sort_key(strong, None) > tool_route_sort_key(weak, None)


async def test_state_persists_typed_time_weighted_tool_outcomes(tmp_path):
    state = StateManager(
        str(tmp_path / "state.sqlite3"),
        [ProviderQuota("test", None, None, None)],
    )
    now = 1_800_000_000
    state._now = lambda: now
    await state.initialize()

    await state.record_route_tool_outcome("route", "test", "model", "valid_call")
    await state.record_route_tool_outcome("route", "test", "model", "schema_invalid")

    snapshots = await state.get_route_tool_reliability(["route", "missing"])

    assert snapshots["route"].successes == 8
    assert snapshots["route"].failures == 8
    assert "missing" not in snapshots


async def test_state_prefers_matching_tool_schema_history(tmp_path):
    state = StateManager(
        str(tmp_path / "state.sqlite3"),
        [ProviderQuota("test", None, None, None)],
    )
    await state.initialize()
    await state.record_route_tool_outcome(
        "route-a",
        "test",
        "model",
        "valid_call",
        request_fingerprint="schema-a",
    )
    await state.record_route_tool_outcome(
        "route-a",
        "test",
        "model",
        "schema_invalid",
        request_fingerprint="schema-b",
    )

    matching = await state.get_route_tool_reliability(
        ["route-a"],
        request_fingerprint="schema-a",
    )

    assert matching["route-a"].successes == 8
    assert matching["route-a"].failures == 0
