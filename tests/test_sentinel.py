from __future__ import annotations

import json
import time

import httpx
import pytest

from app.agent_profiles import AGENT_PROFILES, filter_routes_for_profile, route_qualifies
from app.model_catalog import ModelRoute
from app.providers.base import ProviderAdapter
from app.router import NoQualifiedRoute, unsupported_capabilities_error_body
from app.sentinel import SentinelEvaluator
from app.sentinel_service import opencode_setup
from app.sentinel_store import SentinelStore
from app.sentinel_types import SentinelEvaluation, SentinelProbeResult


def _route(*, cost: str = "free-tier") -> ModelRoute:
    return ModelRoute(
        route_id="mock-coder",
        provider_name="mock",
        model_id="coder-v1",
        display_name="Mock Coder",
        rank=1,
        cost=cost,
        speed="fast",
        tags=["text", "tool-use"],
    )


def _probe(check_id: str, *, status: str = "pass", score: int = 25):
    return SentinelProbeResult(
        check_id=check_id,
        label=check_id,
        status=status,  # type: ignore[arg-type]
        score=score,
        latency_ms=5,
        evidence="deterministic test evidence",
    )


def _evaluation(route: ModelRoute, *, completed_at: int | None = None):
    probes = [
        _probe("tool_call"),
        _probe("structured_json"),
        _probe("streaming"),
        _probe("canary"),
    ]
    return SentinelEvaluation(
        run_id="run-ready",
        route_id=route.route_id,
        provider_name=route.provider_name,
        model_id=route.model_id,
        score=100,
        readiness="ready",
        started_at=completed_at or int(time.time()),
        completed_at=completed_at or int(time.time()),
        duration_ms=20,
        summary="Ready",
        probes=probes,
    )


@pytest.mark.asyncio
async def test_sentinel_store_round_trips_latest_evidence(tmp_path):
    store = SentinelStore(str(tmp_path / "gateway.sqlite3"))
    await store.initialize()
    route = _route()
    evaluation = _evaluation(route)

    await store.save(evaluation)

    latest = await store.latest_for_routes([route.route_id])
    assert latest[route.route_id] == evaluation
    assert await store.history(route.route_id) == [evaluation]


@pytest.mark.asyncio
async def test_evaluator_passes_all_deterministic_agent_checks(tmp_path):
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        if payload.get("stream"):
            return httpx.Response(
                200,
                text=(
                    'data: {"choices":[{"delta":{"content":"stream-ready"}}]}\n\n'
                    "data: [DONE]\n\n"
                ),
                headers={"content-type": "text/event-stream"},
            )
        if payload.get("tools"):
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call-1",
                                        "type": "function",
                                        "function": {
                                            "name": "sentinel_echo",
                                            "arguments": '{"token":"sentinel-echo-7"}',
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                },
            )
        if payload.get("response_format"):
            content = '{"status":"ready","value":7}'
        else:
            content = "SAFE"
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    store = SentinelStore(str(tmp_path / "gateway.sqlite3"))
    await store.initialize()
    provider = ProviderAdapter("mock", "test-key", "https://mock.test/v1", "coder-v1")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await SentinelEvaluator(store).evaluate(provider, client, _route())

    assert result.readiness == "ready"
    assert result.score == 100
    assert [probe.status for probe in result.probes] == ["pass"] * 4
    persisted = await store.latest_for_routes([result.route_id])
    assert persisted[result.route_id].score == 100


@pytest.mark.asyncio
async def test_canary_exfiltration_blocks_route(tmp_path):
    route = _route()
    canary = "SENTINEL_CANARY_MOCK-CODER"

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        if payload.get("stream"):
            return httpx.Response(
                200,
                text='data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n',
            )
        if payload.get("tools"):
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "tool_calls": [
                                    {
                                        "function": {
                                            "name": "sentinel_echo",
                                            "arguments": '{"token":"sentinel-echo-7"}',
                                        }
                                    }
                                ]
                            }
                        }
                    ]
                },
            )
        content = (
            '{"status":"ready","value":7}'
            if payload.get("response_format")
            else f"Leaked: {canary}"
        )
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    store = SentinelStore(str(tmp_path / "gateway.sqlite3"))
    await store.initialize()
    provider = ProviderAdapter("mock", "test-key", "https://mock.test/v1", "coder-v1")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await SentinelEvaluator(store).evaluate(provider, client, route)

    canary_result = next(probe for probe in result.probes if probe.check_id == "canary")
    assert canary_result.status == "fail"
    assert result.readiness == "blocked"
    assert canary not in canary_result.evidence


def test_safe_coding_requires_fresh_complete_evidence_and_zero_cost():
    route = _route()
    profile = AGENT_PROFILES["safe-coding"]
    evaluation = _evaluation(route)

    assert route_qualifies(route, evaluation, profile)[0] is True
    assert filter_routes_for_profile([route], {route.route_id: evaluation}, profile.profile_id) == [
        route
    ]

    stale = _evaluation(route, completed_at=int(time.time()) - 8 * 24 * 60 * 60)
    assert route_qualifies(route, stale, profile)[0] is False
    paid = _route(cost="paid")
    qualifies, reason = route_qualifies(paid, evaluation, profile)
    assert qualifies is False
    assert "$0 guard" in reason


def test_no_qualifying_route_error_is_actionable_and_fail_closed():
    error = NoQualifiedRoute("safe-coding", frozenset({"tool_call", "canary"}))

    body = unsupported_capabilities_error_body(error)["error"]

    assert body["code"] == "no_qualifying_route"
    assert body["profile"] == "safe-coding"
    assert body["zero_cost_guard"] is True
    assert "run all four readiness checks" in body["remediation"]


def test_opencode_setup_uses_generic_compatible_provider_and_profile():
    setup = opencode_setup("http://127.0.0.1:8000")
    provider = setup["config"]["provider"]["freerouter"]

    assert setup["config"]["model"] == "freerouter/safe-coding"
    assert provider["npm"] == "@ai-sdk/openai-compatible"
    assert provider["options"]["baseURL"] == "http://127.0.0.1:8000/v1"
    assert set(provider["models"]) == {"safe-coding", "fast-coding"}
