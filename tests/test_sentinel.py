from __future__ import annotations

import json
import time

import httpx
import pytest

from app.agent_profiles import (
    AGENT_PROFILES,
    filter_routes_for_profile,
    route_qualifies,
    validate_profile_request,
)
from app.live_monitor import APILiveMonitor
from app.model_catalog import ModelCatalog, ModelRoute
from app.providers.base import ProviderAdapter
from app.router import NoQualifiedRoute, unsupported_capabilities_error_body
from app.sentinel import SentinelEvaluator
from app.sentinel_service import SentinelService, opencode_setup
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
                    'data: {"choices":[{"delta":{"content":"stream-ready"}}]}\n\ndata: [DONE]\n\n'
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
    assert set(provider["models"]) == set(AGENT_PROFILES)


def test_safe_study_blocks_tools_but_safe_security_allows_proposals():
    tool_payload = {
        "messages": [{"role": "user", "content": "plan"}],
        "tools": [{"type": "function", "function": {"name": "inspect", "parameters": {}}}],
    }
    with pytest.raises(ValueError, match="safe-study does not permit tool calls"):
        validate_profile_request({**tool_payload, "model": "safe-study"})

    validate_profile_request({**tool_payload, "model": "safe-security"})


@pytest.mark.asyncio
async def test_consumer_preflight_reports_degraded_single_route_and_policy_block(tmp_path):
    route = _route()
    catalog = ModelCatalog(str(tmp_path / "models.json"))
    catalog.initialize()
    catalog.replace_routes(
        [
            {
                "route_id": route.route_id,
                "provider_name": route.provider_name,
                "model_id": route.model_id,
                "display_name": route.display_name,
                "rank": route.rank,
                "enabled": True,
                "cost": route.cost,
                "speed": route.speed,
                "tags": route.tags,
            }
        ]
    )
    store = SentinelStore(str(tmp_path / "gateway.sqlite3"))
    await store.initialize()
    await store.save(_evaluation(route))
    provider = ProviderAdapter("mock", "test-key", "https://mock.test/v1", route.model_id)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200))
    ) as client:
        service = SentinelService([provider], catalog, client, store)
        study = await service.preflight("safe-study", tools=False)
        blocked = await service.preflight("safe-study", tools=True)
        security = await service.preflight("safe-security", tools=True)

    assert study["status"] == "degraded"
    assert study["ok"] is True
    assert study["fallback"]["model"] == "auto"
    assert blocked["status"] == "blocked"
    assert "deliberately blocks" in blocked["reason"]
    assert security["status"] == "degraded"
    assert (
        next(check for check in security["checks"] if check["id"] == "tool_call")["status"]
        == "pass"
    )


@pytest.mark.asyncio
async def test_receipt_monitor_persists_metadata_without_content_or_secrets(tmp_path):
    store = SentinelStore(str(tmp_path / "gateway.sqlite3"))
    await store.initialize()
    monitor = APILiveMonitor(receipt_sink=store.save_receipt)
    await monitor.publish(
        event_type="request_started",
        request_id="run-safe-study",
        payload={
            "path": "/v1/chat/completions",
            "stream": False,
            "model": "safe-study",
            "required_capabilities": ["json-schema"],
            "request_payload": {"messages": [{"content": "private study notes"}]},
        },
    )
    await monitor.publish(
        event_type="request_completed",
        request_id="run-safe-study",
        payload={
            "provider_name": "mock",
            "route_id": "mock-coder",
            "model_id": "coder-v1",
            "latency_ms": 41,
            "attempts": 1,
            "attempts_detail": [{"status": "selected", "route_id": "mock-coder"}],
            "response_body": {"secret": "never-persist-this"},
        },
    )

    receipts = await store.recent_receipts()
    assert receipts[0]["run_id"] == "run-safe-study"
    assert receipts[0]["profile_id"] == "safe-study"
    assert receipts[0]["policy_verdict"] == "allowed"
    assert receipts[0]["capabilities"] == ["json-schema"]
    serialized = json.dumps(receipts)
    assert "private study notes" not in serialized
    assert "never-persist-this" not in serialized
