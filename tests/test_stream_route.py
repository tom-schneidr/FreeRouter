from __future__ import annotations

import json
from typing import Any

import httpx

from app.model_catalog import ModelCatalog
from app.providers.base import ProviderResponse
from app.router import RouteEvent, WaterfallRouter
from app.state import ProviderQuota, StateManager
from app.stream_route import stream_route_chat


class FakeProvider:
    def __init__(
        self,
        name: str,
        *,
        response: dict[str, Any] | None = None,
        error: Exception | None = None,
        configured: bool = True,
    ) -> None:
        self.name = name
        self.api_key = "test-key" if configured else None
        self.max_context_tokens = 8192
        self.response = response or {
            "id": f"chatcmpl-{name}",
            "object": "chat.completion",
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {"total_tokens": 3},
        }
        self.error = error
        self._configured = configured

    @property
    def is_configured(self) -> bool:
        return self._configured

    async def chat_completion(
        self,
        client: httpx.AsyncClient,
        payload: dict[str, Any],
        target_model: str | None = None,
    ) -> ProviderResponse:
        if self.error:
            raise self.error
        return ProviderResponse(self.name, 200, {}, dict(self.response))


def _payload() -> dict[str, Any]:
    return {"model": "auto", "messages": [{"role": "user", "content": "hello"}]}


async def _state(tmp_path) -> StateManager:
    state = StateManager(
        str(tmp_path / "state.sqlite3"),
        [
            ProviderQuota(
                "primary", tokens_per_day=None, requests_per_day=None, requests_per_minute=30
            ),
            ProviderQuota(
                "fallback", tokens_per_day=None, requests_per_day=None, requests_per_minute=30
            ),
        ],
    )
    await state.initialize()
    return state


def _catalog(tmp_path) -> ModelCatalog:
    tmp_path.mkdir(parents=True, exist_ok=True)
    catalog = ModelCatalog(str(tmp_path / "models.json"))
    catalog.replace_routes(
        [
            {
                "route_id": "primary-test",
                "provider_name": "primary",
                "model_id": "primary/model",
                "display_name": "Primary Model",
                "rank": 1,
                "enabled": True,
                "context_window": 8192,
            },
            {
                "route_id": "fallback-test",
                "provider_name": "fallback",
                "model_id": "fallback/model",
                "display_name": "Fallback Model",
                "rank": 2,
                "enabled": True,
                "context_window": 8192,
            },
        ]
    )
    return catalog


def _sse_payloads(chunks: list[str]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for chunk in chunks:
        line = chunk.strip()
        if not line.startswith("data: "):
            continue
        payloads.append(json.loads(line.removeprefix("data: ")))
    return payloads


def _event_shape(event: RouteEvent) -> tuple[str, str | None, str | None]:
    return (event.event_type, event.provider_name, event.reason)


def _sse_shape(event: dict[str, Any]) -> tuple[str, str | None, str | None]:
    event_type = event.get("type")
    if event_type == "route_skip":
        return ("route_skipped", event.get("provider"), event.get("reason"))
    if event_type == "route_trying":
        return ("route_trying", event.get("provider"), None)
    if event_type == "route_fail":
        return ("route_failed", event.get("provider"), event.get("reason"))
    if event_type == "route_flagged":
        return ("route_flagged", event.get("provider"), event.get("reason"))
    if event_type == "route_selected":
        return ("route_selected", event.get("provider"), None)
    return (str(event_type), event.get("provider"), event.get("reason"))


async def test_sse_route_uses_canonical_router_event_sequence(tmp_path):
    route_state = await _state(tmp_path / "route")
    route_router = WaterfallRouter(
        [
            FakeProvider("primary", error=httpx.ReadTimeout("timed out")),
            FakeProvider(
                "fallback",
                response={
                    "id": "chatcmpl-fallback",
                    "object": "chat.completion",
                    "choices": [{"message": {"role": "assistant", "content": "fallback wins"}}],
                    "usage": {"total_tokens": 4},
                },
            ),
        ],
        _catalog(tmp_path / "route"),
        route_state,
        request_timeout_seconds=5,
    )
    route_events = [event async for event in route_router.iter_route_events(_payload())]

    sse_state = await _state(tmp_path / "sse")
    sse_router = WaterfallRouter(
        [
            FakeProvider("primary", error=httpx.ReadTimeout("timed out")),
            FakeProvider(
                "fallback",
                response={
                    "id": "chatcmpl-fallback",
                    "object": "chat.completion",
                    "choices": [{"message": {"role": "assistant", "content": "fallback wins"}}],
                    "usage": {"total_tokens": 4},
                },
            ),
        ],
        _catalog(tmp_path / "sse"),
        sse_state,
        request_timeout_seconds=5,
    )
    sse_chunks = [chunk async for chunk in stream_route_chat(_payload(), sse_router)]
    sse_events = _sse_payloads(sse_chunks)

    routed_core = [
        _event_shape(event)
        for event in route_events
        if event.event_type
        in {"route_trying", "route_failed", "route_flagged", "route_skipped", "route_selected"}
    ]
    sse_core = [
        _sse_shape(event) for event in sse_events if str(event.get("type", "")).startswith("route_")
    ]

    assert sse_core == routed_core
    assert any(event.get("type") == "done" for event in sse_events)


async def test_stream_route_sse_does_not_call_sleep_between_chunks(monkeypatch, tmp_path):
    """Chunk replay uses zero delay by default (no artificial throughput cap)."""
    import app.stream_route as stream_mod

    async def fail_sleep(_seconds: float) -> None:
        raise AssertionError("unexpected asyncio.sleep during SSE replay")

    monkeypatch.setattr(stream_mod.asyncio, "sleep", fail_sleep)

    route_state = await _state(tmp_path / "sleep")
    router = WaterfallRouter(
        [FakeProvider("primary")],
        _catalog(tmp_path / "sleep"),
        route_state,
        request_timeout_seconds=5,
    )
    chunks = [
        c async for c in stream_route_chat(_payload(), router, chunk_replay_sleep_seconds=0.0)
    ]
    assert chunks
