"""API tests for POST /v1/messages."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.app_services import AppServices, attach_app_services
from app.endpoint_diagnosis import EndpointDiagnosisService
from app.factory import build_core_gateway_stack
from app.live_monitor import APILiveMonitor
from app.main import app
from app.model_catalog import ModelCatalog
from app.providers.base import ProviderResponse
from app.request_limiter import GatewayRequestLimiter
from app.router import WaterfallRouter
from app.settings import get_settings
from app.state import ProviderQuota, StateManager


async def _services_with_provider(
    tmp_path,
    provider: Any,
    routes: list[dict[str, Any]],
) -> AppServices:
    state = StateManager(
        str(tmp_path / "state.sqlite3"),
        [
            ProviderQuota(
                provider.name,
                tokens_per_day=None,
                requests_per_day=None,
                requests_per_minute=30,
            )
        ],
    )
    catalog = ModelCatalog(str(tmp_path / "models.json"))
    stack = await build_core_gateway_stack(state=state, model_catalog=catalog)
    catalog.replace_routes(routes)
    router = WaterfallRouter(
        [provider],
        catalog,
        state,
        request_timeout_seconds=stack.settings.request_timeout_seconds,
        http_client=stack.http_client,
    )
    settings = stack.settings
    return AppServices(
        gateway_state=state,
        model_catalog=catalog,
        request_limiter=GatewayRequestLimiter(
            settings.max_concurrent_requests,
            settings.request_queue_timeout_seconds,
            settings.request_queue_max_waiting_requests,
        ),
        live_monitor=APILiveMonitor(max_events=1000),
        http_client=stack.http_client,
        waterfall_router=router,
        endpoint_diagnosis=EndpointDiagnosisService(
            [provider],
            catalog,
            state,
            request_timeout_seconds=settings.request_timeout_seconds,
        ),
        background_endpoint_diagnosis=None,
    )


def _client(tmp_path, monkeypatch) -> TestClient:
    get_settings.cache_clear()
    settings = {
        "DATABASE_PATH": str(tmp_path / "gateway.sqlite3"),
        "MODEL_CATALOG_PATH": str(tmp_path / "model_catalog.json"),
        "AUTO_ENDPOINT_DIAGNOSIS_ENABLED": "false",
        "CEREBRAS_API_KEY": "",
        "GROQ_API_KEY": "",
        "GEMINI_API_KEY": "",
        "NVIDIA_API_KEY": "",
        "OPENROUTER_API_KEY": "",
        "SAMBANOVA_API_KEY": "",
    }
    for key, value in settings.items():
        monkeypatch.setenv(key, value)
    return TestClient(app)


def test_messages_rejects_missing_max_tokens(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        response = client.post(
            "/v1/messages",
            json={"model": "auto", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 400
    payload = response.json()
    assert payload["type"] == "error"
    assert payload["error"]["type"] == "invalid_request_error"


def test_messages_rejects_unsupported_field(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        response = client.post(
            "/v1/messages",
            json={
                "model": "auto",
                "max_tokens": 32,
                "thinking": {"type": "enabled", "budget_tokens": 1000},
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    assert response.status_code == 400
    assert "thinking" in response.json()["error"]["message"]


def test_messages_non_stream_returns_waterfall_exhausted_anthropic_error(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        response = client.post(
            "/v1/messages",
            json={
                "model": "auto",
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
    assert response.status_code == 503
    payload = response.json()
    assert payload["type"] == "error"
    assert payload["error"]["type"] == "api_error"


def test_messages_stream_returns_sse_error_when_exhausted(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        response = client.post(
            "/v1/messages",
            json={
                "model": "auto",
                "max_tokens": 32,
                "stream": True,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
    assert response.status_code == 200
    assert "event: error" in response.text
    assert "api_error" in response.text


@pytest.mark.asyncio
async def test_messages_non_stream_image_without_vision_route_returns_400(tmp_path, monkeypatch):
    class TextOnlyProvider:
        name = "primary"
        api_key = "test-key"
        max_context_tokens = 8192

        @property
        def is_configured(self) -> bool:
            return True

        async def chat_completion(self, client, payload, target_model=None):
            return ProviderResponse("primary", 200, {}, {"choices": [], "usage": {}})

        async def chat_completion_stream(self, client, payload, target_model=None):
            yield ""

    get_settings.cache_clear()
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "gateway.sqlite3"))
    monkeypatch.setenv("MODEL_CATALOG_PATH", str(tmp_path / "model_catalog.json"))
    services = await _services_with_provider(
        tmp_path,
        TextOnlyProvider(),
        [
            {
                "route_id": "text-only",
                "provider_name": "primary",
                "model_id": "primary/model",
                "display_name": "Text Only",
                "rank": 1,
                "enabled": True,
                "tags": ["text"],
            }
        ],
    )
    probe = TestClient(app)
    attach_app_services(probe.app, services)
    response = probe.post(
        "/v1/messages",
        json={
            "model": "auto",
            "max_tokens": 64,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "abc",
                            },
                        },
                    ],
                }
            ],
        },
    )
    assert response.status_code == 400
    payload = response.json()
    assert payload["error"]["type"] == "invalid_request_error"
    assert "vision" in payload["error"]["message"]


@pytest.mark.asyncio
async def test_messages_non_stream_success_shape(tmp_path, monkeypatch):
    class FakeProvider:
        name = "primary"
        api_key = "test-key"
        max_context_tokens = 8192

        @property
        def is_configured(self) -> bool:
            return True

        async def chat_completion(self, client, payload, target_model=None):
            return ProviderResponse(
                "primary",
                200,
                {},
                {
                    "id": "chatcmpl-test",
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "Hello from gateway"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 4, "completion_tokens": 6},
                },
            )

        async def chat_completion_stream(self, client, payload, target_model=None):
            yield ""

    get_settings.cache_clear()
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "gateway.sqlite3"))
    monkeypatch.setenv("MODEL_CATALOG_PATH", str(tmp_path / "model_catalog.json"))
    services = await _services_with_provider(
        tmp_path,
        FakeProvider(),
        [
            {
                "route_id": "primary-test",
                "provider_name": "primary",
                "model_id": "primary/model",
                "display_name": "Primary",
                "rank": 1,
                "enabled": True,
                "tags": ["text"],
            }
        ],
    )

    probe = TestClient(app)
    attach_app_services(probe.app, services)
    response = probe.post(
        "/v1/messages",
        json={
            "model": "auto",
            "max_tokens": 32,
            "system": "Be brief.",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["type"] == "message"
    assert payload["role"] == "assistant"
    assert payload["content"][0]["text"] == "Hello from gateway"
    assert payload["stop_reason"] == "end_turn"
    assert payload["usage"]["input_tokens"] == 4


class FragmentedToolStreamProvider:
    name = "primary"
    api_key = "test-key"
    max_context_tokens = 8192

    @property
    def is_configured(self) -> bool:
        return True

    async def chat_completion(self, client, payload, target_model=None):
        return ProviderResponse("primary", 200, {}, {"choices": [], "usage": {}})

    async def chat_completion_stream(self, client, payload, target_model=None):
        chunks = [
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_a",
                                    "type": "function",
                                    "function": {"name": "alpha", "arguments": ""},
                                },
                                {
                                    "index": 1,
                                    "id": "call_b",
                                    "type": "function",
                                    "function": {"name": "beta", "arguments": ""},
                                },
                            ]
                        },
                    }
                ]
            },
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": '{"a":'}},
                                {"index": 1, "function": {"arguments": '{"b":'}},
                            ]
                        },
                    }
                ]
            },
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": "1}"}},
                                {"index": 1, "function": {"arguments": "2}"}},
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
        ]
        for chunk in chunks:
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_messages_stream_tool_call_lifecycle(tmp_path, monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "gateway.sqlite3"))
    monkeypatch.setenv("MODEL_CATALOG_PATH", str(tmp_path / "model_catalog.json"))
    services = await _services_with_provider(
        tmp_path,
        FragmentedToolStreamProvider(),
        [
            {
                "route_id": "primary-test",
                "provider_name": "primary",
                "model_id": "primary/model",
                "display_name": "Primary",
                "rank": 1,
                "enabled": True,
                "tags": ["text", "tool-use"],
            }
        ],
    )

    probe = TestClient(app)
    attach_app_services(probe.app, services)
    response = probe.post(
        "/v1/messages",
        json={
            "model": "auto",
            "max_tokens": 128,
            "stream": True,
            "messages": [{"role": "user", "content": "run tools"}],
            "tools": [
                {
                    "name": "alpha",
                    "description": "A",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
        },
    )
    assert response.status_code == 200
    text = response.text
    assert "event: message_start" in text
    assert "input_json_delta" in text
    assert "event: message_stop" in text


class TrackableCloseStreamProvider:
    name = "primary"
    api_key = "test-key"
    max_context_tokens = 8192
    generator_closed = False

    @property
    def is_configured(self) -> bool:
        return True

    async def chat_completion(self, client, payload, target_model=None):
        return ProviderResponse("primary", 200, {}, {"choices": [], "usage": {}})

    async def chat_completion_stream(self, client, payload, target_model=None):
        try:
            chunk = json.dumps(
                {
                    "choices": [{"index": 0, "delta": {"content": "hello"}}],
                }
            )
            yield f"data: {chunk}\n\n"
            await asyncio.sleep(3600)
            yield "data: [DONE]\n\n"
        finally:
            self.generator_closed = True


@pytest.mark.asyncio
async def test_messages_stream_closes_upstream_on_cancellation(tmp_path, monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "gateway.sqlite3"))
    monkeypatch.setenv("MODEL_CATALOG_PATH", str(tmp_path / "model_catalog.json"))

    provider = TrackableCloseStreamProvider()
    services = await _services_with_provider(
        tmp_path,
        provider,
        [
            {
                "route_id": "primary-test",
                "provider_name": "primary",
                "model_id": "primary/model",
                "display_name": "Primary",
                "rank": 1,
                "enabled": True,
                "tags": ["text"],
            }
        ],
    )

    router = services.waterfall_router
    from app.anthropic_compat import messages_payload_to_chat
    from app.request_requirements import chat_request_requirements

    chat_payload = messages_payload_to_chat(
        {
            "model": "auto",
            "max_tokens": 32,
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        }
    )
    requirements = chat_request_requirements(chat_payload)
    agen = router.iter_chat_completion_openai_stream(chat_payload, requirements=requirements)
    saw_sse = False
    try:
        async for part in agen:
            if isinstance(part, str) and "hello" in part:
                saw_sse = True
                break
    finally:
        await agen.aclose()

    assert saw_sse is True
    assert provider.generator_closed is True
