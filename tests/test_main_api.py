from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient

from app.main import (
    WEB_SEARCH_TOOL,
    _assistant_text_from_response_body,
    _payload_with_required_web_search,
    app,
)
from app.request_limiter import GatewayRequestLimiter
from app.settings import get_settings


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
    }
    for key, value in settings.items():
        monkeypatch.setenv(key, value)
    return TestClient(app)


def test_v1_models_includes_gateway_alias(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        response = client.get("/v1/models")
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "list"
    assert payload["data"]
    assert payload["data"][0]["id"] == "auto"


def test_gateway_models_update_rejects_invalid_payload(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        response = client.put("/v1/gateway/models", json={"data": "not-a-list"})
    assert response.status_code == 400
    assert "Expected a JSON array" in response.json()["detail"]


def test_chat_completions_rejects_invalid_messages(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        response = client.post("/v1/chat/completions", json={"model": "auto", "messages": []})
    assert response.status_code == 400
    assert "non-empty 'messages'" in response.json()["detail"]


def test_chat_completions_returns_503_when_all_providers_unconfigured(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "auto", "messages": [{"role": "user", "content": "hello"}]},
        )
    assert response.status_code == 503
    payload = response.json()
    assert payload["error"]["code"] == "waterfall_exhausted"
    assert payload["error"]["attempts"]


def test_chat_completions_stream_returns_sse_when_exhausted(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "auto",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        )
    assert response.status_code == 200
    assert response.headers.get("content-type", "").startswith("text/event-stream")
    assert "waterfall_exhausted" in response.text


def test_web_search_payload_requires_web_search_tool():
    payload = _payload_with_required_web_search(
        {
            "model": "auto",
            "messages": [{"role": "user", "content": "latest news"}],
            "tools": [{"type": "function", "function": {"name": "lookup"}}],
        }
    )

    assert payload["tool_choice"] == WEB_SEARCH_TOOL
    assert WEB_SEARCH_TOOL in payload["tools"]
    assert payload["tools"][0]["type"] == "function"


def test_assistant_text_extractor_handles_nested_chat_response():
    assert (
        _assistant_text_from_response_body(
            {"choices": [{"message": {"role": "assistant", "content": "hello from ai"}}]}
        )
        == "hello from ai"
    )
    assert (
        _assistant_text_from_response_body(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "part one"}, {"text": "part two"}],
                        }
                    }
                ]
            }
        )
        == "part one\npart two"
    )


def test_web_search_endpoint_uses_web_search_routes(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        response = client.post(
            "/v1/chat/completions/web-search",
            json={"model": "auto", "messages": [{"role": "user", "content": "hello"}]},
        )

    assert response.status_code == 503
    attempts = response.json()["error"]["attempts"]
    assert attempts
    assert {"groq", "openrouter"}.issubset({attempt["provider_name"] for attempt in attempts})


def test_web_search_endpoint_rejects_non_object_payload(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        response = client.post("/v1/chat/completions/web-search", json=[])

    assert response.status_code == 400
    assert "JSON object" in response.json()["detail"]


def test_stream_route_returns_error_event_on_invalid_payload(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        response = client.post(
            "/v1/chat/completions/stream-route",
            json={"model": "auto", "messages": []},
        )
    assert response.status_code == 200
    body = response.text
    assert "data:" in body
    assert '"type": "error"' in body


def test_auto_rank_endpoint_returns_catalog_payload(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        response = client.post("/v1/gateway/models/auto-rank")
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "list"
    assert payload["data"]
    assert "rank_score" in payload["data"][0]


def test_gateway_models_get_includes_health_blob(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        response = client.get("/v1/gateway/models")
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "list"
    assert payload["data"]
    assert "health" in payload["data"][0]
    assert "status" in payload["data"][0]["health"]


def test_providers_status_returns_models_with_health_and_usage(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        response = client.get("/v1/providers/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "list"
    assert payload["data"]
    first = payload["data"][0]
    assert "available" in first
    assert "models" in first
    if first["models"]:
        assert "health" in first["models"][0]
        assert "usage" in first["models"][0]


def test_live_snapshot_reports_request_activity(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        client.post(
            "/v1/chat/completions",
            json={"model": "auto", "messages": [{"role": "user", "content": "hello"}]},
        )
        response = client.get("/v1/gateway/live/snapshot")
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "list"
    assert payload["data"]
    assert any(item.get("request_id") for item in payload["data"])


async def test_request_limiter_times_out_when_full():
    limiter = GatewayRequestLimiter(max_concurrent_requests=1, queue_timeout_seconds=0.01)
    assert await limiter.acquire() is True

    second = await limiter.acquire()

    limiter.release()
    assert second is False


async def test_request_limiter_allows_waiting_request_after_release():
    limiter = GatewayRequestLimiter(max_concurrent_requests=1, queue_timeout_seconds=1)
    assert await limiter.acquire() is True

    waiter = asyncio.create_task(limiter.acquire())
    await asyncio.sleep(0)
    limiter.release()

    assert await waiter is True
    limiter.release()


async def test_request_limiter_rejects_when_waiting_queue_is_full():
    limiter = GatewayRequestLimiter(
        max_concurrent_requests=1,
        queue_timeout_seconds=1,
        max_waiting_requests=1,
    )
    assert await limiter.acquire() is True

    first_waiter = asyncio.create_task(limiter.acquire())
    await asyncio.sleep(0)
    second_waiter = await limiter.acquire()

    limiter.release()
    assert await first_waiter is True
    limiter.release()
    assert second_waiter is False
