from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.model_catalog import ModelCatalog
from app.providers.base import ProviderRateLimited, ProviderResponse
from app.router import WaterfallRouter
from app.state import ProviderQuota, StateManager
from app.waterfall_resilience import should_retry_waterfall, waterfall_retry_delay_seconds


class _SiblingRouteProvider:
    def __init__(self, name: str, *, error_for: set[str] | None = None) -> None:
        self.name = name
        self.api_key = "test-key"
        self.max_context_tokens = 131_072
        self.error_for = error_for or set()
        self.calls: dict[str, int] = {}

    @property
    def is_configured(self) -> bool:
        return True

    async def chat_completion(self, client, payload, model_id: str) -> ProviderResponse:
        self.calls[model_id] = self.calls.get(model_id, 0) + 1
        if model_id in self.error_for:
            raise ProviderRateLimited("rate limited", status_code=429, headers={"retry-after": "120"})
        return ProviderResponse(
            provider_name=self.name,
            status_code=200,
            headers={},
            body={
                "choices": [{"message": {"role": "assistant", "content": f"ok:{model_id}"}}],
                "usage": {"total_tokens": 3},
            },
        )


def _catalog(tmp_path) -> ModelCatalog:
    catalog = ModelCatalog(str(tmp_path / "models.json"))
    catalog.replace_routes(
        [
            {
                "route_id": "nvidia-a",
                "provider_name": "nvidia",
                "model_id": "model-a",
                "display_name": "A",
                "rank": 1,
                "enabled": True,
                "context_window": 131_072,
                "tags": ["text", "tool-use"],
            },
            {
                "route_id": "nvidia-b",
                "provider_name": "nvidia",
                "model_id": "model-b",
                "display_name": "B",
                "rank": 2,
                "enabled": True,
                "context_window": 131_072,
                "tags": ["text", "tool-use"],
            },
        ]
    )
    return catalog


async def _state(tmp_path) -> StateManager:
    state = StateManager(
        str(tmp_path / "state.sqlite3"),
        quotas=[ProviderQuota(name="nvidia", tokens_per_day=None, requests_per_day=None, requests_per_minute=None)],
    )
    await state.initialize()
    return state


@pytest.mark.asyncio
async def test_route_429_does_not_block_sibling_route_on_same_provider(tmp_path):
    provider = _SiblingRouteProvider("nvidia", error_for={"model-a"})
    state = await _state(tmp_path)
    router = WaterfallRouter([provider], _catalog(tmp_path), state, request_timeout_seconds=5)

    result = await router.route_chat_completion({"model": "auto", "messages": [{"role": "user", "content": "hi"}]})

    assert result.model_id == "model-b"
    assert provider.calls == {"model-a": 1, "model-b": 1}
    provider_state = await state.get_state("nvidia")
    assert provider_state.cooldown_until == 0


def test_should_retry_waterfall_on_cooldown_skips():
    from app.router import ProviderAttempt

    attempts = [
        ProviderAttempt("nvidia", "skipped", "cooldown", route_id="a", model_id="a"),
        ProviderAttempt("groq", "failed", "request_too_large", 413, route_id="b", model_id="b"),
    ]
    assert should_retry_waterfall(attempts) is True
    assert waterfall_retry_delay_seconds() > 0
