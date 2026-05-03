from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.model_catalog import ModelCatalog
from app.providers import PROVIDER_QUOTAS, build_provider_adapters
from app.router import ChatStreamPart, RouteResult, WaterfallRouter
from app.settings import get_settings
from app.state import StateManager


class UnifiedAIClient:
    """A programmatic entry point to the AI Gateway logic.

    Other code can instantiate this to get answers using the best possible
    models configured in the system, transparently failing over to backups.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.state = StateManager(
            self.settings.database_path,
            PROVIDER_QUOTAS,
            busy_timeout_ms=self.settings.sqlite_busy_timeout_ms,
        )
        self.model_catalog = ModelCatalog(self.settings.model_catalog_path)
        self.router: WaterfallRouter | None = None
        self._http_client: httpx.AsyncClient | None = None

    async def initialize(self) -> None:
        """Initialize the necessary async components like the state manager SQLite database."""
        await self.state.initialize()
        self.model_catalog.initialize()
        limits = httpx.Limits(
            max_connections=self.settings.http_max_connections,
            max_keepalive_connections=self.settings.http_max_keepalive_connections,
            keepalive_expiry=self.settings.http_keepalive_expiry_seconds,
        )
        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.settings.request_timeout_seconds),
            limits=limits,
        )
        self.router = WaterfallRouter(
            build_provider_adapters(self.settings),
            self.model_catalog,
            self.state,
            request_timeout_seconds=self.settings.request_timeout_seconds,
            http_client=self._http_client,
        )

    async def aclose(self) -> None:
        """Close outbound HTTP resources (for tests/long-running embeddings)."""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    async def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> RouteResult:
        """Send a chat completion request using the gateway's routing logic.

        Args:
            messages: A list of message dictionaries (e.g., [{"role": "user", "content": "Hi"}])
            kwargs: Any other OpenAI-compatible chat completion parameters
                    (temperature, max_tokens, etc.)

        Returns:
            A RouteResult object containing the raw response body and metadata
            about the provider, model, and attempts.
        """
        if kwargs.get("stream"):
            raise ValueError(
                "UnifiedAIClient.chat() does not support streaming; use chat_stream() instead "
                "or call POST /v1/chat/completions with stream:true."
            )

        if not self.router:
            await self.initialize()

        payload = {"messages": messages, "model": kwargs.pop("model", "auto"), **kwargs}
        return await self.router.route_chat_completion(payload)

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        *,
        required_tag: str | None = None,
        require_assistant_content: bool = False,
        **kwargs: Any,
    ) -> AsyncIterator[ChatStreamPart]:
        """Stream OpenAI-compatible SSE fragments and routing diagnostics from the gateway router."""
        if not self.router:
            await self.initialize()

        kwargs.pop("stream", None)
        payload = {"messages": messages, "model": kwargs.pop("model", "auto"), **kwargs}
        payload["stream"] = True
        async for part in self.router.iter_chat_completion_openai_stream(
            payload,
            required_tag=required_tag,
            require_assistant_content=require_assistant_content,
        ):
            yield part


_default_client: UnifiedAIClient | None = None


async def ask_ai(messages: list[dict[str, str]], **kwargs: Any) -> RouteResult:
    """Convenience function that maintains a singleton client and sends a chat completion.

    This is the simplest entry point for external scripts to request an AI response.
    """
    global _default_client
    if _default_client is None:
        _default_client = UnifiedAIClient()
        await _default_client.initialize()

    return await _default_client.chat(messages, **kwargs)
