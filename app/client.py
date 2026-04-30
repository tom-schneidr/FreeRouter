from __future__ import annotations

from typing import Any

from app.model_catalog import ModelCatalog
from app.providers import PROVIDER_QUOTAS, build_provider_adapters
from app.router import RouteResult, WaterfallRouter
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

    async def initialize(self) -> None:
        """Initialize the necessary async components like the state manager SQLite database."""
        await self.state.initialize()
        self.model_catalog.initialize()
        self.router = WaterfallRouter(
            build_provider_adapters(self.settings),
            self.model_catalog,
            self.state,
            request_timeout_seconds=self.settings.request_timeout_seconds,
        )

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
        if not self.router:
            await self.initialize()

        payload = {"messages": messages, "model": kwargs.pop("model", "auto"), **kwargs}
        return await self.router.route_chat_completion(payload)


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
