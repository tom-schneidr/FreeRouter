from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import httpx


class ProviderError(Exception):
    """Base exception for provider-level errors, carrying HTTP status and response body."""
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        headers: Mapping[str, str] | None = None,
        body: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.headers = headers or {}
        self.body = body


class ProviderRateLimited(ProviderError):
    """Raised specifically for HTTP 429 responses from a provider."""


@dataclass(frozen=True)
class ProviderResponse:
    """Normalized successful response from a provider."""
    provider_name: str
    status_code: int
    headers: Mapping[str, str]
    body: dict[str, Any]


@dataclass(frozen=True)
class ProviderAdapter:
    """Configuration and HTTP logic for a single upstream provider endpoint."""
    name: str
    api_key: str | None
    base_url: str
    default_model: str
    max_context_tokens: int | None = None
    extra_headers: Mapping[str, str] | None = None
    model_aliases: Mapping[str, str] | None = None

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    async def chat_completion(
        self,
        client: httpx.AsyncClient,
        payload: dict[str, Any],
        target_model: str | None = None,
    ) -> ProviderResponse:
        if not self.api_key:
            raise ProviderError(f"{self.name} is missing an API key")

        outbound = dict(payload)
        outbound["model"] = target_model or self._provider_model(payload.get("model"))

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **(self.extra_headers or {}),
        }
        response = await client.post(
            f"{self.base_url.rstrip('/')}/chat/completions",
            headers=headers,
            json=outbound,
        )

        if response.status_code == 429:
            raise ProviderRateLimited(
                f"{self.name} returned 429",
                status_code=response.status_code,
                headers=response.headers,
                body=response.text,
            )

        if response.status_code >= 400:
            raise ProviderError(
                f"{self.name} returned HTTP {response.status_code}",
                status_code=response.status_code,
                headers=response.headers,
                body=response.text,
            )

        try:
            body = response.json()
        except ValueError as exc:
            raise ProviderError(
                f"{self.name} returned a non-JSON response",
                status_code=response.status_code,
                headers=response.headers,
                body=response.text,
            ) from exc

        return ProviderResponse(
            provider_name=self.name,
            status_code=response.status_code,
            headers=response.headers,
            body=body,
        )

    def _provider_model(self, requested_model: Any) -> str:
        if not isinstance(requested_model, str) or requested_model in {"", "auto"}:
            return self.default_model
        if self.model_aliases and requested_model in self.model_aliases:
            return self.model_aliases[requested_model]
        return self.default_model
