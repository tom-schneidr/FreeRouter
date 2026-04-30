from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from app.model_catalog import ModelCatalog
from app.providers.base import ProviderAdapter, ProviderError, ProviderRateLimited
from app.state import StateManager


class NoProviderAvailable(RuntimeError):
    """Raised when the waterfall router exhausts all enabled routes without a successful response."""

    def __init__(self, attempts: list["ProviderAttempt"]) -> None:
        super().__init__("No configured provider is currently available")
        self.attempts = attempts


@dataclass(frozen=True)
class ProviderAttempt:
    """Diagnostic record for a single provider attempt during waterfall routing."""
    provider_name: str
    status: str
    reason: str | None = None
    status_code: int | None = None
    route_id: str | None = None
    model_id: str | None = None


@dataclass(frozen=True)
class RouteResult:
    """Successful routing outcome containing the response body and full attempt history."""
    body: dict[str, Any]
    provider_name: str
    route_id: str
    model_id: str
    attempts: list[ProviderAttempt]


class WaterfallRouter:
    def __init__(
        self,
        providers: list[ProviderAdapter],
        model_catalog: ModelCatalog,
        state: StateManager,
        *,
        request_timeout_seconds: float,
    ) -> None:
        self.providers = providers
        self.provider_by_name = {provider.name: provider for provider in providers}
        self.model_catalog = model_catalog
        self.state = state
        self.request_timeout_seconds = request_timeout_seconds

    async def route_chat_completion(self, payload: dict[str, Any]) -> RouteResult:
        """Walk enabled routes in rank order, attempting each until one succeeds.

        Skips routes for missing API keys, context window violations, and quota exhaustion.
        Automatically handles rate-limit responses, timeouts, and server errors by falling
        through to the next route. Raises NoProviderAvailable if all routes are exhausted.
        """
        validate_chat_completion_payload(payload)

        if payload.get("stream"):
            raise ValueError(
                "Streaming is not supported yet because transparent fallback needs a full response"
            )

        estimated_prompt_tokens = estimate_prompt_tokens(payload)
        estimated_total_tokens = estimated_prompt_tokens + int(
            payload.get("max_completion_tokens") or payload.get("max_tokens") or 0
        )
        attempts: list[ProviderAttempt] = []
        rate_limit_probe_providers: set[str] = set()

        timeout = httpx.Timeout(self.request_timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout) as client:
            for route in self.model_catalog.enabled_routes(payload.get("model")):
                provider = self.provider_by_name.get(route.provider_name)
                if provider is None:
                    attempts.append(
                        ProviderAttempt(
                            route.provider_name,
                            "skipped",
                            "unknown_provider",
                            route_id=route.route_id,
                            model_id=route.model_id,
                        ),
                    )
                    continue

                if not provider.is_configured:
                    attempts.append(
                        ProviderAttempt(
                            provider.name,
                            "skipped",
                            "missing_api_key",
                            route_id=route.route_id,
                            model_id=route.model_id,
                        ),
                    )
                    continue

                max_context_tokens = route.context_window or provider.max_context_tokens
                if (
                    max_context_tokens is not None
                    and estimated_prompt_tokens > max_context_tokens
                ):
                    attempts.append(
                        ProviderAttempt(
                            provider.name,
                            "skipped",
                            "context_window_exceeded",
                            route_id=route.route_id,
                            model_id=route.model_id,
                        ),
                    )
                    continue

                provider_availability = await self.state.check_available(
                    provider.name,
                    estimated_total_tokens,
                )
                if not provider_availability.available:
                    attempts.append(
                        ProviderAttempt(
                            provider.name,
                            "skipped",
                            provider_availability.reason,
                            route_id=route.route_id,
                            model_id=route.model_id,
                        ),
                    )
                    continue

                route_availability = await self.state.check_route_available(
                    route.route_id,
                    provider.name,
                    route.model_id,
                    allow_rate_limit_probe=provider.name not in rate_limit_probe_providers,
                )
                if not route_availability.available:
                    attempts.append(
                        ProviderAttempt(
                            provider.name,
                            "skipped",
                            route_availability.reason,
                            route_id=route.route_id,
                            model_id=route.model_id,
                        ),
                    )
                    continue
                if route_availability.reason in {"rate_limit_probe", "too_slow_probe"}:
                    rate_limit_probe_providers.add(provider.name)

                availability = await self.state.try_reserve_request(
                    provider.name,
                    estimated_total_tokens,
                )
                if not availability.available:
                    attempts.append(
                        ProviderAttempt(
                            provider.name,
                            "skipped",
                            availability.reason,
                            route_id=route.route_id,
                            model_id=route.model_id,
                        ),
                    )
                    continue

                try:
                    response = await provider.chat_completion(client, payload, route.model_id)
                except ProviderRateLimited as exc:
                    await self.state.mark_route_rate_limited(
                        route.route_id,
                        provider.name,
                        route.model_id,
                        headers=exc.headers,
                        status_code=exc.status_code,
                    )
                    await self.state.mark_exhausted(
                        provider.name,
                        headers=exc.headers,
                        status_code=exc.status_code,
                    )
                    attempts.append(
                        ProviderAttempt(
                            provider.name,
                            "rate_limited",
                            "provider_429",
                            exc.status_code,
                            route.route_id,
                            route.model_id,
                        ),
                    )
                    continue
                except httpx.TimeoutException:
                    timeout_state = await self.state.mark_route_timeout(
                        route.route_id,
                        provider.name,
                        route.model_id,
                    )
                    attempts.append(
                        ProviderAttempt(
                            provider.name,
                            "failed",
                            "timeout",
                            route_id=route.route_id,
                            model_id=route.model_id,
                        )
                    )
                    if timeout_state.status == "too_slow":
                        attempts.append(
                            ProviderAttempt(
                                provider.name,
                                "flagged",
                                "too_slow",
                                route_id=route.route_id,
                                model_id=route.model_id,
                            )
                        )
                    continue
                except httpx.RequestError as exc:
                    attempts.append(
                        ProviderAttempt(
                            provider.name,
                            "failed",
                            exc.__class__.__name__,
                            route_id=route.route_id,
                            model_id=route.model_id,
                        )
                    )
                    continue
                except ProviderError as exc:
                    if exc.status_code is not None and 500 <= exc.status_code < 600:
                        attempts.append(
                            ProviderAttempt(
                                provider.name,
                                "failed",
                                "provider_5xx",
                                exc.status_code,
                                route.route_id,
                                route.model_id,
                            ),
                        )
                        continue
                    if exc.status_code in {401, 403}:
                        attempts.append(
                            ProviderAttempt(
                                provider.name,
                                "failed",
                                "auth_error",
                                exc.status_code,
                                route.route_id,
                                route.model_id,
                            ),
                        )
                        continue
                    if exc.status_code == 404:
                        await self.state.mark_route_not_found(
                            route.route_id,
                            provider.name,
                            route.model_id,
                            status_code=exc.status_code,
                        )
                        attempts.append(
                            ProviderAttempt(
                                provider.name,
                                "failed",
                                "model_not_found",
                                exc.status_code,
                                route.route_id,
                                route.model_id,
                            ),
                        )
                        continue
                    if _looks_like_missing_endpoint(exc):
                        await self.state.mark_route_not_found(
                            route.route_id,
                            provider.name,
                            route.model_id,
                            status_code=exc.status_code,
                        )
                        attempts.append(
                            ProviderAttempt(
                                provider.name,
                                "failed",
                                "model_not_found",
                                exc.status_code,
                                route.route_id,
                                route.model_id,
                            ),
                        )
                        continue
                    raise

                usage = response.body.get("usage")
                await self.state.record_route_success(
                    route.route_id,
                    provider.name,
                    route.model_id,
                )
                await self.state.record_success(
                    provider.name,
                    usage=usage if isinstance(usage, dict) else None,
                    headers=response.headers,
                    status_code=response.status_code,
                )
                attempts.append(
                    ProviderAttempt(
                        provider.name,
                        "selected",
                        route_id=route.route_id,
                        model_id=route.model_id,
                    )
                )
                return RouteResult(
                    response.body,
                    provider.name,
                    route.route_id,
                    route.model_id,
                    attempts,
                )

        raise NoProviderAvailable(attempts)


def estimate_prompt_tokens(payload: dict[str, Any]) -> int:
    """Cheap local estimator used only for preflight routing decisions.

    A provider's response usage remains authoritative for accounting. The rough 4 chars/token
    heuristic avoids spending tokens on doomed calls when a request is obviously too large.
    """

    text_parts: list[str] = []
    for message in payload.get("messages") or []:
        if not isinstance(message, dict):
            continue
        content = message.get("content", "")
        text_parts.append(_content_to_text(content))

    for item in payload.get("input") or []:
        text_parts.append(_content_to_text(item))

    character_count = sum(len(part) for part in text_parts)
    message_overhead = 4 * len(payload.get("messages") or [])
    return max(1, character_count // 4 + message_overhead)


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(_content_to_text(item) for item in content)
    if isinstance(content, dict):
        if "text" in content:
            return str(content["text"])
        if "content" in content:
            return _content_to_text(content["content"])
    return str(content)


def _looks_like_missing_endpoint(exc: ProviderError) -> bool:
    if exc.status_code in {404, 410}:
        return True
    if exc.status_code not in {400, 422} or not exc.body:
        return False
    body = exc.body.lower()
    return (
        "model" in body
        and (
            "not found" in body
            or "does not exist" in body
            or "not exist" in body
            or "unknown model" in body
        )
    )


def validate_chat_completion_payload(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("Request body must include a non-empty 'messages' array")
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"messages[{index}] must be an object")
        if not isinstance(message.get("role"), str):
            raise ValueError(f"messages[{index}].role must be a string")
        if "content" not in message:
            raise ValueError(f"messages[{index}].content is required")
