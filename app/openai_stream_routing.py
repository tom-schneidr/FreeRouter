"""OpenAI-style SSE streaming with commit-on-first-chunk waterfall routing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx

from app.provider_errors import looks_like_missing_model
from app.providers.base import ProviderError, ProviderRateLimited
from app.router import (
    _SSE_DONE,
    NoProviderAvailable,
    ProviderAttempt,
    RouteStreamDiag,
    _event_block_data_payload,
    _payload_commits_openai_stream,
    _split_sse_event_blocks,
    _usage_from_openai_chunk,
    estimate_prompt_tokens,
    validate_chat_completion_payload,
)
from app.state import Availability, StateManager

if TYPE_CHECKING:
    from app.model_catalog import ModelCatalog
    from app.providers.base import ProviderAdapter


async def waterfall_openai_stream(
    *,
    providers_by_name: dict[str, ProviderAdapter],
    model_catalog: ModelCatalog,
    state: StateManager,
    client: httpx.AsyncClient,
    payload: dict[str, Any],
    required_tag: str | None = None,
    require_assistant_content: bool = False,
) -> Any:
    """Yield :class:`RouteStreamDiag` plus raw OpenAI ``text/event-stream`` fragments (``str``)."""
    validate_chat_completion_payload(payload)
    outbound_payload = dict(payload)
    outbound_payload["stream"] = True

    estimated_prompt_tokens = estimate_prompt_tokens(outbound_payload)
    estimated_total_tokens = estimated_prompt_tokens + int(
        outbound_payload.get("max_completion_tokens") or outbound_payload.get("max_tokens") or 0
    )
    attempts: list[ProviderAttempt] = []
    rate_limit_probe_providers: set[str] = set()

    routes_list = [
        route
        for route in model_catalog.enabled_routes(outbound_payload.get("model"))
        if required_tag is None or required_tag in route.tags
    ]
    prefetch_provider_names = sorted(
        {
            route.provider_name
            for route in routes_list
            if route.provider_name in providers_by_name
            and providers_by_name[route.provider_name].is_configured
        }
    )
    provider_availability_prefetch: dict[str, Availability] = {}
    if prefetch_provider_names:
        provider_availability_prefetch = await state.snapshot_providers_availability(
            prefetch_provider_names,
            estimated_tokens=estimated_total_tokens,
        )
    route_states_prefetch = await state.get_route_states_batch(
        [(route.route_id, route.provider_name, route.model_id) for route in routes_list]
    )

    for route in routes_list:
        provider = providers_by_name.get(route.provider_name)
        if provider is None:
            attempt = ProviderAttempt(
                route.provider_name,
                "skipped",
                "unknown_provider",
                route_id=route.route_id,
                model_id=route.model_id,
            )
            attempts.append(attempt)
            yield RouteStreamDiag(
                event_type="route_skipped",
                provider_name=route.provider_name,
                route_id=route.route_id,
                model_id=route.model_id,
                reason=attempt.reason,
            )
            continue

        if not provider.is_configured:
            attempt = ProviderAttempt(
                provider.name,
                "skipped",
                "missing_api_key",
                route_id=route.route_id,
                model_id=route.model_id,
            )
            attempts.append(attempt)
            yield RouteStreamDiag(
                event_type="route_skipped",
                provider_name=provider.name,
                route_id=route.route_id,
                model_id=route.model_id,
                reason=attempt.reason,
            )
            continue

        max_context_tokens = route.context_window or provider.max_context_tokens
        if max_context_tokens is not None and estimated_prompt_tokens > max_context_tokens:
            attempt = ProviderAttempt(
                provider.name,
                "skipped",
                "context_window_exceeded",
                route_id=route.route_id,
                model_id=route.model_id,
            )
            attempts.append(attempt)
            yield RouteStreamDiag(
                event_type="route_skipped",
                provider_name=provider.name,
                route_id=route.route_id,
                model_id=route.model_id,
                reason=attempt.reason,
            )
            continue

        prefetched = provider_availability_prefetch.get(provider.name)
        provider_availability = (
            prefetched
            if prefetched is not None
            else await state.check_available(provider.name, estimated_total_tokens)
        )
        if not provider_availability.available:
            attempt = ProviderAttempt(
                provider.name,
                "skipped",
                provider_availability.reason,
                route_id=route.route_id,
                model_id=route.model_id,
            )
            attempts.append(attempt)
            yield RouteStreamDiag(
                event_type="route_skipped",
                provider_name=provider.name,
                route_id=route.route_id,
                model_id=route.model_id,
                reason=attempt.reason,
            )
            continue

        route_state = route_states_prefetch.get(route.route_id)
        route_availability = (
            state.route_availability_from_state(
                route_state,
                allow_rate_limit_probe=provider.name not in rate_limit_probe_providers,
            )
            if route_state is not None
            else await state.check_route_available(
                route.route_id,
                provider.name,
                route.model_id,
                allow_rate_limit_probe=provider.name not in rate_limit_probe_providers,
            )
        )
        if not route_availability.available:
            attempt = ProviderAttempt(
                provider.name,
                "skipped",
                route_availability.reason,
                route_id=route.route_id,
                model_id=route.model_id,
            )
            attempts.append(attempt)
            yield RouteStreamDiag(
                event_type="route_skipped",
                provider_name=provider.name,
                route_id=route.route_id,
                model_id=route.model_id,
                reason=attempt.reason,
            )
            continue
        if route_availability.reason in {"rate_limit_probe", "too_slow_probe"}:
            rate_limit_probe_providers.add(provider.name)

        availability = await state.try_reserve_request(provider.name, estimated_total_tokens)
        if not availability.available:
            attempt = ProviderAttempt(
                provider.name,
                "skipped",
                availability.reason,
                route_id=route.route_id,
                model_id=route.model_id,
            )
            attempts.append(attempt)
            yield RouteStreamDiag(
                event_type="route_skipped",
                provider_name=provider.name,
                route_id=route.route_id,
                model_id=route.model_id,
                reason=attempt.reason,
            )
            continue

        yield RouteStreamDiag(
            event_type="route_trying",
            provider_name=provider.name,
            route_id=route.route_id,
            model_id=route.model_id,
        )

        agen = provider.chat_completion_stream(client, outbound_payload, route.model_id)
        carry = ""
        buffered_before_commit: list[str] = []
        committed = False
        usage: dict[str, Any] | None = None
        last_status = 200
        stop_route_attempt = False
        try:
            async for piece in agen:
                carry += piece
                blocks, carry = _split_sse_event_blocks(carry)
                for event_block in blocks:
                    pl = _event_block_data_payload(event_block)
                    text = event_block + "\n\n"
                    if isinstance(pl, dict):
                        u = _usage_from_openai_chunk(pl)
                        if u is not None:
                            usage = u

                    if not committed:
                        if pl is _SSE_DONE:
                            attempt = ProviderAttempt(
                                provider.name,
                                "failed",
                                "empty_stream",
                                route_id=route.route_id,
                                model_id=route.model_id,
                            )
                            attempts.append(attempt)
                            yield RouteStreamDiag(
                                event_type="route_failed",
                                provider_name=provider.name,
                                route_id=route.route_id,
                                model_id=route.model_id,
                                reason=attempt.reason,
                            )
                            stop_route_attempt = True
                            break
                        if isinstance(pl, dict) and "error" in pl:
                            attempt = ProviderAttempt(
                                provider.name,
                                "failed",
                                "provider_stream_error",
                                route_id=route.route_id,
                                model_id=route.model_id,
                            )
                            attempts.append(attempt)
                            yield RouteStreamDiag(
                                event_type="route_failed",
                                provider_name=provider.name,
                                route_id=route.route_id,
                                model_id=route.model_id,
                                reason=attempt.reason,
                            )
                            stop_route_attempt = True
                            break
                        if _payload_commits_openai_stream(
                            pl,
                            require_substantive_assistant=require_assistant_content,
                        ):
                            committed = True
                            selected = ProviderAttempt(
                                provider.name,
                                "selected",
                                route_id=route.route_id,
                                model_id=route.model_id,
                            )
                            attempts.append(selected)
                            yield RouteStreamDiag(
                                event_type="route_selected",
                                provider_name=provider.name,
                                route_id=route.route_id,
                                model_id=route.model_id,
                            )
                            for b in buffered_before_commit:
                                yield b
                            buffered_before_commit.clear()
                            yield text
                        else:
                            buffered_before_commit.append(text)
                    else:
                        if pl is _SSE_DONE:
                            await state.record_route_success(
                                route.route_id,
                                provider.name,
                                route.model_id,
                                usage=usage,
                                status_code=last_status,
                            )
                            await state.record_success(
                                provider.name,
                                usage=usage,
                                headers={},
                                status_code=last_status,
                            )
                            yield text
                            return
                        yield text
                if stop_route_attempt:
                    break

            if committed:
                await state.record_route_success(
                    route.route_id,
                    provider.name,
                    route.model_id,
                    usage=usage,
                    status_code=last_status,
                )
                await state.record_success(
                    provider.name,
                    usage=usage,
                    headers={},
                    status_code=last_status,
                )
                yield "data: [DONE]\n\n"
                return
        except ProviderRateLimited as exc:
            if committed:
                await state.record_route_success(
                    route.route_id,
                    provider.name,
                    route.model_id,
                    usage=usage,
                    status_code=last_status,
                )
                await state.record_success(
                    provider.name,
                    usage=usage,
                    headers={},
                    status_code=last_status,
                )
                yield "data: [DONE]\n\n"
                return
            flagged_state = await state.mark_route_rate_limited(
                route.route_id,
                provider.name,
                route.model_id,
                headers=exc.headers,
                status_code=exc.status_code,
            )
            await state.mark_exhausted(
                provider.name,
                headers=exc.headers,
                status_code=exc.status_code,
            )
            attempt = ProviderAttempt(
                provider.name,
                "rate_limited",
                "provider_429",
                exc.status_code,
                route.route_id,
                route.model_id,
            )
            attempts.append(attempt)
            yield RouteStreamDiag(
                event_type="route_failed",
                provider_name=provider.name,
                route_id=route.route_id,
                model_id=route.model_id,
                reason=attempt.reason,
                status_code=exc.status_code,
            )
            yield RouteStreamDiag(
                event_type="route_flagged",
                provider_name=provider.name,
                route_id=route.route_id,
                model_id=route.model_id,
                reason=flagged_state.status,
            )
            continue
        except httpx.TimeoutException:
            if committed:
                await state.record_route_success(
                    route.route_id,
                    provider.name,
                    route.model_id,
                    usage=usage,
                    status_code=last_status,
                )
                await state.record_success(
                    provider.name,
                    usage=usage,
                    headers={},
                    status_code=last_status,
                )
                yield "data: [DONE]\n\n"
                return
            timeout_state = await state.mark_route_timeout(
                route.route_id,
                provider.name,
                route.model_id,
            )
            attempt = ProviderAttempt(
                provider.name,
                "failed",
                "timeout",
                route_id=route.route_id,
                model_id=route.model_id,
            )
            attempts.append(attempt)
            yield RouteStreamDiag(
                event_type="route_failed",
                provider_name=provider.name,
                route_id=route.route_id,
                model_id=route.model_id,
                reason=attempt.reason,
            )
            if timeout_state.status == "too_slow":
                flagged = ProviderAttempt(
                    provider.name,
                    "flagged",
                    "too_slow",
                    route_id=route.route_id,
                    model_id=route.model_id,
                )
                attempts.append(flagged)
                yield RouteStreamDiag(
                    event_type="route_flagged",
                    provider_name=provider.name,
                    route_id=route.route_id,
                    model_id=route.model_id,
                    reason=flagged.reason,
                )
            continue
        except httpx.RequestError as exc:
            if committed:
                await state.record_route_success(
                    route.route_id,
                    provider.name,
                    route.model_id,
                    usage=usage,
                    status_code=last_status,
                )
                await state.record_success(
                    provider.name,
                    usage=usage,
                    headers={},
                    status_code=last_status,
                )
                yield "data: [DONE]\n\n"
                return
            attempt = ProviderAttempt(
                provider.name,
                "failed",
                exc.__class__.__name__,
                route_id=route.route_id,
                model_id=route.model_id,
            )
            attempts.append(attempt)
            yield RouteStreamDiag(
                event_type="route_failed",
                provider_name=provider.name,
                route_id=route.route_id,
                model_id=route.model_id,
                reason=attempt.reason,
            )
            continue
        except ProviderError as exc:
            if committed:
                await state.record_route_success(
                    route.route_id,
                    provider.name,
                    route.model_id,
                    usage=usage,
                    status_code=last_status,
                )
                await state.record_success(
                    provider.name,
                    usage=usage,
                    headers={},
                    status_code=last_status,
                )
                yield "data: [DONE]\n\n"
                return
            if exc.status_code is not None and 500 <= exc.status_code < 600:
                attempt = ProviderAttempt(
                    provider.name,
                    "failed",
                    "provider_5xx",
                    exc.status_code,
                    route.route_id,
                    route.model_id,
                )
                attempts.append(attempt)
                yield RouteStreamDiag(
                    event_type="route_failed",
                    provider_name=provider.name,
                    route_id=route.route_id,
                    model_id=route.model_id,
                    reason=attempt.reason,
                    status_code=exc.status_code,
                )
                continue
            if exc.status_code == 413:
                attempt = ProviderAttempt(
                    provider.name,
                    "failed",
                    "request_too_large",
                    exc.status_code,
                    route.route_id,
                    route.model_id,
                )
                attempts.append(attempt)
                yield RouteStreamDiag(
                    event_type="route_failed",
                    provider_name=provider.name,
                    route_id=route.route_id,
                    model_id=route.model_id,
                    reason=attempt.reason,
                    status_code=exc.status_code,
                )
                continue
            if exc.status_code in {401, 403}:
                attempt = ProviderAttempt(
                    provider.name,
                    "failed",
                    "auth_error",
                    exc.status_code,
                    route.route_id,
                    route.model_id,
                )
                attempts.append(attempt)
                yield RouteStreamDiag(
                    event_type="route_failed",
                    provider_name=provider.name,
                    route_id=route.route_id,
                    model_id=route.model_id,
                    reason=attempt.reason,
                    status_code=exc.status_code,
                )
                continue
            if looks_like_missing_model(exc):
                route_state_nf = await state.mark_route_not_found(
                    route.route_id,
                    provider.name,
                    route.model_id,
                    status_code=exc.status_code,
                )
                attempt = ProviderAttempt(
                    provider.name,
                    "failed",
                    "model_not_found",
                    exc.status_code,
                    route.route_id,
                    route.model_id,
                )
                attempts.append(attempt)
                yield RouteStreamDiag(
                    event_type="route_failed",
                    provider_name=provider.name,
                    route_id=route.route_id,
                    model_id=route.model_id,
                    reason=attempt.reason,
                    status_code=exc.status_code,
                )
                if route_state_nf.status == "potentially_outdated":
                    flagged = ProviderAttempt(
                        provider.name,
                        "flagged",
                        "potentially_outdated",
                        route_id=route.route_id,
                        model_id=route.model_id,
                    )
                    attempts.append(flagged)
                    yield RouteStreamDiag(
                        event_type="route_flagged",
                        provider_name=provider.name,
                        route_id=route.route_id,
                        model_id=route.model_id,
                        reason=flagged.reason,
                    )
                continue
            attempt = ProviderAttempt(
                provider.name,
                "failed",
                "provider_error",
                exc.status_code,
                route.route_id,
                route.model_id,
            )
            attempts.append(attempt)
            yield RouteStreamDiag(
                event_type="route_failed",
                provider_name=provider.name,
                route_id=route.route_id,
                model_id=route.model_id,
                reason=attempt.reason,
                status_code=exc.status_code,
            )
            continue
        finally:
            await agen.aclose()

    raise NoProviderAvailable(attempts)
