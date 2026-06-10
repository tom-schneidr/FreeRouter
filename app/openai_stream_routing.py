"""OpenAI-style SSE streaming with commit-on-first-chunk waterfall routing."""

from __future__ import annotations

from contextlib import aclosing
from typing import TYPE_CHECKING, Any

import httpx

from app.capability_runtime import adjust_capabilities_from_traffic
from app.provider_errors import looks_like_missing_model
from app.providers.base import ProviderError, ProviderRateLimited
from app.request_requirements import RequestRequirements, chat_request_requirements
from app.router import (
    _SSE_DONE,
    NoProviderAvailable,
    ProviderAttempt,
    RouteStreamDiag,
    UnsupportedCapabilities,
    _delta_has_tool_calls,
    _delta_visible_text_from_chunk,
    _event_block_data_payload,
    _payload_commits_openai_stream,
    _split_sse_event_blocks,
    _usage_from_openai_chunk,
    estimate_prompt_tokens,
    validate_chat_completion_payload,
)
from app.routing_policy import (
    configured_provider_names,
    enabled_routes_for_request,
    static_route_skip_reason,
)
from app.state import Availability, StateManager
from app.tool_use_validation import (
    evaluate_tool_use_outcome,
    payload_requires_function_tools,
    should_abort_tool_stream_early,
)

if TYPE_CHECKING:
    from app.model_catalog import ModelCatalog
    from app.providers.base import ProviderAdapter


def _synthetic_stream_response_body(*, saw_tool_calls: bool, text: str) -> dict[str, Any]:
    if saw_tool_calls:
        return {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {"name": "stream", "arguments": "{}"},
                            }
                        ]
                    }
                }
            ]
        }
    return {"choices": [{"message": {"content": text}}]}


def _usage_summary_diag(usage: dict[str, Any] | None) -> RouteStreamDiag | None:
    """Emit once per completed stream so dashboards can show upstream usage totals."""

    if not usage:
        return None
    return RouteStreamDiag(event_type="usage_summary", usage=dict(usage))


async def waterfall_openai_stream(
    *,
    providers_by_name: dict[str, ProviderAdapter],
    model_catalog: ModelCatalog,
    state: StateManager,
    client: httpx.AsyncClient,
    payload: dict[str, Any],
    requirements: RequestRequirements | None = None,
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

    resolved_requirements = requirements or chat_request_requirements(outbound_payload)
    required_capabilities = resolved_requirements.required_capabilities
    requested_model = outbound_payload.get("model")
    routes_list = enabled_routes_for_request(
        model_catalog,
        requested_model=requested_model,
        required_capabilities=required_capabilities,
    )
    if not routes_list:
        raise UnsupportedCapabilities(
            required_capabilities,
            requested_model if isinstance(requested_model, str) else None,
        )
    prefetch_provider_names = configured_provider_names(routes_list, providers_by_name)
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
        skip_reason = static_route_skip_reason(
            provider,
            route,
            estimated_prompt_tokens=estimated_prompt_tokens,
        )
        if skip_reason == "unknown_provider":
            attempt = ProviderAttempt(
                route.provider_name,
                "skipped",
                skip_reason,
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

        assert provider is not None

        if skip_reason is not None:
            attempt = ProviderAttempt(
                provider.name,
                "skipped",
                skip_reason,
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

        carry = ""
        buffered_before_commit: list[str] = []
        committed = False
        precommit_saw_tool_calls = False
        precommit_text = ""
        usage: dict[str, Any] | None = None
        last_status = 200
        stop_route_attempt = False
        try:
            async with aclosing(
                provider.chat_completion_stream(client, outbound_payload, route.model_id)
            ) as agen:
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
                            if isinstance(pl, dict):
                                if _delta_has_tool_calls(pl):
                                    precommit_saw_tool_calls = True
                                precommit_text += _delta_visible_text_from_chunk(pl)
                                if should_abort_tool_stream_early(
                                    outbound_payload,
                                    text=precommit_text,
                                    saw_tool_calls=precommit_saw_tool_calls,
                                ):
                                    synthetic_body = _synthetic_stream_response_body(
                                        saw_tool_calls=False,
                                        text=precommit_text,
                                    )
                                    adjust_capabilities_from_traffic(
                                        model_catalog,
                                        route_id=route.route_id,
                                        required_capabilities=required_capabilities,
                                        payload=outbound_payload,
                                        response_body=synthetic_body,
                                    )
                                    attempt = ProviderAttempt(
                                        provider.name,
                                        "failed",
                                        "invalid_tool_response",
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
                            if pl is _SSE_DONE:
                                fail_reason = "empty_stream"
                                if payload_requires_function_tools(outbound_payload):
                                    synthetic_body = _synthetic_stream_response_body(
                                        saw_tool_calls=precommit_saw_tool_calls,
                                        text=precommit_text,
                                    )
                                    if (
                                        evaluate_tool_use_outcome(
                                            outbound_payload,
                                            synthetic_body,
                                        )
                                        == "unsupported"
                                    ):
                                        adjust_capabilities_from_traffic(
                                            model_catalog,
                                            route_id=route.route_id,
                                            required_capabilities=required_capabilities,
                                            payload=outbound_payload,
                                            response_body=synthetic_body,
                                        )
                                        fail_reason = "invalid_tool_response"
                                attempt = ProviderAttempt(
                                    provider.name,
                                    "failed",
                                    fail_reason,
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
                                outbound_payload=outbound_payload,
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
                                    route_tags=tuple(route.tags),
                                    required_capabilities=required_capabilities,
                                )
                                if isinstance(pl, dict) and _delta_has_tool_calls(pl):
                                    adjust_capabilities_from_traffic(
                                        model_catalog,
                                        route_id=route.route_id,
                                        required_capabilities=required_capabilities,
                                        payload=outbound_payload,
                                        response_body={
                                            "choices": [
                                                {
                                                    "message": {
                                                        "tool_calls": [
                                                            {
                                                                "type": "function",
                                                                "function": {
                                                                    "name": "probe",
                                                                    "arguments": "{}",
                                                                },
                                                            }
                                                        ]
                                                    }
                                                }
                                            ]
                                        },
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
                                summary = _usage_summary_diag(usage)
                                if summary is not None:
                                    yield summary
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
                    summary = _usage_summary_diag(usage)
                    if summary is not None:
                        yield summary
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
                summary = _usage_summary_diag(usage)
                if summary is not None:
                    yield summary
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
                summary = _usage_summary_diag(usage)
                if summary is not None:
                    yield summary
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
                summary = _usage_summary_diag(usage)
                if summary is not None:
                    yield summary
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
                summary = _usage_summary_diag(usage)
                if summary is not None:
                    yield summary
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
            if "tool-use" in required_capabilities and exc.status_code == 400:
                adjust_capabilities_from_traffic(
                    model_catalog,
                    route_id=route.route_id,
                    required_capabilities=required_capabilities,
                    payload=outbound_payload,
                    error=exc,
                )
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

    raise NoProviderAvailable(attempts)
