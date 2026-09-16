"""OpenAI-style SSE streaming with commit-on-first-chunk waterfall routing."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from contextlib import aclosing
from typing import TYPE_CHECKING, Any

import httpx

from app.agent_profiles import AGENT_PROFILES, is_agent_profile
from app.capability_runtime import adjust_capabilities_from_traffic
from app.provider_errors import looks_like_missing_model
from app.providers.base import ProviderError, ProviderRateLimited
from app.request_requirements import RequestRequirements, chat_request_requirements
from app.request_sizing import estimate_total_request_tokens
from app.router import (
    _SSE_DONE,
    NoProviderAvailable,
    NoQualifiedRoute,
    ProviderAttempt,
    RouteStreamDiag,
    UnsupportedCapabilities,
    _delta_has_tool_calls,
    _delta_visible_text_from_chunk,
    _event_block_data_payload,
    _payload_commits_openai_stream,
    _split_sse_event_blocks,
    _usage_from_openai_chunk,
    validate_chat_completion_payload,
)
from app.routing_policy import (
    configured_provider_names,
    enabled_routes_for_request,
    static_route_skip_reason,
)
from app.sentinel_store import SentinelStore
from app.state import Availability, StateManager
from app.tool_call_stream import ToolCallStreamAccumulator
from app.tool_reliability import (
    tool_failure_outcome_category,
    tool_request_fingerprint,
    tool_route_sort_key,
)
from app.tool_use_validation import (
    payload_requires_function_tools,
    response_fakes_tool_use_in_text,
    response_promises_action_in_text,
    should_abort_tool_stream_early,
    tool_loop_already_started,
    validate_and_normalize_tool_response,
)
from app.waterfall_resilience import (
    MAX_WATERFALL_PASSES,
    route_exhausted_for_request,
    should_retry_waterfall,
    waterfall_retry_delay_seconds,
)

if TYPE_CHECKING:
    from app.model_catalog import ModelCatalog
    from app.providers.base import ProviderAdapter


def _usage_summary_diag(usage: dict[str, Any] | None) -> RouteStreamDiag | None:
    """Emit once per completed stream so dashboards can show upstream usage totals."""

    if not usage:
        return None
    return RouteStreamDiag(event_type="usage_summary", usage=dict(usage))


def _canonical_tool_sse_blocks(
    body: dict[str, Any],
    *,
    requested_model: str,
) -> list[str]:
    """Serialize a fully validated response as canonical OpenAI stream chunks."""

    choices = body.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return []
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict):
        return []
    delta: dict[str, Any] = {"role": "assistant"}
    content = message.get("content")
    if isinstance(content, str) and content:
        delta["content"] = content
    calls = message.get("tool_calls")
    if isinstance(calls, list) and calls:
        delta["tool_calls"] = [
            {"index": index, **call} for index, call in enumerate(calls) if isinstance(call, dict)
        ]
    common: dict[str, Any] = {
        "id": body.get("id") if isinstance(body.get("id"), str) else f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion.chunk",
        "created": body.get("created")
        if isinstance(body.get("created"), int)
        else int(time.time()),
        "model": body.get("model") if isinstance(body.get("model"), str) else requested_model,
    }
    for key in ("system_fingerprint", "service_tier"):
        if key in body:
            common[key] = body[key]
    first = {
        **common,
        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
    }
    terminal = {
        **common,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": choice.get("finish_reason")
                or ("tool_calls" if delta.get("tool_calls") else "stop"),
            }
        ],
    }
    blocks = [
        f"data: {json.dumps(first, separators=(',', ':'))}\n\n",
        f"data: {json.dumps(terminal, separators=(',', ':'))}\n\n",
    ]
    usage = body.get("usage")
    if isinstance(usage, dict):
        blocks.append(
            f"data: {json.dumps({**common, 'choices': [], 'usage': usage}, separators=(',', ':'))}\n\n"
        )
    blocks.append("data: [DONE]\n\n")
    return blocks


def _stream_error_block(code: str, message: str) -> str:
    payload = {"error": {"type": "stream_error", "code": code, "message": message}}
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


def _chunk_has_terminal_finish(payload: dict[str, Any]) -> bool:
    choices = payload.get("choices")
    return isinstance(choices, list) and any(
        isinstance(choice, dict) and isinstance(choice.get("finish_reason"), str)
        for choice in choices
    )


async def waterfall_openai_stream(
    *,
    providers_by_name: dict[str, ProviderAdapter],
    model_catalog: ModelCatalog,
    state: StateManager,
    client: httpx.AsyncClient,
    payload: dict[str, Any],
    requirements: RequestRequirements | None = None,
    require_assistant_content: bool = False,
    tool_requests_require: frozenset[str] | None = None,
    normal_requests_avoid: frozenset[str] | None = None,
    reject_initial_action_promise: bool = False,
    allow_unconfirmed_tool_use_fallback: bool = False,
    sentinel_store: SentinelStore | None = None,
) -> Any:
    """Yield :class:`RouteStreamDiag` plus raw OpenAI ``text/event-stream`` fragments (``str``)."""
    validate_chat_completion_payload(payload)
    outbound_payload = dict(payload)
    outbound_payload["stream"] = True

    estimated_prompt_tokens, estimated_total_tokens = estimate_total_request_tokens(
        outbound_payload
    )
    attempts: list[ProviderAttempt] = []
    rate_limit_probed_routes: set[str] = set()
    exhausted_routes: dict[str, str] = {}

    resolved_requirements = requirements or chat_request_requirements(outbound_payload)
    if resolved_requirements.request_class == "tool-use":
        resolved_requirements = RequestRequirements(
            required_capabilities=(
                resolved_requirements.required_capabilities
                | (tool_requests_require or frozenset({"tool-use"}))
            ),
            request_class=resolved_requirements.request_class,
        )
    required_capabilities = resolved_requirements.required_capabilities
    request_tool_fingerprint = tool_request_fingerprint(outbound_payload)
    requested_model = outbound_payload.get("model")
    sentinel_evaluations = None
    if is_agent_profile(requested_model) and sentinel_store is not None:
        sentinel_evaluations = await sentinel_store.latest_for_routes(
            [route.route_id for route in model_catalog.enabled_routes()]
        )
    routes_list = enabled_routes_for_request(
        model_catalog,
        requested_model=requested_model,
        required_capabilities=required_capabilities,
        avoid_capabilities=(
            (normal_requests_avoid or frozenset({"tool-use"}))
            if resolved_requirements.request_class == "normal"
            else frozenset()
        ),
        allow_unconfirmed_tool_use_fallback=allow_unconfirmed_tool_use_fallback,
        sentinel_evaluations=sentinel_evaluations,
    )
    if not routes_list:
        if is_agent_profile(requested_model):
            raise NoQualifiedRoute(
                requested_model,
                required_capabilities
                | frozenset(AGENT_PROFILES[requested_model].required_checks),
            )
        raise UnsupportedCapabilities(
            required_capabilities,
            requested_model if isinstance(requested_model, str) else None,
        )
    if "tool-use" in required_capabilities:
        reliability = await state.get_route_tool_reliability(
            [route.route_id for route in routes_list],
            request_fingerprint=request_tool_fingerprint,
        )
        routes_list.sort(
            key=lambda route: tool_route_sort_key(route, reliability.get(route.route_id)),
            reverse=True,
        )
    prefetch_provider_names = configured_provider_names(routes_list, providers_by_name)
    defer_initial_tool_text_commit = (
        reject_initial_action_promise
        and payload_requires_function_tools(outbound_payload)
        and not tool_loop_already_started(outbound_payload)
    )
    buffer_tool_response = payload_requires_function_tools(outbound_payload)
    stream_options = outbound_payload.get("stream_options")
    include_usage = isinstance(stream_options, dict) and stream_options.get("include_usage") is True

    for pass_index in range(MAX_WATERFALL_PASSES):
        if pass_index > 0:
            if not should_retry_waterfall(attempts):
                break
            delay = waterfall_retry_delay_seconds()
            await asyncio.sleep(delay)
            yield RouteStreamDiag(
                event_type="waterfall_retry",
                reason=f"retry_after_{delay:g}s",
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
        if pass_index > 0:
            rate_limit_probed_routes.clear()

        for route in routes_list:
            prior_reason = exhausted_routes.get(route.route_id)
            if prior_reason is not None:
                attempt = ProviderAttempt(
                    route.provider_name,
                    "skipped",
                    prior_reason,
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

            provider = providers_by_name.get(route.provider_name)
            skip_reason = static_route_skip_reason(
                provider,
                route,
                estimated_prompt_tokens=estimated_prompt_tokens,
                estimated_total_tokens=estimated_total_tokens,
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
                if route_exhausted_for_request(skip_reason):
                    exhausted_routes[route.route_id] = skip_reason
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
                    allow_rate_limit_probe=route.route_id not in rate_limit_probed_routes,
                )
                if route_state is not None
                else await state.check_route_available(
                    route.route_id,
                    provider.name,
                    route.model_id,
                    allow_rate_limit_probe=route.route_id not in rate_limit_probed_routes,
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
                rate_limit_probed_routes.add(route.route_id)

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
            precommit_text = ""
            tool_stream = ToolCallStreamAccumulator()
            buffer_current_tool_response = buffer_tool_response
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
                            if (
                                isinstance(pl, dict)
                                and _chunk_has_terminal_finish(pl)
                                and not include_usage
                            ):
                                # Some OpenAI-compatible servers close cleanly after
                                # a terminal finish_reason without sending [DONE].
                                blocks.append("data: [DONE]")
                            text = event_block + "\n\n"
                            if isinstance(pl, dict):
                                u = _usage_from_openai_chunk(pl)
                                if u is not None:
                                    usage = u

                            if not committed:
                                if isinstance(pl, dict):
                                    tool_stream.ingest(pl)
                                    if _delta_has_tool_calls(pl):
                                        buffer_current_tool_response = True
                                    precommit_text += _delta_visible_text_from_chunk(pl)

                                if isinstance(pl, dict) and "error" in pl:
                                    if buffer_current_tool_response:
                                        await state.record_route_tool_outcome(
                                            route.route_id,
                                            provider.name,
                                            route.model_id,
                                            "stream_error",
                                            request_fingerprint=request_tool_fingerprint,
                                        )
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

                                if pl is _SSE_DONE:
                                    if not buffer_current_tool_response:
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

                                    assembled = tool_stream.to_chat_body(text=precommit_text)
                                    validation = validate_and_normalize_tool_response(
                                        outbound_payload,
                                        assembled,
                                    )
                                    promise_rejected = (
                                        defer_initial_tool_text_commit
                                        and not tool_stream.has_tool_calls
                                        and response_promises_action_in_text(assembled)
                                    )
                                    fake_required_call = should_abort_tool_stream_early(
                                        outbound_payload,
                                        text=precommit_text,
                                        saw_tool_calls=tool_stream.has_tool_calls,
                                    ) or (
                                        not tool_stream.has_tool_calls
                                        and response_fakes_tool_use_in_text(assembled)
                                    )
                                    empty_response = (
                                        not precommit_text.strip()
                                        and not tool_stream.has_tool_calls
                                    )
                                    if (
                                        validation.failures
                                        or promise_rejected
                                        or fake_required_call
                                        or empty_response
                                    ):
                                        if promise_rejected:
                                            failure_category = "action_promise"
                                            fail_reason = "action_promise_without_tool_call"
                                        elif empty_response:
                                            failure_category = "truncated_stream"
                                            fail_reason = "empty_stream"
                                        elif validation.failures:
                                            failure_category = tool_failure_outcome_category(
                                                validation.failures[0].category
                                            )
                                            fail_reason = validation.failures[0].category
                                        else:
                                            failure_category = "malformed_call"
                                            fail_reason = "invalid_tool_response"
                                        await state.record_route_tool_outcome(
                                            route.route_id,
                                            provider.name,
                                            route.model_id,
                                            failure_category,
                                            request_fingerprint=request_tool_fingerprint,
                                        )
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
                                    if validation.outcome == "supported":
                                        adjust_capabilities_from_traffic(
                                            model_catalog,
                                            route_id=route.route_id,
                                            required_capabilities=required_capabilities,
                                            payload=outbound_payload,
                                            response_body=validation.normalized_body,
                                        )
                                        await state.record_route_tool_outcome(
                                            route.route_id,
                                            provider.name,
                                            route.model_id,
                                            (
                                                "continuation_success"
                                                if tool_loop_already_started(outbound_payload)
                                                else "valid_call"
                                            ),
                                            request_fingerprint=request_tool_fingerprint,
                                        )
                                        for block in _canonical_tool_sse_blocks(
                                            validation.normalized_body,
                                            requested_model=route.model_id,
                                        ):
                                            yield block
                                    else:
                                        if tool_loop_already_started(outbound_payload):
                                            await state.record_route_tool_outcome(
                                                route.route_id,
                                                provider.name,
                                                route.model_id,
                                                "continuation_success",
                                                request_fingerprint=request_tool_fingerprint,
                                            )
                                        for block in buffered_before_commit:
                                            yield block
                                        yield text
                                    buffered_before_commit.clear()
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
                                    return

                                if buffer_current_tool_response:
                                    buffered_before_commit.append(text)
                                    continue

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
                                    for buffered in buffered_before_commit:
                                        yield buffered
                                    buffered_before_commit.clear()
                                    yield text
                                else:
                                    buffered_before_commit.append(text)
                            else:
                                if isinstance(pl, dict) and "error" in pl:
                                    yield RouteStreamDiag(
                                        event_type="route_failed",
                                        provider_name=provider.name,
                                        route_id=route.route_id,
                                        model_id=route.model_id,
                                        reason="provider_stream_error_after_commit",
                                    )
                                    yield text
                                    return
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

                    if stop_route_attempt:
                        continue
                    if committed:
                        yield RouteStreamDiag(
                            event_type="route_failed",
                            provider_name=provider.name,
                            route_id=route.route_id,
                            model_id=route.model_id,
                            reason="stream_ended_without_terminal",
                        )
                        yield _stream_error_block(
                            "stream_ended_without_terminal",
                            "The selected provider ended its stream without a terminal event.",
                        )
                        return
                    if buffer_current_tool_response and buffered_before_commit:
                        await state.record_route_tool_outcome(
                            route.route_id,
                            provider.name,
                            route.model_id,
                            "truncated_stream",
                            request_fingerprint=request_tool_fingerprint,
                        )
                        attempt = ProviderAttempt(
                            provider.name,
                            "failed",
                            "stream_ended_without_terminal",
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
            except ProviderRateLimited as exc:
                if committed:
                    await state.mark_route_rate_limited(
                        route.route_id,
                        provider.name,
                        route.model_id,
                        headers=exc.headers,
                        status_code=exc.status_code,
                    )
                    yield RouteStreamDiag(
                        event_type="route_failed",
                        provider_name=provider.name,
                        route_id=route.route_id,
                        model_id=route.model_id,
                        reason="provider_429_after_commit",
                        status_code=exc.status_code,
                    )
                    yield _stream_error_block(
                        "provider_429_after_commit",
                        "The selected provider rate-limited an incomplete stream.",
                    )
                    return
                flagged_state = await state.mark_route_rate_limited(
                    route.route_id,
                    provider.name,
                    route.model_id,
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
                if request_tool_fingerprint is not None:
                    await state.record_route_tool_outcome(
                        route.route_id,
                        provider.name,
                        route.model_id,
                        "stream_error",
                        request_fingerprint=request_tool_fingerprint,
                    )
                if committed:
                    await state.mark_route_timeout(
                        route.route_id,
                        provider.name,
                        route.model_id,
                    )
                    yield RouteStreamDiag(
                        event_type="route_failed",
                        provider_name=provider.name,
                        route_id=route.route_id,
                        model_id=route.model_id,
                        reason="timeout_after_commit",
                    )
                    yield _stream_error_block(
                        "timeout_after_commit",
                        "The selected provider timed out before completing its stream.",
                    )
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
                if request_tool_fingerprint is not None:
                    await state.record_route_tool_outcome(
                        route.route_id,
                        provider.name,
                        route.model_id,
                        "stream_error",
                        request_fingerprint=request_tool_fingerprint,
                    )
                if committed:
                    yield RouteStreamDiag(
                        event_type="route_failed",
                        provider_name=provider.name,
                        route_id=route.route_id,
                        model_id=route.model_id,
                        reason="transport_error_after_commit",
                    )
                    yield _stream_error_block(
                        "transport_error_after_commit",
                        "The selected provider connection ended before stream completion.",
                    )
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
                if request_tool_fingerprint is not None and (
                    exc.status_code == 400
                    or (exc.status_code is not None and exc.status_code >= 500)
                ):
                    await state.record_route_tool_outcome(
                        route.route_id,
                        provider.name,
                        route.model_id,
                        "provider_rejected" if exc.status_code == 400 else "stream_error",
                        request_fingerprint=request_tool_fingerprint,
                    )
                if committed:
                    yield RouteStreamDiag(
                        event_type="route_failed",
                        provider_name=provider.name,
                        route_id=route.route_id,
                        model_id=route.model_id,
                        reason="provider_error_after_commit",
                        status_code=exc.status_code,
                    )
                    yield _stream_error_block(
                        "provider_error_after_commit",
                        "The selected provider returned an error before stream completion.",
                    )
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
        if pass_index + 1 >= MAX_WATERFALL_PASSES or not should_retry_waterfall(attempts):
            break

    raise NoProviderAvailable(attempts)
