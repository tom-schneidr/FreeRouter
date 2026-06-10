from __future__ import annotations

from typing import Any

from app.capability_probes import _provider_error_indicates_unsupported
from app.model_catalog import ModelCatalog
from app.providers.base import ProviderError
from app.tool_use_validation import (
    evaluate_tool_use_outcome,
    payload_requires_function_tools,
    response_fakes_tool_use_in_text,
    response_has_valid_function_tool_calls,
    tool_use_response_mandatory,
)


def response_has_tool_calls(body: dict[str, Any]) -> bool:
    return response_has_valid_function_tool_calls(body)


def adjust_capabilities_from_traffic(
    catalog: ModelCatalog,
    *,
    route_id: str,
    required_capabilities: frozenset[str],
    payload: dict[str, Any],
    response_body: dict[str, Any] | None = None,
    error: ProviderError | None = None,
) -> None:
    """Promote or demote capability claims from real request outcomes."""
    if "tool-use" not in required_capabilities and not payload_requires_function_tools(payload):
        return

    if error is not None and _provider_error_indicates_unsupported("tool-use", error):
        catalog.note_runtime_capability(
            route_id,
            "tool-use",
            status="unsupported",
            evidence=str(error)[:240],
        )
        return

    if response_body is None:
        return

    outcome = evaluate_tool_use_outcome(payload, response_body)
    if outcome == "supported":
        catalog.note_runtime_capability(
            route_id,
            "tool-use",
            status="supported",
            evidence="Valid function tool_calls in live traffic",
        )
        return

    if outcome == "unsupported":
        if response_fakes_tool_use_in_text(response_body):
            evidence = "Assistant returned prose/JSON instead of tool_calls"
        elif tool_use_response_mandatory(payload):
            evidence = "Function tools required but response had no valid tool_calls"
        else:
            evidence = "Invalid tool-use response"
        catalog.note_runtime_capability(
            route_id,
            "tool-use",
            status="unsupported",
            evidence=evidence,
        )
