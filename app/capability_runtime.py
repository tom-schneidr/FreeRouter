from __future__ import annotations

from typing import Any

from app.capability_probes import _provider_error_indicates_unsupported
from app.model_catalog import ModelCatalog
from app.providers.base import ProviderError
from app.tool_use_validation import (
    evaluate_tool_use_outcome,
    payload_requires_function_tools,
    response_has_valid_function_tool_calls,
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
    """Update transport capability from decisive real request outcomes.

    Behavioral misses and malformed generations are tracked separately by the
    route reliability store. A single model miss must not erase previously
    confirmed transport support.
    """
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

    # Unsupported model output is intentionally not a transport-capability
    # demotion. Repeated typed failures lower request-conditioned reliability;
    # only an explicit upstream "tools unsupported" error removes eligibility.
