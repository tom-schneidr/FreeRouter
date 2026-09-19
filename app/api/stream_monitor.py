from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from app.router import (
    RouteStreamDiag,
    _delta_visible_text_from_chunk,
    _event_block_data_payload,
    _split_sse_event_blocks,
)
from app.tool_call_stream import ToolCallStreamAccumulator


def monitor_live_value(
    value: Any,
    *,
    max_string: int = 800,
    max_items: int = 12,
    max_depth: int = 3,
    _depth: int = 0,
) -> Any:
    """Bound live-monitor payloads so one long agent turn cannot freeze the history view.

    The monitor is diagnostic metadata, not an archival copy of prompts and responses. Responses
    requests can contain the complete accumulated agent transcript, so retaining each event's full
    payload grows quadratically during a stalled client loop.
    """
    if _depth >= max_depth:
        return "<truncated-depth>"
    if isinstance(value, str):
        if len(value) <= max_string:
            return value
        return value[:max_string] + f"... <truncated {len(value) - max_string} chars>"
    if isinstance(value, list):
        items = [
            monitor_live_value(
                item,
                max_string=max_string,
                max_items=max_items,
                max_depth=max_depth,
                _depth=_depth + 1,
            )
            for item in value[:max_items]
        ]
        if len(value) > max_items:
            items.append(f"<truncated {len(value) - max_items} items>")
        return items
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= max_items:
                out["__truncated__"] = f"{len(value) - max_items} keys omitted"
                break
            out[str(key)] = monitor_live_value(
                item,
                max_string=max_string,
                max_items=max_items,
                max_depth=max_depth,
                _depth=_depth + 1,
            )
        return out
    return value


class StreamMonitorTracker:
    """Collect routing progress and streamed assistant text for live monitor completion."""

    def __init__(self, *, requested_model: Any) -> None:
        self.requested_model = (
            requested_model
            if isinstance(requested_model, str) and requested_model.strip()
            else "auto"
        )
        self.attempts: list[dict[str, Any]] = []
        self.assistant_text = ""
        self.usage: dict[str, Any] | None = None
        self.provider_name = ""
        self.route_id = ""
        self.model_id = ""
        self._sse_carry = ""
        self._tool_calls = ToolCallStreamAccumulator()

    def record_diag(self, diag: RouteStreamDiag) -> None:
        if diag.event_type == "usage_summary":
            if diag.usage:
                self.usage = dict(diag.usage)
            return

        if diag.event_type == "route_selected":
            self.provider_name = diag.provider_name or ""
            self.route_id = diag.route_id or ""
            self.model_id = diag.model_id or ""

        attempt = _attempt_from_diag(diag)
        if attempt is not None:
            self.attempts.append(attempt)

    def record_openai_sse(self, fragment: str) -> None:
        if not fragment:
            return
        self._sse_carry += fragment
        blocks, self._sse_carry = _split_sse_event_blocks(self._sse_carry)
        for block in blocks:
            payload = _event_block_data_payload(block)
            if not isinstance(payload, dict):
                continue
            self._tool_calls.ingest(payload)
            delta = _delta_visible_text_from_chunk(payload)
            if delta:
                self.assistant_text += delta
            usage = payload.get("usage")
            if isinstance(usage, dict) and usage:
                self.usage = dict(usage)

    def flush_sse_carry(self) -> None:
        carry = self._sse_carry.strip()
        self._sse_carry = ""
        if not carry:
            return
        for raw_line in carry.splitlines():
            line = raw_line.strip()
            if not line.startswith("data: "):
                continue
            inner = line[6:].strip()
            if inner == "[DONE]":
                continue
            try:
                payload = json.loads(inner)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                self._tool_calls.ingest(payload)
                delta = _delta_visible_text_from_chunk(payload)
                if delta:
                    self.assistant_text += delta

    def build_response_body(self) -> dict[str, Any] | None:
        if not self.assistant_text and not self.usage and not self._tool_calls.has_tool_calls:
            return None
        body = self._tool_calls.to_chat_body(text=self.assistant_text)
        body["object"] = "chat.completion"
        body["model"] = self.requested_model
        if not self._tool_calls.has_tool_calls:
            body["choices"][0]["finish_reason"] = body["choices"][0].get("finish_reason") or "stop"
        if self.usage:
            body["usage"] = dict(self.usage)
        return body

    def completed_payload(self, *, status_code: int, latency_ms: int) -> dict[str, Any]:
        self.flush_sse_carry()
        response_body = self.build_response_body()
        payload: dict[str, Any] = {
            "status_code": status_code,
            "model": self.requested_model,
            "provider_name": self.provider_name,
            "route_id": self.route_id,
            "model_id": self.model_id,
            "latency_ms": latency_ms,
            "attempts": len(self.attempts),
            "attempts_detail": [dict(attempt) for attempt in self.attempts],
            "assistant_text": self.assistant_text,
        }
        if self.usage:
            payload["usage"] = dict(self.usage)
        if response_body is not None:
            payload["response_body"] = response_body
        return payload


def _attempt_from_diag(diag: RouteStreamDiag) -> dict[str, Any] | None:
    base = {
        "provider_name": diag.provider_name or "",
        "route_id": diag.route_id or "",
        "model_id": diag.model_id or "",
        "reason": diag.reason,
        "status_code": diag.status_code,
    }
    if diag.event_type == "route_trying":
        return {**base, "status": "trying"}
    if diag.event_type == "route_skipped":
        status = _status_from_skip_reason(diag.reason)
        return {**base, "status": status}
    if diag.event_type == "route_failed":
        return {**base, "status": "failed"}
    if diag.event_type == "route_flagged":
        return {**base, "status": "flagged"}
    if diag.event_type == "route_selected":
        return {**base, "status": "selected", "reason": None}
    return None


def _status_from_skip_reason(reason: str | None) -> str:
    if not reason:
        return "skipped"
    normalized = reason.strip().lower()
    if normalized in {"rate_limited", "cooldown", "missing_api_key", "context_window_exceeded"}:
        return normalized
    if "rate" in normalized and "limit" in normalized:
        return "rate_limited"
    return "skipped"


def attempts_detail_from_provider_attempts(attempts: list[Any]) -> list[dict[str, Any]]:
    return [
        asdict(attempt) if hasattr(attempt, "__dataclass_fields__") else dict(attempt)
        for attempt in attempts
    ]
