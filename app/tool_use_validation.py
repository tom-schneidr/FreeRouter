from __future__ import annotations

import hashlib
import json
import math
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal

from app.request_requirements import (
    _tool_choice_requires_function_tools,
)

ToolUseOutcome = Literal["supported", "unsupported", "neutral"]
ToolCallFailureCategory = Literal[
    "malformed_tool_calls",
    "unsupported_tool_call_type",
    "missing_function",
    "missing_function_name",
    "undeclared_function",
    "invalid_arguments_json",
    "arguments_not_object",
    "arguments_schema_mismatch",
    "invalid_call_id",
    "duplicate_call_id",
    "tool_choice_none_violation",
    "tool_choice_required_missing",
    "tool_choice_named_mismatch",
    "parallel_tool_calls_disabled",
    "inconsistent_finish_reason",
]


@dataclass(frozen=True)
class ToolCallValidationFailure:
    """A machine-readable reason an upstream tool response cannot be executed."""

    category: ToolCallFailureCategory
    message: str
    call_index: int | None = None
    path: str | None = None


@dataclass(frozen=True)
class ToolCallValidationResult:
    """Request-aware validation result plus an OpenAI-normalized response body."""

    outcome: ToolUseOutcome
    normalized_body: dict[str, Any]
    failures: tuple[ToolCallValidationFailure, ...] = ()

    @property
    def is_valid(self) -> bool:
        return not self.failures


_FAKE_TOOL_TEXT_MARKERS = (
    '"tool_calls"',
    "'tool_calls'",
    "tool_call",
    "<tool",
    "</tool",
    "function_call",
    '"type": "function"',
    '"type":"function"',
    '"name":',
    '{"name":',
    '{"function"',
    "```json",
    "<function=",
    "invoke(",
)

_FAKE_TOOL_TEXT_RE = re.compile(
    r"(\{\s*\"name\"\s*:\s*\"[^\"]+\"\s*,\s*\"arguments\"|\[\s*\{\s*\"type\"\s*:\s*\"function\")",
    re.IGNORECASE,
)
_PROMISE_TO_ACT_RE = re.compile(
    r"\b("
    r"i\s*(?:will|'ll|am\s+going\s+to|m\s+going\s+to)\s+"
    r"|let\s+me\s+"
    r"|i\s+can\s+"
    r"|i\s+should\s+"
    r"|i\s+need\s+to\s+"
    r")"
    r"(?:actually\s+|now\s+|go\s+ahead\s+and\s+|just\s+)?"
    r"(?:do|run|check|build|create|write|edit|update|fix|inspect|look|open|restart|install|configure|test|verify)\b",
    re.IGNORECASE,
)


def payload_requires_function_tools(payload: dict[str, Any]) -> bool:
    tools = payload.get("tools")
    if not isinstance(tools, list):
        return False
    return any(isinstance(tool, dict) and tool.get("type") == "function" for tool in tools)


def tool_use_response_mandatory(payload: dict[str, Any]) -> bool:
    """True when tool_choice requires structured tool_calls (not optional auto/none).

    Prior tool or assistant tool_calls in message history only means the session
    can use tools — it does not require every reply to be a tool_call. Agent
    clients (e.g. OpenClaw) always send tool history while still allowing text.
    """
    return _tool_choice_requires_function_tools(payload.get("tool_choice"))


def tool_loop_already_started(payload: dict[str, Any]) -> bool:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return False
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "tool":
            return True
        if message.get("role") == "assistant" and isinstance(message.get("tool_calls"), list):
            return True
    return False


def assistant_text_from_body(body: dict[str, Any]) -> str:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def function_tool_calls_from_body(body: dict[str, Any]) -> list[dict[str, Any]]:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return []
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        return []
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        return []
    return [call for call in tool_calls if isinstance(call, dict)]


def parse_function_tool_arguments(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        parsed = json.loads(
            raw,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"Non-finite JSON number: {value}")
            ),
        )
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def validate_and_normalize_tool_response(
    payload: dict[str, Any],
    body: dict[str, Any],
) -> ToolCallValidationResult:
    """Validate tool calls against the request and normalize executable calls.

    Optional ``tool_choice=auto`` permits a normal text response, but any tool
    calls that *are* returned must still be executable. Missing call IDs are
    synthesized deterministically because a tool result must be able to refer
    back to the call on the next agent turn.
    """
    normalized = deepcopy(body)
    failures: list[ToolCallValidationFailure] = []
    choice, message = _first_choice_and_message(normalized)
    raw_tool_calls = message.get("tool_calls") if message is not None else None
    calls_present = raw_tool_calls is not None

    if calls_present and not isinstance(raw_tool_calls, list):
        failures.append(
            ToolCallValidationFailure(
                "malformed_tool_calls",
                "message.tool_calls must be an array",
                path="$.choices[0].message.tool_calls",
            )
        )
        raw_tool_calls = []
    elif not isinstance(raw_tool_calls, list):
        raw_tool_calls = []

    if not raw_tool_calls:
        finish_reason = choice.get("finish_reason") if choice is not None else None
        if finish_reason == "tool_calls":
            failures.append(
                ToolCallValidationFailure(
                    "inconsistent_finish_reason",
                    "finish_reason is tool_calls but no tool calls were returned",
                    path="$.choices[0].finish_reason",
                )
            )

    declared_tools = _declared_function_tools(payload)
    named_choice = _named_tool_choice(payload.get("tool_choice"))
    tool_choice = payload.get("tool_choice")
    if raw_tool_calls and tool_choice == "none":
        failures.append(
            ToolCallValidationFailure(
                "tool_choice_none_violation",
                "tool calls were returned even though tool_choice is none",
                path="$.tool_choice",
            )
        )
    if len(raw_tool_calls) > 1 and payload.get("parallel_tool_calls") is False:
        failures.append(
            ToolCallValidationFailure(
                "parallel_tool_calls_disabled",
                "multiple tool calls were returned while parallel_tool_calls is false",
                path="$.parallel_tool_calls",
            )
        )

    normalized_calls: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    call_id_context = {
        "response_id": normalized.get("id"),
        "messages": payload.get("messages"),
    }
    for index, raw_call in enumerate(raw_tool_calls):
        normalized_call = _validate_and_normalize_call(
            raw_call,
            index=index,
            declared_tools=declared_tools,
            named_choice=named_choice,
            failures=failures,
        )
        if normalized_call is None:
            continue

        call_id = normalized_call.get("id")
        if call_id is None or call_id == "":
            call_id = _stable_call_id(
                normalized_call,
                index=index,
                used_ids=seen_ids,
                context=call_id_context,
            )
            normalized_call["id"] = call_id
        elif not isinstance(call_id, str) or not call_id.strip():
            failures.append(
                ToolCallValidationFailure(
                    "invalid_call_id",
                    "tool call id must be a non-empty string when provided",
                    call_index=index,
                    path=f"$.choices[0].message.tool_calls[{index}].id",
                )
            )
        elif call_id in seen_ids:
            failures.append(
                ToolCallValidationFailure(
                    "duplicate_call_id",
                    f"tool call id {call_id!r} is not unique",
                    call_index=index,
                    path=f"$.choices[0].message.tool_calls[{index}].id",
                )
            )

        if isinstance(call_id, str) and call_id:
            seen_ids.add(call_id)
        normalized_calls.append(normalized_call)

    requires_call = tool_use_response_mandatory(payload)
    if requires_call and not raw_tool_calls:
        failures.append(
            ToolCallValidationFailure(
                "tool_choice_required_missing",
                "tool_choice requires at least one function tool call",
                path="$.tool_choice",
            )
        )

    if message is not None and isinstance(raw_tool_calls, list) and raw_tool_calls:
        message["tool_calls"] = normalized_calls
        if choice is not None:
            choice["finish_reason"] = "tool_calls"

    if failures:
        outcome: ToolUseOutcome = "unsupported"
    elif normalized_calls:
        outcome = "supported"
    else:
        outcome = "neutral"
    return ToolCallValidationResult(outcome, normalized, tuple(failures))


def _first_choice_and_message(
    body: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None, None
    choice = choices[0]
    message = choice.get("message")
    return choice, message if isinstance(message, dict) else None


def _declared_function_tools(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    declared: dict[str, dict[str, Any]] = {}
    tools = payload.get("tools")
    if not isinstance(tools, list):
        return declared
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        function = tool.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if isinstance(name, str) and name.strip():
            declared[name.strip()] = function
    return declared


def _named_tool_choice(tool_choice: Any) -> str | None:
    if not isinstance(tool_choice, dict):
        return None
    function = tool_choice.get("function")
    if isinstance(function, dict):
        name = function.get("name")
    else:
        name = tool_choice.get("name") if tool_choice.get("type") == "function" else None
    return name.strip() if isinstance(name, str) and name.strip() else None


def _validate_and_normalize_call(
    raw_call: Any,
    *,
    index: int,
    declared_tools: dict[str, dict[str, Any]],
    named_choice: str | None,
    failures: list[ToolCallValidationFailure],
) -> dict[str, Any] | None:
    base_path = f"$.choices[0].message.tool_calls[{index}]"
    if not isinstance(raw_call, dict):
        failures.append(
            ToolCallValidationFailure(
                "malformed_tool_calls",
                "each tool call must be an object",
                call_index=index,
                path=base_path,
            )
        )
        return None
    normalized_call = deepcopy(raw_call)
    call_type = normalized_call.get("type")
    if call_type not in {None, "function"}:
        failures.append(
            ToolCallValidationFailure(
                "unsupported_tool_call_type",
                f"unsupported tool call type {call_type!r}",
                call_index=index,
                path=f"{base_path}.type",
            )
        )
    normalized_call["type"] = "function"

    function = normalized_call.get("function")
    if not isinstance(function, dict):
        failures.append(
            ToolCallValidationFailure(
                "missing_function",
                "tool call must contain a function object",
                call_index=index,
                path=f"{base_path}.function",
            )
        )
        return normalized_call
    name = function.get("name")
    if not isinstance(name, str) or not name.strip():
        failures.append(
            ToolCallValidationFailure(
                "missing_function_name",
                "tool call function name must be a non-empty string",
                call_index=index,
                path=f"{base_path}.function.name",
            )
        )
        return normalized_call
    name = name.strip()
    function["name"] = name
    definition = declared_tools.get(name)
    if definition is None:
        failures.append(
            ToolCallValidationFailure(
                "undeclared_function",
                f"function {name!r} was not declared in the request",
                call_index=index,
                path=f"{base_path}.function.name",
            )
        )
    if named_choice is not None and name != named_choice:
        failures.append(
            ToolCallValidationFailure(
                "tool_choice_named_mismatch",
                f"tool_choice requires {named_choice!r}, but the response called {name!r}",
                call_index=index,
                path=f"{base_path}.function.name",
            )
        )

    raw_arguments = function.get("arguments")
    arguments: dict[str, Any] | None
    if isinstance(raw_arguments, dict):
        arguments = raw_arguments
    elif isinstance(raw_arguments, str):
        try:
            parsed = json.loads(
                raw_arguments,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"Non-finite JSON number: {value}")
                ),
            )
        except (json.JSONDecodeError, ValueError) as exc:
            failures.append(
                ToolCallValidationFailure(
                    "invalid_arguments_json",
                    f"function arguments are not valid JSON: {exc}",
                    call_index=index,
                    path=f"{base_path}.function.arguments",
                )
            )
            arguments = None
        else:
            if not isinstance(parsed, dict):
                failures.append(
                    ToolCallValidationFailure(
                        "arguments_not_object",
                        "function arguments must decode to a JSON object",
                        call_index=index,
                        path=f"{base_path}.function.arguments",
                    )
                )
                arguments = None
            else:
                arguments = parsed
    else:
        failures.append(
            ToolCallValidationFailure(
                "arguments_not_object",
                "function arguments must be a JSON object or an encoded JSON object",
                call_index=index,
                path=f"{base_path}.function.arguments",
            )
        )
        arguments = None

    if arguments is not None:
        schema = definition.get("parameters") if definition is not None else None
        schema_error = _json_schema_error(arguments, schema, root_schema=schema)
        if schema_error is not None:
            error_path, error_message = schema_error
            failures.append(
                ToolCallValidationFailure(
                    "arguments_schema_mismatch",
                    error_message,
                    call_index=index,
                    path=f"{base_path}.function.arguments{error_path[1:]}",
                )
            )
        function["arguments"] = json.dumps(
            arguments,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    return normalized_call


def _stable_call_id(
    call: dict[str, Any],
    *,
    index: int,
    used_ids: set[str],
    context: dict[str, Any],
) -> str:
    function = call.get("function")
    material = json.dumps(
        {"context": context, "function": function, "index": index},
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]
    candidate = f"call_fr_{digest}"
    suffix = 2
    while candidate in used_ids:
        candidate = f"call_fr_{digest}_{suffix}"
        suffix += 1
    return candidate


def _json_schema_error(
    value: Any,
    schema: Any,
    *,
    root_schema: Any,
    path: str = "$",
    seen_refs: frozenset[str] = frozenset(),
) -> tuple[str, str] | None:
    """Return the first practical JSON Schema mismatch.

    Tool definitions overwhelmingly rely on the supported core: local refs,
    composition, object properties/required/additionalProperties, arrays,
    primitive types, enum/const, and common scalar bounds.
    """
    if schema is None or schema is True:
        return None
    if schema is False:
        return path, f"arguments at {path} are forbidden by the tool schema"
    if not isinstance(schema, dict):
        return None

    ref = schema.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/") and ref not in seen_refs:
        target = _resolve_local_schema_ref(root_schema, ref)
        if target is not None:
            return _json_schema_error(
                value,
                target,
                root_schema=root_schema,
                path=path,
                seen_refs=seen_refs | {ref},
            )

    for branch in schema.get("allOf", []) if isinstance(schema.get("allOf"), list) else []:
        error = _json_schema_error(value, branch, root_schema=root_schema, path=path)
        if error is not None:
            return error
    any_of = schema.get("anyOf")
    if isinstance(any_of, list) and any_of:
        if not any(
            _json_schema_error(value, branch, root_schema=root_schema, path=path) is None
            for branch in any_of
        ):
            return path, f"value at {path} does not match any allowed schema"
    one_of = schema.get("oneOf")
    if isinstance(one_of, list) and one_of:
        matches = sum(
            _json_schema_error(value, branch, root_schema=root_schema, path=path) is None
            for branch in one_of
        )
        if matches != 1:
            return path, f"value at {path} must match exactly one allowed schema"

    if "const" in schema and not _json_values_equal(value, schema["const"]):
        return path, f"value at {path} must equal {schema['const']!r}"
    enum = schema.get("enum")
    if isinstance(enum, list) and not any(_json_values_equal(value, item) for item in enum):
        return path, f"value at {path} must be one of {enum!r}"

    expected_type = schema.get("type")
    if schema.get("nullable") is True and value is None:
        return None
    if isinstance(expected_type, str):
        allowed_types = [expected_type]
    elif isinstance(expected_type, list):
        allowed_types = [item for item in expected_type if isinstance(item, str)]
    else:
        allowed_types = []
    if allowed_types and not any(_json_value_has_type(value, item) for item in allowed_types):
        return path, f"value at {path} must have type {' or '.join(allowed_types)}"

    if isinstance(value, dict):
        required = schema.get("required")
        if isinstance(required, list):
            for key in required:
                if isinstance(key, str) and key not in value:
                    return f"{path}.{key}", f"required property {key!r} is missing at {path}"
        properties = schema.get("properties")
        properties = properties if isinstance(properties, dict) else {}
        for key, child_value in value.items():
            child_schema = properties.get(key)
            if child_schema is not None:
                error = _json_schema_error(
                    child_value,
                    child_schema,
                    root_schema=root_schema,
                    path=f"{path}.{key}",
                )
                if error is not None:
                    return error
                continue
            additional = schema.get("additionalProperties", True)
            if additional is False:
                return f"{path}.{key}", f"additional property {key!r} is not allowed at {path}"
            if isinstance(additional, dict):
                error = _json_schema_error(
                    child_value,
                    additional,
                    root_schema=root_schema,
                    path=f"{path}.{key}",
                )
                if error is not None:
                    return error
        min_properties = schema.get("minProperties")
        max_properties = schema.get("maxProperties")
        if isinstance(min_properties, int) and len(value) < min_properties:
            return path, f"object at {path} has fewer than {min_properties} properties"
        if isinstance(max_properties, int) and len(value) > max_properties:
            return path, f"object at {path} has more than {max_properties} properties"

    if isinstance(value, list):
        items = schema.get("items")
        if isinstance(items, (dict, bool)):
            for item_index, item in enumerate(value):
                error = _json_schema_error(
                    item,
                    items,
                    root_schema=root_schema,
                    path=f"{path}[{item_index}]",
                )
                if error is not None:
                    return error
        min_items = schema.get("minItems")
        max_items = schema.get("maxItems")
        if isinstance(min_items, int) and len(value) < min_items:
            return path, f"array at {path} has fewer than {min_items} items"
        if isinstance(max_items, int) and len(value) > max_items:
            return path, f"array at {path} has more than {max_items} items"

    if isinstance(value, str):
        min_length = schema.get("minLength")
        max_length = schema.get("maxLength")
        pattern = schema.get("pattern")
        if isinstance(min_length, int) and len(value) < min_length:
            return path, f"string at {path} is shorter than {min_length} characters"
        if isinstance(max_length, int) and len(value) > max_length:
            return path, f"string at {path} is longer than {max_length} characters"
        if isinstance(pattern, str):
            try:
                matches = re.search(pattern, value) is not None
            except re.error:
                matches = True
            if not matches:
                return path, f"string at {path} does not match required pattern"

    if _json_value_has_type(value, "number"):
        for keyword, comparator, description in (
            ("minimum", lambda actual, bound: actual >= bound, "at least"),
            ("maximum", lambda actual, bound: actual <= bound, "at most"),
            ("exclusiveMinimum", lambda actual, bound: actual > bound, "greater than"),
            ("exclusiveMaximum", lambda actual, bound: actual < bound, "less than"),
        ):
            bound = schema.get(keyword)
            if isinstance(bound, (int, float)) and not isinstance(bound, bool):
                if not comparator(value, bound):
                    return path, f"number at {path} must be {description} {bound}"
    return None


def _resolve_local_schema_ref(root_schema: Any, ref: str) -> Any:
    current = root_schema
    for encoded_part in ref[2:].split("/"):
        part = encoded_part.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _json_value_has_type(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return (
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        )
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    return True


def _json_values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    return left == right


def function_tool_call_is_valid(call: dict[str, Any]) -> bool:
    if call.get("type") not in {None, "function"}:
        return False
    fn = call.get("function")
    if not isinstance(fn, dict):
        return False
    name = fn.get("name")
    if not isinstance(name, str) or not name.strip():
        return False
    return parse_function_tool_arguments(fn.get("arguments")) is not None


def response_has_valid_function_tool_calls(body: dict[str, Any]) -> bool:
    calls = function_tool_calls_from_body(body)
    return bool(calls) and all(function_tool_call_is_valid(call) for call in calls)


def response_fakes_tool_use_in_text(body: dict[str, Any]) -> bool:
    text = assistant_text_from_body(body).strip()
    if not text or response_has_valid_function_tool_calls(body):
        return False
    lowered = text.lower()
    if any(marker in lowered for marker in _FAKE_TOOL_TEXT_MARKERS):
        return True
    return _FAKE_TOOL_TEXT_RE.search(text) is not None


def response_promises_action_in_text(body: dict[str, Any]) -> bool:
    text = assistant_text_from_body(body).strip()
    if not text or response_has_valid_function_tool_calls(body):
        return False
    return _PROMISE_TO_ACT_RE.search(text) is not None


def evaluate_tool_use_outcome(
    payload: dict[str, Any],
    body: dict[str, Any],
    *,
    reject_initial_action_promise: bool = False,
) -> ToolUseOutcome:
    if not payload_requires_function_tools(payload):
        return "neutral"
    validation = validate_and_normalize_tool_response(payload, body)
    if validation.outcome == "supported":
        return "supported"
    if validation.failures:
        return "unsupported"
    if (
        reject_initial_action_promise
        and not tool_loop_already_started(payload)
        and response_promises_action_in_text(body)
    ):
        return "unsupported"
    if tool_use_response_mandatory(payload) or response_fakes_tool_use_in_text(body):
        return "unsupported"
    return "neutral"


def stream_chunk_commits_for_tool_use(outbound_payload: dict[str, Any]) -> bool:
    """When tool calls are mandatory, do not commit a stream on assistant text alone."""
    return not tool_use_response_mandatory(outbound_payload)


def should_abort_tool_stream_early(
    payload: dict[str, Any],
    *,
    text: str,
    saw_tool_calls: bool,
) -> bool:
    """Stop waiting on a stream that is faking tool_calls in prose when tools are required."""
    if saw_tool_calls or not tool_use_response_mandatory(payload):
        return False
    return response_fakes_tool_use_in_text({"choices": [{"message": {"content": text}}]})


ROUTING_SSE_KEEPALIVE = ": freerouter routing\r\n\r\n"
