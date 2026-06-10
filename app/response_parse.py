from __future__ import annotations

import json
from typing import Any


def chat_response_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list):
        return ""

    chunks: list[str] = []
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                chunks.append(content)
            elif isinstance(content, list):
                chunks.extend(
                    str(item.get("text"))
                    for item in content
                    if isinstance(item, dict) and isinstance(item.get("text"), str)
                )
        text = choice.get("text")
        if isinstance(text, str):
            chunks.append(text)
    return "\n".join(chunks)


def json_object_from_text(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        parsed = json.loads(stripped)
    except ValueError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(stripped[start : end + 1])
        except ValueError:
            return None
    return parsed if isinstance(parsed, dict) else None
