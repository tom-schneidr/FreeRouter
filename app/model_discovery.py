from __future__ import annotations

from typing import Any

from app.model_catalog import ModelRoute, route_from_discovered_model
from app.providers.base import ProviderAdapter


def routes_from_payload(provider: ProviderAdapter, payload: dict[str, Any]) -> list[ModelRoute]:
    """Build discoverable free chat routes from a provider `/models` payload."""
    routes: list[ModelRoute] = []
    for item in payload.get("data", []):
        if not isinstance(item, dict):
            continue
        route = route_from_catalog_item(provider, item)
        if route is not None:
            routes.append(route)
    return routes


def route_from_catalog_item(
    provider: ProviderAdapter,
    item: dict[str, Any],
    *,
    assume_free: bool = False,
    supervisor_note: str = "",
) -> ModelRoute | None:
    """Convert a provider catalog item into a reviewable chat route when free use is proven."""
    model_id = item.get("id")
    if not isinstance(model_id, str) or not model_id:
        return None

    if not is_chat_model(item):
        return None

    free_by_payload = is_discoverable_free_model(provider, item)
    if not assume_free and not free_by_payload:
        return None

    route_model_id = route_model_id_from_catalog_id(model_id)
    display_name = str(item.get("name") or route_model_id)
    tags = tags_for_model(item)

    if provider.name == "openrouter":
        enabled = False
        notes = (
            "Discovered automatically from OpenRouter /api/v1/models as a zero-cost or :free model. "
            "Disabled by default until reviewed."
        )
    elif assume_free and not free_by_payload:
        enabled = False
        notes = (
            "Discovered automatically after free-tier verification. Disabled by default until "
            "reviewed because provider free-tier limits can still vary by account."
        )
    else:
        enabled = False
        notes = (
            "Discovered automatically from structured zero pricing in the provider /models endpoint. "
            "Disabled by default until reviewed because provider free-tier limits can still vary by account."
        )

    if supervisor_note:
        notes = f"{notes} {supervisor_note}".strip()

    return route_from_discovered_model(
        provider.name,
        route_model_id,
        display_name=display_name,
        context_window=_int_or_none(item.get("context_length")),
        tags=tags,
        source_url=f"{provider.base_url.rstrip('/')}/models",
        notes=notes,
        enabled=enabled,
    )


def is_discoverable_free_model(provider: ProviderAdapter, item: dict[str, Any]) -> bool:
    """Return true only when the provider payload itself proves a free route."""
    if provider.name == "openrouter":
        return is_openrouter_free_model(item)
    return has_structured_zero_price(item)


def is_openrouter_free_model(item: dict[str, Any]) -> bool:
    model_id = item.get("id")
    if isinstance(model_id, str) and model_id.endswith(":free"):
        return True
    return has_structured_zero_price(item)


def has_structured_zero_price(item: dict[str, Any]) -> bool:
    pricing = item.get("pricing")
    if not isinstance(pricing, dict):
        return False

    price_pairs = (
        ("prompt", "completion"),
        ("input", "output"),
        ("input_price", "output_price"),
        ("prompt_price", "completion_price"),
    )
    price_keys = {key for pair in price_pairs for key in pair}
    present_prices = [_float_or_none(pricing.get(key)) for key in price_keys if key in pricing]
    has_complete_pair = any(
        input_key in pricing and output_key in pricing for input_key, output_key in price_pairs
    )

    if (
        not has_complete_pair
        or not present_prices
        or any(price is None for price in present_prices)
    ):
        return False
    return all(price == 0 for price in present_prices)


def route_model_id_from_catalog_id(model_id: str) -> str:
    return model_id.removeprefix("models/")


def tags_for_model(item: dict[str, Any]) -> list[str]:
    text = _model_search_text(item)
    capability_text = _capability_search_text(item)

    tags = ["text"] if is_chat_model(item) else []
    if any(term in text for term in ("vision", "image", "vl", "pixtral", "gemma-3")):
        tags.append("vision")
    if any(term in text for term in ("audio", "speech")):
        tags.append("audio")
    if any(term in text for term in ("reason", "thinking", "qwen", "hermes", "gpt-oss")):
        tags.append("reasoning")
    if any(term in text for term in ("coder", "code")):
        tags.append("coding")
    if _supports_tool_use(capability_text):
        tags.append("tool-use")
    if _supports_web_search(capability_text):
        tags.append("web-search")
    return list(dict.fromkeys(tags))


def is_chat_model(item: dict[str, Any]) -> bool:
    """Classify chat/text-generation LLMs and reject generation/specialized endpoints."""
    text = _model_search_text(item)
    if any(term in text for term in _NON_CHAT_MODEL_TERMS):
        return False

    architecture = item.get("architecture")
    if isinstance(architecture, dict):
        input_modalities = _string_list(architecture.get("input_modalities"))
        output_modalities = _string_list(architecture.get("output_modalities"))
        if input_modalities and "text" not in input_modalities:
            return False
        if output_modalities and "text" not in output_modalities:
            return False
        modality = str(architecture.get("modality") or "").lower()
        if modality:
            parts = [
                part.strip() for part in modality.replace(",", "->").split("->") if part.strip()
            ]
            if parts and ("text" not in parts[0] or "text" not in parts[-1]):
                return False
        if input_modalities and output_modalities:
            return "text" in input_modalities and "text" in output_modalities

    if any(term in text for term in _CHAT_MODEL_TERMS):
        return True
    if not isinstance(architecture, dict):
        return False
    return "text" in text and not any(term in text for term in ("generate", "generation"))


_CHAT_MODEL_TERMS = (
    "chat",
    "instruct",
    "llm",
    "language model",
    "text->text",
    "text to text",
    "gemini",
    "gemma",
    "llama",
    "qwen",
    "deepseek",
    "mistral",
    "mixtral",
    "gpt",
    "claude",
    "command",
    "nemotron",
    "hermes",
    "kimi",
    "minimax",
    "glm",
    "yi",
    "phi",
    "dolphin",
    "ling",
)

_NON_CHAT_MODEL_TERMS = (
    "veo",
    "video generation",
    "ocr",
    "generate-preview",
    "generate-001",
    "imagen",
    "image generation",
    "lyria",
    "music generation",
    "tts",
    "text-to-speech",
    "speech generation",
    "embedding",
    "embed",
    "rerank",
    "moderation",
    "safety",
    "guard",
    "translat",
)


def _model_search_text(item: dict[str, Any]) -> str:
    architecture = item.get("architecture")
    architecture_values: list[str] = []
    if isinstance(architecture, dict):
        architecture_values = [str(value) for value in architecture.values()]
    return " ".join(
        str(value)
        for value in (
            item.get("id"),
            item.get("name"),
            item.get("description"),
            *architecture_values,
        )
    ).lower()


def _capability_search_text(item: dict[str, Any]) -> str:
    values: list[str] = []
    for key in (
        "supported_parameters",
        "supported_features",
        "capabilities",
        "features",
        "tools",
        "tool_types",
    ):
        values.extend(_flatten_text_values(item.get(key)))
    return " ".join(values).lower()


def _flatten_text_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (int, float, bool)):
        return [str(value)]
    if isinstance(value, list):
        values: list[str] = []
        for item in value:
            values.extend(_flatten_text_values(item))
        return values
    if isinstance(value, dict):
        values = []
        for key, item in value.items():
            values.append(str(key))
            values.extend(_flatten_text_values(item))
        return values
    return [str(value)]


def _supports_tool_use(capability_text: str) -> bool:
    return any(
        term in capability_text
        for term in (
            "tools",
            "tool_choice",
            "function_call",
            "function calling",
            "function_calling",
        )
    )


def _supports_web_search(capability_text: str) -> bool:
    normalized = capability_text.replace("-", "_").replace(" ", "_")
    return any(
        term in normalized
        for term in (
            "web_search",
            "web_search_options",
            "web_search_preview",
            "browser_search",
            "internet_search",
            "grounding_with_google_search",
        )
    )


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).lower() for item in value if isinstance(item, str)]
    if isinstance(value, str):
        return [part.strip().lower() for part in value.replace(",", " ").split() if part.strip()]
    return []


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
