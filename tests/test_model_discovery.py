from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.model_discovery import route_from_catalog_item, routes_from_payload


@dataclass
class FakeProvider:
    name: str
    base_url: str = "https://example.test/v1"


def test_openrouter_catalog_payload_creates_free_routes_only():
    provider = FakeProvider("openrouter")
    routes = routes_from_payload(
        provider,
        {
            "data": [
                {
                    "id": "new/free-model:free",
                    "name": "New Free Model",
                    "context_length": 65536,
                    "architecture": {"modality": "text->text"},
                    "pricing": {"prompt": "0", "completion": "0"},
                },
                {
                    "id": "paid/model",
                    "name": "Paid Model",
                    "architecture": {"modality": "text->text"},
                    "pricing": {"prompt": "1", "completion": "1"},
                },
            ]
        },
    )

    assert [route.model_id for route in routes] == ["new/free-model:free"]
    assert routes[0].display_name == "New Free Model"
    assert routes[0].context_window == 65536
    assert "text" in routes[0].tags
    assert routes[0].enabled is False


def test_non_openrouter_catalog_payload_requires_structured_free_pricing():
    provider = FakeProvider("groq")
    routes = routes_from_payload(
        provider,
        {
            "data": [
                {
                    "id": "llama-new-70b",
                    "name": "Llama New 70B",
                    "context_length": 131072,
                    "architecture": {"modality": "text->text"},
                },
                {
                    "id": "llama-free-70b",
                    "name": "Llama Free 70B",
                    "context_length": 131072,
                    "architecture": {"modality": "text->text"},
                    "pricing": {"prompt": "0", "completion": "0"},
                },
                {
                    "id": "llama-paid-70b",
                    "name": "Llama Paid 70B",
                    "context_length": 131072,
                    "architecture": {"modality": "text->text"},
                    "pricing": {"prompt": "0.1", "completion": "0"},
                },
                {
                    "id": "embedding-model",
                    "name": "Embedding Model",
                    "architecture": {"modality": "text->embedding"},
                },
                {
                    "id": "veo-3.1-generate-preview",
                    "name": "Veo 3.1",
                    "description": "Video generation model",
                    "architecture": {"modality": "text->video"},
                },
                {
                    "id": "imagen-4.0-generate-001",
                    "name": "Imagen 4",
                    "description": "Image generation model",
                    "architecture": {"input_modalities": ["text"], "output_modalities": ["image"]},
                },
            ]
        },
    )

    assert [route.model_id for route in routes] == ["llama-free-70b"]
    assert routes[0].provider_name == "groq"
    assert routes[0].enabled is False
    assert "free-tier limits can still vary" in routes[0].notes


def test_openrouter_catalog_payload_allows_multimodal_text_models():
    provider = FakeProvider("openrouter")
    routes = routes_from_payload(
        provider,
        {
            "data": [
                {
                    "id": "google/lyria-3-pro-preview",
                    "name": "Google: Lyria 3 Pro Preview",
                    "description": "Music generation model",
                    "pricing": {"prompt": "0", "completion": "0"},
                },
                {
                    "id": "mistralai/pixtral-12b:free",
                    "name": "Mistral: Pixtral 12B",
                    "architecture": {"input_modalities": ["text", "image"], "output_modalities": ["text"]},
                    "pricing": {"prompt": "0", "completion": "0"},
                },
                {
                    "id": "qwen/qwen3-coder:free",
                    "name": "Qwen3 Coder",
                    "architecture": {"modality": "text->text"},
                    "pricing": {"prompt": "0", "completion": "0"},
                },
            ]
        },
    )

    assert [route.model_id for route in routes] == [
        "mistralai/pixtral-12b:free",
        "qwen/qwen3-coder:free",
    ]
    assert routes[0].tags[0] == "text"
    assert "vision" in routes[0].tags


def test_catalog_payload_allows_multimodal_text_exchange_models():
    provider = FakeProvider("openrouter")
    routes = routes_from_payload(
        provider,
        {
            "data": [
                {
                    "id": "vision-chat:free",
                    "name": "Vision Chat",
                    "architecture": {"input_modalities": ["text", "image"], "output_modalities": ["text"]},
                    "pricing": {"prompt": "0", "completion": "0"},
                },
                {
                    "id": "plain-chat:free",
                    "name": "Plain Chat",
                    "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
                    "pricing": {"prompt": "0", "completion": "0"},
                },
            ]
        },
    )

    assert [route.model_id for route in routes] == ["vision-chat:free", "plain-chat:free"]


def test_malformed_or_missing_price_fields_do_not_count_as_free():
    provider = FakeProvider("openrouter")
    routes = routes_from_payload(
        provider,
        {
            "data": [
                _chat_model("missing-completion", {"prompt": "0"}),
                _chat_model("malformed-price", {"prompt": "0", "completion": "free"}),
                _chat_model("zero-price", {"prompt": "0", "completion": "0"}),
            ]
        },
    )

    assert [route.model_id for route in routes] == ["zero-price"]


def test_supervisor_verified_missing_pricing_routes_remain_disabled_by_default():
    provider = FakeProvider("gemini")
    route = route_from_catalog_item(
        provider,
        _chat_model("models/gemini-new-free", pricing=None),
        assume_free=True,
        supervisor_note="Supervisor found official free-tier evidence.",
    )

    assert route is not None
    assert route.model_id == "gemini-new-free"
    assert route.enabled is False
    assert "free-tier verification" in route.notes
    assert "Supervisor found official free-tier evidence." in route.notes


def _chat_model(model_id: str, pricing: dict[str, Any] | None) -> dict[str, Any]:
    item: dict[str, Any] = {
        "id": model_id,
        "name": model_id,
        "architecture": {"modality": "text->text"},
    }
    if pricing is not None:
        item["pricing"] = pricing
    return item
