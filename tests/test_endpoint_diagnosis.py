from __future__ import annotations

from typing import Any

import httpx

from app.endpoint_diagnosis import (
    EndpointSupervisor,
    EndpointDiagnosisService,
    SupervisorVerdict,
    _routes_from_payload,
)
from app.model_catalog import ModelCatalog, ModelRoute, route_id_for
from app.providers.base import ProviderError, ProviderRateLimited, ProviderResponse
from app.state import ProviderQuota, StateManager


class FakeCatalogProvider:
    def __init__(
        self,
        name: str,
        payload: dict[str, Any],
        configured: bool = True,
        *,
        missing_models: set[str] | None = None,
        rate_limited_models: set[str] | None = None,
        supervisor_verdicts: dict[str, SupervisorVerdict] | None = None,
    ) -> None:
        self.name = name
        self.api_key = "key" if configured else None
        self.base_url = f"https://example.test/{name}/v1"
        self.payload = payload
        self.max_context_tokens = None
        self.missing_models = missing_models or set()
        self.rate_limited_models = rate_limited_models or set()
        self.supervisor_verdicts = supervisor_verdicts or {}

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    async def list_models(self, client: httpx.AsyncClient) -> dict[str, Any]:
        return self.payload

    async def chat_completion(
        self,
        client: httpx.AsyncClient,
        payload: dict[str, Any],
        target_model: str | None = None,
    ) -> ProviderResponse:
        model = str(target_model or payload.get("model") or "")
        if model in self.rate_limited_models:
            raise ProviderRateLimited("rate limited", status_code=429, headers={}, body="")
        if model in self.missing_models:
            raise ProviderError(
                "missing model",
                status_code=404,
                headers={},
                body='{"error":{"message":"model not found"}}',
            )
        if self.supervisor_verdicts:
            user_text = " ".join(
                str(message.get("content") or "")
                for message in payload.get("messages", [])
                if isinstance(message, dict)
            )
            for target_model_id, verdict in self.supervisor_verdicts.items():
                if target_model_id in user_text:
                    return ProviderResponse(
                        self.name,
                        200,
                        {},
                        {
                            "choices": [
                                {
                                    "message": {
                                        "content": (
                                            '{"free_chat_model": '
                                            f"{str(verdict.free_chat_model).lower()}, "
                                            f'"confidence": "{verdict.confidence}", '
                                            f'"reason": "{verdict.reason}", '
                                            f'"sources": {verdict.sources!r}'
                                            "}"
                                        ).replace("'", '"')
                                    }
                                }
                            ]
                        },
                    )
        return ProviderResponse(self.name, 200, {}, {"id": "probe", "choices": []})


class FakeSupervisor:
    def __init__(self, verdicts: dict[str, SupervisorVerdict]) -> None:
        self.verdicts = verdicts
        self.calls: list[tuple[str, str]] = []

    async def verify_free_chat_model(
        self,
        client: httpx.AsyncClient,
        *,
        provider_name: str,
        model_id: str,
        display_name: str,
    ) -> SupervisorVerdict:
        self.calls.append((provider_name, model_id))
        return self.verdicts.get(
            model_id,
            SupervisorVerdict(False, "low", "No official free-tier evidence."),
        )


def test_openrouter_catalog_payload_creates_free_routes_only():
    provider = FakeCatalogProvider("openrouter", {})
    routes = _routes_from_payload(
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
                    "pricing": {"prompt": "1", "completion": "1"},
                },
            ]
        },
    )

    assert [route.model_id for route in routes] == ["new/free-model:free"]
    assert routes[0].display_name == "New Free Model"
    assert routes[0].context_window == 65536
    assert "text" in routes[0].tags


def test_non_openrouter_catalog_payload_requires_structured_free_pricing():
    provider = FakeCatalogProvider("groq", {})
    routes = _routes_from_payload(
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


def test_openrouter_catalog_payload_excludes_non_chat_free_models():
    provider = FakeCatalogProvider("openrouter", {})
    routes = _routes_from_payload(
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
                    "id": "qwen/qwen3-coder:free",
                    "name": "Qwen3 Coder",
                    "architecture": {"modality": "text->text"},
                    "pricing": {"prompt": "0", "completion": "0"},
                },
            ]
        },
    )

    assert [route.model_id for route in routes] == ["qwen/qwen3-coder:free"]
    assert routes[0].tags[0] == "text"


async def test_endpoint_diagnosis_suggests_openrouter_free_routes_and_stale_routes(tmp_path):
    state = StateManager(
        str(tmp_path / "state.sqlite3"),
        [ProviderQuota("openrouter", tokens_per_day=None, requests_per_day=None, requests_per_minute=None)],
    )
    await state.initialize()

    catalog = ModelCatalog(str(tmp_path / "models.json"))
    catalog.replace_routes(
        [
            {
                "route_id": "openrouter-existing-free",
                "provider_name": "openrouter",
                "model_id": "existing/free:free",
                "display_name": "Existing Free",
                "rank": 1,
                "enabled": True,
            },
            {
                "route_id": "openrouter-stale-free",
                "provider_name": "openrouter",
                "model_id": "stale/free:free",
                "display_name": "Stale Free",
                "rank": 2,
                "enabled": True,
            },
        ]
    )

    provider = FakeCatalogProvider(
        "openrouter",
        {
            "data": [
                {
                    "id": "existing/free:free",
                    "name": "Existing Free",
                    "context_length": 32768,
                    "pricing": {"prompt": "0", "completion": "0"},
                },
                {
                    "id": "new/llama-chat:free",
                    "name": "New Llama Chat",
                    "context_length": 65536,
                    "architecture": {"modality": "text->text"},
                    "pricing": {"prompt": "0", "completion": "0"},
                },
            ]
        },
        missing_models={"stale/free:free"},
    )
    service = EndpointDiagnosisService([provider], catalog, state, request_timeout_seconds=5)

    report = await service.run_once()
    stale_state = await state.get_route_state(
        "openrouter-stale-free",
        "openrouter",
        "stale/free:free",
    )

    assert report.providers[0].ok is True
    assert report.providers[0].new_route_suggestion_count == 1
    assert report.providers[0].stale_route_suggestion_count == 1
    assert {suggestion.action for suggestion in report.suggestions} == {"add_route", "remove_route"}
    assert not any(route.model_id == "new/llama-chat:free" for route in catalog.all_routes())
    assert stale_state.status == "active"


async def test_endpoint_diagnosis_uses_supervisor_for_missing_pricing_discovery(tmp_path):
    state = StateManager(
        str(tmp_path / "state.sqlite3"),
        [ProviderQuota("groq", tokens_per_day=None, requests_per_day=None, requests_per_minute=None)],
    )
    await state.initialize()

    catalog = ModelCatalog(str(tmp_path / "models.json"))
    catalog.replace_routes([])
    provider = FakeCatalogProvider(
        "groq",
        {
            "data": [
                {
                    "id": "llama-free-70b",
                    "name": "Llama Free 70B",
                    "context_length": 131072,
                    "architecture": {"modality": "text->text"},
                },
                {
                    "id": "llama-unknown-70b",
                    "name": "Llama Unknown 70B",
                    "context_length": 131072,
                    "architecture": {"modality": "text->text"},
                },
            ]
        },
    )
    supervisor = FakeSupervisor(
        {
            "llama-free-70b": SupervisorVerdict(
                True,
                "high",
                "Official docs list this as free tier.",
                ["https://example.test/groq/free"],
            ),
            "llama-unknown-70b": SupervisorVerdict(False, "medium", "Unclear."),
        }
    )
    service = EndpointDiagnosisService(
        [provider],
        catalog,
        state,
        request_timeout_seconds=5,
        supervisor=supervisor,
    )

    report = await service.run_once()

    assert supervisor.calls == [("groq", "llama-free-70b"), ("groq", "llama-unknown-70b")]
    assert [suggestion.model_id for suggestion in report.suggestions] == ["llama-free-70b"]
    assert "Supervisor web search verified" in str(report.suggestions[0].route["notes"])


async def test_local_endpoint_supervisor_uses_enabled_free_catalog_route(tmp_path):
    state = StateManager(
        str(tmp_path / "state.sqlite3"),
        [ProviderQuota("groq", tokens_per_day=None, requests_per_day=None, requests_per_minute=None)],
    )
    await state.initialize()

    catalog = ModelCatalog(str(tmp_path / "models.json"))
    catalog.replace_routes(
        [
            {
                "route_id": "groq-supervisor",
                "provider_name": "groq",
                "model_id": "supervisor-model",
                "display_name": "Supervisor Model",
                "rank": 1,
                "enabled": True,
                "tags": ["text", "web-search"],
            },
        ]
    )
    provider = FakeCatalogProvider(
        "groq",
        {"data": []},
        supervisor_verdicts={
            "llama-free-70b": SupervisorVerdict(
                True,
                "high",
                "Official docs list this as free tier.",
                ["https://example.test/groq/free"],
            )
        },
    )
    supervisor = EndpointSupervisor(
        enabled=True,
        providers=[provider],
        catalog=catalog,
        state=state,
    )

    async with httpx.AsyncClient() as client:
        verdict = await supervisor.verify_free_chat_model(
            client,
            provider_name="groq",
            model_id="llama-free-70b",
            display_name="Llama Free 70B",
        )

    assert verdict.free_chat_model is True
    assert verdict.confidence == "high"


async def test_endpoint_diagnosis_does_not_mark_openrouter_router_route_stale(tmp_path):
    state = StateManager(
        str(tmp_path / "state.sqlite3"),
        [ProviderQuota("openrouter", tokens_per_day=None, requests_per_day=None, requests_per_minute=None)],
    )
    await state.initialize()

    catalog = ModelCatalog(str(tmp_path / "models.json"))
    catalog.replace_routes(
        [
            {
                "route_id": "openrouter-openrouter-free",
                "provider_name": "openrouter",
                "model_id": "openrouter/free",
                "display_name": "Free Models Router",
                "rank": 1,
                "enabled": True,
            },
        ]
    )
    provider = FakeCatalogProvider("openrouter", {"data": []})
    service = EndpointDiagnosisService([provider], catalog, state, request_timeout_seconds=5)

    report = await service.run_once()

    assert report.providers[0].stale_route_suggestion_count == 0
    assert report.suggestions == []


async def test_endpoint_diagnosis_does_not_mark_gemini_models_stale_when_catalog_omits_them(tmp_path):
    state = StateManager(
        str(tmp_path / "state.sqlite3"),
        [ProviderQuota("gemini", tokens_per_day=None, requests_per_day=None, requests_per_minute=None)],
    )
    await state.initialize()

    catalog = ModelCatalog(str(tmp_path / "models.json"))
    catalog.replace_routes(
        [
            {
                "route_id": "gemini-gemini-3-1-pro-preview",
                "provider_name": "gemini",
                "model_id": "gemini-3.1-pro-preview",
                "display_name": "Gemini 3.1 Pro Preview",
                "rank": 1,
                "enabled": True,
            },
        ]
    )
    provider = FakeCatalogProvider(
        "gemini",
        {
            "data": [
                {"id": "models/gemini-2.5-flash", "name": "Gemini 2.5 Flash"},
            ]
        },
    )
    service = EndpointDiagnosisService([provider], catalog, state, request_timeout_seconds=5)

    report = await service.run_once()
    route_state = await state.get_route_state(
        "gemini-gemini-3-1-pro-preview",
        "gemini",
        "gemini-3.1-pro-preview",
    )

    assert report.providers[0].ok is True
    assert report.providers[0].stale_route_suggestion_count == 0
    assert report.suggestions == []
    assert route_state.status == "active"


async def test_endpoint_diagnosis_marks_other_provider_stale_only_after_missing_probe(tmp_path):
    state = StateManager(
        str(tmp_path / "state.sqlite3"),
        [ProviderQuota("groq", tokens_per_day=None, requests_per_day=None, requests_per_minute=None)],
    )
    await state.initialize()

    catalog = ModelCatalog(str(tmp_path / "models.json"))
    catalog.replace_routes(
        [
            {
                "route_id": "groq-old-model",
                "provider_name": "groq",
                "model_id": "old-model",
                "display_name": "Old Model",
                "rank": 1,
                "enabled": True,
            },
        ]
    )
    provider = FakeCatalogProvider("groq", {"data": []}, missing_models={"old-model"})
    service = EndpointDiagnosisService([provider], catalog, state, request_timeout_seconds=5)

    report = await service.run_once()

    assert report.providers[0].stale_route_suggestion_count == 1
    assert [suggestion.action for suggestion in report.suggestions] == ["remove_route"]


async def test_endpoint_diagnosis_does_not_mark_other_provider_stale_when_probe_is_rate_limited(tmp_path):
    state = StateManager(
        str(tmp_path / "state.sqlite3"),
        [ProviderQuota("groq", tokens_per_day=None, requests_per_day=None, requests_per_minute=None)],
    )
    await state.initialize()

    catalog = ModelCatalog(str(tmp_path / "models.json"))
    catalog.replace_routes(
        [
            {
                "route_id": "groq-rate-limited-model",
                "provider_name": "groq",
                "model_id": "rate-limited-model",
                "display_name": "Rate Limited Model",
                "rank": 1,
                "enabled": True,
            },
        ]
    )
    provider = FakeCatalogProvider("groq", {"data": []}, rate_limited_models={"rate-limited-model"})
    service = EndpointDiagnosisService([provider], catalog, state, request_timeout_seconds=5)

    report = await service.run_once()

    assert report.providers[0].stale_route_suggestion_count == 0
    assert report.suggestions == []


async def test_endpoint_diagnosis_matches_provider_prefixed_model_ids(tmp_path):
    state = StateManager(
        str(tmp_path / "state.sqlite3"),
        [ProviderQuota("gemini", tokens_per_day=None, requests_per_day=None, requests_per_minute=None)],
    )
    await state.initialize()

    catalog = ModelCatalog(str(tmp_path / "models.json"))
    catalog.replace_routes(
        [
            {
                "route_id": "gemini-gemini-2-5-flash",
                "provider_name": "gemini",
                "model_id": "gemini-2.5-flash",
                "display_name": "Gemini 2.5 Flash",
                "rank": 1,
                "enabled": True,
            },
        ]
    )
    provider = FakeCatalogProvider("gemini", {"data": [{"id": "models/gemini-2.5-flash"}]})
    service = EndpointDiagnosisService([provider], catalog, state, request_timeout_seconds=5)

    report = await service.run_once()

    assert report.providers[0].confirmed_route_count == 1
    assert report.suggestions == []


async def test_endpoint_diagnosis_applies_selected_suggestions_only(tmp_path):
    state = StateManager(
        str(tmp_path / "state.sqlite3"),
        [ProviderQuota("openrouter", tokens_per_day=None, requests_per_day=None, requests_per_minute=None)],
    )
    await state.initialize()

    catalog = ModelCatalog(str(tmp_path / "models.json"))
    catalog.replace_routes(
        [
            {
                "route_id": "openrouter-stale-free",
                "provider_name": "openrouter",
                "model_id": "stale/free:free",
                "display_name": "Stale Free",
                "rank": 1,
                "enabled": True,
            },
        ]
    )
    provider = FakeCatalogProvider(
        "openrouter",
        {
            "data": [
                {
                    "id": "new/llama-chat:free",
                    "name": "New Llama Chat",
                    "context_length": 65536,
                    "architecture": {"modality": "text->text"},
                    "pricing": {"prompt": "0", "completion": "0"},
                },
            ]
        },
        missing_models={"stale/free:free"},
    )
    service = EndpointDiagnosisService([provider], catalog, state, request_timeout_seconds=5)
    report = await service.run_once()

    add_id = next(item.suggestion_id for item in report.suggestions if item.action == "add_route")
    applied = await service.apply_suggestions([add_id])
    stale_state = await state.get_route_state(
        "openrouter-stale-free",
        "openrouter",
        "stale/free:free",
    )

    assert [item.suggestion_id for item in applied] == [add_id]
    assert any(route.model_id == "new/llama-chat:free" for route in catalog.all_routes())
    assert stale_state.status == "active"
    assert service.last_report is not None
    assert {item.action for item in service.last_report.suggestions} == {"remove_route"}


async def test_endpoint_diagnosis_applies_remove_route_suggestion(tmp_path):
    state = StateManager(
        str(tmp_path / "state.sqlite3"),
        [ProviderQuota("groq", tokens_per_day=None, requests_per_day=None, requests_per_minute=None)],
    )
    await state.initialize()

    catalog = ModelCatalog(str(tmp_path / "models.json"))
    catalog.replace_routes(
        [
            {
                "route_id": "groq-dead-model",
                "provider_name": "groq",
                "model_id": "dead-model",
                "display_name": "Dead Model",
                "rank": 1,
                "enabled": True,
            },
        ]
    )
    provider = FakeCatalogProvider("groq", {"data": []}, missing_models={"dead-model"})
    service = EndpointDiagnosisService([provider], catalog, state, request_timeout_seconds=5)
    report = await service.run_once()

    remove_id = next(item.suggestion_id for item in report.suggestions if item.action == "remove_route")
    applied = await service.apply_suggestions([remove_id])

    assert [item.suggestion_id for item in applied] == [remove_id]
    assert catalog.all_routes() == []
    assert service.last_report is not None
    assert service.last_report.suggestions == []


async def test_approved_added_routes_are_promoted_to_reset_defaults(tmp_path):
    state = StateManager(
        str(tmp_path / "state.sqlite3"),
        [ProviderQuota("openrouter", tokens_per_day=None, requests_per_day=None, requests_per_minute=None)],
    )
    await state.initialize()

    catalog = ModelCatalog(str(tmp_path / "models.json"))
    catalog.initialize()
    provider = FakeCatalogProvider(
        "openrouter",
        {
            "data": [
                {
                    "id": "unit-test/new-llama-chat:free",
                    "name": "Unit Test New Llama Chat",
                    "context_length": 65536,
                    "architecture": {"modality": "text->text"},
                    "pricing": {"prompt": "0", "completion": "0"},
                },
            ]
        },
    )
    service = EndpointDiagnosisService([provider], catalog, state, request_timeout_seconds=5)
    report = await service.run_once()
    add_id = next(item.suggestion_id for item in report.suggestions if item.action == "add_route")

    await service.apply_suggestions([add_id])
    reset_routes = catalog.reset_to_defaults()

    assert any(route.route_id == route_id_for("openrouter", "unit-test/new-llama-chat:free") for route in reset_routes)
