from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass, field
from time import time
from typing import Any

import httpx

from app.model_catalog import (
    ModelCatalog,
    ModelRoute,
    promote_routes_to_default_catalog,
    remove_routes_from_default_catalog,
    route_from_discovered_model,
)
from app.providers.base import ProviderAdapter, ProviderError, ProviderRateLimited
from app.router import _looks_like_missing_endpoint
from app.state import StateManager


@dataclass(frozen=True)
class EndpointSuggestion:
    suggestion_id: str
    action: str
    provider_name: str
    model_id: str
    route_id: str
    title: str
    details: str
    route: dict[str, Any] | None = None


@dataclass(frozen=True)
class ProviderDiagnosis:
    provider_name: str
    configured: bool
    ok: bool
    discovered_model_count: int = 0
    new_route_suggestion_count: int = 0
    confirmed_route_count: int = 0
    stale_route_suggestion_count: int = 0
    recovered_route_suggestion_count: int = 0
    error: str | None = None


@dataclass(frozen=True)
class DiagnosisReport:
    checked_at: int
    providers: list[ProviderDiagnosis]
    suggestions: list[EndpointSuggestion] = field(default_factory=list)


@dataclass(frozen=True)
class SupervisorVerdict:
    free_chat_model: bool
    confidence: str
    reason: str
    sources: list[str] = field(default_factory=list)


class EndpointSupervisor:
    """Optional local supervisor for provider catalogs that omit pricing."""

    def __init__(
        self,
        *,
        enabled: bool,
        providers: list[ProviderAdapter],
        catalog: ModelCatalog,
        state: StateManager,
        preferred_model: str | None = None,
    ) -> None:
        self.enabled = enabled
        self.providers = {provider.name: provider for provider in providers}
        self.catalog = catalog
        self.state = state
        self.preferred_model = preferred_model

    async def verify_free_chat_model(
        self,
        client: httpx.AsyncClient,
        *,
        provider_name: str,
        model_id: str,
        display_name: str,
    ) -> SupervisorVerdict:
        if not self.enabled:
            return SupervisorVerdict(False, "none", "Supervisor disabled.")

        route = self._select_supervisor_route()
        if route is None:
            return SupervisorVerdict(False, "none", "No enabled free supervisor route is available.")

        provider = self.providers.get(route.provider_name)
        if provider is None or not provider.is_configured:
            return SupervisorVerdict(False, "none", "Supervisor provider is not configured.")

        availability = await self.state.check_available(provider.name, estimated_tokens=800)
        if not availability.available:
            return SupervisorVerdict(False, "none", f"Supervisor unavailable: {availability.reason}.")

        prompt = (
            "Determine whether this provider model is currently available as a no-cost/free-tier "
            "chat or text-generation LLM endpoint through the provider API. If you have web search "
            "or browsing tools, use official provider documentation, pricing pages, model catalogs, "
            "or API docs. If you do not have enough reliable evidence, answer false. Do not "
            "count free trials, paid-tier-only models, non-chat models, embeddings, image/video/audio "
            "generators, moderation, rerankers, or examples that do not establish free-tier API usage.\n\n"
            f"Provider: {provider_name}\n"
            f"Model ID: {model_id}\n"
            f"Display name: {display_name}\n\n"
            "Return only JSON with this schema: "
            '{"free_chat_model": boolean, "confidence": "high|medium|low", '
            '"reason": string, "sources": [string]}. '
            "Set free_chat_model true only when official/current sources clearly support it."
        )
        try:
            response = await provider.chat_completion(
                client,
                {
                    "model": "auto",
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a conservative endpoint-pricing verifier. Return only JSON.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0,
                    "max_tokens": 500,
                },
                route.model_id,
            )
        except ProviderRateLimited as exc:
            await self.state.mark_exhausted(
                provider.name,
                headers=exc.headers,
                status_code=exc.status_code,
            )
            return SupervisorVerdict(False, "none", "Supervisor route is rate limited.")
        except (ProviderError, httpx.RequestError, httpx.TimeoutException) as exc:
            return SupervisorVerdict(False, "none", f"Supervisor failed: {exc.__class__.__name__}.")

        raw_text = _chat_response_text(response.body)
        parsed = _json_object_from_text(raw_text)
        if not isinstance(parsed, dict):
            return SupervisorVerdict(False, "none", "Supervisor returned unparsable output.")

        free_chat_model = parsed.get("free_chat_model") is True
        confidence = str(parsed.get("confidence") or "low").lower()
        sources = [str(source) for source in parsed.get("sources", []) if isinstance(source, str)]
        reason = str(parsed.get("reason") or "No reason provided.")
        if confidence != "high":
            free_chat_model = False
        return SupervisorVerdict(free_chat_model, confidence, reason, sources)

    def _select_supervisor_route(self) -> ModelRoute | None:
        routes = [
            route
            for route in self.catalog.enabled_routes(self.preferred_model)
            if "text" in route.tags
            and not any(tag in route.tags for tag in ("safety", "moderation", "classification"))
            and route.provider_name in self.providers
        ]
        if not routes:
            return None
        return sorted(
            routes,
            key=lambda route: (
                "web-search" in route.tags,
                "tool-use" in route.tags,
                -route.rank,
            ),
            reverse=True,
        )[0]


class EndpointDiagnosisService:
    """Finds provider availability changes and stores them as user-reviewable suggestions."""

    def __init__(
        self,
        providers: list[ProviderAdapter],
        catalog: ModelCatalog,
        state: StateManager,
        *,
        request_timeout_seconds: float,
        supervisor: EndpointSupervisor | None = None,
    ) -> None:
        self.providers = providers
        self.catalog = catalog
        self.state = state
        self.request_timeout_seconds = request_timeout_seconds
        self.supervisor = supervisor or EndpointSupervisor(
            enabled=False,
            providers=[],
            catalog=catalog,
            state=state,
        )
        self._lock = asyncio.Lock()
        self.last_report: DiagnosisReport | None = None

    async def run_once(self) -> DiagnosisReport:
        async with self._lock:
            provider_reports: list[ProviderDiagnosis] = []
            all_suggestions: list[EndpointSuggestion] = []

            timeout = httpx.Timeout(self.request_timeout_seconds)
            async with httpx.AsyncClient(timeout=timeout) as client:
                for provider in self.providers:
                    report, suggestions = await self._diagnose_provider(client, provider)
                    provider_reports.append(report)
                    all_suggestions.extend(suggestions)

            report = DiagnosisReport(
                checked_at=int(time()),
                providers=provider_reports,
                suggestions=all_suggestions,
            )
            self.last_report = report
            return report

    async def apply_suggestions(self, suggestion_ids: list[str]) -> list[EndpointSuggestion]:
        async with self._lock:
            if self.last_report is None:
                return []

            selected_ids = set(suggestion_ids)
            selected = [
                suggestion
                for suggestion in self.last_report.suggestions
                if suggestion.suggestion_id in selected_ids
            ]

            for suggestion in selected:
                if suggestion.action == "add_route" and suggestion.route is not None:
                    added = self.catalog.add_discovered_routes([ModelCatalog._route_from_dict(suggestion.route)])
                    promote_routes_to_default_catalog(added)
                elif suggestion.action == "remove_route":
                    removed = self.catalog.remove_routes({suggestion.route_id})
                    remove_routes_from_default_catalog({route.route_id for route in removed})
                elif suggestion.action == "clear_stale":
                    await self.state.clear_route_health(
                        suggestion.route_id,
                        suggestion.provider_name,
                        suggestion.model_id,
                    )

            remaining = [
                suggestion
                for suggestion in self.last_report.suggestions
                if suggestion.suggestion_id not in selected_ids
            ]
            self.last_report = DiagnosisReport(
                checked_at=self.last_report.checked_at,
                providers=self.last_report.providers,
                suggestions=remaining,
            )
            return selected

    async def _diagnose_provider(
        self,
        client: httpx.AsyncClient,
        provider: ProviderAdapter,
    ) -> tuple[ProviderDiagnosis, list[EndpointSuggestion]]:
        if not provider.is_configured:
            return ProviderDiagnosis(provider.name, configured=False, ok=False, error="missing_api_key"), []

        availability = await self.state.check_available(provider.name)
        if not availability.available:
            return (
                ProviderDiagnosis(
                    provider.name,
                    configured=True,
                    ok=False,
                    error=availability.reason or "provider_unavailable",
                ),
                [],
            )

        try:
            payload = await provider.list_models(client)
        except ProviderRateLimited as exc:
            await self.state.mark_exhausted(
                provider.name,
                headers=exc.headers,
                status_code=exc.status_code,
            )
            return (
                ProviderDiagnosis(
                    provider.name,
                    configured=True,
                    ok=False,
                    error=f"rate_limited_{exc.status_code or 429}",
                ),
                [],
            )
        except (ProviderError, httpx.RequestError, httpx.TimeoutException) as exc:
            return ProviderDiagnosis(provider.name, configured=True, ok=False, error=exc.__class__.__name__), []

        raw_discovered_ids = _raw_model_ids_from_payload(payload)
        discovered_ids = _model_id_variants_from_ids(raw_discovered_ids)
        discovered_routes = await self._routes_from_payload(client, provider, payload)
        routeable_ids = _model_ids_from_routes(discovered_routes)
        new_route_suggestions = self._new_route_suggestions(discovered_routes)
        confirmed, stale_suggestions, recovered_suggestions = await self._route_health_suggestions(
            client,
            provider,
            discovered_ids,
            routeable_ids,
        )
        suggestions = new_route_suggestions + stale_suggestions + recovered_suggestions

        return (
            ProviderDiagnosis(
                provider.name,
                configured=True,
                ok=True,
                discovered_model_count=len(raw_discovered_ids),
                new_route_suggestion_count=len(new_route_suggestions),
                confirmed_route_count=confirmed,
                stale_route_suggestion_count=len(stale_suggestions),
                recovered_route_suggestion_count=len(recovered_suggestions),
            ),
            suggestions,
        )

    def _new_route_suggestions(self, discovered_routes: list[ModelRoute]) -> list[EndpointSuggestion]:
        existing_ids = {route.route_id for route in self.catalog.all_routes()}
        existing_targets = {(route.provider_name, route.model_id) for route in self.catalog.all_routes()}
        suggestions: list[EndpointSuggestion] = []

        for route in discovered_routes:
            if route.route_id in existing_ids or (route.provider_name, route.model_id) in existing_targets:
                continue
            suggestions.append(
                EndpointSuggestion(
                    suggestion_id=f"add:{route.route_id}",
                    action="add_route",
                    provider_name=route.provider_name,
                    model_id=route.model_id,
                    route_id=route.route_id,
                    title=f"Add {route.display_name}",
                    details="New free route discovered in the provider model catalog.",
                    route=asdict(route),
                )
            )
        return suggestions

    async def _routes_from_payload(
        self,
        client: httpx.AsyncClient,
        provider: ProviderAdapter,
        payload: dict[str, Any],
    ) -> list[ModelRoute]:
        routes: list[ModelRoute] = []
        for item in payload.get("data", []):
            if not isinstance(item, dict):
                continue
            route = await self._route_from_catalog_item(client, provider, item)
            if route is not None:
                routes.append(route)
        return routes

    async def _route_from_catalog_item(
        self,
        client: httpx.AsyncClient,
        provider: ProviderAdapter,
        item: dict[str, Any],
    ) -> ModelRoute | None:
        model_id = item.get("id")
        if not isinstance(model_id, str) or not model_id:
            return None
        model_id = _route_model_id_from_catalog_id(model_id)
        display_name = str(item.get("name") or model_id)

        if not _is_chat_model(item):
            return None

        supervisor_note = ""
        if _is_discoverable_free_model(provider, item):
            supervisor_verified = False
        else:
            verdict = await self.supervisor.verify_free_chat_model(
                client,
                provider_name=provider.name,
                model_id=model_id,
                display_name=display_name,
            )
            if not verdict.free_chat_model:
                return None
            supervisor_verified = True
            supervisor_note = _supervisor_note(verdict)

        if provider.name == "openrouter":
            enabled = True
            notes = "Discovered automatically from OpenRouter /api/v1/models as a zero-cost or :free model."
        else:
            enabled = False
            notes = (
                "Discovered automatically after free-tier verification. Disabled by default until reviewed "
                "because provider free-tier limits can still vary by account."
            )
        if supervisor_verified:
            notes = f"{notes} {supervisor_note}".strip()

        return route_from_discovered_model(
            provider.name,
            model_id,
            display_name=display_name,
            context_window=_int_or_none(item.get("context_length")),
            tags=_tags_for_model(item),
            source_url=f"{provider.base_url.rstrip('/')}/models",
            notes=notes,
            enabled=enabled,
        )

    async def _route_health_suggestions(
        self,
        client: httpx.AsyncClient,
        provider: ProviderAdapter,
        discovered_ids: set[str],
        routeable_ids: set[str],
    ) -> tuple[int, list[EndpointSuggestion], list[EndpointSuggestion]]:
        confirmed = 0
        stale: list[EndpointSuggestion] = []
        recovered: list[EndpointSuggestion] = []
        for route in self.catalog.all_routes():
            if route.provider_name != provider.name:
                continue
            presence_ids = _presence_ids_for_route(provider.name, route, discovered_ids, routeable_ids)
            route_state = await self.state.get_route_state(
                route.route_id,
                route.provider_name,
                route.model_id,
            )
            if _model_id_in_catalog(route.model_id, presence_ids):
                if route_state.status == "potentially_outdated":
                    recovered.append(
                        EndpointSuggestion(
                            suggestion_id=f"clear_stale:{route.route_id}",
                            action="clear_stale",
                            provider_name=route.provider_name,
                            model_id=route.model_id,
                            route_id=route.route_id,
                            title=f"Restore {route.display_name}",
                            details="This route appears in the provider catalog again.",
                        )
                    )
                confirmed += 1
                continue

            probe_result = await self._probe_missing_route(client, provider, route)
            if probe_result == "available":
                if route_state.status == "potentially_outdated":
                    recovered.append(
                        EndpointSuggestion(
                            suggestion_id=f"clear_stale:{route.route_id}",
                            action="clear_stale",
                            provider_name=route.provider_name,
                            model_id=route.model_id,
                            route_id=route.route_id,
                            title=f"Restore {route.display_name}",
                            details="A direct model probe succeeded even though the catalog did not list this route.",
                        )
                    )
                confirmed += 1
                continue

            if probe_result == "missing":
                stale.append(
                    EndpointSuggestion(
                        suggestion_id=f"remove:{route.route_id}",
                        action="remove_route",
                        provider_name=route.provider_name,
                        model_id=route.model_id,
                        route_id=route.route_id,
                        title=f"Remove {route.display_name}",
                        details="This route was absent from the provider catalog and a direct model probe returned a missing-model error.",
                    )
                )
        return confirmed, stale, recovered

    async def _probe_missing_route(
        self,
        client: httpx.AsyncClient,
        provider: ProviderAdapter,
        route: ModelRoute,
    ) -> str:
        if not route.enabled:
            return "inconclusive"
        if provider.name == "openrouter" and not route.model_id.endswith(":free"):
            return "inconclusive"

        availability = await self.state.check_available(provider.name, estimated_tokens=8)
        if not availability.available:
            return "inconclusive"

        try:
            await provider.chat_completion(client, _probe_payload(), route.model_id)
        except ProviderRateLimited as exc:
            await self.state.mark_exhausted(
                provider.name,
                headers=exc.headers,
                status_code=exc.status_code,
            )
            return "inconclusive"
        except httpx.TimeoutException:
            return "inconclusive"
        except httpx.RequestError:
            return "inconclusive"
        except ProviderError as exc:
            return "missing" if _looks_like_missing_endpoint(exc) else "inconclusive"
        return "available"


class BackgroundEndpointDiagnosis:
    def __init__(
        self,
        service: EndpointDiagnosisService,
        *,
        interval_seconds: int,
        startup_delay_seconds: int,
    ) -> None:
        self.service = service
        self.interval_seconds = max(60, interval_seconds)
        self.startup_delay_seconds = max(0, startup_delay_seconds)
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass

    async def _run(self) -> None:
        await asyncio.sleep(self.startup_delay_seconds)
        while True:
            try:
                await self.service.run_once()
            except Exception:
                # The request path must never depend on background refresh health.
                pass
            await asyncio.sleep(self.interval_seconds)


def _model_ids_from_payload(payload: dict[str, Any]) -> set[str]:
    return _model_id_variants_from_ids(_raw_model_ids_from_payload(payload))


def _raw_model_ids_from_payload(payload: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for item in payload.get("data", []):
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        if isinstance(model_id, str) and model_id:
            ids.add(model_id.strip())
    return ids


def _model_id_variants_from_ids(model_ids: set[str]) -> set[str]:
    ids: set[str] = set()
    for model_id in model_ids:
        ids.update(_model_id_variants(model_id))
    return ids


def _model_ids_from_routes(routes: list[ModelRoute]) -> set[str]:
    ids: set[str] = set()
    for route in routes:
        ids.update(_model_id_variants(route.model_id))
    return ids


def _route_can_safely_be_marked_missing(provider_name: str, route: ModelRoute) -> bool:
    return provider_name == "openrouter" and route.model_id.endswith(":free")


def _presence_ids_for_route(
    provider_name: str,
    route: ModelRoute,
    discovered_ids: set[str],
    routeable_ids: set[str],
) -> set[str]:
    if _route_can_safely_be_marked_missing(provider_name, route):
        return routeable_ids
    return discovered_ids


def _model_id_in_catalog(model_id: str, discovered_ids: set[str]) -> bool:
    return any(variant in discovered_ids for variant in _model_id_variants(model_id))


def _model_id_variants(model_id: str) -> set[str]:
    normalized = model_id.strip()
    if not normalized:
        return set()

    variants = {normalized}
    if normalized.startswith("models/"):
        variants.add(normalized.removeprefix("models/"))
    else:
        variants.add(f"models/{normalized}")
    return variants


def _probe_payload() -> dict[str, Any]:
    return {
        "model": "auto",
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
    }


def _routes_from_payload(provider: ProviderAdapter, payload: dict[str, Any]) -> list[ModelRoute]:
    routes: list[ModelRoute] = []
    for item in payload.get("data", []):
        if not isinstance(item, dict):
            continue
        route = _route_from_structured_free_catalog_item(provider, item)
        if route is not None:
            routes.append(route)
    return routes


def _route_from_structured_free_catalog_item(
    provider: ProviderAdapter,
    item: dict[str, Any],
) -> ModelRoute | None:
    model_id = item.get("id")
    if not isinstance(model_id, str) or not model_id:
        return None
    model_id = _route_model_id_from_catalog_id(model_id)

    if not _is_discoverable_free_model(provider, item) or not _is_chat_model(item):
        return None

    if provider.name == "openrouter":
        enabled = True
        notes = "Discovered automatically from OpenRouter /api/v1/models as a zero-cost or :free model."
    else:
        enabled = False
        notes = (
            "Discovered automatically from structured zero pricing in the provider /models endpoint. "
            "Disabled by default until reviewed because provider free-tier limits can still vary by account."
        )

    return route_from_discovered_model(
        provider.name,
        model_id,
        display_name=str(item.get("name") or model_id),
        context_window=_int_or_none(item.get("context_length")),
        tags=_tags_for_model(item),
        source_url=f"{provider.base_url.rstrip('/')}/models",
        notes=notes,
        enabled=enabled,
    )


def _is_discoverable_free_model(provider: ProviderAdapter, item: dict[str, Any]) -> bool:
    if provider.name == "openrouter":
        return _is_openrouter_free_model(item)
    return _has_structured_zero_price(item)


def _is_openrouter_free_model(item: dict[str, Any]) -> bool:
    model_id = item.get("id")
    if isinstance(model_id, str) and model_id.endswith(":free"):
        return True
    return _has_structured_zero_price(item)


def _has_structured_zero_price(item: dict[str, Any]) -> bool:
    pricing = item.get("pricing")
    if not isinstance(pricing, dict):
        return False

    price_keys = (
        "prompt",
        "completion",
        "input",
        "output",
        "input_price",
        "output_price",
        "prompt_price",
        "completion_price",
    )
    prices = [_float_or_none(pricing.get(key)) for key in price_keys if key in pricing]
    if not prices or any(price is None for price in prices):
        return False
    return all(price == 0 for price in prices)


def _chat_response_text(payload: dict[str, Any]) -> str:
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


def _json_object_from_text(text: str) -> dict[str, Any] | None:
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


def _supervisor_note(verdict: SupervisorVerdict) -> str:
    source_text = ", ".join(verdict.sources[:3])
    if source_text:
        return f"Supervisor web search verified free-tier status: {verdict.reason} Sources: {source_text}"
    return f"Supervisor web search verified free-tier status: {verdict.reason}"


def _route_model_id_from_catalog_id(model_id: str) -> str:
    return model_id.removeprefix("models/")


def _tags_for_model(item: dict[str, Any]) -> list[str]:
    text = _model_search_text(item)

    tags = ["text"] if _is_chat_model(item) else []
    if any(term in text for term in ("vision", "image", "vl", "ocr", "pixtral", "gemma-3")):
        tags.append("vision")
    if any(term in text for term in ("audio", "lyria", "speech", "music")):
        tags.append("audio")
    if any(term in text for term in ("reason", "thinking", "qwen", "hermes", "gpt-oss")):
        tags.append("reasoning")
    if any(term in text for term in ("coder", "code")):
        tags.append("coding")
    if "ocr" in text:
        tags.append("classification")
    return list(dict.fromkeys(tags))


def _is_chat_model(item: dict[str, Any]) -> bool:
    text = _model_search_text(item)
    if any(term in text for term in _NON_CHAT_MODEL_TERMS):
        return False

    architecture = item.get("architecture")
    if isinstance(architecture, dict):
        input_modalities = _string_list(architecture.get("input_modalities"))
        output_modalities = _string_list(architecture.get("output_modalities"))
        if output_modalities and "text" not in output_modalities:
            return False
        modality = str(architecture.get("modality") or "").lower()
        if modality:
            parts = [part.strip() for part in modality.replace(",", "->").split("->") if part.strip()]
            if parts and "text" not in parts[-1]:
                return False
        if input_modalities and output_modalities:
            return "text" in output_modalities

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
