from __future__ import annotations

from dataclasses import dataclass
from time import time
from typing import TYPE_CHECKING, Any

from app.web_search_payload import payload_with_required_web_search
from app.benchmark_store import BenchmarkStore
from app.model_catalog import ModelCatalog, is_text_chat_route
from app.request_requirements import chat_request_requirements, with_extra_capabilities
from app.response_parse import chat_response_text, json_object_from_text

if TYPE_CHECKING:
    from app.router import WaterfallRouter


@dataclass(frozen=True)
class BenchmarkRefreshReport:
    ok: bool
    checked_at: int
    scores_merged: int
    models_requested: int
    route_id: str | None = None
    provider_name: str | None = None
    model_id: str | None = None
    source_url: str | None = None
    error: str | None = None
    skipped: bool = False
    skip_reason: str | None = None


class BenchmarkResearchService:
    """Refresh benchmark scores by routing a web-search request through FreeRouter itself."""

    def __init__(
        self,
        router: WaterfallRouter,
        catalog: ModelCatalog,
        store: BenchmarkStore,
        *,
        enabled: bool,
        max_age_seconds: int,
        max_models: int,
        min_scores_to_apply: int,
    ) -> None:
        self.router = router
        self.catalog = catalog
        self.store = store
        self.enabled = enabled
        self.max_age_seconds = max_age_seconds
        self.max_models = max(1, max_models)
        self.min_scores_to_apply = max(1, min_scores_to_apply)
        self.last_report: BenchmarkRefreshReport | None = None

    async def refresh_if_stale(self) -> BenchmarkRefreshReport | None:
        if not self.enabled:
            return None
        if not self.store.is_stale(max_age_seconds=self.max_age_seconds):
            return None
        report = await self.refresh()
        self.last_report = report
        return report

    async def refresh(self) -> BenchmarkRefreshReport:
        checked_at = int(time())
        if not self.enabled:
            report = BenchmarkRefreshReport(
                ok=False,
                checked_at=checked_at,
                scores_merged=0,
                models_requested=0,
                error="Benchmark refresh is disabled.",
                skipped=True,
                skip_reason="disabled",
            )
            self.last_report = report
            return report

        routes = [
            route
            for route in self.catalog.enabled_routes()
            if route.enabled and is_text_chat_route(route)
        ]
        targets = self._select_target_routes(routes)
        if not targets:
            report = BenchmarkRefreshReport(
                ok=False,
                checked_at=checked_at,
                scores_merged=0,
                models_requested=0,
                error="No enabled text routes available for benchmark research.",
            )
            self.last_report = report
            return report

        payload = self._build_payload(targets)
        try:
            prepared = payload_with_required_web_search(payload)
            requirements = with_extra_capabilities(
                chat_request_requirements(prepared),
                "web-search",
            )
            result = await self.router.route_chat_completion(
                prepared,
                requirements=requirements,
                require_assistant_content=True,
            )
        except Exception as exc:
            report = BenchmarkRefreshReport(
                ok=False,
                checked_at=checked_at,
                scores_merged=0,
                models_requested=len(targets),
                error=f"{exc.__class__.__name__}: {exc}",
            )
            self.last_report = report
            return report

        parsed_scores, source_url = _parse_research_response(result.body)
        if len(parsed_scores) < self.min_scores_to_apply:
            report = BenchmarkRefreshReport(
                ok=False,
                checked_at=checked_at,
                scores_merged=0,
                models_requested=len(targets),
                route_id=result.route_id,
                provider_name=result.provider_name,
                model_id=result.model_id,
                source_url=source_url,
                error=(
                    f"Research returned {len(parsed_scores)} scores; "
                    f"need at least {self.min_scores_to_apply}."
                ),
            )
            self.last_report = report
            return report

        merged = self.store.merge_scores(
            parsed_scores,
            source="freerouter-web-search",
            confidence="medium",
            source_url=source_url,
            updated_at=checked_at,
        )
        self.catalog.auto_rank_routes()
        self.catalog.save()

        report = BenchmarkRefreshReport(
            ok=True,
            checked_at=checked_at,
            scores_merged=merged,
            models_requested=len(targets),
            route_id=result.route_id,
            provider_name=result.provider_name,
            model_id=result.model_id,
            source_url=source_url,
        )
        self.last_report = report
        return report

    def _select_target_routes(self, routes: list[Any]) -> list[Any]:
        missing = self.store.routes_without_dynamic_match(routes)
        if missing:
            ordered = sorted(missing, key=lambda route: route.rank)
            return ordered[: self.max_models]
        ordered = sorted(routes, key=lambda route: route.rank)
        return ordered[: self.max_models]

    def _build_payload(self, routes: list[Any]) -> dict[str, Any]:
        model_lines = "\n".join(
            f"- {route.provider_name}/{route.model_id} ({route.display_name})"
            for route in routes
        )
        user_prompt = (
            "Use web search to find current Artificial Analysis Intelligence Index scores "
            "(or the closest official quality index on artificialanalysis.ai) for the models below.\n\n"
            f"Models:\n{model_lines}\n\n"
            "Return ONLY JSON with this schema:\n"
            "{\n"
            '  "scores": {"kimi-k2.6": 58, "gemma-4-31b": 39},\n'
            '  "source_url": "https://...",\n'
            '  "confidence": "high|medium|low"\n'
            "}\n\n"
            "Rules:\n"
            "- Use lowercase hyphenated keys that match model names/versions.\n"
            "- Index scores are typically integers from 1 to 60.\n"
            "- Only include models where you found reliable leaderboard evidence.\n"
            "- Do not invent scores."
        )
        return {
            "model": "auto",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You research public LLM benchmark leaderboards. "
                        "Return only valid JSON."
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "max_tokens": 4000,
        }


def _parse_research_response(body: dict[str, Any]) -> tuple[dict[str, int], str | None]:
    raw_text = chat_response_text(body)
    parsed = json_object_from_text(raw_text)
    if not isinstance(parsed, dict):
        return {}, None

    source_url = parsed.get("source_url")
    source = source_url if isinstance(source_url, str) else None

    raw_scores = parsed.get("scores")
    if not isinstance(raw_scores, dict):
        return {}, source

    scores: dict[str, int] = {}
    for key, value in raw_scores.items():
        if not isinstance(key, str):
            continue
        if isinstance(value, (int, float)):
            scores[key.strip().lower()] = int(value)
        elif isinstance(value, dict):
            index = value.get("index")
            if isinstance(index, (int, float)):
                scores[key.strip().lower()] = int(index)
    return scores, source
