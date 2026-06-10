from __future__ import annotations

import json
import os
from unittest.mock import AsyncMock

import pytest

from app.benchmark_research import BenchmarkResearchService, _parse_research_response
from app.benchmark_store import BenchmarkStore, reset_benchmark_store_for_tests
from app.model_catalog import ModelCatalog, ModelRoute, route_id_for
from app.model_ranking import compute_rank_score, invalidate_dynamic_benchmark_cache
from app.router import RouteResult


@pytest.fixture(autouse=True)
def _reset_benchmark_state():
    reset_benchmark_store_for_tests()
    invalidate_dynamic_benchmark_cache()
    yield
    reset_benchmark_store_for_tests()
    invalidate_dynamic_benchmark_cache()


def _route(model_id: str, display_name: str, **kwargs) -> ModelRoute:
    provider = kwargs.pop("provider_name", "openrouter")
    return ModelRoute(
        route_id=route_id_for(provider, model_id),
        provider_name=provider,
        model_id=model_id,
        display_name=display_name,
        rank=kwargs.pop("rank", 1),
        tags=kwargs.pop("tags", ["text"]),
        quality=kwargs.pop("quality", "unknown"),
        **kwargs,
    )


def test_parse_research_response_extracts_scores():
    body = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "scores": {"kimi-k2.6": 58, "gemma-4-31b": 39},
                            "source_url": "https://artificialanalysis.ai/",
                        }
                    )
                }
            }
        ]
    }
    scores, source = _parse_research_response(body)
    assert scores == {"kimi-k2.6": 58, "gemma-4-31b": 39}
    assert source == "https://artificialanalysis.ai/"


def test_benchmark_store_merge_persists_and_overrides_ranking(tmp_path, monkeypatch):
    reset_benchmark_store_for_tests()
    path = str(tmp_path / "benchmark_scores.json")
    store = BenchmarkStore(path)
    merged = store.merge_scores(
        {"kimi-k2.6": 60},
        source="test",
        confidence="high",
        updated_at=1_700_000_000,
    )
    assert merged == 1

    reloaded = BenchmarkStore(path)
    assert reloaded.index_scores_map()["kimi-k2.6"] == 60

    monkeypatch.setenv("BENCHMARK_SCORES_PATH", path)
    from app.settings import get_settings

    get_settings.cache_clear()
    invalidate_dynamic_benchmark_cache()

    kimi = _route("moonshotai/kimi-k2.6:free", "Kimi K2.6")
    assert compute_rank_score(kimi) >= 60 * 3000


@pytest.mark.asyncio
async def test_benchmark_research_refresh_merges_and_auto_ranks(tmp_path):
    reset_benchmark_store_for_tests()
    catalog_path = str(tmp_path / "catalog.json")
    benchmark_path = str(tmp_path / "benchmark_scores.json")

    catalog = ModelCatalog(catalog_path)
    catalog.initialize()
    kimi = _route("moonshotai/kimi-k2.6:free", "Kimi K2.6", rank=50)
    gemma = _route("google/gemma-4-31b-it:free", "Gemma 4 31B", rank=1, quality="good")
    catalog.replace_routes(
        [
            {
                "route_id": kimi.route_id,
                "provider_name": kimi.provider_name,
                "model_id": kimi.model_id,
                "display_name": kimi.display_name,
                "rank": kimi.rank,
                "enabled": True,
                "tags": kimi.tags,
                "quality": kimi.quality,
            },
            {
                "route_id": gemma.route_id,
                "provider_name": gemma.provider_name,
                "model_id": gemma.model_id,
                "display_name": gemma.display_name,
                "rank": gemma.rank,
                "enabled": True,
                "tags": gemma.tags,
                "quality": gemma.quality,
            },
        ]
    )

    store = BenchmarkStore(benchmark_path)
    router = AsyncMock()
    router.route_chat_completion.return_value = RouteResult(
        {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "scores": {"kimi-k2.6": 60, "gemma-4-31b": 20},
                                "source_url": "https://example.test/",
                            }
                        )
                    }
                }
            ]
        },
        "openrouter",
        "openrouter:moonshotai/kimi-k2.6:free",
        "moonshotai/kimi-k2.6:free",
        [],
    )

    service = BenchmarkResearchService(
        router,
        catalog,
        store,
        enabled=True,
        max_age_seconds=0,
        max_models=10,
        min_scores_to_apply=2,
    )
    report = await service.refresh()

    assert report.ok is True
    assert report.scores_merged == 2
    assert store.index_scores_map()["kimi-k2.6"] == 60
    assert os.path.exists(benchmark_path)

    ranked = {route.model_id: route.rank for route in catalog.all_routes()}
    assert ranked["moonshotai/kimi-k2.6:free"] < ranked["google/gemma-4-31b-it:free"]
