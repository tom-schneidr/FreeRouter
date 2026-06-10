from __future__ import annotations

from app.benchmark_store import reset_benchmark_store_for_tests
from app.model_catalog import ModelRoute, route_id_for
from app.model_ranking import compute_rank_score, invalidate_dynamic_benchmark_cache


def setup_function() -> None:
    reset_benchmark_store_for_tests()
    invalidate_dynamic_benchmark_cache()


def _route(model_id: str, display_name: str, **kwargs) -> ModelRoute:
    provider = kwargs.pop("provider_name", "openrouter")
    return ModelRoute(
        route_id=route_id_for(provider, model_id),
        provider_name=provider,
        model_id=model_id,
        display_name=display_name,
        rank=1,
        tags=kwargs.pop("tags", ["text"]),
        quality=kwargs.pop("quality", "unknown"),
        **kwargs,
    )


def test_kimi_k2_6_scores_above_gemma_4():
    kimi = _route("moonshotai/kimi-k2.6:free", "MoonshotAI: Kimi K2.6")
    gemma = _route("google/gemma-4-31b-it:free", "Google: Gemma 4 31B", quality="good")
    assert compute_rank_score(kimi) > compute_rank_score(gemma)


def test_kimi_k2_6_matches_dotted_and_hyphenated_ids():
    dotted = _route("moonshotai/kimi-k2.6", "Kimi K2.6", provider_name="nvidia")
    hyphen = _route("moonshotai/kimi-k2-6", "Kimi K2-6", provider_name="nvidia")
    assert compute_rank_score(dotted) == compute_rank_score(hyphen)
    assert compute_rank_score(dotted) >= 54 * 3000


def test_aa_matching_uses_model_identity_not_misleading_tags():
    kimi = _route(
        "moonshotai/kimi-k2.6:free",
        "Kimi K2.6",
        tags=["text", "reasoning", "tool-use"],
    )
    # Without AA entry this would be ~30000 (unknown quality fallback).
    assert compute_rank_score(kimi) > 150_000


def test_gemini_3_1_pro_preview_matches_benchmark_entry():
    gemini = _route("gemini-3.1-pro-preview", "Gemini 3.1 Pro Preview", provider_name="gemini")
    assert compute_rank_score(gemini) >= 57 * 3000


def test_deprecated_kimi_k2_instruct_scores_below_kimi_k2_6():
    old = _route(
        "moonshotai/kimi-k2-instruct",
        "Kimi K2 Instruct",
        provider_name="nvidia",
        quality="high",
    )
    current = _route(
        "moonshotai/kimi-k2.6",
        "Kimi K2.6",
        provider_name="nvidia",
        quality="very high",
    )
    assert compute_rank_score(current) > compute_rank_score(old)
