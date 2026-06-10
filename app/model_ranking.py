from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.model_catalog import ModelRoute


_AA_INTELLIGENCE_INDEX_SCALE = 3000

# Artificial Analysis Intelligence Index scores (and manual overrides for models absent
# from the public table). Matched against normalized model_id + display_name only.
# Longer keys are checked first to avoid partial matches (e.g. kimi-k2.6 before kimi-k2).
_AA_INTELLIGENCE_INDEX_SCORES: dict[str, int] = {
    "gemini-3.1-pro": 57,
    "kimi-k2.6": 58,
    "kimi-k2-6": 58,
    "kimi-k2.5": 56,
    "kimi-k2-5": 56,
    "kimi-k2-thinking": 54,
    "deepseek-v4-pro": 52,
    "minimax-m2.7": 50,
    "deepseek-v4-flash": 47,
    "gemini-3-flash": 46,
    "qwen-3-235b": 42,
    "hy3-preview": 42,
    "deepseek-v3.1": 39,
    "glm-4.5-air": 39,
    "gemma-4-31b": 39,
    "step-3.5-flash": 38,
    "kimi-k2-instruct-0905": 32,
    "kimi-k2-instruct": 32,
    "nemotron-3-super-120b": 36,
    "gemini-2.5-pro": 35,
    "ling-2.6-1t": 34,
    "gemini-3.1-flash-lite": 34,
    "seed-oss-36b": 34,
    "gpt-oss-120b": 33,
    "gemma-4-26b": 31,
    "gemini-2.5-flash": 30,
    "qwen3-coder": 28,
    "qwen3-next-80b": 27,
    "ling-2.6-flash": 26,
    "qwen3-32b": 25,
    "gpt-oss-20b": 24,
    "nemotron-3-nano-30b": 24,
    "nemotron-3-nano-omni": 24,
    "mistral-large-3": 23,
    "devstral-2": 22,
    "gemini-2.5-flash-lite": 22,
    "mistral-nemotron": 19,
    "magistral-small": 18,
    "llama-4-maverick": 18,
    "llama-3.1-405b": 17,
    "nemotron-nano-12b": 15,
    "nemotron-nano-9b": 15,
    "llama-3.3-70b": 14,
    "llama-4-scout": 14,
    "gemma-3-27b": 12,
    "dolphin-mistral-24b": 12,
    "pixtral-12b": 10,
    "phi-4-multimodal": 10,
    "gemma-3-12b": 9,
    "lfm-2.5-1.2b": 8,
    "gemma-3-4b": 8,
    "gemma-3n-e4b": 8,
    "gemma-2-2b": 7,
    "gemma-3n-e2b": 6,
    "openrouter/free": 1,
}

_DYNAMIC_SCORES_CACHE: dict[str, int] | None = None


def invalidate_dynamic_benchmark_cache() -> None:
    global _DYNAMIC_SCORES_CACHE
    _DYNAMIC_SCORES_CACHE = None


def _expand_benchmark_key_variants(scores: dict[str, int]) -> dict[str, int]:
    expanded = dict(scores)
    for key, value in scores.items():
        hyphenated = key.replace(".", "-")
        if hyphenated != key:
            expanded.setdefault(hyphenated, value)
    return expanded


def _dynamic_aa_scores() -> dict[str, int]:
    global _DYNAMIC_SCORES_CACHE
    if _DYNAMIC_SCORES_CACHE is not None:
        return _DYNAMIC_SCORES_CACHE
    try:
        from app.benchmark_store import get_benchmark_store
        from app.settings import get_settings

        _DYNAMIC_SCORES_CACHE = _expand_benchmark_key_variants(
            get_benchmark_store(get_settings().benchmark_scores_path).index_scores_map()
        )
    except (RuntimeError, OSError, ValueError):
        _DYNAMIC_SCORES_CACHE = {}
    return _DYNAMIC_SCORES_CACHE


def _merged_aa_scores() -> dict[str, int]:
    merged = dict(_AA_INTELLIGENCE_INDEX_SCORES)
    merged.update(_dynamic_aa_scores())
    return merged


def _aa_match_keys() -> list[str]:
    return sorted(_merged_aa_scores(), key=len, reverse=True)


def route_has_dynamic_benchmark_match(
    route: ModelRoute,
    dynamic_scores: dict[str, int] | None = None,
) -> bool:
    scores = (
        dynamic_scores
        if dynamic_scores is not None
        else _dynamic_aa_scores()
    )
    if not scores:
        return False
    keys = sorted(scores, key=len, reverse=True)
    for search_text in _aa_search_variants(route):
        for key in keys:
            if key in search_text:
                return True
    return False

_PROVIDER_SCORES = {
    "gemini": 100,
    "groq": 90,
    "cerebras": 80,
    "nvidia": 70,
    "sambanova": 65,
    "openrouter": 60,
}

_QUALITY_FALLBACK_INDEX = {
    "very high": 26,
    "high": 22,
    "agentic": 21,
    "good": 14,
    "vision": 10,
    "utility": 8,
    "translation": 6,
    "safety": -100,
    "unknown": 10,
}


def _normalize_ranking_text(*parts: str) -> str:
    text = " ".join(part for part in parts if part).lower()
    return text.replace("_", "-").replace("/", " ").replace(":", " ")


def _aa_search_variants(route: ModelRoute) -> list[str]:
    """Build search strings for benchmark lookup (with and without version dots)."""
    base = _normalize_ranking_text(route.model_id, route.display_name)
    variants = [base]
    dotted_to_hyphen = base.replace(".", "-")
    if dotted_to_hyphen != base:
        variants.append(dotted_to_hyphen)
    return variants


def _aa_index_for_route(route: ModelRoute) -> int | None:
    scores = _merged_aa_scores()
    for search_text in _aa_search_variants(route):
        for key in _aa_match_keys():
            if key in search_text:
                return scores[key]
    return None


def compute_rank_score(route: ModelRoute) -> int:
    """Capability score guided by the Artificial Analysis Intelligence Index."""
    identity_text = _normalize_ranking_text(route.model_id, route.display_name)

    if any(
        term in identity_text
        for term in ("safety", "guard", "pii", "translate", "paligemma")
    ):
        return -500000

    aa_index = _aa_index_for_route(route)
    if aa_index is not None:
        score = aa_index * _AA_INTELLIGENCE_INDEX_SCALE
    else:
        score = _QUALITY_FALLBACK_INDEX.get(route.quality.lower(), 10) * _AA_INTELLIGENCE_INDEX_SCALE

    size_match = re.search(r"(\d+(?:\.\d+)?)b", identity_text)
    if size_match:
        score += min(int(float(size_match.group(1)) * 2), 900)

    size_t_match = re.search(r"(\d+(?:\.\d+)?)t", identity_text)
    if size_t_match:
        score += min(int(float(size_t_match.group(1)) * 600), 900)

    tag_text = " ".join(route.tags).lower()
    combined = f"{identity_text} {tag_text}"

    if "pro" in identity_text:
        score += 25
    if "large" in identity_text:
        score += 20
    if "versatile" in identity_text:
        score += 10
    if "flash" in identity_text:
        score += 5
    if "lite" in identity_text:
        score -= 5
    if "mini" in identity_text:
        score -= 10
    if "nano" in identity_text:
        score -= 15
    if "vision" in tag_text:
        score -= 5
    if "coder" in combined:
        score += 10
    if "reasoning" in tag_text:
        score += 15

    return score


def compute_provider_score(provider_name: str) -> int:
    return _PROVIDER_SCORES.get(provider_name.lower(), 0)


def compute_composite_rank_score(route: ModelRoute) -> int:
    return compute_rank_score(route) + compute_provider_score(route.provider_name)


def rank_sort_key(route: ModelRoute) -> tuple[int, int, str, str]:
    return (
        compute_rank_score(route),
        compute_provider_score(route.provider_name),
        route.provider_name,
        route.model_id,
    )
