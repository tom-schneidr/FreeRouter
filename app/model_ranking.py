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

_TOOL_PROFILE_ADJUSTMENTS = {
    "supported": 1,
    "inconclusive": 0,
    "unknown": 0,
    "unsupported": -1,
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


def _tool_profile_status(route: ModelRoute, suffix: str) -> str:
    """Read a structured tool profile dimension with legacy evidence fallback."""
    claim = route.capabilities.get(f"tool-use.{suffix}")
    if claim is not None:
        return claim.status
    base = route.capabilities.get("tool-use")
    evidence = base.evidence.lower() if base is not None else ""
    if "openclaw tool profile:" in evidence:
        evidence = evidence.split(":", 1)[1]
    marker = f"{suffix.replace('-', '_')}="
    for part in evidence.split(";"):
        item = part.strip()
        if item.startswith(marker):
            return item.removeprefix(marker).strip()
    return "unknown"


def tool_use_behavior_score(route: ModelRoute) -> int:
    """Return a bounded automatic-rank adjustment for verified tool behavior.

    This is intentionally much smaller than an intelligence-index difference.
    It lets equivalent candidates with stronger protocol evidence sort first
    without allowing one lucky probe to displace a materially stronger model.
    """
    if "tool-use" not in route.tags:
        return 0
    exact = _tool_profile_status(route, "required-exact-call")
    auto = _tool_profile_status(route, "auto-selection")
    continuation = _tool_profile_status(route, "tool-result-continuation")
    stability = _tool_profile_status(route, "multi-turn-stability")
    score = 0
    # Later dimensions only become meaningful after the protocol guarantees
    # they depend on have passed.  This prevents an incomplete profile from
    # receiving continuation/stability credit merely because those fields were
    # present in an older or partially completed probe.
    score += 40 * _TOOL_PROFILE_ADJUSTMENTS.get(exact, 0)
    if exact == "supported":
        score += 80 * _TOOL_PROFILE_ADJUSTMENTS.get(auto, 0)
        if auto == "supported":
            score += 100 * _TOOL_PROFILE_ADJUSTMENTS.get(continuation, 0)
            if continuation == "supported":
                score += 160 * _TOOL_PROFILE_ADJUSTMENTS.get(stability, 0)
    claim = route.capabilities.get("tool-use")
    if claim is not None and claim.source in {"probe", "runtime"} and claim.status == "supported":
        score += 20
    return max(-500, min(500, score))


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

    score += tool_use_behavior_score(route)

    return score


def ranking_factors(route: ModelRoute) -> dict[str, int | str | None]:
    """Expose the automatic decision inputs for diagnostics and API clients."""
    benchmark_source: str | None = None
    benchmark_confidence: str | None = None
    benchmark_updated_at: int | None = None
    try:
        from app.benchmark_store import get_benchmark_store
        from app.settings import get_settings

        snapshot = get_benchmark_store(get_settings().benchmark_scores_path).snapshot()
        for key in sorted(snapshot.scores, key=len, reverse=True):
            if any(key in variant for variant in _aa_search_variants(route)):
                entry = snapshot.scores[key]
                benchmark_source = entry.source
                benchmark_confidence = entry.confidence
                benchmark_updated_at = entry.updated_at
                break
    except (RuntimeError, OSError, ValueError):
        pass
    return {
        "intelligence_index": _aa_index_for_route(route),
        "intelligence_source": benchmark_source or "bundled_index",
        "intelligence_confidence": benchmark_confidence,
        "intelligence_updated_at": benchmark_updated_at,
        "tool_behavior_adjustment": tool_use_behavior_score(route),
        "provider_score": compute_provider_score(route.provider_name),
        "computed_score": compute_rank_score(route),
        "rank_source": route.rank_source,
    }


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
