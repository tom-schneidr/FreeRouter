from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.model_catalog import ModelRoute


_AA_INTELLIGENCE_INDEX_SCALE = 3000

# Artificial Analysis Intelligence Index scores for models represented in the
# default catalog. Values are kept as leaderboard-style index scores and scaled
# in compute_rank_score so small tie-breakers cannot overwhelm capability rank.
_AA_INTELLIGENCE_INDEX_SCORES = {
    "gemini-3.1-pro": 57,
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
    "kimi-k2-instruct-0905": 37,
    "kimi-k2-instruct": 37,
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

_PROVIDER_SCORES = {
    "gemini": 100,      # Dynamic but highly generous token limits
    "groq": 90,         # 30 RPM, 14.4K RPD
    "cerebras": 80,     # 30 RPM, 1M TPD
    "nvidia": 70,       # Generous but undocumented limits
    "openrouter": 60,   # Aggressive free tier rate limits
}


def compute_rank_score(route: ModelRoute) -> int:
    """Capability score guided by the Artificial Analysis Intelligence Index."""
    text = (route.display_name + " " + route.model_id + " " + " ".join(route.tags)).lower()

    if "safety" in text or "guard" in text or "pii" in text or "translate" in text or "paligemma" in text:
        return -500000

    score = 0
    for key, val in sorted(_AA_INTELLIGENCE_INDEX_SCORES.items(), key=lambda item: len(item[0]), reverse=True):
        if key in text:
            score = val * _AA_INTELLIGENCE_INDEX_SCALE
            break

    if score == 0:
        quality_scores = {
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
        score = quality_scores.get(route.quality.lower(), 10) * _AA_INTELLIGENCE_INDEX_SCALE

    size_match = re.search(r"(\d+(?:\.\d+)?)b", text)
    if size_match:
        score += min(int(float(size_match.group(1)) * 2), 900)

    size_t_match = re.search(r"(\d+(?:\.\d+)?)t", text)
    if size_t_match:
        score += min(int(float(size_t_match.group(1)) * 600), 900)

    if "pro" in text:
        score += 25
    if "large" in text:
        score += 20
    if "versatile" in text:
        score += 10
    if "flash" in text:
        score += 5
    if "lite" in text:
        score -= 5
    if "mini" in text:
        score -= 10
    if "nano" in text:
        score -= 15
    if "vision" in text:
        score -= 5
    if "coder" in text:
        score += 10
    if "reasoning" in text:
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

