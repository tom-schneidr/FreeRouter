from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any


CANONICAL_MODEL_TAGS = {
    "text",
    "reasoning",
    "coding",
    "tool-use",
    "web-search",
    "vision",
    "audio",
    "safety",
    "moderation",
    "translation",
    "classification",
    "rag",
}


@dataclass(frozen=True)
class ModelRoute:
    """Immutable descriptor for a single provider + model routing target."""

    route_id: str
    provider_name: str
    model_id: str
    display_name: str
    rank: int
    enabled: bool = True
    context_window: int | None = None
    quality: str = "unknown"
    speed: str = "unknown"
    cost: str = "free-tier"
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    source_url: str = ""


def _route(
    provider_name: str,
    model_id: str,
    display_name: str,
    context_window: int | None,
    *,
    quality: str = "good",
    speed: str = "variable",
    tags: list[str] | None = None,
    notes: str = "",
    source_url: str = "",
    enabled: bool = True,
    text: bool = True,
) -> ModelRoute:
    """Build a ModelRoute with an auto-generated route_id slug."""
    raw_id = f"{provider_name}-{model_id}".lower()
    route_id = re.sub(r"[/:._]+", "-", raw_id).strip("-")
    route_tags = list(dict.fromkeys((["text"] if text else []) + (tags or [])))
    return ModelRoute(
        route_id=route_id,
        provider_name=provider_name,
        model_id=model_id,
        display_name=display_name,
        rank=0,
        enabled=enabled,
        context_window=context_window,
        quality=quality,
        speed=speed,
        tags=route_tags,
        notes=notes,
        source_url=source_url,
    )


def _canonical_tags(tags: list[str]) -> list[str]:
    return list(dict.fromkeys(tag for tag in tags if tag in CANONICAL_MODEL_TAGS))


DEFAULT_MODEL_ROUTES = [

    _route("cerebras",
        "llama3.1-8b",
        "Llama 3.1 8B",
        8192,
        speed="very fast",
        tags=[],
        notes="Official Cerebras model. Free tier: 60K TPM, 1M TPD, 30 RPM. Scheduled for deprecation May 27, 2026.",
        source_url="https://inference-docs.cerebras.ai/models/overview",
    ),
    _route("cerebras",
        "qwen-3-235b-a22b-instruct-2507",
        "Qwen 3 235B Instruct",
        8192,
        quality="high",
        speed="very fast",
        tags=["reasoning"],
        notes="Official Cerebras preview model. Free tier: 60K TPM, 1M TPD, 30 RPM. Scheduled for deprecation May 27, 2026.",
        source_url="https://inference-docs.cerebras.ai/models/overview",
    ),

    _route("groq", "openai/gpt-oss-120b", "GPT OSS 120B", 131072, quality="high", speed="very fast", tags=["reasoning"], notes="Free-plan limits listed by Groq: 30 RPM, 1K RPD, 8K TPM, 200K TPD.", source_url="https://console.groq.com/docs/rate-limits"),
    _route("groq", "openai/gpt-oss-20b", "GPT OSS 20B", 131072, speed="very fast", tags=["reasoning"], notes="Free-plan limits listed by Groq: 30 RPM, 1K RPD, 8K TPM, 200K TPD.", source_url="https://console.groq.com/docs/rate-limits"),
    _route("groq", "llama-3.3-70b-versatile", "Llama 3.3 70B Versatile", 131072, quality="high", speed="very fast", tags=[], notes="Free-plan limits listed by Groq: 30 RPM, 1K RPD, 12K TPM, 100K TPD.", source_url="https://console.groq.com/docs/rate-limits"),
    _route("groq", "llama-3.1-8b-instant", "Llama 3.1 8B Instant", 131072, speed="very fast", tags=[], notes="Free-plan limits listed by Groq: 30 RPM, 14.4K RPD, 6K TPM, 500K TPD.", source_url="https://console.groq.com/docs/rate-limits"),
    _route("groq", "meta-llama/llama-4-scout-17b-16e-instruct", "Llama 4 Scout 17B 16E", 131072, quality="high", speed="very fast", tags=["vision"], notes="Free-plan limits listed by Groq: 30 RPM, 1K RPD, 30K TPM, 500K TPD.", source_url="https://console.groq.com/docs/rate-limits"),
    _route("groq", "qwen/qwen3-32b", "Qwen3 32B", 131072, quality="high", speed="very fast", tags=["reasoning"], notes="Free-plan limits listed by Groq: 60 RPM, 1K RPD, 6K TPM, 500K TPD.", source_url="https://console.groq.com/docs/rate-limits"),
    _route("groq", "groq/compound", "Groq Compound", 131072, quality="agentic", speed="fast", tags=["tool-use", "web-search"], notes="Free-plan limits listed by Groq: 30 RPM, 250 RPD, 70K TPM.", source_url="https://console.groq.com/docs/rate-limits"),
    _route("groq", "groq/compound-mini", "Groq Compound Mini", 131072, quality="agentic", speed="fast", tags=["tool-use", "web-search"], notes="Free-plan limits listed by Groq: 30 RPM, 250 RPD, 70K TPM.", source_url="https://console.groq.com/docs/rate-limits"),
    _route("gemini", "gemini-3.1-pro-preview", "Gemini 3.1 Pro Preview", 2_097_152, quality="very high", speed="medium", tags=["reasoning", "vision", "audio"], notes="Listed in Gemini docs. Free-tier limits are dynamic and must be checked in AI Studio. 2 RPM.", source_url="https://ai.google.dev/gemini-api/docs/models"),
    _route("gemini", "gemini-3-flash-preview", "Gemini 3 Flash Preview", 1_048_576, quality="high", speed="fast", tags=["vision", "audio"], notes="Frontier-class performance rivaling larger models. Free-tier limits are dynamic. 15 RPM.", source_url="https://ai.google.dev/gemini-api/docs/models"),
    _route("gemini", "gemini-3.1-flash-lite-preview", "Gemini 3.1 Flash-Lite Preview", 1_048_576, quality="good", speed="very fast", tags=["vision", "audio"], notes="Listed in Gemini docs. Free-tier limits are dynamic and must be checked in AI Studio. 15 RPM.", source_url="https://ai.google.dev/gemini-api/docs/models"),
    _route("gemini", "gemini-2.5-flash", "Gemini 2.5 Flash", 1_000_000, quality="high", speed="fast", tags=["vision", "audio"], notes="Current stable Flash family model. Free-tier limits are dynamic and per project.", source_url="https://ai.google.dev/gemini-api/docs/models"),
    _route("gemini", "gemini-2.5-flash-lite", "Gemini 2.5 Flash-Lite", 1_000_000, quality="good", speed="very fast", tags=["vision", "audio"], notes="Fastest 2.5 family model. Free-tier limits are dynamic and per project.", source_url="https://ai.google.dev/gemini-api/docs/models"),
    _route("gemini", "gemini-2.5-pro", "Gemini 2.5 Pro", 1_000_000, quality="very high", speed="medium", tags=["reasoning", "vision", "audio"], notes="Listed in Gemini docs; availability/free limits vary by project and tier.", source_url="https://ai.google.dev/gemini-api/docs/models"),
    _route("nvidia", "deepseek-ai/deepseek-v4-pro", "DeepSeek V4 Pro", 1_000_000, quality="very high", speed="fast", tags=["coding"], notes="Verified on NVIDIA Build Free Endpoint filtered catalog. DeepSeek V4 Pro is optimized for coding tasks and 1M-token context.", source_url="https://build.nvidia.com/deepseek-ai/deepseek-v4-pro"),
    _route("nvidia", "deepseek-ai/deepseek-v4-flash", "DeepSeek V4 Flash", 1_000_000, quality="high", speed="very fast", tags=["coding"], notes="Verified on NVIDIA Build Free Endpoint filtered catalog. DeepSeek V4 Flash is a 284B MoE optimized for fast coding and agents.", source_url="https://build.nvidia.com/deepseek-ai/deepseek-v4-flash"),

    _route("nvidia", "minimaxai/minimax-m2.7", "MiniMax M2.7", 128000, quality="high", speed="fast", tags=["coding", "reasoning"], notes="Verified on NVIDIA Build Free Endpoint filtered catalog. 230B text-to-text model for coding, reasoning, and office tasks.", source_url="https://build.nvidia.com/minimaxai/minimax-m2.7"),

    _route("nvidia", "deepseek-ai/deepseek-v3.1-terminus", "DeepSeek V3.1 Terminus", 128000, quality="high", speed="fast", tags=["tool-use"], notes="Verified on NVIDIA Build Free Endpoint filtered catalog, but marked with near-term deprecation.", source_url="https://build.nvidia.com/deepseek-ai/deepseek-v3_1-terminus"),
    _route("nvidia", "stepfun-ai/step-3.5-flash", "Step 3.5 Flash", 128000, quality="high", speed="fast", tags=["reasoning"], notes="Verified on NVIDIA Build Free Endpoint filtered catalog.", source_url="https://build.nvidia.com/stepfun-ai/step-3.5-flash"),
    _route("nvidia", "mistralai/devstral-2-123b-instruct-2512", "Devstral 2 123B Instruct", 256000, quality="high", speed="fast", tags=["coding", "reasoning"], notes="Verified on NVIDIA Build Free Endpoint filtered catalog.", source_url="https://build.nvidia.com/mistralai/devstral-2-123b-instruct-2512"),
    _route("nvidia", "moonshotai/kimi-k2-thinking", "Kimi K2 Thinking", 256000, quality="high", speed="fast", tags=["reasoning", "tool-use"], notes="Verified on NVIDIA Build Free Endpoint filtered catalog.", source_url="https://build.nvidia.com/moonshotai/kimi-k2-thinking"),
    _route("nvidia", "mistralai/mistral-large-3-675b-instruct-2512", "Mistral Large 3 675B Instruct", 256000, quality="very high", speed="fast", tags=["vision"], notes="Verified on NVIDIA Build Free Endpoint filtered catalog.", source_url="https://build.nvidia.com/mistralai/mistral-large-3-675b-instruct-2512"),
    _route("nvidia", "moonshotai/kimi-k2-instruct-0905", "Kimi K2 Instruct 0905", 256000, quality="high", speed="fast", tags=["reasoning"], notes="Verified on NVIDIA Build Free Endpoint filtered catalog.", source_url="https://build.nvidia.com/moonshotai/kimi-k2-instruct-0905"),
    _route("nvidia", "bytedance/seed-oss-36b-instruct", "Seed OSS 36B Instruct", 128000, quality="good", speed="fast", tags=[], notes="Verified on NVIDIA Build Free Endpoint filtered catalog.", source_url="https://build.nvidia.com/bytedance/seed-oss-36b-instruct"),
    _route("nvidia", "qwen/qwen3-coder-480b-a35b-instruct", "Qwen3 Coder 480B A35B", 256000, quality="high", speed="fast", tags=["coding"], notes="Verified on NVIDIA Build Free Endpoint filtered catalog.", source_url="https://build.nvidia.com/qwen/qwen3-coder-480b-a35b-instruct"),
    _route("nvidia", "moonshotai/kimi-k2-instruct", "Kimi K2 Instruct", 128000, quality="high", speed="fast", tags=["coding"], notes="Verified on NVIDIA Build Free Endpoint filtered catalog.", source_url="https://build.nvidia.com/moonshotai/kimi-k2-instruct"),
    _route("nvidia", "mistralai/magistral-small-2506", "Magistral Small 2506", 128000, quality="good", speed="fast", tags=["reasoning", "coding"], notes="Verified on NVIDIA Build Free Endpoint filtered catalog.", source_url="https://build.nvidia.com/mistralai/magistral-small-2506"),
    _route("nvidia", "mistralai/mistral-nemotron", "Mistral Nemotron", 128000, quality="good", speed="fast", tags=["coding", "tool-use"], notes="Verified on NVIDIA Build Free Endpoint filtered catalog.", source_url="https://build.nvidia.com/mistralai/mistral-nemotron"),

    _route("nvidia", "meta/llama-4-maverick-17b-128e-instruct", "Llama 4 Maverick 17B 128E", 128000, quality="high", speed="fast", tags=["vision"], notes="Verified on NVIDIA Build Free Endpoint filtered catalog.", source_url="https://build.nvidia.com/meta/llama-4-maverick-17b-128e-instruct"),
    _route("nvidia", "google/gemma-3n-e4b-it", "Gemma 3n E4B IT", 8192, quality="good", speed="fast", tags=["vision", "audio"], notes="Verified on NVIDIA Build Free Endpoint filtered catalog.", source_url="https://build.nvidia.com/google/gemma-3n-e4b-it"),
    _route("nvidia", "google/gemma-3n-e2b-it", "Gemma 3n E2B IT", 8192, quality="good", speed="fast", tags=["vision", "audio"], notes="Verified on NVIDIA Build Free Endpoint filtered catalog.", source_url="https://build.nvidia.com/google/gemma-3n-e2b-it"),
    _route("nvidia", "google/gemma-3-27b-it", "Gemma 3 27B IT", 131072, quality="good", speed="fast", tags=["vision"], notes="Verified on NVIDIA Build Free Endpoint filtered catalog.", source_url="https://build.nvidia.com/google/gemma-3-27b-it"),
    _route("nvidia", "microsoft/phi-4-multimodal-instruct", "Phi 4 Multimodal Instruct", 128000, quality="good", speed="fast", tags=["vision", "audio"], notes="Verified on NVIDIA Build Free Endpoint filtered catalog.", source_url="https://build.nvidia.com/microsoft/phi-4-multimodal-instruct"),
    _route("nvidia", "abacusai/dracarys-llama-3.1-70b-instruct", "Dracarys Llama 3.1 70B Instruct", 128000, quality="good", speed="fast", tags=["coding"], notes="Verified on NVIDIA Build Free Endpoint filtered catalog.", source_url="https://build.nvidia.com/abacusai/dracarys-llama-3_1-70b-instruct"),
    _route("nvidia", "nvidia/nemotron-mini-4b-instruct", "Nemotron Mini 4B Instruct", 8192, quality="utility", speed="very fast", tags=["rag", "tool-use"], notes="Verified on NVIDIA Build Free Endpoint filtered catalog.", source_url="https://build.nvidia.com/nvidia/nemotron-mini-4b-instruct"),
    _route("nvidia", "google/gemma-2-2b-it", "Gemma 2 2B IT", 8192, quality="utility", speed="very fast", tags=[], notes="Verified on NVIDIA Build Free Endpoint filtered catalog.", source_url="https://build.nvidia.com/google/gemma-2-2b-it"),
    _route("nvidia", "upstage/solar-10.7b-instruct", "Solar 10.7B Instruct", 4096, quality="good", speed="fast", tags=[], notes="Verified on NVIDIA Build Free Endpoint filtered catalog. Non-commercial use label shown in catalog.", source_url="https://build.nvidia.com/upstage/solar-10_7b-instruct"),
    _route("nvidia", "google/google-paligemma", "PaliGemma", 8192, quality="vision", speed="fast", tags=["vision"], notes="Verified on NVIDIA Build Free Endpoint filtered catalog. Vision-specialized route; disabled by default for chat routing.", source_url="https://build.nvidia.com/google/google-paligemma", enabled=False),
    _route("nvidia", "nvidia/nemotron-3-content-safety", "Nemotron 3 Content Safety", 128000, quality="safety", speed="fast", tags=["safety", "moderation"], notes="Verified on NVIDIA Build Free Endpoint filtered catalog. Safety classifier; disabled by default for chat routing.", source_url="https://build.nvidia.com/nvidia/nemotron-3-content-safety", enabled=False),
    _route("nvidia", "nvidia/nemotron-content-safety-reasoning-4b", "Nemotron Content Safety Reasoning 4B", 8192, quality="safety", speed="fast", tags=["safety", "moderation", "reasoning"], notes="Verified on NVIDIA Build Free Endpoint filtered catalog. Safety model; disabled by default for chat routing.", source_url="https://build.nvidia.com/nvidia/nemotron-content-safety-reasoning-4b", enabled=False),
    _route("nvidia", "meta/llama-guard-4-12b", "Llama Guard 4 12B", 128000, quality="safety", speed="fast", tags=["safety", "moderation", "vision"], notes="Verified on NVIDIA Build Free Endpoint filtered catalog. Safety classifier; disabled by default for chat routing.", source_url="https://build.nvidia.com/meta/llama-guard-4-12b", enabled=False),
    _route("nvidia", "nvidia/llama-3.1-nemotron-safety-guard-8b-v3", "Llama 3.1 Nemotron Safety Guard 8B V3", 8192, quality="safety", speed="fast", tags=["safety", "moderation"], notes="Verified on NVIDIA Build Free Endpoint filtered catalog. Safety classifier; disabled by default for chat routing.", source_url="https://build.nvidia.com/nvidia/llama-3_1-nemotron-safety-guard-8b-v3", enabled=False),
    _route("nvidia", "nvidia/riva-translate-4b-instruct-v1_1", "Riva Translate 4B Instruct V1.1", 8192, quality="translation", speed="fast", tags=["translation"], notes="Verified on NVIDIA Build Free Endpoint filtered catalog. Translation-specific route; disabled by default for chat routing.", source_url="https://build.nvidia.com/nvidia/riva-translate-4b-instruct-v1_1", enabled=False),
    _route("nvidia", "nvidia/gliner-pii", "GLiNER PII", 8192, quality="utility", speed="fast", tags=["classification", "moderation"], notes="Verified on NVIDIA Build Free Endpoint filtered catalog. PII detector; disabled by default for chat routing.", source_url="https://build.nvidia.com/nvidia/gliner-pii", enabled=False),
    _route("nvidia", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning", "Nemotron-3 Nano Omni 30B", 256000, quality="high", speed="fast", tags=["reasoning"], notes="Verified on NVIDIA Build Free Endpoint filtered catalog.", source_url="https://build.nvidia.com/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"),

    _route("openrouter", "openrouter/free", "Free Models Router", 200000, speed="variable", tags=[], notes="OpenRouter's free router automatically selects from available zero-cost models.", source_url="https://openrouter.ai/openrouter/free"),
]

_OPENROUTER_FREE_MODELS = [
    ("meta-llama/llama-3.1-405b-instruct:free", "Meta: Llama 3.1 405B Instruct", 128000, "text"),
    ("mistralai/pixtral-12b:free", "Mistral: Pixtral 12B", 128000, "text"),
    ("poolside/laguna-xs.2:free", "Poolside: Laguna XS.2", 131072, "text"),
    ("poolside/laguna-m.1:free", "Poolside: Laguna M.1", 131072, "text"),
    ("inclusionai/ling-2.6-1t:free", "inclusionAI: Ling-2.6-1T", 262144, "text"),
    ("tencent/hy3-preview:free", "Tencent: Hy3 Preview", 262144, "text"),
    ("inclusionai/ling-2.6-flash:free", "inclusionAI: Ling-2.6 Flash", 262144, "text"),
    ("baidu/qianfan-ocr-fast:free", "Baidu: Qianfan OCR Fast", 65536, "vision"),
    ("google/gemma-4-26b-a4b-it:free", "Google: Gemma 4 26B A4B", 262144, "vision"),
    ("google/gemma-4-31b-it:free", "Google: Gemma 4 31B", 262144, "vision"),
    ("google/lyria-3-pro-preview", "Google: Lyria 3 Pro Preview", 1048576, "audio"),
    ("google/lyria-3-clip-preview", "Google: Lyria 3 Clip Preview", 1048576, "audio"),
    ("nvidia/nemotron-3-super-120b-a12b:free", "NVIDIA: Nemotron 3 Super", 262144, "text"),
    ("minimax/minimax-m2.5:free", "MiniMax: MiniMax M2.5", 196608, "text"),
    ("liquid/lfm-2.5-1.2b-thinking:free", "LiquidAI: LFM2.5 1.2B Thinking", 32768, "text"),
    ("liquid/lfm-2.5-1.2b-instruct:free", "LiquidAI: LFM2.5 1.2B Instruct", 32768, "text"),
    ("nvidia/nemotron-3-nano-30b-a3b:free", "NVIDIA: Nemotron 3 Nano 30B A3B", 256000, "text"),
    ("nvidia/nemotron-nano-12b-v2-vl:free", "NVIDIA: Nemotron Nano 12B V2 VL", 128000, "vision"),
    ("qwen/qwen3-next-80b-a3b-instruct:free", "Qwen: Qwen3 Next 80B A3B Instruct", 262144, "text"),
    ("nvidia/nemotron-nano-9b-v2:free", "NVIDIA: Nemotron Nano 9B V2", 128000, "text"),
    ("openai/gpt-oss-120b:free", "OpenAI: GPT OSS 120B", 131072, "text"),
    ("openai/gpt-oss-20b:free", "OpenAI: GPT OSS 20B", 131072, "text"),
    ("z-ai/glm-4.5-air:free", "Z.ai: GLM 4.5 Air", 131072, "text"),
    ("qwen/qwen3-coder:free", "Qwen: Qwen3 Coder 480B A35B", 262000, "text"),
    ("cognitivecomputations/dolphin-mistral-24b-venice-edition:free", "Venice: Uncensored", 32768, "text"),
    ("google/gemma-3n-e2b-it:free", "Google: Gemma 3n 2B", 8192, "text"),
    ("google/gemma-3n-e4b-it:free", "Google: Gemma 3n 4B", 8192, "text"),
    ("google/gemma-3-4b-it:free", "Google: Gemma 3 4B", 32768, "vision"),
    ("google/gemma-3-12b-it:free", "Google: Gemma 3 12B", 32768, "vision"),
    ("google/gemma-3-27b-it:free", "Google: Gemma 3 27B", 131072, "vision"),
    ("meta-llama/llama-3.3-70b-instruct:free", "Meta: Llama 3.3 70B Instruct", 65536, "text"),
    ("meta-llama/llama-3.2-3b-instruct:free", "Meta: Llama 3.2 3B Instruct", 131072, "text"),
    ("nousresearch/hermes-3-llama-3.1-405b:free", "Nous: Hermes 3 405B Instruct", 131072, "text"),
]


def _openrouter_tags(model_id: str, display_name: str, context_window: int, modality: str) -> list[str]:
    text = f"{model_id} {display_name}".lower()
    tags: list[str] = []

    if modality in {"text", "vision"}:
        tags.append("text")
    if modality == "vision" or any(term in text for term in ("pixtral", "gemma-3", "vl", "ocr")):
        tags.append("vision")
    if modality == "audio" or "lyria" in text:
        tags.append("audio")
    if any(term in text for term in ("thinking", "reasoning", "gpt-oss", "qwen", "hermes", "ling")):
        tags.append("reasoning")
    if any(term in text for term in ("coder", "code")):
        tags.append("coding")
    if "ocr" in text:
        tags.append("classification")

    return list(dict.fromkeys(tags))


DEFAULT_MODEL_ROUTES.extend(
    _route(
        "openrouter",
        model_id,
        display_name,
        context_window,
        speed="variable",
        tags=_openrouter_tags(model_id, display_name, context_window, modality),
        text=False,
        notes="Verified from OpenRouter public /api/v1/models as zero prompt and completion price.",
        source_url="https://openrouter.ai/api/v1/models",
    )
    for model_id, display_name, context_window, modality in _OPENROUTER_FREE_MODELS
)

_AA_INTELLIGENCE_INDEX_SCALE = 3000

# Artificial Analysis Intelligence Index scores for models represented in the
# default catalog. Values are kept as leaderboard-style index scores and scaled
# in _get_model_score so small tie-breakers cannot overwhelm capability rank.
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


def _get_model_score(route: ModelRoute) -> int:
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

def _get_provider_score(provider_name: str) -> int:
    """Score providers based on the generosity of their free usage limits."""
    scores = {
        "gemini": 100,      # Dynamic but highly generous token limits
        "groq": 90,         # 30 RPM, 14.4K RPD
        "cerebras": 80,     # 30 RPM, 1M TPD
        "nvidia": 70,       # Generous but undocumented limits
        "openrouter": 60,   # Aggressive free tier rate limits
    }
    return scores.get(provider_name.lower(), 0)


def _assign_default_ranks() -> None:
    """Sort DEFAULT_MODEL_ROUTES by benchmark score, breaking ties by provider free limits."""
    DEFAULT_MODEL_ROUTES.sort(
        key=lambda x: (_get_model_score(x), _get_provider_score(x.provider_name)),
        reverse=True,
    )
    for i, route in enumerate(DEFAULT_MODEL_ROUTES):
        object.__setattr__(route, "rank", i + 1)


_assign_default_ranks()


class ModelCatalog:
    """JSON-backed model catalog that persists user ranking customizations."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._routes: list[ModelRoute] = []

    def initialize(self) -> None:
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        if not os.path.exists(self.path):
            self._routes = DEFAULT_MODEL_ROUTES
            self.save()
            return

        self._routes = self._load_routes()
        self._merge_new_defaults()
        self.save()

    def all_routes(self) -> list[ModelRoute]:
        return sorted(self._routes, key=lambda route: (route.rank, route.provider_name, route.model_id))

    def enabled_routes(self, requested_model: str | None = None) -> list[ModelRoute]:
        routes = [route for route in self.all_routes() if route.enabled]
        if not requested_model or requested_model == "auto":
            return routes

        exact = [
            route
            for route in routes
            if requested_model in {route.route_id, route.model_id, route.display_name}
        ]
        if exact:
            return exact + [route for route in routes if route not in exact]
        return routes

    def replace_routes(self, raw_routes: list[dict[str, Any]]) -> list[ModelRoute]:
        routes = [self._route_from_dict(raw_route) for raw_route in raw_routes]
        seen = set()
        for route in routes:
            if route.route_id in seen:
                raise ValueError(f"Duplicate route_id: {route.route_id}")
            seen.add(route.route_id)
        self._routes = routes
        self.save()
        return self.all_routes()

    def set_route_enabled(self, route_id: str, enabled: bool) -> ModelRoute:
        for index, route in enumerate(self._routes):
            if route.route_id != route_id:
                continue
            updated = ModelRoute(
                route_id=route.route_id,
                provider_name=route.provider_name,
                model_id=route.model_id,
                display_name=route.display_name,
                rank=route.rank,
                enabled=enabled,
                context_window=route.context_window,
                quality=route.quality,
                speed=route.speed,
                cost=route.cost,
                tags=route.tags,
                notes=route.notes,
                source_url=route.source_url,
            )
            self._routes[index] = updated
            self.save()
            return updated
        raise KeyError(f"Unknown route_id: {route_id}")

    def reset_to_defaults(self) -> list[ModelRoute]:
        self._routes = DEFAULT_MODEL_ROUTES.copy()
        self.save()
        return self.all_routes()

    def to_payload(self) -> dict[str, Any]:
        return {
            "object": "list",
            "catalog_path": self.path,
            "data": [asdict(route) for route in self.all_routes()],
        }

    def save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump([asdict(route) for route in self.all_routes()], handle, indent=2)
            handle.write("\n")

    def _load_routes(self) -> list[ModelRoute]:
        with open(self.path, encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, list):
            raise ValueError(f"Model catalog must be a JSON array: {self.path}")
        return [self._route_from_dict(item) for item in raw]

    def _merge_new_defaults(self) -> None:
        defaults = {route.route_id: route for route in DEFAULT_MODEL_ROUTES}
        refreshed: list[ModelRoute] = []

        for route in self._routes:
            default = defaults.get(route.route_id)
            if not default:
                refreshed.append(
                    ModelRoute(
                        route_id=route.route_id,
                        provider_name=route.provider_name,
                        model_id=route.model_id,
                        display_name=route.display_name,
                        rank=route.rank,
                        enabled=route.enabled,
                        context_window=route.context_window,
                        quality=route.quality,
                        speed=route.speed,
                        cost=route.cost,
                        tags=_canonical_tags(route.tags),
                        notes=route.notes,
                        source_url=route.source_url,
                    )
                )
                continue

            refreshed.append(
                ModelRoute(
                    route_id=default.route_id,
                    provider_name=default.provider_name,
                    model_id=default.model_id,
                    display_name=default.display_name,
                    rank=route.rank,
                    enabled=route.enabled,
                    context_window=default.context_window,
                    quality=default.quality,
                    speed=default.speed,
                    cost=default.cost,
                    tags=_canonical_tags(default.tags),
                    notes=default.notes,
                    source_url=default.source_url,
                )
            )

        existing = {route.route_id for route in refreshed}
        refreshed.extend(route for route in DEFAULT_MODEL_ROUTES if route.route_id not in existing)
        self._routes = refreshed

    @staticmethod
    def _route_from_dict(raw: dict[str, Any]) -> ModelRoute:
        if not isinstance(raw, dict):
            raise ValueError("Each model catalog entry must be an object")
        return ModelRoute(
            route_id=str(raw["route_id"]),
            provider_name=str(raw["provider_name"]),
            model_id=str(raw["model_id"]),
            display_name=str(raw.get("display_name") or raw["model_id"]),
            rank=int(raw["rank"]),
            enabled=bool(raw.get("enabled", True)),
            context_window=(
                int(raw["context_window"]) if raw.get("context_window") is not None else None
            ),
            quality=str(raw.get("quality", "unknown")),
            speed=str(raw.get("speed", "unknown")),
            cost=str(raw.get("cost", "free-tier")),
            tags=_canonical_tags([str(tag) for tag in raw.get("tags", [])]),
            notes=str(raw.get("notes", "")),
            source_url=str(raw.get("source_url", "")),
        )
