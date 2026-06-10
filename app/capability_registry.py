from __future__ import annotations

import fnmatch
from typing import Literal

CapabilityStatus = Literal["unknown", "supported", "unsupported", "inconclusive"]

# (provider_name, model_id_glob) -> tag -> (status, evidence)
_REGISTRY: list[tuple[tuple[str, str], dict[str, tuple[CapabilityStatus, str]]]] = [
    # Gemini — function calling + vision on all chat models
    (
        ("gemini", "*"),
        {
            "tool-use": ("supported", "Gemini API supports function calling on chat models"),
            "vision": ("supported", "Gemini chat models accept image parts"),
        },
    ),
    # Groq Compound — native search only, not OpenAI function tools
    (
        ("groq", "groq/compound"),
        {
            "web-search": ("supported", "Groq Compound native web search"),
            "tool-use": ("unsupported", "Native search only; not function tools"),
        },
    ),
    (
        ("groq", "groq/compound-mini"),
        {
            "web-search": ("supported", "Groq Compound Mini native web search"),
            "tool-use": ("unsupported", "Native search only; not function tools"),
        },
    ),
    # Groq chat models (Compound entries above must stay more specific)
    (
        ("groq", "openai/gpt-oss*"),
        {"tool-use": ("supported", "Groq GPT-OSS supports tools")},
    ),
    (
        ("groq", "llama-*"),
        {"tool-use": ("supported", "Groq Llama models support tools")},
    ),
    (
        ("groq", "meta-llama/*"),
        {"tool-use": ("supported", "Groq Llama models support tools")},
    ),
    (
        ("groq", "qwen/*"),
        {"tool-use": ("supported", "Groq Qwen models support tools")},
    ),
    # Cerebras
    (
        ("cerebras", "*"),
        {"tool-use": ("supported", "Cerebras chat models support function calling")},
    ),
    # NVIDIA Build — agentic / coding / instruct models
    (
        ("nvidia", "deepseek-ai/*"),
        {"tool-use": ("supported", "DeepSeek on NVIDIA Build supports function calling")},
    ),
    (
        ("nvidia", "moonshotai/kimi-k2.6"),
        {
            "tool-use": ("supported", "Kimi K2.6 supports tools on NVIDIA Build"),
            "vision": ("supported", "Kimi K2.6 is natively multimodal"),
        },
    ),
    (
        ("nvidia", "moonshotai/kimi-k2*"),
        {"tool-use": ("supported", "Kimi K2 supports tools on NVIDIA Build")},
    ),
    (
        ("openrouter", "moonshotai/kimi-k2.6*"),
        {
            "tool-use": ("supported", "Kimi K2.6 supports tools via OpenRouter"),
            "reasoning": ("supported", "Kimi K2.6 supports thinking mode via OpenRouter"),
        },
    ),
    (
        ("nvidia", "mistralai/mistral-nemotron"),
        {"tool-use": ("supported", "Mistral Nemotron supports tools on NVIDIA Build")},
    ),
    (
        ("nvidia", "mistralai/devstral*"),
        {"tool-use": ("supported", "Devstral coding agent supports tools on NVIDIA Build")},
    ),
    (
        ("nvidia", "mistralai/magistral*"),
        {"tool-use": ("supported", "Magistral supports tools on NVIDIA Build")},
    ),
    (
        ("nvidia", "minimaxai/*"),
        {"tool-use": ("supported", "MiniMax on NVIDIA Build supports function calling")},
    ),
    (
        ("nvidia", "qwen/qwen3-coder*"),
        {"tool-use": ("supported", "Qwen3 Coder supports tools on NVIDIA Build")},
    ),
    (
        ("nvidia", "qwen/*"),
        {"tool-use": ("supported", "Qwen instruct models support tools on NVIDIA Build")},
    ),
    (
        ("nvidia", "meta/llama-*"),
        {"tool-use": ("supported", "Llama instruct models support tools on NVIDIA Build")},
    ),
    (
        ("nvidia", "abacusai/*"),
        {"tool-use": ("supported", "Dracarys Llama instruct supports tools on NVIDIA Build")},
    ),
    (
        ("nvidia", "bytedance/seed-oss*"),
        {"tool-use": ("supported", "Seed OSS instruct supports tools on NVIDIA Build")},
    ),
    (
        ("nvidia", "stepfun-ai/*"),
        {"tool-use": ("supported", "Step models support tools on NVIDIA Build")},
    ),
    (
        ("nvidia", "nvidia/nemotron*"),
        {"tool-use": ("supported", "Nemotron models support tools on NVIDIA Build")},
    ),
    (
        ("nvidia", "google/gemma-*-it"),
        {"tool-use": ("supported", "Gemma IT models support function calling")},
    ),
    (
        ("nvidia", "microsoft/phi-4-multimodal-instruct"),
        {
            "tool-use": ("supported", "Phi-4 multimodal supports tools on NVIDIA Build"),
            "vision": ("supported", "Phi-4 multimodal accepts image/audio parts"),
        },
    ),
    (
        ("nvidia", "mistralai/mistral-large*"),
        {
            "tool-use": ("supported", "Mistral Large 3 supports tools on NVIDIA Build"),
            "vision": ("supported", "Mistral Large 3 accepts vision inputs"),
        },
    ),
    # OpenRouter free router
    (
        ("openrouter", "openrouter/free"),
        {
            "tool-use": ("supported", "OpenRouter free router selects tool-capable models"),
            "web-search": ("supported", "OpenRouter server-side web_search tool"),
        },
    ),
    # OpenRouter free models — curated patterns (no blanket tagging)
    (
        ("openrouter", "nvidia/nemotron*"),
        {"tool-use": ("supported", "Nemotron models support tools via OpenRouter")},
    ),
    (
        ("openrouter", "openai/gpt-oss*"),
        {"tool-use": ("supported", "GPT-OSS supports tools via OpenRouter")},
    ),
    (
        ("openrouter", "qwen/*"),
        {"tool-use": ("supported", "Qwen models support tools via OpenRouter")},
    ),
    (
        ("openrouter", "meta-llama/*"),
        {"tool-use": ("supported", "Llama instruct models support tools via OpenRouter")},
    ),
    (
        ("openrouter", "nousresearch/hermes-*"),
        {"tool-use": ("supported", "Hermes models support tools via OpenRouter")},
    ),
    (
        ("openrouter", "z-ai/glm-*"),
        {"tool-use": ("supported", "GLM models support tools via OpenRouter")},
    ),
    (
        ("openrouter", "minimax/*"),
        {"tool-use": ("supported", "MiniMax models support tools via OpenRouter")},
    ),
    (
        ("openrouter", "inclusionai/ling-*"),
        {"tool-use": ("supported", "Ling models support tools via OpenRouter")},
    ),
    (
        ("openrouter", "poolside/*"),
        {"tool-use": ("supported", "Poolside Laguna models support tools via OpenRouter")},
    ),
    (
        ("openrouter", "tencent/*"),
        {"tool-use": ("supported", "Tencent Hy models support tools via OpenRouter")},
    ),
    (
        ("openrouter", "liquid/*"),
        {"tool-use": ("supported", "Liquid LFM models support tools via OpenRouter")},
    ),
    (
        ("openrouter", "cognitivecomputations/dolphin*"),
        {"tool-use": ("supported", "Dolphin instruct models support tools via OpenRouter")},
    ),
    (
        ("openrouter", "google/gemma-*"),
        {"tool-use": ("supported", "Gemma IT models support tools via OpenRouter")},
    ),
    (
        ("openrouter", "mistralai/pixtral*"),
        {
            "tool-use": ("supported", "Pixtral supports tools via OpenRouter"),
            "vision": ("supported", "Pixtral accepts image inputs"),
        },
    ),
    # SambaNova instruct / agent models
    (
        ("sambanova", "DeepSeek-*"),
        {"tool-use": ("supported", "DeepSeek on SambaNova supports function calling")},
    ),
    (
        ("sambanova", "MiniMax-*"),
        {"tool-use": ("supported", "MiniMax on SambaNova supports function calling")},
    ),
    (
        ("sambanova", "gpt-oss-*"),
        {"tool-use": ("supported", "GPT-OSS on SambaNova supports function calling")},
    ),
    (
        ("sambanova", "Meta-Llama-*"),
        {"tool-use": ("supported", "Llama instruct on SambaNova supports function calling")},
    ),
    (
        ("sambanova", "Llama-*"),
        {"tool-use": ("supported", "Llama instruct on SambaNova supports function calling")},
    ),
    (
        ("sambanova", "gemma-*-it"),
        {"tool-use": ("supported", "Gemma IT on SambaNova supports function calling")},
    ),
]


def registry_claims_for(
    provider_name: str,
    model_id: str,
) -> list[tuple[str, CapabilityStatus, str]]:
    """Return registry capability claims for a provider + model, highest priority last."""
    claims: dict[str, tuple[CapabilityStatus, str]] = {}
    for (provider_pattern, model_pattern), tag_map in _REGISTRY:
        if provider_pattern != "*" and provider_name != provider_pattern:
            continue
        if not fnmatch.fnmatchcase(model_id, model_pattern):
            continue
        claims.update(tag_map)
    return [(tag, status, evidence) for tag, (status, evidence) in sorted(claims.items())]
