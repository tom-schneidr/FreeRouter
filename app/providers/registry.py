from __future__ import annotations

from app.providers.base import ProviderAdapter
from app.settings import Settings
from app.state import ProviderQuota

PROVIDER_QUOTAS = [
    ProviderQuota(
        name="cerebras",
        tokens_per_day=1_000_000,
        requests_per_day=None,
        requests_per_minute=None,
    ),
    ProviderQuota(
        name="groq",
        tokens_per_day=500_000,
        requests_per_day=None,
        requests_per_minute=30,
    ),
    ProviderQuota(
        name="gemini",
        tokens_per_day=None,
        requests_per_day=250,
        requests_per_minute=None,
    ),
    ProviderQuota(
        name="nvidia",
        tokens_per_day=None,
        requests_per_day=None,
        requests_per_minute=None,
    ),
    ProviderQuota(
        name="openrouter",
        tokens_per_day=None,
        requests_per_day=None,
        requests_per_minute=None,
    ),
]


def build_provider_adapters(settings: Settings) -> list[ProviderAdapter]:
    return [
        ProviderAdapter(
            name="cerebras",
            api_key=settings.cerebras_api_key,
            base_url="https://api.cerebras.ai/v1",
            default_model="llama3.1-8b",
            max_context_tokens=8192,
        ),
        ProviderAdapter(
            name="groq",
            api_key=settings.groq_api_key,
            base_url="https://api.groq.com/openai/v1",
            default_model="llama-3.1-8b-instant",
            max_context_tokens=8192,
        ),
        ProviderAdapter(
            name="gemini",
            api_key=settings.gemini_api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai",
            default_model="gemini-2.0-flash",
            max_context_tokens=1_000_000,
        ),
        ProviderAdapter(
            name="nvidia",
            api_key=settings.nvidia_api_key,
            base_url="https://integrate.api.nvidia.com/v1",
            default_model="meta/llama-3.1-8b-instruct",
            max_context_tokens=None,
        ),
        ProviderAdapter(
            name="openrouter",
            api_key=settings.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
            default_model="meta-llama/llama-3.1-8b-instruct:free",
            max_context_tokens=None,
            extra_headers={
                "HTTP-Referer": "http://localhost:8000",
                "X-Title": "FreeRouter",
            },
        ),
    ]
