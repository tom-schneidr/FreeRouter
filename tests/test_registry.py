from __future__ import annotations

from app.providers.registry import PROVIDER_QUOTAS, build_provider_adapters
from app.settings import Settings


def test_all_providers_have_matching_quotas():
    """Every adapter built must have a corresponding quota definition."""
    settings = Settings()
    adapters = build_provider_adapters(settings)
    adapter_names = {a.name for a in adapters}
    quota_names = {q.name for q in PROVIDER_QUOTAS}
    assert adapter_names == quota_names


def test_nvidia_adapter_uses_openai_compatible_api_settings():
    settings = Settings(
        nvidia_api_key="test-nvidia",
    )

    adapters = {adapter.name: adapter for adapter in build_provider_adapters(settings)}

    assert adapters["nvidia"].api_key == "test-nvidia"
    assert adapters["nvidia"].base_url == "https://integrate.api.nvidia.com/v1"
    assert adapters["nvidia"].default_model == "meta/llama-3.1-8b-instruct"


def test_openrouter_adapter_has_required_headers():
    """OpenRouter free models require HTTP-Referer and X-Title headers."""
    settings = Settings(openrouter_api_key="or-key")
    adapters = {a.name: a for a in build_provider_adapters(settings)}
    headers = adapters["openrouter"].extra_headers
    assert headers is not None
    assert "HTTP-Referer" in headers
    assert "X-Title" in headers


def test_unconfigured_providers_report_not_configured():
    """Providers without API keys should report is_configured = False."""
    settings = Settings(
        cerebras_api_key=None,
        groq_api_key=None,
        gemini_api_key=None,
        nvidia_api_key=None,
        openrouter_api_key=None,
        _env_file=None,
    )
    adapters = build_provider_adapters(settings)
    for adapter in adapters:
        assert adapter.is_configured is False
