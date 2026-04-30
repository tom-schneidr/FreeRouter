from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-driven gateway configuration loaded from .env and environment variables."""
    cerebras_api_key: str | None = None
    groq_api_key: str | None = None
    gemini_api_key: str | None = None
    nvidia_api_key: str | None = None
    openrouter_api_key: str | None = None

    database_path: str = "./data/gateway.sqlite3"
    model_catalog_path: str = "./data/model_catalog.json"
    request_timeout_seconds: float = 90
    gateway_model_name: str = "auto"
    auto_endpoint_diagnosis_enabled: bool = True
    auto_endpoint_diagnosis_interval_seconds: int = 21_600
    auto_endpoint_diagnosis_startup_delay_seconds: int = 10
    endpoint_diagnosis_supervisor_enabled: bool = False
    endpoint_diagnosis_supervisor_model: str | None = None

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
