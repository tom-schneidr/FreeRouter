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
    max_concurrent_requests: int = 20
    #: When True, SSE chat completions release the concurrency slot once a route is committed
    #: (first streamed chunk). Long-running streams then no longer block other requests from
    #: acquiring MAX_CONCURRENT_REQUESTS. Set False to bound simultaneous open streams strictly.
    streaming_release_slot_after_route_selected: bool = True
    request_queue_timeout_seconds: float = 30
    request_queue_max_waiting_requests: int | None = 200
    sqlite_busy_timeout_ms: int = 5000
    http_max_connections: int = 400
    http_max_keepalive_connections: int = 200
    http_keepalive_expiry_seconds: float = 30
    #: Delay between SSE content chunks when replaying a full completion (0 = no delay).
    sse_chunk_replay_sleep_seconds: float = 0.0
    gateway_model_name: str = "auto"
    auto_endpoint_diagnosis_enabled: bool = True
    auto_endpoint_maintenance_enabled: bool = True
    auto_endpoint_diagnosis_interval_seconds: int = 21_600
    auto_endpoint_diagnosis_startup_delay_seconds: int = 10
    endpoint_diagnosis_supervisor_enabled: bool = False
    endpoint_diagnosis_supervisor_model: str | None = None

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
