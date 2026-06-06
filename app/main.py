from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import (
    anthropic_routes,
    catalog_routes,
    desktop_routes,
    diagnosis_routes,
    monitoring_routes,
    openai_routes,
    static_routes,
)
from app.api.chat_handlers import (
    WEB_SEARCH_TOOL,
    _assistant_text_from_response_body,
    _payload_with_required_web_search,
)
from app.app_services import attach_app_services
from app.factory import build_app_services
from app.react_app import mount_react_app

__all__ = [
    "WEB_SEARCH_TOOL",
    "_assistant_text_from_response_body",
    "_payload_with_required_web_search",
    "app",
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    services = await build_app_services()
    attach_app_services(app, services)
    try:
        yield
    finally:
        await services.shutdown()


app = FastAPI(
    title="FreeRouter",
    version="0.1.0",
    lifespan=lifespan,
    docs_url=None,
)
mount_react_app(app)

app.include_router(static_routes.router)
app.include_router(desktop_routes.router)
app.include_router(monitoring_routes.router)
app.include_router(catalog_routes.router)
app.include_router(diagnosis_routes.router)
app.include_router(openai_routes.router)
app.include_router(anthropic_routes.router)
