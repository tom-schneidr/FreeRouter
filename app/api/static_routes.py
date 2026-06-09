from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response

from app.ui.brand import FAVICON_PATH, LOGO_PATH
from app.ui.docs_page import swagger_docs_html

router = APIRouter()


@router.get("/", include_in_schema=False)
async def index() -> RedirectResponse:
    return RedirectResponse(url="/app", status_code=307)


@router.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Response:
    return Response(content=FAVICON_PATH.read_bytes(), media_type="image/png")


@router.get("/brand/favicon.png", include_in_schema=False)
async def brand_favicon() -> Response:
    return Response(content=FAVICON_PATH.read_bytes(), media_type="image/png")


@router.get("/brand/logo.png", include_in_schema=False)
async def brand_logo() -> Response:
    return Response(content=LOGO_PATH.read_bytes(), media_type="image/png")


@router.get("/docs", include_in_schema=False)
async def swagger_docs_page(request: Request) -> HTMLResponse:
    return swagger_docs_html(
        openapi_url=request.app.openapi_url,
        title=f"{request.app.title} API",
    )
