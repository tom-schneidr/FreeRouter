from __future__ import annotations

from app.providers.base import ProviderError


def looks_like_missing_model(exc: ProviderError) -> bool:
    """Classify provider errors that strongly indicate an unavailable model id."""
    if exc.status_code in {404, 410}:
        return True
    if exc.status_code not in {400, 422} or not exc.body:
        return False
    body = exc.body.lower()
    return (
        "model" in body
        and (
            "not found" in body
            or "does not exist" in body
            or "not exist" in body
            or "unknown model" in body
        )
    )
