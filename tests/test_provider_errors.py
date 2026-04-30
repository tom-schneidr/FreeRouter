from __future__ import annotations

from app.provider_errors import looks_like_missing_model
from app.providers.base import ProviderError


def test_missing_model_classifier_accepts_clear_not_found_statuses():
    assert looks_like_missing_model(ProviderError("missing", status_code=404))
    assert looks_like_missing_model(ProviderError("gone", status_code=410))


def test_missing_model_classifier_accepts_model_specific_bad_request_body():
    error = ProviderError(
        "bad request",
        status_code=400,
        body='{"error": {"message": "Unknown model: old-model"}}',
    )

    assert looks_like_missing_model(error)


def test_missing_model_classifier_rejects_non_model_provider_errors():
    error = ProviderError(
        "bad request",
        status_code=400,
        body='{"error": {"message": "Invalid request payload"}}',
    )

    assert not looks_like_missing_model(error)
