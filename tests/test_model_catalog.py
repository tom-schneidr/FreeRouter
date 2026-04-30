from __future__ import annotations

from app.model_catalog import DEFAULT_MODEL_ROUTES, ModelCatalog, ModelRoute, _get_model_score


def test_default_routes_have_contiguous_ranks():
    """Ranks should be 1, 2, 3, ... with no gaps."""
    ranks = [route.rank for route in DEFAULT_MODEL_ROUTES]
    assert ranks == list(range(1, len(DEFAULT_MODEL_ROUTES) + 1))


def test_default_routes_are_sorted_descending_by_score():
    """Higher-scored models should have lower rank numbers (tried first)."""
    scores = [_get_model_score(route) for route in DEFAULT_MODEL_ROUTES]
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1], (
            f"Route rank {i + 1} (score {scores[i]}) should be >= "
            f"route rank {i + 2} (score {scores[i + 1]})"
        )


def test_safety_models_ranked_last():
    """Safety/guard/PII models should always sort below chat models."""
    safety_routes = [r for r in DEFAULT_MODEL_ROUTES if not r.enabled]
    chat_routes = [r for r in DEFAULT_MODEL_ROUTES if r.enabled]
    if safety_routes and chat_routes:
        worst_chat = max(r.rank for r in chat_routes)
        best_safety = min(r.rank for r in safety_routes)
        assert best_safety > worst_chat, "Disabled safety models should rank below all enabled chat models"


def test_no_duplicate_route_ids():
    """Every route_id in the default catalog must be unique."""
    ids = [route.route_id for route in DEFAULT_MODEL_ROUTES]
    assert len(ids) == len(set(ids)), f"Duplicate route_ids found: {[x for x in ids if ids.count(x) > 1]}"


def test_every_route_has_provider_and_model_id():
    """Every route must have non-empty provider_name and model_id."""
    for route in DEFAULT_MODEL_ROUTES:
        assert route.provider_name, f"Route {route.route_id} has empty provider_name"
        assert route.model_id, f"Route {route.route_id} has empty model_id"


def test_catalog_initialize_creates_file(tmp_path):
    """Catalog should create the JSON file if it doesn't exist."""
    catalog = ModelCatalog(str(tmp_path / "models.json"))
    catalog.initialize()
    assert (tmp_path / "models.json").exists()
    assert len(catalog.all_routes()) == len(DEFAULT_MODEL_ROUTES)


def test_catalog_merge_preserves_user_customizations(tmp_path):
    """Re-initialization should keep user routes and add new defaults."""
    catalog = ModelCatalog(str(tmp_path / "models.json"))
    catalog.initialize()

    # Simulate user customization: change rank of first route
    routes = catalog.all_routes()
    customized = [
        {
            "route_id": routes[0].route_id,
            "provider_name": routes[0].provider_name,
            "model_id": routes[0].model_id,
            "display_name": "User Custom Name",
            "rank": 999,
            "enabled": False,
        }
    ]
    catalog.replace_routes(customized)

    # Re-init should merge defaults back in
    catalog2 = ModelCatalog(str(tmp_path / "models.json"))
    catalog2.initialize()

    # The user's customized route should still exist
    user_route = next(r for r in catalog2.all_routes() if r.route_id == routes[0].route_id)
    assert user_route.display_name == "User Custom Name"
    assert user_route.rank == 999
    assert user_route.enabled is False

    # And defaults should have been merged back
    assert len(catalog2.all_routes()) > 1


def test_catalog_enabled_routes_filters_disabled(tmp_path):
    """enabled_routes() should not return disabled routes."""
    catalog = ModelCatalog(str(tmp_path / "models.json"))
    catalog.replace_routes(
        [
            {"route_id": "a", "provider_name": "p", "model_id": "m1", "display_name": "A", "rank": 1, "enabled": True},
            {"route_id": "b", "provider_name": "p", "model_id": "m2", "display_name": "B", "rank": 2, "enabled": False},
        ]
    )
    enabled = catalog.enabled_routes()
    assert len(enabled) == 1
    assert enabled[0].route_id == "a"


def test_catalog_enabled_routes_prioritizes_requested_model(tmp_path):
    """When a specific model is requested, it should be returned first."""
    catalog = ModelCatalog(str(tmp_path / "models.json"))
    catalog.replace_routes(
        [
            {"route_id": "a", "provider_name": "p", "model_id": "m1", "display_name": "Model A", "rank": 1, "enabled": True},
            {"route_id": "b", "provider_name": "p", "model_id": "m2", "display_name": "Model B", "rank": 2, "enabled": True},
        ]
    )
    routes = catalog.enabled_routes("b")
    assert routes[0].route_id == "b"
    assert len(routes) == 2  # fallbacks still included


def test_catalog_replace_rejects_duplicates(tmp_path):
    """replace_routes should reject duplicate route_ids."""
    catalog = ModelCatalog(str(tmp_path / "models.json"))
    import pytest
    with pytest.raises(ValueError, match="Duplicate"):
        catalog.replace_routes(
            [
                {"route_id": "dup", "provider_name": "p", "model_id": "m", "display_name": "D", "rank": 1},
                {"route_id": "dup", "provider_name": "p", "model_id": "m", "display_name": "D", "rank": 2},
            ]
        )


def test_get_model_score_gpt_oss_high():
    """GPT-OSS should score very high (benchmark leader)."""
    route = ModelRoute(
        route_id="test", provider_name="groq", model_id="openai/gpt-oss-120b",
        display_name="GPT OSS 120B", rank=1, tags=["chat", "reasoning"],
    )
    assert _get_model_score(route) > 90000


def test_get_model_score_safety_very_low():
    """Safety models should score far below zero."""
    route = ModelRoute(
        route_id="test", provider_name="nvidia", model_id="nvidia/safety-guard",
        display_name="Safety Guard", rank=1, tags=["safety", "moderation"],
    )
    assert _get_model_score(route) < 0
