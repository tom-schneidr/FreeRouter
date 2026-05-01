from __future__ import annotations

from app.model_catalog import (
    CANONICAL_MODEL_TAGS,
    DEFAULT_MODEL_ROUTES,
    NON_TEXT_ROUTE_TAGS,
    ModelCatalog,
    ModelRoute,
    _get_model_score,
    is_text_chat_route,
)


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


def test_specialized_models_are_excluded_from_defaults():
    """Non-chat/specialized routes should not appear in the default catalog."""
    assert all(is_text_chat_route(route) for route in DEFAULT_MODEL_ROUTES)


def test_no_duplicate_route_ids():
    """Every route_id in the default catalog must be unique."""
    ids = [route.route_id for route in DEFAULT_MODEL_ROUTES]
    assert len(ids) == len(set(ids)), (
        f"Duplicate route_ids found: {[x for x in ids if ids.count(x) > 1]}"
    )


def test_default_routes_only_use_canonical_tags():
    """Tags should stay focused on user-visible capabilities."""
    unknown_tags = sorted(
        {
            tag
            for route in DEFAULT_MODEL_ROUTES
            for tag in route.tags
            if tag not in CANONICAL_MODEL_TAGS
        }
    )
    assert unknown_tags == []


def test_text_is_available_as_a_filterable_capability():
    """Text should be represented explicitly in the capability filters."""
    assert "text" in CANONICAL_MODEL_TAGS
    assert any("text" in route.tags for route in DEFAULT_MODEL_ROUTES)


def test_default_routes_are_text_input_text_output_only():
    assert DEFAULT_MODEL_ROUTES
    assert all(is_text_chat_route(route) for route in DEFAULT_MODEL_ROUTES)
    assert any("vision" in route.tags or "audio" in route.tags for route in DEFAULT_MODEL_ROUTES)
    assert not any(set(route.tags) & NON_TEXT_ROUTE_TAGS for route in DEFAULT_MODEL_ROUTES)


def test_every_route_has_provider_and_model_id():
    """Every route must have non-empty provider_name and model_id."""
    for route in DEFAULT_MODEL_ROUTES:
        assert route.provider_name, f"Route {route.route_id} has empty provider_name"
        assert route.model_id, f"Route {route.route_id} has empty model_id"


def test_catalog_replace_builds_sorted_cache_for_consecutive_reads(tmp_path, monkeypatch):
    import app.model_catalog as mc

    sorts: list[bool] = []

    def wrap(routes: list[ModelRoute]) -> list[ModelRoute]:
        sorts.append(True)
        return sorted(routes, key=lambda route: (route.rank, route.provider_name, route.model_id))

    monkeypatch.setattr(mc, "_sort_routes_for_catalog", wrap)
    catalog = ModelCatalog(str(tmp_path / "models.json"))
    catalog.replace_routes(
        [
            {
                "route_id": "z",
                "provider_name": "p",
                "model_id": "m2",
                "display_name": "Z",
                "rank": 2,
                "enabled": True,
            },
            {
                "route_id": "a",
                "provider_name": "p",
                "model_id": "m1",
                "display_name": "A",
                "rank": 1,
                "enabled": True,
            },
        ]
    )
    after_replace = len(sorts)
    catalog.all_routes()
    catalog.all_routes()
    assert len(sorts) == after_replace
    assert len(catalog.all_routes()) == 2


def test_catalog_initialize_creates_file(tmp_path):
    """Catalog should create the JSON file if it doesn't exist."""
    catalog = ModelCatalog(str(tmp_path / "models.json"))
    catalog.initialize()
    assert (tmp_path / "models.json").exists()


def test_catalog_merge_refreshes_default_metadata_and_preserves_routing(tmp_path):
    """Re-initialization should refresh model facts while preserving routing choices."""
    catalog = ModelCatalog(str(tmp_path / "models.json"))
    catalog.initialize()

    # Simulate stale persisted metadata plus user routing choices.
    routes = catalog.all_routes()
    original = routes[0]
    customized = [
        {
            "route_id": original.route_id,
            "provider_name": original.provider_name,
            "model_id": original.model_id,
            "display_name": "User Custom Name",
            "rank": 999,
            "enabled": False,
            "tags": ["chat", "deprecated-soon", "fast"],
        }
    ]
    catalog.replace_routes(customized)

    # Re-init should merge defaults back in
    catalog2 = ModelCatalog(str(tmp_path / "models.json"))
    catalog2.initialize()

    user_route = next(r for r in catalog2.all_routes() if r.route_id == original.route_id)
    assert user_route.display_name == original.display_name
    assert user_route.rank == 999
    assert user_route.enabled is False
    assert user_route.tags == original.tags

    # And defaults should have been merged back
    assert len(catalog2.all_routes()) > 1


def test_catalog_enabled_routes_filters_disabled(tmp_path):
    """enabled_routes() should not return disabled routes."""
    catalog = ModelCatalog(str(tmp_path / "models.json"))
    catalog.replace_routes(
        [
            {
                "route_id": "a",
                "provider_name": "p",
                "model_id": "m1",
                "display_name": "A",
                "rank": 1,
                "enabled": True,
            },
            {
                "route_id": "b",
                "provider_name": "p",
                "model_id": "m2",
                "display_name": "B",
                "rank": 2,
                "enabled": False,
            },
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
            {
                "route_id": "a",
                "provider_name": "p",
                "model_id": "m1",
                "display_name": "Model A",
                "rank": 1,
                "enabled": True,
            },
            {
                "route_id": "b",
                "provider_name": "p",
                "model_id": "m2",
                "display_name": "Model B",
                "rank": 2,
                "enabled": True,
            },
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
                {
                    "route_id": "dup",
                    "provider_name": "p",
                    "model_id": "m",
                    "display_name": "D",
                    "rank": 1,
                },
                {
                    "route_id": "dup",
                    "provider_name": "p",
                    "model_id": "m",
                    "display_name": "D",
                    "rank": 2,
                },
            ]
        )


def test_get_model_score_gpt_oss_high():
    """GPT-OSS should score very high (benchmark leader)."""
    route = ModelRoute(
        route_id="test",
        provider_name="groq",
        model_id="openai/gpt-oss-120b",
        display_name="GPT OSS 120B",
        rank=1,
        tags=["reasoning"],
    )
    assert _get_model_score(route) > 90000


def test_get_model_score_safety_very_low():
    """Safety models should score far below zero."""
    route = ModelRoute(
        route_id="test",
        provider_name="nvidia",
        model_id="nvidia/safety-guard",
        display_name="Safety Guard",
        rank=1,
        tags=["safety", "moderation"],
    )
    assert _get_model_score(route) < 0


def test_discovered_routes_are_inserted_by_auto_rank_and_disabled(tmp_path):
    catalog = ModelCatalog(str(tmp_path / "models.json"))
    catalog.replace_routes(
        [
            {
                "route_id": "baseline",
                "provider_name": "groq",
                "model_id": "openai/gpt-oss-20b",
                "display_name": "GPT OSS 20B",
                "rank": 1,
                "enabled": True,
            }
        ]
    )
    discovered = ModelRoute(
        route_id="new-high",
        provider_name="gemini",
        model_id="gemini-3.1-pro-preview",
        display_name="Gemini 3.1 Pro Preview",
        rank=999,
        enabled=True,
        tags=["reasoning", "text"],
    )
    added = catalog.add_discovered_routes([discovered])
    all_routes = catalog.all_routes()

    assert len(added) == 1
    assert added[0].enabled is False
    assert all_routes[0].route_id == "new-high"
    assert all_routes[0].rank_score is not None


def test_auto_rank_keeps_multimodal_text_routes(tmp_path):
    catalog = ModelCatalog(str(tmp_path / "models.json"))
    catalog.replace_routes(
        [
            {
                "route_id": "text-route",
                "provider_name": "p",
                "model_id": "text-model",
                "display_name": "Text Model",
                "rank": 1,
                "enabled": True,
                "tags": ["text"],
            },
            {
                "route_id": "vision-route",
                "provider_name": "p",
                "model_id": "vision-chat-model",
                "display_name": "Vision Chat Model",
                "rank": 2,
                "enabled": True,
                "tags": ["text", "vision"],
            },
        ]
    )

    assert [route.route_id for route in catalog.all_routes()] == ["text-route", "vision-route"]
