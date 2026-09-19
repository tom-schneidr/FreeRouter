from __future__ import annotations

import json
from dataclasses import replace

from app.capability_probe_schedule import (
    next_capability_probe_at,
    select_routes_for_capability_probe,
)
from app.capability_tags import CapabilityClaim, merge_capability_claim
from app.model_catalog import ModelCatalog, ModelRoute, route_id_for
from app.model_discovery import route_from_catalog_item
from app.model_ranking import compute_rank_score, tool_use_behavior_score


def _route(*, rank: int = 1, capabilities: dict[str, CapabilityClaim] | None = None) -> ModelRoute:
    return ModelRoute(
        route_id=route_id_for("openrouter", f"evidence-{rank}"),
        provider_name="openrouter",
        model_id=f"evidence-{rank}:free",
        display_name=f"Evidence {rank}",
        rank=rank,
        tags=["text", "tool-use"],
        capabilities=capabilities or {},
    )


def test_transient_probe_does_not_erase_verified_capability() -> None:
    verified = CapabilityClaim(
        tag="tool-use",
        status="supported",
        source="probe",
        confidence="high",
        checked_at=100,
        last_attempted_at=100,
        evidence="verified profile",
    )
    transient = CapabilityClaim(
        tag="tool-use",
        status="inconclusive",
        source="probe",
        confidence="low",
        checked_at=200,
        last_attempted_at=200,
        next_probe_at=500,
        reason="rate_limited",
        evidence="Rate limited",
    )

    merged = merge_capability_claim(verified, transient, locked=False)

    assert merged.status == "supported"
    assert merged.checked_at == 100
    assert merged.last_attempted_at == 200
    assert merged.next_probe_at == 500
    assert merged.reason == "rate_limited"


def test_probe_scheduler_waits_for_transient_retry_window() -> None:
    route = _route(
        capabilities={
            "tool-use": CapabilityClaim(
                tag="tool-use",
                status="inconclusive",
                source="probe",
                confidence="low",
                checked_at=100,
                last_attempted_at=100,
                next_probe_at=500,
                reason="timeout",
            )
        }
    )

    assert next_capability_probe_at(route) == 500
    assert select_routes_for_capability_probe([route], provider_name="openrouter", now=499) == []
    assert select_routes_for_capability_probe([route], provider_name="openrouter", now=500)


def test_discovery_keeps_structured_provenance_without_confirming_metadata_tools() -> None:
    class Provider:
        name = "openrouter"
        base_url = "https://openrouter.ai/api/v1"

    route = route_from_catalog_item(
        Provider(),
        {
            "id": "evidence/model:free",
            "name": "Evidence Model",
            "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
            "pricing": {"prompt": "0", "completion": "0"},
            "supported_parameters": ["tools", "tool_choice"],
        },
    )

    assert route is not None
    assert route.discovery_source == "openrouter:/models"
    assert route.discovery_metadata["free_evidence"] == "structured_zero_price"
    assert route.discovery_metadata["tool_use_metadata"] == "supported"
    assert "tool-use" not in route.tags


def test_structured_tool_profile_changes_auto_score_with_bounded_adjustment() -> None:
    base = _route()
    claims = {
        "tool-use": CapabilityClaim(
            tag="tool-use",
            status="supported",
            source="probe",
            confidence="high",
            checked_at=100,
        ),
        "tool-use.required-exact-call": CapabilityClaim(
            tag="tool-use.required-exact-call",
            status="supported",
            source="probe",
            confidence="high",
            checked_at=100,
        ),
        "tool-use.auto-selection": CapabilityClaim(
            tag="tool-use.auto-selection",
            status="supported",
            source="probe",
            confidence="high",
            checked_at=100,
        ),
        "tool-use.tool-result-continuation": CapabilityClaim(
            tag="tool-use.tool-result-continuation",
            status="supported",
            source="probe",
            confidence="high",
            checked_at=100,
        ),
        "tool-use.multi-turn-stability": CapabilityClaim(
            tag="tool-use.multi-turn-stability",
            status="supported",
            source="probe",
            confidence="high",
            checked_at=100,
        ),
    }
    verified = replace(base, capabilities=claims)

    assert tool_use_behavior_score(verified) > tool_use_behavior_score(base)
    assert compute_rank_score(verified) > compute_rank_score(base)
    assert compute_rank_score(verified) - compute_rank_score(base) <= 500


def test_incomplete_tool_profile_does_not_receive_later_dimension_credit() -> None:
    base = _route()
    partial = replace(
        base,
        capabilities={
            "tool-use": CapabilityClaim(
                tag="tool-use",
                status="supported",
                source="probe",
                confidence="high",
                checked_at=100,
            ),
            "tool-use.required-exact-call": CapabilityClaim(
                tag="tool-use.required-exact-call",
                status="supported",
                source="probe",
                confidence="high",
                checked_at=100,
            ),
            "tool-use.auto-selection": CapabilityClaim(
                tag="tool-use.auto-selection",
                status="supported",
                source="probe",
                confidence="high",
                checked_at=100,
            ),
            "tool-use.tool-result-continuation": CapabilityClaim(
                tag="tool-use.tool-result-continuation",
                status="inconclusive",
                source="probe",
                confidence="low",
                checked_at=100,
            ),
            "tool-use.multi-turn-stability": CapabilityClaim(
                tag="tool-use.multi-turn-stability",
                status="supported",
                source="probe",
                confidence="high",
                checked_at=100,
            ),
        },
    )
    stability_only = replace(
        partial,
        capabilities={
            tag: claim
            for tag, claim in partial.capabilities.items()
            if tag != "tool-use.auto-selection"
        },
    )

    assert tool_use_behavior_score(partial) == tool_use_behavior_score(stability_only) + 80


def test_catalog_restart_preserves_probe_evidence(tmp_path) -> None:
    path = str(tmp_path / "models.json")
    catalog = ModelCatalog(path)
    catalog.replace_routes(
        [
            {
                "route_id": "openrouter-evidence-model-free",
                "provider_name": "openrouter",
                "model_id": "evidence/model:free",
                "display_name": "Evidence Model",
                "rank": 1,
                "enabled": True,
                "tags": ["text", "tool-use"],
                "capabilities": {
                    "text": {
                        "tag": "text",
                        "status": "supported",
                        "source": "manual",
                        "confidence": "high",
                    },
                    "tool-use": {
                        "tag": "tool-use",
                        "status": "supported",
                        "source": "probe",
                        "confidence": "high",
                        "checked_at": 123,
                        "evidence": "verified profile",
                    },
                },
            }
        ]
    )
    reloaded = ModelCatalog(path)
    reloaded.initialize()

    route = next(
        route
        for route in reloaded.all_routes()
        if route.route_id == "openrouter-evidence-model-free"
    )
    assert route.capabilities["tool-use"].source == "probe"
    assert route.capabilities["tool-use"].checked_at == 123
    assert "tool-use" in route.tags


def test_catalog_save_is_versioned_but_legacy_arrays_still_load(tmp_path) -> None:
    path = str(tmp_path / "models.json")
    catalog = ModelCatalog(path)
    catalog.replace_routes([])

    saved = json.loads((tmp_path / "models.json").read_text(encoding="utf-8"))
    assert saved["schema_version"] == 2
    assert saved["routes"] == []

    (tmp_path / "legacy.json").write_text("[]\n", encoding="utf-8")
    legacy = ModelCatalog(str(tmp_path / "legacy.json"))
    legacy.initialize()
    assert legacy.all_routes()
