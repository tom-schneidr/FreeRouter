from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

from app.capability_registry import registry_claims_for

if TYPE_CHECKING:
    from app.model_catalog import ModelRoute

CANONICAL_MODEL_TAGS = frozenset({
    "text",
    "reasoning",
    "coding",
    "tool-use",
    "json-schema",
    "web-search",
    "vision",
    "audio",
    "safety",
    "moderation",
    "translation",
    "classification",
    "rag",
})

CapabilityStatus = Literal["unknown", "supported", "unsupported", "inconclusive"]
CapabilitySource = Literal["provider_metadata", "registry", "probe", "runtime", "manual"]
CapabilityConfidence = Literal["high", "medium", "low"]

HARD_ROUTING_TAGS = frozenset({"tool-use", "web-search", "vision", "json-schema"})
SOFT_ROUTING_TAGS = frozenset({"reasoning", "coding"})
CONFIRMED_CAPABILITY_SOURCES = frozenset({"probe", "runtime"})
TAGS_REQUIRING_CONFIRMATION = frozenset({"tool-use"})


@dataclass(frozen=True)
class CapabilityClaim:
    tag: str
    status: CapabilityStatus
    source: CapabilitySource
    confidence: CapabilityConfidence
    checked_at: int | None = None
    evidence: str = ""


def capability_qualifies_for_tag(claim: CapabilityClaim, tag: str) -> bool:
    if claim.status != "supported":
        return False
    if tag in TAGS_REQUIRING_CONFIRMATION:
        return claim.source in CONFIRMED_CAPABILITY_SOURCES
    return True


def derive_tags_from_capabilities(capabilities: dict[str, CapabilityClaim]) -> list[str]:
    return [
        tag
        for tag in sorted(capabilities)
        if tag in CANONICAL_MODEL_TAGS
        and capability_qualifies_for_tag(capabilities[tag], tag)
    ]


def should_probe_tool_use(route: ModelRoute) -> bool:  # noqa: F821
    """Probe tool-use when the registry hints support or a prior probe needs refresh."""
    claim = route.capabilities.get("tool-use")
    if claim and claim.source in CONFIRMED_CAPABILITY_SOURCES:
        return True
    for tag, status, _ in registry_claims_for(route.provider_name, route.model_id):
        if tag == "tool-use":
            return status != "unsupported"
    return False


def normalize_route_tool_use_policy(route: ModelRoute) -> ModelRoute:  # noqa: F821
    """Remove manual/registry-only tool-use tags; keep probe/runtime confirmations."""
    capabilities = {
        tag: claim
        for tag, claim in route.capabilities.items()
        if not (tag == "tool-use" and claim.source == "manual")
    }
    locks = {tag for tag in route.tag_locks if tag != "tool-use"}
    interim = replace(
        route,
        capabilities=capabilities,
        tag_locks=frozenset(locks),
    )
    pipelined = apply_capability_pipeline(interim)
    derived = derive_tags_from_capabilities(pipelined.capabilities)
    tags = derived if derived else [tag for tag in pipelined.tags if tag != "tool-use"]
    if "text" not in tags:
        tags = ["text", *tags]
    return replace(pipelined, tags=list(dict.fromkeys(tags)))


def tags_to_capabilities(
    tags: list[str],
    *,
    source: CapabilitySource = "manual",
    confidence: CapabilityConfidence = "high",
    checked_at: int | None = None,
) -> dict[str, CapabilityClaim]:
    timestamp = checked_at if checked_at is not None else int(time.time())
    return {
        tag: CapabilityClaim(
            tag=tag,
            status="supported",
            source=source,
            confidence=confidence,
            checked_at=timestamp,
        )
        for tag in tags
        if tag in CANONICAL_MODEL_TAGS
    }


def capability_claim_from_dict(raw: dict[str, object]) -> CapabilityClaim:
    return CapabilityClaim(
        tag=str(raw["tag"]),
        status=str(raw["status"]),  # type: ignore[arg-type]
        source=str(raw["source"]),  # type: ignore[arg-type]
        confidence=str(raw["confidence"]),  # type: ignore[arg-type]
        checked_at=int(raw["checked_at"]) if raw.get("checked_at") is not None else None,
        evidence=str(raw.get("evidence") or ""),
    )


def capability_claim_to_dict(claim: CapabilityClaim) -> dict[str, object]:
    return {
        "tag": claim.tag,
        "status": claim.status,
        "source": claim.source,
        "confidence": claim.confidence,
        "checked_at": claim.checked_at,
        "evidence": claim.evidence,
    }


def merge_capability_claim(
    existing: CapabilityClaim | None,
    incoming: CapabilityClaim,
    *,
    locked: bool,
) -> CapabilityClaim:
    if locked and existing is not None:
        return existing
    if existing is None:
        return incoming
    if (
        incoming.status == "unsupported"
        and incoming.source in CONFIRMED_CAPABILITY_SOURCES
        and existing.status == "supported"
    ):
        return incoming
    priority = {
        "manual": 5,
        "probe": 4,
        "runtime": 3,
        "registry": 2,
        "provider_metadata": 1,
    }
    if priority.get(incoming.source, 0) > priority.get(existing.source, 0):
        return incoming
    if priority.get(incoming.source, 0) < priority.get(existing.source, 0):
        return existing
    confidence_rank = {"high": 3, "medium": 2, "low": 1}
    if confidence_rank.get(incoming.confidence, 0) >= confidence_rank.get(existing.confidence, 0):
        return incoming
    return existing


def apply_capability_pipeline(
    route: ModelRoute,
    *,
    metadata_tags: list[str] | None = None,
) -> ModelRoute:  # noqa: F821
    """Merge manual tags, provider metadata, and registry into capabilities + derived tags."""
    capabilities = dict(route.capabilities)
    if not capabilities:
        capabilities = tags_to_capabilities(route.tags, source="manual")

    if "text" in route.tags and "text" not in capabilities:
        capabilities["text"] = CapabilityClaim(
            tag="text",
            status="supported",
            source="registry",
            confidence="high",
            checked_at=int(time.time()),
            evidence="Baseline text chat route",
        )

    if metadata_tags:
        for tag in metadata_tags:
            if tag not in CANONICAL_MODEL_TAGS:
                continue
            claim = CapabilityClaim(
                tag=tag,
                status="supported",
                source="provider_metadata",
                confidence="medium",
                checked_at=int(time.time()),
            )
            capabilities[tag] = merge_capability_claim(
                capabilities.get(tag),
                claim,
                locked=tag in route.tag_locks,
            )

    for tag, status, evidence in registry_claims_for(route.provider_name, route.model_id):
        if tag not in CANONICAL_MODEL_TAGS:
            continue
        claim = CapabilityClaim(
            tag=tag,
            status=status,
            source="registry",
            confidence="high",
            checked_at=int(time.time()),
            evidence=evidence,
        )
        capabilities[tag] = merge_capability_claim(
            capabilities.get(tag),
            claim,
            locked=tag in route.tag_locks,
        )

    derived_tags = derive_tags_from_capabilities(capabilities)
    return replace(route, capabilities=capabilities, tags=derived_tags)


def finalize_route(route: ModelRoute, *, metadata_tags: list[str] | None = None) -> ModelRoute:  # noqa: F821
    return apply_capability_pipeline(route, metadata_tags=metadata_tags)


def route_satisfies_capabilities(route: ModelRoute, required: frozenset[str]) -> bool:  # noqa: F821
    """Return true when route tags cover requirements and hard tags are verified."""
    tag_set = set(route.tags)
    if not required.issubset(tag_set):
        return False
    if not route.capabilities:
        if "tool-use" in required:
            return "tool-use" in tag_set
        return True
    for tag in required:
        if tag not in HARD_ROUTING_TAGS:
            continue
        claim = route.capabilities.get(tag)
        if tag in TAGS_REQUIRING_CONFIRMATION:
            if claim is None or not capability_qualifies_for_tag(claim, tag):
                return False
            continue
        if claim is None:
            continue
        if claim.status != "supported":
            return False
    return True


def apply_probe_claims(
    route: ModelRoute,  # noqa: F821
    probe_claims: dict[str, CapabilityClaim],
) -> ModelRoute:  # noqa: F821
    capabilities = dict(route.capabilities)
    if not capabilities:
        capabilities = tags_to_capabilities(route.tags, source="manual")
    for tag, claim in probe_claims.items():
        capabilities[tag] = merge_capability_claim(
            capabilities.get(tag),
            claim,
            locked=tag in route.tag_locks,
        )
    derived_tags = derive_tags_from_capabilities(capabilities)
    return replace(route, capabilities=capabilities, tags=derived_tags)


def apply_runtime_claim(
    route: ModelRoute,  # noqa: F821
    tag: str,
    *,
    status: CapabilityStatus,
    evidence: str,
) -> ModelRoute:  # noqa: F821
    if tag in route.tag_locks:
        return route
    capabilities = dict(route.capabilities)
    if not capabilities:
        capabilities = tags_to_capabilities(route.tags, source="manual")
    claim = CapabilityClaim(
        tag=tag,
        status=status,
        source="runtime",
        confidence="medium",
        checked_at=int(time.time()),
        evidence=evidence[:240],
    )
    capabilities[tag] = merge_capability_claim(capabilities.get(tag), claim, locked=False)
    return replace(route, capabilities=capabilities, tags=derive_tags_from_capabilities(capabilities))


def with_manual_tags(
    route: ModelRoute,  # noqa: F821
    tags: list[str],
    *,
    lock_tags: bool = True,
) -> ModelRoute:  # noqa: F821
    """Attach manually curated tags (and optional locks) before running the pipeline."""
    locks = set(route.tag_locks)
    if lock_tags:
        locks.update(tag for tag in tags if tag in CANONICAL_MODEL_TAGS)
    base = replace(
        route,
        tags=list(dict.fromkeys(tags)),
        capabilities=tags_to_capabilities(tags, source="manual"),
        tag_locks=frozenset(locks),
    )
    return apply_capability_pipeline(base)
