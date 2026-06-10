from __future__ import annotations

from time import time

from app.model_catalog import ModelRoute

# Per provider, per diagnosis refresh (see auto_endpoint_diagnosis_interval_seconds).
PROBE_BUDGET_PER_PROVIDER = 12

# After this many seconds, a route is considered fully "due" for re-probe (staleness → 1.0).
PROBE_RECHECK_SECONDS = 7 * 24 * 3600

# Never-probed routes outrank freshly probed top-ranked routes, but lose to equal staleness
# when rank gap is large enough that routing priority should win a tie-break.
NEVER_PROBED_STALENESS = 1.25

STALENESS_WEIGHT = 0.55
RANK_WEIGHT = 0.45


def last_capability_probe_at(route: ModelRoute) -> int | None:
    """Return the newest probe timestamp on this route, if any."""
    probe_times = [
        claim.checked_at
        for claim in route.capabilities.values()
        if claim.source == "probe" and claim.checked_at is not None
    ]
    return max(probe_times) if probe_times else None


def capability_probe_staleness(route: ModelRoute, *, now: int) -> float:
    """0 = just probed, 1 = due for recheck, NEVER_PROBED_STALENESS = never verified."""
    last_probe = last_capability_probe_at(route)
    if last_probe is None:
        return NEVER_PROBED_STALENESS
    age_seconds = max(0, now - last_probe)
    return min(age_seconds / PROBE_RECHECK_SECONDS, 1.0)


def capability_probe_rank_score(route: ModelRoute, *, max_rank: int) -> float:
    """1.0 for rank 1, approaching 0 for the lowest-priority route in the pool."""
    if max_rank <= 0:
        return 1.0
    return (max_rank - route.rank + 1) / max_rank


def capability_probe_priority(
    route: ModelRoute,
    *,
    now: int,
    max_rank: int,
) -> float:
    return STALENESS_WEIGHT * capability_probe_staleness(
        route, now=now
    ) + RANK_WEIGHT * capability_probe_rank_score(route, max_rank=max_rank)


def select_routes_for_capability_probe(
    routes: list[ModelRoute],
    *,
    provider_name: str,
    budget: int = PROBE_BUDGET_PER_PROVIDER,
    now: int | None = None,
) -> list[ModelRoute]:
    """Choose which enabled routes to probe this refresh.

    Algorithm (per provider, each diagnosis refresh):

    1. **Pool** — enabled catalog routes for the provider.
    2. **Staleness** — seconds since the latest ``source: probe`` claim on the route,
       normalized to ``[0, 1]`` over ``PROBE_RECHECK_SECONDS`` (7 days).
       Never-probed routes get ``NEVER_PROBED_STALENESS`` (1.25) so they are checked
       before recently verified low-priority routes.
    3. **Rank bias** — ``(max_rank - rank + 1) / max_rank`` so rank 1 scores 1.0 and
       lower-priority routes score closer to 0.
    4. **Priority** — ``0.55 * staleness + 0.45 * rank_score`` (higher probes first).
    5. **Budget** — take the top ``budget`` routes (default 12 per provider per refresh).

    Full-catalog coverage: with ``N`` enabled routes and budget ``B``, every route is
    due within ``PROBE_RECHECK_SECONDS`` and the priority score ensures never-probed
    and stale routes rotate in while top ranks are re-checked more often when equally stale.
    Expected full sweep time ≈ ``ceil(N / B) * diagnosis_interval`` per provider
    (e.g. 15 routes, B=12, 6h interval → ~12h to touch all routes at least once
    in a cold start; thereafter each route is re-probed at least every 7 days).
    """
    timestamp = now if now is not None else int(time())
    candidates = [
        route
        for route in routes
        if route.provider_name == provider_name and route.enabled
    ]
    if not candidates or budget <= 0:
        return []

    max_rank = max(route.rank for route in candidates)
    scored: list[tuple[float, int, str, ModelRoute]] = []
    for route in candidates:
        scored.append(
            (
                capability_probe_priority(route, now=timestamp, max_rank=max_rank),
                route.rank,
                route.route_id,
                route,
            )
        )

    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [route for _, _, _, route in scored[:budget]]
