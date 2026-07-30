import { describe, expect, it } from "vitest";
import {
  filterSentinelRoutes,
  formatEvidenceAge,
  routeTrustLabel,
  type SentinelRoute,
} from "./sentinelLogic";

function route(overrides: Partial<SentinelRoute> = {}): SentinelRoute {
  return {
    route_id: "route-1",
    provider_name: "test",
    model_id: "model",
    display_name: "Test model",
    rank: 1,
    enabled: true,
    configured: true,
    cost: "free-tier",
    zero_cost: true,
    speed: "fast",
    quality: "good",
    evaluation: null,
    ...overrides,
  };
}

describe("Sentinel route presentation", () => {
  it("makes missing evidence explicit", () => {
    expect(routeTrustLabel(route())).toBe("Needs evidence");
  });

  it("filters by readiness and searchable provider metadata", () => {
    const ready = route({
      evaluation: {
        run_id: "run",
        score: 100,
        readiness: "ready",
        completed_at: 100,
        duration_ms: 12,
        summary: "Ready",
        probes: [],
      },
    });
    const blocked = route({
      route_id: "paid",
      provider_name: "paid-cloud",
      zero_cost: false,
    });
    expect(filterSentinelRoutes([ready, blocked], "", "ready")).toEqual([ready]);
    expect(filterSentinelRoutes([ready, blocked], "cloud", "all")).toEqual([blocked]);
  });

  it("formats compact evidence ages", () => {
    expect(formatEvidenceAge(950, 1000)).toBe("just now");
    expect(formatEvidenceAge(100, 1000)).toBe("15m ago");
    expect(formatEvidenceAge(0, 90000)).toBe("1d ago");
  });
});
