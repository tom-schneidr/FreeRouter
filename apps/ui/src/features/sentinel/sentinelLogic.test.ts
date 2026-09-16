import { describe, expect, it } from "vitest";
import {
  filterSentinelRoutes,
  formatEvidenceAge,
  receiptStatusLabel,
  receiptStatusTone,
  routeTrustLabel,
  type SentinelReceipt,
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

describe("Sentinel consumer receipts", () => {
  const receipt: SentinelReceipt = {
    run_id: "run-1",
    created_at: 100,
    consumer_id: "agentrange",
    profile_id: "safe-security",
    status: "degraded",
    policy_verdict: "allowed",
    provider_name: "openrouter",
    route_id: "route-1",
    model_id: "model:free",
    latency_ms: 400,
    attempts: 2,
    fallback_used: true,
    fallback_reason: "primary: rate_limited",
    stream: false,
    capabilities: ["tool-use"],
    tool_policy: "proposal-only",
    request_path: "/v1/responses",
  };

  it("makes fallback receipts scan-friendly", () => {
    expect(receiptStatusLabel(receipt)).toBe("Fallback used");
    expect(receiptStatusTone(receipt)).toBe("warn");
    expect(receiptStatusTone({ ...receipt, status: "blocked" })).toBe("bad");
  });
});
