import { describe, expect, it } from "vitest";
import { healthSummaryText, limitedRoutes } from "./healthLogic";

describe("healthLogic", () => {
  it("filters non-active routes", () => {
    const routes = [
      { route_id: "a", health: { status: "active" } },
      { route_id: "b", health: { status: "rate_limited" } },
    ] as never[];
    expect(limitedRoutes(routes).map((route) => route.route_id)).toEqual(["b"]);
  });

  it("formats summary text", () => {
    expect(healthSummaryText(1)).toBe("1 automatically limited route");
    expect(healthSummaryText(2)).toBe("2 automatically limited routes");
  });
});
