import { describe, expect, it } from "vitest";
import { endpointUpdatesStatusText, shouldAutoOpenEndpointUpdates } from "./ModelsView";

describe("shouldAutoOpenEndpointUpdates", () => {
  it("opens once for a report with suggestions, but not again after dismissal", () => {
    expect(shouldAutoOpenEndpointUpdates(123, 4, null)).toBe(true);
    expect(shouldAutoOpenEndpointUpdates(123, 4, 123)).toBe(false);
  });

  it("opens again for a newer report and ignores empty reports", () => {
    expect(shouldAutoOpenEndpointUpdates(124, 2, 123)).toBe(true);
    expect(shouldAutoOpenEndpointUpdates(125, 0, 124)).toBe(false);
  });
});

describe("endpointUpdatesStatusText", () => {
  it("shows loading while refresh is in flight even with an empty summary", () => {
    expect(
      endpointUpdatesStatusText({
        suggestionCount: 0,
        isRefreshing: true,
        isApplying: false,
        summary: "No pending endpoint updates.",
      }),
    ).toContain("Checking provider catalogs");
  });

  it("does not duplicate the empty-state message", () => {
    expect(
      endpointUpdatesStatusText({
        suggestionCount: 0,
        isRefreshing: false,
        isApplying: false,
        summary: "No pending endpoint updates.",
      }),
    ).toBe("No pending endpoint updates.");
  });
});
