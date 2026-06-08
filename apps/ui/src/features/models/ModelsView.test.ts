import { describe, expect, it } from "vitest";
import { shouldAutoOpenEndpointUpdates } from "./ModelsView";

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
