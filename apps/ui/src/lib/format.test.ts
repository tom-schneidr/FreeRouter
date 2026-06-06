import { describe, expect, it } from "vitest";
import { escapeHtml, fmtCompact, healthLabel, modelIdLeaf } from "./format";

describe("format helpers", () => {
  it("escapes unsafe HTML characters", () => {
    expect(escapeHtml(`<script data-x="1">'&</script>`)).toBe(
      "&lt;script data-x=&quot;1&quot;&gt;&#39;&amp;&lt;/script&gt;",
    );
  });

  it("formats compact counts at useful thresholds", () => {
    expect(fmtCompact(999)).toBe("999");
    expect(fmtCompact(1_250)).toBe("1.3k");
    expect(fmtCompact(25_400)).toBe("25k");
    expect(fmtCompact(1_250_000)).toBe("1.3M");
  });

  it("extracts model leaves and normalizes health labels", () => {
    expect(modelIdLeaf("provider/models/model-a")).toBe("model-a");
    expect(modelIdLeaf("model-a")).toBe("model-a");
    expect(healthLabel("rate_limited")).toBe("rate limited");
    expect(healthLabel(undefined)).toBe("active");
  });
});
