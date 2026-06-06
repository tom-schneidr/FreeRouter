import { describe, expect, it } from "vitest";
import { embedSrcWithReload } from "./embedSrc";

describe("embedSrcWithReload", () => {
  it("leaves the initial URL unchanged", () => {
    expect(embedSrcWithReload("/chat?embed=1", 0)).toBe("/chat?embed=1");
  });

  it("adds cache-busting parameters correctly", () => {
    expect(embedSrcWithReload("/chat", 2)).toBe("/chat?_=2");
    expect(embedSrcWithReload("/chat?embed=1", 2)).toBe("/chat?embed=1&_=2");
  });
});
