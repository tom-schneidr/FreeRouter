import { describe, expect, it } from "vitest";
import { isFlaggedSkip, parseSseChunk } from "./chatLogic";

describe("chatLogic", () => {
  it("parses stream-route SSE chunks", () => {
    const buffer = 'data: {"type":"content","text":"hi"}\n\ndata: {"type":"done"}\n\nleftover';
    const parsed = parseSseChunk(buffer);
    expect(parsed.events).toHaveLength(2);
    expect(parsed.events[0]).toEqual({ type: "content", text: "hi" });
    expect(parsed.rest).toBe("leftover");
  });

  it("detects flagged skip reasons", () => {
    expect(isFlaggedSkip("potentially_outdated")).toBe(true);
    expect(isFlaggedSkip("timeout")).toBe(false);
  });
});
