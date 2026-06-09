import type { StreamRouteEvent } from "../../api/types";
import { extractAssistantText } from "../../lib/markdown_extract";

export type ChatMessage = {
  id: string;
  role: "user" | "assistant" | "system-error";
  content: string;
  meta?: string;
};

export type RouteEventStatus = "trying" | "selected" | "skipped" | "failed" | "flagged";

export type RouteEvent = {
  id: string;
  status: RouteEventStatus;
  providerName: string;
  modelId: string;
  reason?: string;
  durationMs?: number | null;
  routeId?: string;
};

export type RouteGroup = {
  id: number;
  statusText: string;
  statusColor?: string;
  events: RouteEvent[];
};

export function isFlaggedSkip(reason: string | undefined): boolean {
  return ["potentially_outdated", "route_rate_limited", "route_too_slow"].includes(reason || "");
}

export function parseSseChunk(buffer: string): { events: StreamRouteEvent[]; rest: string } {
  const SEP = "\n\n";
  const events: StreamRouteEvent[] = [];
  let remaining = buffer;
  while (true) {
    const idx = remaining.indexOf(SEP);
    if (idx === -1) break;
    const chunk = remaining.slice(0, idx).trim();
    remaining = remaining.slice(idx + 2);
    if (!chunk.startsWith("data: ")) continue;
    const raw = chunk.slice(6);
    if (raw === "[DONE]") continue;
    try {
      events.push(JSON.parse(raw) as StreamRouteEvent);
    } catch {
      /* ignore */
    }
  }
  return { events, rest: remaining };
}

export { extractAssistantText };
