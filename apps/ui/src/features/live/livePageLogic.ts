import type { LiveRequestEvent } from "../../api/types";
import { extractAssistantText } from "../../lib/markdown_extract";

export type LiveRow = {
  request_id: string;
  first_ts: number;
  last_ts: number;
  path: string;
  model: string;
  tokens: string;
  status_code: number | null;
  provider_name: string;
  route_id: string;
  latency_ms: number | null;
  phase: string;
  reason: string;
  request_payload: Record<string, unknown> | null;
  response_body: unknown;
  stream_event: unknown;
  assistant_text: string;
  route_attempts: Array<{
    status: string;
    provider_name: string;
    route_id: string;
    model_id: string;
    reason: string;
    status_code: number | null;
  }>;
};

export function formatProviderUsage(u: Record<string, unknown> | undefined | null): string {
  if (!u || typeof u !== "object") return "";
  const num = (x: unknown) => (x == null || x === "" ? NaN : Number(x));
  const total = num(u.total_tokens);
  if (!Number.isNaN(total)) return String(total);
  const prompt = num(u.prompt_tokens);
  const completion = num(u.completion_tokens);
  const parts: number[] = [];
  if (!Number.isNaN(prompt)) parts.push(prompt);
  if (!Number.isNaN(completion)) parts.push(completion);
  return parts.length ? parts.join(" + ") : "";
}

export function stripTruncationMarkers(value: string): string {
  return value
    .replace(/\.\.\. <truncated \d+ chars>/g, "")
    .replace(/<truncated-depth>/g, "")
    .replace(/<truncated \d+ items>/g, "")
    .replace(/"__truncated__":\s*"[^"]*"/g, "")
    .trim();
}

export function isUsefulText(value: unknown): value is string {
  return typeof value === "string" && stripTruncationMarkers(value).length > 0;
}

export function statusClass(status: number | null | undefined): string {
  if (status == null) return "";
  if (Number(status) >= 500 || Number(status) === 499) return "bad";
  if (Number(status) >= 400) return "warn";
  return "ok";
}

function messageContent(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((item) => (typeof item === "string" ? item : (item as { text?: string; content?: string })?.text || (item as { content?: string })?.content || ""))
      .filter(Boolean)
      .join("\n");
  }
  return String(content || "");
}

export function userMessage(payload: Record<string, unknown> | null): string {
  const messages = payload?.messages;
  if (!Array.isArray(messages)) return "";
  const lastUser = [...messages].reverse().find((message) => message?.role === "user");
  return messageContent(lastUser?.content ?? "");
}

export function systemMessage(payload: Record<string, unknown> | null): string {
  if (!payload || typeof payload !== "object") return "";
  const parts: string[] = [];
  if (typeof payload.instructions === "string" && payload.instructions.trim()) {
    parts.push(payload.instructions.trim());
  }
  const messages = payload?.messages;
  if (Array.isArray(messages)) {
    for (const message of messages) {
      if (message?.role !== "system") continue;
      const text = messageContent(message?.content ?? "").trim();
      if (text) parts.push(text);
    }
  }
  const unique: string[] = [];
  for (const part of parts) {
    if (!unique.includes(part)) unique.push(part);
  }
  return unique.join("\n\n");
}

export function aiResponse(body: unknown): string {
  if (typeof body === "string") {
    return body === "<truncated-depth>" ? "" : body;
  }
  if (!body || typeof body !== "object") return "";
  const record = body as Record<string, unknown>;
  if (isUsefulText(record.content as string)) return record.content as string;
  if (isUsefulText(record.text as string)) return record.text as string;
  const message = record.message as { content?: string } | undefined;
  if (isUsefulText(message?.content)) return message!.content!;
  const choiceContent = (record.choices as Array<{ message?: { content?: unknown }; text?: string }> | undefined)?.[0]
    ?.message?.content;
  if (isUsefulText(choiceContent as string)) return choiceContent as string;
  if (Array.isArray(choiceContent)) {
    return choiceContent
      .map((item) => (typeof item === "string" ? item : (item as { text?: string; content?: string })?.text || (item as { content?: string })?.content || ""))
      .filter(isUsefulText)
      .join("\n");
  }
  const choiceText = (record.choices as Array<{ text?: string }> | undefined)?.[0]?.text;
  if (isUsefulText(choiceText)) return choiceText!;
  return extractAssistantText(record);
}

export function normalizeLiveEvent(
  evt: LiveRequestEvent,
  requests: Map<string, LiveRow>,
  seenEventIds: Set<number>,
): LiveRow | null {
  if (!evt?.request_id) return null;
  if (evt.event_id != null) {
    if (seenEventIds.has(evt.event_id)) return null;
    seenEventIds.add(evt.event_id);
  }
  const payload = evt.payload || {};
  const current =
    requests.get(evt.request_id) ||
    ({
      request_id: evt.request_id,
      first_ts: evt.timestamp,
      last_ts: evt.timestamp,
      path: "",
      model: "",
      tokens: "",
      status_code: null,
      provider_name: "",
      route_id: "",
      latency_ms: null,
      phase: "started",
      reason: "",
      request_payload: null,
      response_body: null,
      stream_event: null,
      assistant_text: "",
      route_attempts: [],
    } satisfies LiveRow);

  current.last_ts = evt.timestamp || current.last_ts;
  current.path = (payload.path as string) || current.path;
  current.model = (payload.model as string) || (payload.model_id as string) || current.model;
  if (payload.usage && typeof payload.usage === "object") {
    const formatted = formatProviderUsage(payload.usage as Record<string, unknown>);
    current.tokens = formatted || "Unknown";
  } else if (
    (payload.request_payload as Record<string, unknown> | undefined)?.max_tokens ||
    (payload.request_payload as Record<string, unknown> | undefined)?.max_completion_tokens
  ) {
    current.tokens = "Unknown";
  }
  current.provider_name = (payload.provider_name as string) || current.provider_name;
  current.route_id = (payload.route_id as string) || current.route_id;
  if (payload.status_code != null) current.status_code = payload.status_code as number;
  if (payload.latency_ms != null) current.latency_ms = payload.latency_ms as number;
  current.reason = (payload.reason as string) || (payload.message as string) || current.reason;
  if (payload.request_payload != null) current.request_payload = payload.request_payload as Record<string, unknown>;
  if (isUsefulText(payload.assistant_text as string)) {
    current.assistant_text = stripTruncationMarkers(payload.assistant_text as string);
  }
  if (payload.response_body != null) {
    current.response_body = payload.response_body;
    const extracted = aiResponse(payload.response_body);
    if (isUsefulText(extracted)) current.assistant_text = stripTruncationMarkers(extracted);
  }
  if (payload.stream_event != null) current.stream_event = payload.stream_event;
  if (evt.event_type === "response_content" && typeof payload.content === "string") {
    current.assistant_text = (current.assistant_text || "") + payload.content;
  }
  if (Array.isArray(payload.attempts_detail)) {
    current.route_attempts = payload.attempts_detail.map((attempt: Record<string, unknown>) => ({
      status: (attempt.status as string) || "",
      provider_name: (attempt.provider_name as string) || "",
      route_id: (attempt.route_id as string) || "",
      model_id: (attempt.model_id as string) || "",
      reason: (attempt.reason as string) || "",
      status_code: (attempt.status_code as number | null) ?? null,
    }));
  }
  const routeEvent = payload.route_event as Record<string, unknown> | undefined;
  if (routeEvent) {
    current.route_attempts.push({
      status: (routeEvent.type as string) || "",
      provider_name: (routeEvent.provider_name as string) || "",
      route_id: (routeEvent.route_id as string) || "",
      model_id: (routeEvent.model_id as string) || "",
      reason: (routeEvent.reason as string) || "",
      status_code: (routeEvent.status_code as number | null) ?? null,
    });
  }

  if (evt.event_type === "request_started") current.phase = "in_progress";
  else if (evt.event_type === "route_selected" && current.phase !== "completed") current.phase = "routing";
  else if (evt.event_type === "request_completed") current.phase = "completed";
  else if (evt.event_type === "request_failed") current.phase = "failed";
  else if (evt.event_type === "request_rejected") current.phase = "rejected";
  else if (evt.event_type === "request_closed") current.phase = "closed";

  requests.set(evt.request_id, current);
  return current;
}

export function trimLiveRequests(requests: Map<string, LiveRow>, maxRows: number) {
  if (requests.size <= maxRows) return;
  const oldest = [...requests.values()].sort((a, b) => (a.last_ts || 0) - (b.last_ts || 0))[0];
  if (oldest) requests.delete(oldest.request_id);
}

export function phaseLabel(phase: string): string {
  if (phase === "completed") return "Done";
  if (phase === "failed") return "Failed";
  if (phase === "rejected") return "Rejected";
  if (phase === "closed") return "Closed";
  if (phase === "routing") return "Routing";
  return "In progress";
}

export function phaseClass(phase: string, statusCode: number | null): string {
  if (phase === "completed") return "ok";
  if (phase === "failed" || phase === "rejected" || phase === "closed") return "bad";
  return statusClass(statusCode);
}

export function statusLabel(status: string): string {
  const raw = String(status || "").trim();
  if (!raw) return "Attempt";
  return raw.replace(/_/g, " ");
}

export function canonicalAttemptStatus(raw: string): string {
  const status = String(raw || "").toLowerCase();
  if (status === "route_trying" || status === "trying") return "trying";
  if (status === "route_selected" || status === "selected") return "success";
  if (status.includes("selected") || status.includes("done") || status === "success") return "success";
  if (status.includes("flag")) return "flagged";
  if (status.includes("fail") || status.includes("error") || status === "timeout") return "failed";
  if (status.includes("skip")) return "skipped";
  return status || "unknown";
}

export function consolidateAttempts(attempts: LiveRow["route_attempts"]) {
  const priority = (status: string) => {
    if (status === "success") return 5;
    if (status === "flagged") return 4;
    if (status === "failed") return 3;
    if (status === "skipped") return 2;
    return 1;
  };
  const grouped = new Map<string, LiveRow["route_attempts"][number] & { status: string; _order: number }>();
  for (let index = 0; index < attempts.length; index += 1) {
    const item = attempts[index];
    const status = canonicalAttemptStatus(item.status);
    const key = [item.provider_name || "", item.model_id || "", item.route_id || "", status].join("|");
    const existing = grouped.get(key);
    if (!existing) {
      grouped.set(key, { ...item, status, _order: index });
      continue;
    }
    if (priority(status) >= priority(existing.status)) {
      grouped.set(key, { ...existing, ...item, status, _order: existing._order });
    }
  }
  return [...grouped.values()].sort((a, b) => {
    const successDelta = Number(a.status === "success") - Number(b.status === "success");
    if (successDelta !== 0) return successDelta;
    return (a._order || 0) - (b._order || 0);
  });
}

export function prettyJson(value: unknown): string {
  try {
    return JSON.stringify(value ?? {}, null, 2);
  } catch {
    return String(value);
  }
}
