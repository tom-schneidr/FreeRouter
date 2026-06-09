import React from "react";
import type { StreamRouteEvent } from "../../api/types";
import {
  type ChatMessage,
  type RouteEvent,
  type RouteGroup,
  extractAssistantText,
  isFlaggedSkip,
  parseSseChunk,
} from "./chatLogic";

let messageCounter = 0;
let routeGroupCounter = 0;

function nextMessageId() {
  messageCounter += 1;
  return `msg-${messageCounter}`;
}

function nextRouteGroupId() {
  routeGroupCounter += 1;
  return routeGroupCounter;
}

function nextRouteEventId() {
  return `route-${Math.random().toString(36).slice(2, 9)}`;
}

export function useChatStream() {
  const [messages, setMessages] = React.useState<ChatMessage[]>([]);
  const [routeGroups, setRouteGroups] = React.useState<RouteGroup[]>([]);
  const [typing, setTyping] = React.useState(false);
  const [sending, setSending] = React.useState(false);
  const [webSearch, setWebSearch] = React.useState(false);
  const chatHistoryRef = React.useRef<Array<{ role: string; content: string }>>([]);
  const isFirstMsgRef = React.useRef(true);
  const currentGroupRef = React.useRef<RouteGroup | null>(null);
  const tryingEventsRef = React.useRef<Record<string, { eventId: string; startedAt: number }>>({});

  const clearEmptyState = React.useCallback(() => {
    if (isFirstMsgRef.current) {
      isFirstMsgRef.current = false;
      if (messages.length === 0) return;
    }
  }, [messages.length]);

  const addUserMessage = React.useCallback((text: string) => {
    isFirstMsgRef.current = false;
    setMessages((current) => [...current, { id: nextMessageId(), role: "user", content: text }]);
  }, []);

  const addErrorMessage = React.useCallback((text: string) => {
    isFirstMsgRef.current = false;
    setMessages((current) => [...current, { id: nextMessageId(), role: "system-error", content: text }]);
  }, []);

  const startRouteGroup = React.useCallback(() => {
    const id = nextRouteGroupId();
    const group: RouteGroup = {
      id,
      statusText: "routing...",
      events: [],
    };
    currentGroupRef.current = group;
    tryingEventsRef.current = {};
    setRouteGroups((current) => [group, ...current]);
    return group;
  }, []);

  const updateCurrentGroup = React.useCallback((updater: (group: RouteGroup) => RouteGroup) => {
    const current = currentGroupRef.current;
    if (!current) return;
    const updated = updater(current);
    currentGroupRef.current = updated;
    setRouteGroups((groups) => groups.map((group) => (group.id === updated.id ? updated : group)));
  }, []);

  const appendRouteEvent = React.useCallback(
    (event: Omit<RouteEvent, "id">) => {
      const id = nextRouteEventId();
      updateCurrentGroup((group) => ({
        ...group,
        events: [...group.events, { ...event, id }],
      }));
      return id;
    },
    [updateCurrentGroup],
  );

  const updateTryingEvent = React.useCallback(
    (
      eventId: string,
      status: RouteEvent["status"],
      reason?: string,
      durationMs?: number | null,
    ) => {
      updateCurrentGroup((group) => ({
        ...group,
        events: group.events.map((event) =>
          event.id === eventId
            ? {
                ...event,
                status,
                reason: reason ?? event.reason,
                durationMs: durationMs ?? event.durationMs,
              }
            : event,
        ),
      }));
    },
    [updateCurrentGroup],
  );

  const handleStreamEvent = React.useCallback(
    (
      evt: StreamRouteEvent,
      state: {
        assistantId: string | null;
        setAssistantId: (id: string | null) => void;
        assistantMarkdown: string;
        setAssistantMarkdown: (value: string) => void;
        finalProvider: string;
        setFinalProvider: (value: string) => void;
        finalModel: string;
        setFinalModel: (value: string) => void;
        t0: number;
      },
    ) => {
      if (evt.type === "route_trying") {
        const eventId = appendRouteEvent({
          status: "trying",
          providerName: evt.provider,
          modelId: evt.model_id,
          routeId: evt.route_id,
        });
        tryingEventsRef.current[`${evt.provider}/${evt.model_id}`] = {
          eventId,
          startedAt: performance.now(),
        };
        return;
      }
      if (evt.type === "route_skip" || evt.type === "route_fail") {
        const key = `${evt.provider}/${evt.model_id}`;
        const te = tryingEventsRef.current[key];
        const dur = te ? Math.round(performance.now() - te.startedAt) : null;
        const status = evt.type === "route_skip" ? "skipped" : "failed";
        if (te) {
          updateTryingEvent(te.eventId, status, evt.reason, dur);
        } else {
          appendRouteEvent({
            status,
            providerName: evt.provider,
            modelId: evt.model_id,
            reason: evt.reason,
            durationMs: dur,
            routeId: evt.route_id,
          });
        }
        return;
      }
      if (evt.type === "route_flagged") {
        appendRouteEvent({
          status: "flagged",
          providerName: evt.provider,
          modelId: evt.model_id,
          reason: `Automatically flagged: ${evt.reason}`,
          routeId: evt.route_id,
        });
        return;
      }
      if (evt.type === "route_selected") {
        const key = `${evt.provider}/${evt.model_id}`;
        const te = tryingEventsRef.current[key];
        const dur = te ? Math.round(performance.now() - te.startedAt) : null;
        if (te) updateTryingEvent(te.eventId, "selected", undefined, dur);
        else {
          appendRouteEvent({
            status: "selected",
            providerName: evt.provider,
            modelId: evt.model_id,
            durationMs: dur,
            routeId: evt.route_id,
          });
        }
        state.setFinalProvider(evt.provider);
        state.setFinalModel(evt.model_id);
        return;
      }
      if (evt.type === "content") {
        setTyping(false);
        let assistantId = state.assistantId;
        if (!assistantId) {
          assistantId = nextMessageId();
          state.setAssistantId(assistantId);
          setMessages((current) => [
            ...current,
            { id: assistantId!, role: "assistant", content: "" },
          ]);
        }
        const nextMarkdown = state.assistantMarkdown + evt.text;
        state.setAssistantMarkdown(nextMarkdown);
        setMessages((current) =>
          current.map((message) =>
            message.id === assistantId ? { ...message, content: nextMarkdown } : message,
          ),
        );
        return;
      }
      if (evt.type === "done") {
        const totalMs = Math.round(performance.now() - state.t0);
        const meta = `${state.finalProvider} · ${state.finalModel} · ${totalMs}ms`;
        if (state.assistantId) {
          setMessages((current) =>
            current.map((message) =>
              message.id === state.assistantId ? { ...message, meta } : message,
            ),
          );
        }
        if (evt.content) chatHistoryRef.current.push({ role: "assistant", content: evt.content });
        updateCurrentGroup((group) => ({
          ...group,
          statusText: `✓ ${totalMs}ms`,
          statusColor: "#22c55e",
        }));
        return;
      }
      if (evt.type === "error") {
        setTyping(false);
        addErrorMessage(evt.message || "All providers exhausted.");
        updateCurrentGroup((group) => ({
          ...group,
          statusText: "✗ failed",
          statusColor: "#ef4444",
        }));
      }
    },
    [addErrorMessage, appendRouteEvent, updateCurrentGroup, updateTryingEvent],
  );

  const sendWebSearch = React.useCallback(
    async (group: RouteGroup, t0: number) => {
      updateCurrentGroup((g) => ({ ...g, statusText: "searching web..." }));
      const resp = await fetch("/v1/chat/completions/web-search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: chatHistoryRef.current, max_tokens: 4096 }),
      });
      const totalMs = Math.round(performance.now() - t0);
      const provider = resp.headers.get("X-Gateway-Provider") || "";
      const model = resp.headers.get("X-Gateway-Model") || "";
      const route = resp.headers.get("X-Gateway-Route") || "";
      let payload: Record<string, unknown> = {};
      try {
        payload = await resp.json();
      } catch {
        /* ignore */
      }
      if (!resp.ok) {
        const message =
          (payload?.error as { message?: string } | undefined)?.message ||
          `Web search failed with HTTP ${resp.status}`;
        addErrorMessage(message);
        updateCurrentGroup((g) => ({ ...g, statusText: "✗ failed", statusColor: "#ef4444" }));
        return;
      }
      appendRouteEvent({
        status: "selected",
        providerName: provider || "web-search",
        modelId: model || "web-search model",
        durationMs: totalMs,
        routeId: route,
      });
      const content = extractAssistantText(payload);
      if (content) {
        setMessages((current) => [
          ...current,
          {
            id: nextMessageId(),
            role: "assistant",
            content,
            meta: `${provider} · ${model} · web search · ${totalMs}ms`,
          },
        ]);
        chatHistoryRef.current.push({ role: "assistant", content });
      } else {
        addErrorMessage(
          "Web search returned successfully, but the provider sent an empty final assistant message.",
        );
      }
      updateCurrentGroup((g) => ({
        ...g,
        statusText: `✓ ${totalMs}ms`,
        statusColor: "#22c55e",
      }));
    },
    [addErrorMessage, appendRouteEvent, updateCurrentGroup],
  );

  const sendMessage = React.useCallback(
    async (text: string) => {
      if (!text.trim() || sending) return;
      setSending(true);
      addUserMessage(text);
      chatHistoryRef.current.push({ role: "user", content: text });
      setTyping(true);
      const group = startRouteGroup();
      currentGroupRef.current = group;
      const t0 = performance.now();

      let assistantId: string | null = null;
      let assistantMarkdown = "";
      let finalProvider = "";
      let finalModel = "";

      const eventState = {
        assistantId,
        setAssistantId: (id: string | null) => {
          assistantId = id;
        },
        assistantMarkdown,
        setAssistantMarkdown: (value: string) => {
          assistantMarkdown = value;
        },
        finalProvider,
        setFinalProvider: (value: string) => {
          finalProvider = value;
        },
        finalModel,
        setFinalModel: (value: string) => {
          finalModel = value;
        },
        t0,
      };

      try {
        if (webSearch) {
          await sendWebSearch(group, t0);
          return;
        }

        const resp = await fetch("/v1/chat/completions/stream-route", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ messages: chatHistoryRef.current, max_tokens: 4096 }),
        });

        const reader = resp.body?.getReader();
        if (!reader) throw new Error("No response body");
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const parsed = parseSseChunk(buffer);
          buffer = parsed.rest;
          for (const evt of parsed.events) {
            handleStreamEvent(evt, eventState);
          }
        }
      } catch (err) {
        setTyping(false);
        addErrorMessage(`Network error: ${err instanceof Error ? err.message : String(err)}`);
        updateCurrentGroup((g) => ({ ...g, statusText: "✗ error", statusColor: "#ef4444" }));
      } finally {
        setTyping(false);
        setSending(false);
      }
    },
    [
      addErrorMessage,
      addUserMessage,
      handleStreamEvent,
      sendWebSearch,
      sending,
      startRouteGroup,
      updateCurrentGroup,
      webSearch,
    ],
  );

  const toggleRouteEnabled = React.useCallback(
    async (routeId: string, currentlyEnabled: boolean) => {
      if (!routeId) return;
      const nextEnabled = !currentlyEnabled;
      if (!nextEnabled && !confirm("Disable this model route?")) return;
      const action = nextEnabled ? "enable" : "disable";
      const response = await fetch(
        `/v1/gateway/models/${encodeURIComponent(routeId)}/${action}`,
        { method: "POST" },
      );
      return response.ok ? nextEnabled : null;
    },
    [],
  );

  return {
    messages,
    routeGroups,
    typing,
    sending,
    webSearch,
    setWebSearch,
    sendMessage,
    toggleRouteEnabled,
    isEmpty: messages.length === 0,
  };
}
