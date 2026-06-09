import React from "react";
import { renderMarkdown } from "../../lib/markdown";
import { isFlaggedSkip } from "./chatLogic";
import type { RouteEvent } from "./chatLogic";
import { useChatStream } from "./useChatStream";
import "../../lib/markdown_styles.css";
import "../../lib/section_embed_layout.css";
import "./chat.css";

export function ChatView() {
  const chat = useChatStream();
  const [input, setInput] = React.useState("");
  const textareaRef = React.useRef<HTMLTextAreaElement>(null);

  function autoResize() {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 150)}px`;
  }

  React.useEffect(() => {
    autoResize();
  }, [input]);

  async function handleSend() {
    const text = input.trim();
    if (!text) return;
    setInput("");
    autoResize();
    await chat.sendMessage(text);
    textareaRef.current?.focus();
  }

  return (
    <div className="section-embed-layout chat-page embed-mode">
      <div className="app">
        <div className="chat-panel">
          <div className="messages" id="messages">
            {chat.isEmpty ? (
              <div className="empty-state">
                <span className="icon-lg">💬</span>
                Send a message to start chatting.
                <br />
                The gateway will route to the best available model.
              </div>
            ) : (
              chat.messages.map((message) => <ChatMessage key={message.id} message={message} />)
            )}
          </div>
          <div className="typing" id="typing" style={{ display: chat.typing ? "block" : "none" }}>
            <span />
            <span />
            <span />
          </div>
          <div className="input-bar">
            <textarea
              ref={textareaRef}
              id="input"
              rows={1}
              placeholder="Type a message..."
              value={input}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  void handleSend();
                }
              }}
            />
            <label
              className="input-options"
              title="Route this message through /v1/chat/completions/web-search"
            >
              <input
                type="checkbox"
                id="web-search-toggle"
                checked={chat.webSearch}
                onChange={(event) => chat.setWebSearch(event.target.checked)}
              />
              Web search
            </label>
            <button
              className="send-btn"
              id="send-btn"
              type="button"
              title="Send"
              disabled={chat.sending}
              onClick={() => void handleSend()}
            >
              <svg
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <line x1="22" y1="2" x2="11" y2="13" />
                <polygon points="22 2 15 22 11 13 2 9 22 2" />
              </svg>
            </button>
          </div>
        </div>

        <div className="route-panel">
          <div className="route-panel-header">
            <span className="dot" />
            Routing Activity
          </div>
          <div className="route-log" id="route-log">
            {chat.routeGroups.length === 0 ? (
              <div className="empty-state">
                <span className="icon-lg">📡</span>
                Routing events will appear here in real time.
              </div>
            ) : (
              chat.routeGroups.map((group) => (
                <RouteGroupCard
                  key={group.id}
                  group={group}
                  onToggleRoute={chat.toggleRouteEnabled}
                />
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function ChatMessage(props: {
  message: { id: string; role: string; content: string; meta?: string };
}) {
  const { message } = props;
  if (message.role === "system-error") {
    return <div className="msg system-error">{message.content}</div>;
  }
  return (
    <div className={`msg ${message.role}`}>
      {message.role === "assistant" ? (
        <span
          className="content-text md-body"
          dangerouslySetInnerHTML={{ __html: renderMarkdown(message.content || "") }}
        />
      ) : (
        <span className="content-text">{message.content}</span>
      )}
      {message.meta ? <div className="msg-meta">{message.meta}</div> : null}
    </div>
  );
}

function RouteGroupCard(props: {
  group: {
    id: number;
    statusText: string;
    statusColor?: string;
    events: RouteEvent[];
  };
  onToggleRoute: (routeId: string, enabled: boolean) => Promise<boolean | null | undefined>;
}) {
  const [routeEnabled, setRouteEnabled] = React.useState<Record<string, boolean>>({});
  const [routeBusy, setRouteBusy] = React.useState<string | null>(null);
  const [routeFailed, setRouteFailed] = React.useState<string | null>(null);

  return (
    <div className="route-group">
      <div className="route-group-header">
        <span>Request #{props.group.id}</span>
        <span className="req-id" style={props.group.statusColor ? { color: props.group.statusColor } : undefined}>
          {props.group.statusText}
        </span>
      </div>
      {props.group.events.map((event) => (
        <RouteEventRow
          key={event.id}
          event={event}
          busy={routeBusy === event.routeId}
          failed={routeFailed === event.routeId}
          enabled={event.routeId ? routeEnabled[event.routeId] !== false : true}
          onToggle={async () => {
            if (!event.routeId || event.status === "flagged") return;
            setRouteBusy(event.routeId);
            setRouteFailed(null);
            const currentlyEnabled = routeEnabled[event.routeId] !== false;
            const result = await props.onToggleRoute(event.routeId, currentlyEnabled);
            setRouteBusy(null);
            if (result == null) {
              setRouteFailed(event.routeId);
              window.setTimeout(() => setRouteFailed(null), 1500);
            } else {
              setRouteEnabled((current) => ({ ...current, [event.routeId!]: result }));
            }
          }}
        />
      ))}
    </div>
  );
}

function RouteEventRow(props: {
  event: RouteEvent;
  busy: boolean;
  failed: boolean;
  enabled: boolean;
  onToggle: () => void;
}) {
  const { event } = props;
  let iconClass = "fail";
  let iconText = "✗";
  if (event.status === "trying") {
    iconClass = "trying";
    iconText = "⟳";
  } else if (event.status === "selected") {
    iconClass = "ok";
    iconText = "✓";
  } else if (event.status === "skipped") {
    iconClass = isFlaggedSkip(event.reason) ? "flagged-skip" : "skip";
    iconText = "⏭";
  } else if (event.status === "flagged") {
    iconClass = "flagged";
    iconText = "!";
  }
  const showDisable = event.routeId && event.status !== "flagged";
  return (
    <div className={`route-event${event.status === "flagged" ? " flagged" : ""}`}>
      <div className={`icon ${iconClass}`}>{iconText}</div>
      <div className="info">
        <span className="provider-name">{event.providerName}</span>
        <span className="model-name">{event.modelId || ""}</span>
      </div>
      {event.reason ? <span className="reason">{event.reason}</span> : null}
      {event.durationMs != null ? <span className="duration">{event.durationMs}ms</span> : null}
      {showDisable ? (
        <button
          type="button"
          className={`route-disable${props.enabled ? "" : " enable"}`}
          disabled={props.busy}
          onClick={() => void props.onToggle()}
        >
          {props.busy
            ? props.enabled
              ? "Disabling..."
              : "Enabling..."
            : props.failed
              ? "Failed"
              : props.enabled
                ? "Disable"
                : "Enable"}
        </button>
      ) : null}
    </div>
  );
}
