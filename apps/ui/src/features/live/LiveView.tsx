import React from "react";
import { escapeHtml } from "../../lib/format";
import { renderMarkdown } from "../../lib/markdown";
import {
  aiResponse,
  consolidateAttempts,
  isUsefulText,
  phaseClass,
  phaseLabel,
  prettyJson,
  statusClass,
  statusLabel,
  stripTruncationMarkers,
  systemMessage,
  userMessage,
  type LiveRow,
} from "./livePageLogic";
import { useLiveTraffic } from "./useLiveTraffic";
import "../../lib/markdown_styles.css";
import "../../lib/section_embed_layout.css";
import "./live.css";

export function LiveView() {
  const { rows, liveBadge, liveBadgeClass, countBadge } = useLiveTraffic();
  const [expanded, setExpanded] = React.useState<Set<string>>(() => new Set());

  function toggleExpand(requestId: string) {
    setExpanded((current) => {
      const next = new Set(current);
      if (next.has(requestId)) next.delete(requestId);
      else next.add(requestId);
      return next;
    });
  }

  return (
    <div className="section-embed-layout live-page embed-mode">
      <main className="section-embed-main">
        <div className="toolbar">
          <div>
            <h2>Live API Traffic</h2>
            <p className="muted">Recent gateway requests with expandable payloads.</p>
          </div>
          <div style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
            <span id="countBadge" className="badge">
              {countBadge}
            </span>
            <span id="liveBadge" className={liveBadgeClass}>
              {liveBadge}
            </span>
          </div>
        </div>
        <div className="table-wrap">
          <table className="live-table">
            <thead>
              <tr>
                <th>Time</th>
                <th>Phase</th>
                <th>Model</th>
                <th>Status</th>
                <th>Route</th>
                <th>Latency</th>
                <th>Reason</th>
                <th>Details</th>
              </tr>
            </thead>
            <tbody id="rows">
              {rows.map((row) => (
                <LiveTableRows
                  key={row.request_id}
                  row={row}
                  open={expanded.has(row.request_id)}
                  onToggle={() => toggleExpand(row.request_id)}
                />
              ))}
            </tbody>
          </table>
        </div>
      </main>
    </div>
  );
}

function LiveTableRows(props: { row: LiveRow; open: boolean; onToggle: () => void }) {
  const { row } = props;
  const status = row.status_code ?? "";
  const when = row.last_ts
    ? new Date(Number(row.last_ts) * 1000).toLocaleTimeString()
    : "";

  const assistantText = isUsefulText(row.assistant_text)
    ? stripTruncationMarkers(row.assistant_text)
    : stripTruncationMarkers(aiResponse(row.response_body || row.stream_event));
  const userPlain = userMessage(row.request_payload);
  const systemPlain = systemMessage(row.request_payload);

  return (
    <>
      <tr>
        <td>{when}</td>
        <td className={phaseClass(row.phase, row.status_code)}>{phaseLabel(row.phase)}</td>
        <td>{row.model || ""}</td>
        <td className={statusClass(row.status_code)}>{status}</td>
        <td>
          {row.provider_name || ""}
          <div className="muted">
            <code>{row.route_id || ""}</code>
          </div>
        </td>
        <td>{row.latency_ms != null ? `${row.latency_ms}ms` : ""}</td>
        <td className="muted">{row.reason || ""}</td>
        <td>
          <button type="button" className="expand-btn" onClick={props.onToggle}>
            {props.open ? "Hide" : "Show"}
          </button>
        </td>
      </tr>
      {props.open ? (
        <tr className="details-row">
          <td colSpan={8}>
            <div className="details-wrap">
              {systemPlain.trim() ? (
                <details>
                  <summary>System prompt</summary>
                  <div
                    className="md-body"
                    dangerouslySetInnerHTML={{ __html: renderMarkdown(systemPlain) }}
                  />
                </details>
              ) : null}
              <div>
                <strong>User message</strong>
                <div
                  className="md-body"
                  dangerouslySetInnerHTML={{
                    __html: userPlain.trim() ? renderMarkdown(userPlain) : '<p class="muted">(empty)</p>',
                  }}
                />
              </div>
              <div>
                <strong>AI response</strong>
                <div
                  className="md-body"
                  dangerouslySetInnerHTML={{
                    __html: assistantText.trim()
                      ? renderMarkdown(assistantText)
                      : '<p class="muted">(empty)</p>',
                  }}
                />
              </div>
              <details>
                <summary>Raw response payload</summary>
                <pre>{escapeHtml(prettyJson(row.response_body || row.stream_event || {}))}</pre>
              </details>
              <div>
                <strong>Route attempts</strong>
                <RouteAttempts attempts={row.route_attempts || []} />
              </div>
            </div>
          </td>
        </tr>
      ) : null}
    </>
  );
}

function RouteAttempts(props: { attempts: LiveRow["route_attempts"] }) {
  const consolidated = consolidateAttempts(props.attempts);
  if (!consolidated.length) {
    return <div className="attempt-empty">No route attempts captured for this request.</div>;
  }
  return (
    <div className="attempts">
      {consolidated.map((item, idx) => {
        const status = item.status;
        const provider = item.provider_name || "unknown";
        const model = item.model_id || "unknown-model";
        const route = item.route_id || "n/a";
        const reason = item.reason || "";
        const code = item.status_code != null ? `HTTP ${item.status_code}` : "";
        const marker =
          status === "success"
            ? "✓"
            : status === "failed"
              ? "x"
              : status === "flagged"
                ? "⚑"
                : status === "trying"
                  ? "…"
                  : "!";
        const tone =
          status === "success" ? "ok" : status === "failed" ? "bad" : status === "trying" ? "" : "warn";
        const reasonText = reason ? reason.replace(/_/g, " ") : "";
        return (
          <div key={`${provider}-${model}-${route}-${idx}`} className={`attempt ${tone}`}>
            <span className="attempt-marker">{marker}</span>
            <div className="attempt-main">
              <div className="attempt-title">
                {idx + 1}. {provider} / <code>{model}</code>
              </div>
              <div className="attempt-sub">
                <span className="label">Route:</span> <code>{route}</code>
                {reasonText ? (
                  <>
                    {" "}
                    · <span className="label">Reason:</span> {reasonText}
                  </>
                ) : null}
              </div>
            </div>
            <div className="attempt-meta">
              <span className={`attempt-status ${status}`}>{statusLabel(status)}</span>
              {code ? <span className="attempt-http">{code}</span> : null}
            </div>
          </div>
        );
      })}
    </div>
  );
}
