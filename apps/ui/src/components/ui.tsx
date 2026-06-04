import React from "react";
import type { ProviderStatus } from "../api/types";

export function SectionIntro({ title, copy }: { title: string; copy: string }) {
  return (
    <div className="section-heading">
      <div>
        <h1>{title}</h1>
        <p>{copy}</p>
      </div>
    </div>
  );
}

export function DesktopRequired() {
  return (
    <div className="disabled-box">
      <div>
        <strong>Desktop app required</strong>
        <span>
          Open FreeRouter from the desktop shortcut to use settings, backups, logs, restart, and tray
          controls. Normal gateway pages still work in a browser.
        </span>
      </div>
    </div>
  );
}

export function Notice({ tone, children }: { tone: "ok" | "bad"; children: React.ReactNode }) {
  return <div className={`notice ${tone}`}>{children}</div>;
}

export function Metric({
  label,
  value,
  note,
}: {
  label: string;
  value: React.ReactNode;
  note: string;
}) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{note}</small>
    </div>
  );
}

export function Panel({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="panel">
      <header>
        <h2>{title}</h2>
      </header>
      {children}
    </section>
  );
}

export function StatusPill({
  tone,
  children,
}: {
  tone: "ok" | "warn" | "bad" | "muted";
  children: React.ReactNode;
}) {
  return <span className={`pill ${tone}`}>{children}</span>;
}

export function EmptyState({ message }: { message: string }) {
  return <div className="empty">{message}</div>;
}

export function ProviderTable({ providers }: { providers: ProviderStatus[] }) {
  if (!providers.length) return <EmptyState message="No provider state loaded yet." />;
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Provider</th>
            <th>Status</th>
            <th>Requests today</th>
            <th>Tokens today</th>
          </tr>
        </thead>
        <tbody>
          {providers.map((provider) => (
            <tr key={provider.name}>
              <td>
                <strong>{provider.name}</strong>
                <span>{provider.configured ? "API key configured" : "Missing API key"}</span>
              </td>
              <td>
                <StatusPill tone={provider.available ? "ok" : provider.configured ? "warn" : "bad"}>
                  {provider.available ? "Available" : provider.unavailable_reason || "Unavailable"}
                </StatusPill>
              </td>
              <td>{provider.requests_today}</td>
              <td>{provider.tokens_used_today}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function numericFieldKind(kind: string) {
  return kind === "int" || kind === "optional_int" || kind === "float";
}
