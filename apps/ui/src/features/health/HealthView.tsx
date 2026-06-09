import React from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { fetchJson } from "../../api/client";
import type { ModelRoute, ModelsListResponse } from "../../api/types";
import {
  formatHealthTime,
  healthStatusLabel,
  healthSummaryText,
  limitedRoutes,
} from "./healthLogic";
import "../../lib/section_embed_layout.css";
import "./health.css";

export function HealthView() {
  const queryClient = useQueryClient();
  const [clearingId, setClearingId] = React.useState<string | null>(null);
  const [failedId, setFailedId] = React.useState<string | null>(null);

  const catalog = useQuery({
    queryKey: ["gateway-models"],
    queryFn: () => fetchJson<ModelsListResponse>("/v1/gateway/models"),
    refetchInterval: 10_000,
  });

  const routes = limitedRoutes(catalog.data?.data ?? []);
  const summary = catalog.isLoading
    ? "Loading..."
    : catalog.error
      ? "Load failed"
      : healthSummaryText(routes.length);

  async function reload() {
    await queryClient.invalidateQueries({ queryKey: ["gateway-models"] });
  }

  async function clearFlag(routeId: string) {
    setClearingId(routeId);
    setFailedId(null);
    try {
      const response = await fetch(
        `/v1/gateway/models/${encodeURIComponent(routeId)}/health/reset`,
        { method: "POST" },
      );
      if (response.ok) {
        await reload();
      } else {
        setFailedId(routeId);
      }
    } finally {
      setClearingId(null);
    }
  }

  return (
    <div className="section-embed-layout health-page">
      <main className="section-embed-main">
        <h2>Route Health</h2>
        <p className="muted">
          Routes automatically limited by FreeRouter are listed here. Active routes are hidden.
        </p>
        <div className="toolbar">
          <span className="muted">{summary}</span>
          <button type="button" id="reload" onClick={() => void reload()}>
            Reload
          </button>
        </div>
        <div id="routes" className="grid">
          {!catalog.isLoading && routes.length === 0 ? (
            <div className="empty">No routes are currently automatically limited.</div>
          ) : (
            routes.map((route) => (
              <HealthRouteCard
                key={route.route_id}
                route={route}
                clearing={clearingId === route.route_id}
                failed={failedId === route.route_id}
                onClear={() => void clearFlag(route.route_id)}
              />
            ))
          )}
        </div>
      </main>
    </div>
  );
}

function HealthRouteCard(props: {
  route: ModelRoute;
  clearing: boolean;
  failed: boolean;
  onClear: () => void;
}) {
  const { route } = props;
  const health = route.health!;
  return (
    <div className="card">
      <div>
        <div className="provider">{route.provider_name}</div>
        <strong>{route.display_name}</strong>
        <div className="model">{route.model_id}</div>
        <div className="model">
          Reason: {health.status_reason || health.status} · failures:{" "}
          {health.consecutive_failures}
        </div>
        <div className="model">Next probe: {formatHealthTime(health.next_probe_at)}</div>
        <div className="model">
          <button
            type="button"
            className="secondary"
            disabled={props.clearing}
            onClick={props.onClear}
          >
            {props.clearing ? "Clearing..." : props.failed ? "Failed" : "Clear flag"}
          </button>
        </div>
      </div>
      <span className={`pill ${health.status}`}>{healthStatusLabel(health.status)}</span>
    </div>
  );
}
