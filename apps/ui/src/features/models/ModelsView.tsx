import React from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { RefreshCw, Save, Sparkles, Undo2, Wrench } from "lucide-react";
import { fetchJson, queryClient } from "../../api/client";
import type {
  EndpointDiagnosisRefreshResponse,
  EndpointDiagnosisResponse,
  EndpointSuggestion,
  ModelRoute,
  ModelsListResponse,
} from "../../api/types";
import { EmptyState, Notice, StatusPill } from "../../components/ui";
import { healthLabel } from "../../lib/format";

const CAPABILITY_LABELS: Record<string, string> = {
  text: "Text",
  reasoning: "Reasoning",
  coding: "Coding",
  "tool-use": "Tool use",
  "web-search": "Web search",
  vision: "Vision",
  audio: "Audio",
  safety: "Safety",
  moderation: "Moderation",
  translation: "Translation",
  classification: "Classification",
  rag: "RAG",
};

const CONTEXT_OPTIONS = [
  { value: "", label: "Any context" },
  { value: "32000", label: "32K+ context" },
  { value: "128000", label: "128K+ context" },
  { value: "256000", label: "256K+ context" },
  { value: "1000000", label: "1M+ context" },
];

function normalize(value: unknown) {
  return String(value || "").toLowerCase();
}

function suggestionActionLabel(action: string) {
  return (
    { add_route: "New route", remove_route: "Remove route", clear_stale: "Recovered route" }[
      action
    ] || action
  );
}

export function shouldAutoOpenEndpointUpdates(
  checkedAt: number | undefined,
  suggestionCount: number,
  lastAutoOpenedAt: number | null,
) {
  return Boolean(checkedAt && suggestionCount > 0 && checkedAt !== lastAutoOpenedAt);
}

export function ModelsView() {
  const [routes, setRoutes] = React.useState<ModelRoute[]>([]);
  const [search, setSearch] = React.useState("");
  const [capabilityFilter, setCapabilityFilter] = React.useState<string[]>([]);
  const [providerFilter, setProviderFilter] = React.useState<string[]>([]);
  const [contextFilter, setContextFilter] = React.useState("");
  const [routeability, setRouteability] = React.useState("");
  const [filterOpen, setFilterOpen] = React.useState(false);
  const [statusText, setStatusText] = React.useState("Loading model catalog...");
  const [draggedId, setDraggedId] = React.useState<string | null>(null);
  const [updateModalOpen, setUpdateModalOpen] = React.useState(false);
  const [endpointSuggestions, setEndpointSuggestions] = React.useState<EndpointSuggestion[]>([]);
  const [updateSummary, setUpdateSummary] = React.useState("Loading suggestions...");
  const [selectedSuggestions, setSelectedSuggestions] = React.useState<Set<string>>(() => new Set());
  const lastAutoOpenedAt = React.useRef<number | null>(null);

  const catalog = useQuery({
    queryKey: ["gateway-models-catalog"],
    queryFn: () => fetchJson<ModelsListResponse>("/v1/gateway/models"),
  });

  React.useEffect(() => {
    if (!catalog.data) return;
    setRoutes(catalog.data.data);
    setStatusText(
      `Loaded ${catalog.data.data.length} model routes${catalog.data.catalog_path ? ` from ${catalog.data.catalog_path}` : ""}.`,
    );
  }, [catalog.data]);

  const diagnosis = useQuery({
    queryKey: ["endpoint-diagnosis"],
    queryFn: () => fetchJson<EndpointDiagnosisResponse>("/v1/gateway/endpoint-diagnosis"),
    refetchInterval: 60000,
  });

  React.useEffect(() => {
    const suggestions = diagnosis.data?.last_report?.suggestions ?? [];
    setEndpointSuggestions(suggestions);
  }, [diagnosis.data]);

  const saveMutation = useMutation({
    mutationFn: (nextRoutes: ModelRoute[]) =>
      fetchJson<ModelsListResponse>("/v1/gateway/models", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ data: nextRoutes }),
      }),
    onSuccess: (payload) => {
      setRoutes(payload.data);
      queryClient.invalidateQueries({ queryKey: ["gateway-models-catalog"] });
      setStatusText("Saved. New requests will use the updated ranking immediately.");
    },
    onError: (error: Error) => setStatusText(error.message || "Save failed"),
  });

  const autoRankMutation = useMutation({
    mutationFn: () =>
      fetchJson<ModelsListResponse>("/v1/gateway/models/auto-rank", { method: "POST" }),
    onSuccess: (payload) => {
      setRoutes(payload.data);
      setStatusText(`Auto-ranked ${payload.data.length} text-capable routes.`);
    },
    onError: (error: Error) => setStatusText(error.message || "Auto-rank failed"),
  });

  const resetMutation = useMutation({
    mutationFn: () => fetchJson<ModelsListResponse>("/v1/gateway/models/reset", { method: "POST" }),
    onSuccess: (payload) => {
      setRoutes(payload.data);
      setStatusText("Restored default model rankings.");
    },
    onError: (error: Error) => setStatusText(error.message || "Reset failed"),
  });

  const refreshSuggestions = useMutation({
    mutationFn: () =>
      fetchJson<EndpointDiagnosisRefreshResponse>("/v1/gateway/endpoint-diagnosis/refresh", {
        method: "POST",
      }),
    onMutate: () => setUpdateSummary("Checking provider catalogs..."),
    onSuccess: (payload) => {
      setEndpointSuggestions(payload.data.suggestions || []);
      setUpdateSummary(
        payload.data.suggestions?.length
          ? `${payload.data.suggestions.length} suggested update(s) found.`
          : "No pending endpoint updates.",
      );
    },
    onError: () => setUpdateSummary("Could not refresh endpoint suggestions."),
  });

  const applySuggestions = useMutation({
    mutationFn: (suggestionIds: string[]) =>
      fetchJson("/v1/gateway/endpoint-diagnosis/apply", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ suggestion_ids: suggestionIds }),
      }),
    onSuccess: async () => {
      setUpdateSummary("Applied selected updates.");
      setSelectedSuggestions(new Set());
      await catalog.refetch();
      await diagnosis.refetch();
      setUpdateModalOpen(false);
    },
    onError: () => setUpdateSummary("Could not apply selected updates."),
  });

  const allCapabilities = [...new Set(routes.flatMap((route) => route.tags || []))].sort((a, b) =>
    (CAPABILITY_LABELS[a] || a).localeCompare(CAPABILITY_LABELS[b] || b),
  );
  const allProviders = [...new Set(routes.map((route) => route.provider_name))].sort();

  const filterCount =
    capabilityFilter.length +
    providerFilter.length +
    (contextFilter ? 1 : 0) +
    (routeability ? 1 : 0);

  const visible = routes
    .slice()
    .sort((a, b) => a.rank - b.rank || a.provider_name.localeCompare(b.provider_name))
    .filter(
      (route) =>
        !capabilityFilter.length ||
        capabilityFilter.every((tag) => (route.tags || []).includes(tag)),
    )
    .filter((route) => !providerFilter.length || providerFilter.includes(route.provider_name))
    .filter((route) => !contextFilter || Number(route.context_window || 0) >= Number(contextFilter))
    .filter((route) => {
      if (routeability === "routeable") return route.enabled;
      if (routeability === "disabled") return !route.enabled;
      return true;
    })
    .filter((route) => {
      const haystack = normalize(
        [
          route.route_id,
          route.provider_name,
          route.model_id,
          route.display_name,
          route.quality,
          route.speed,
          route.tags?.join(" "),
          route.notes,
        ].join(" "),
      );
      const query = normalize(search);
      return !query || haystack.includes(query);
    });

  function persist(nextRoutes: ModelRoute[], message?: string) {
    setRoutes(nextRoutes);
    if (message) setStatusText(message);
    saveMutation.mutate(nextRoutes);
  }

  function toggleRouteEnabled(routeId: string) {
    const route = routes.find((item) => item.route_id === routeId);
    if (!route) return;
    const next = routes.map((item) =>
      item.route_id === routeId ? { ...item, enabled: !item.enabled } : item,
    );
    persist(next);
  }

  function updateField(routeId: string, key: keyof ModelRoute, value: string) {
    const next = routes.map((route) => {
      if (route.route_id !== routeId) return route;
      if (key === "rank" || key === "context_window") {
        return { ...route, [key]: value === "" ? null : Number(value) };
      }
      if (key === "tags") {
        return {
          ...route,
          tags: value
            .split(",")
            .map((tag) => tag.trim())
            .filter(Boolean),
        };
      }
      return { ...route, [key]: value };
    });
    setRoutes(next);
  }

  function handleDrop(dropId: string) {
    if (!draggedId || draggedId === dropId) return;
    const ordered = routes.slice().sort((a, b) => a.rank - b.rank);
    const fromIndex = ordered.findIndex((r) => r.route_id === draggedId);
    const toIndex = ordered.findIndex((r) => r.route_id === dropId);
    if (fromIndex < 0 || toIndex < 0) return;
    const [moved] = ordered.splice(fromIndex, 1);
    ordered.splice(toIndex, 0, moved);
    const reindexed = ordered.map((route, index) => ({ ...route, rank: index + 1 }));
    persist(reindexed);
    setDraggedId(null);
  }

  function openUpdateModal() {
    setUpdateModalOpen(true);
    setUpdateSummary(
      endpointSuggestions.length
        ? `${endpointSuggestions.length} suggested update${endpointSuggestions.length === 1 ? "" : "s"} found. Nothing changes until you apply them.`
        : "No pending endpoint updates.",
    );
  }

  React.useEffect(() => {
    if (!diagnosis.data) return;
    const checkedAt = diagnosis.data.last_report?.checked_at;
    const count = diagnosis.data.last_report?.suggestions?.length ?? 0;
    if (shouldAutoOpenEndpointUpdates(checkedAt, count, lastAutoOpenedAt.current)) {
      lastAutoOpenedAt.current = checkedAt ?? null;
      const timer = window.setTimeout(() => openUpdateModal(), 1500);
      return () => window.clearTimeout(timer);
    }
    return undefined;
  }, [diagnosis.data]);

  return (
    <div className="section-stack models-view">
      <div className="section-heading">
        <div>
          <h1>Models</h1>
          <p>Edit waterfall priority, enable routes, and apply endpoint catalog updates.</p>
        </div>
        <div className="toolbar-actions">
          <button type="button" onClick={() => catalog.refetch()} disabled={catalog.isFetching}>
            <RefreshCw size={16} />
            Reload
          </button>
          <button
            type="button"
            className={endpointSuggestions.length ? "primary-action" : ""}
            onClick={openUpdateModal}
          >
            <Wrench size={16} />
            Updates
            {endpointSuggestions.length > 0 ? (
              <span className="update-count active">{endpointSuggestions.length}</span>
            ) : null}
          </button>
          <button
            type="button"
            disabled={autoRankMutation.isPending}
            onClick={() => autoRankMutation.mutate()}
          >
            <Sparkles size={16} />
            Auto-rank
          </button>
          <button
            type="button"
            disabled={resetMutation.isPending}
            onClick={() => {
              if (
                !window.confirm(
                  "Restore default model rankings? This overwrites your custom order.",
                )
              )
                return;
              resetMutation.mutate();
            }}
          >
            <Undo2 size={16} />
            Reset
          </button>
          <button
            className="primary-action"
            type="button"
            disabled={saveMutation.isPending}
            onClick={() => saveMutation.mutate(routes)}
          >
            <Save size={16} />
            Save
          </button>
        </div>
      </div>

      <p className="path-note">{statusText}</p>
      {catalog.error && <Notice tone="bad">{catalog.error.message}</Notice>}

      <div className="models-toolbar">
        <input
          className="models-search"
          placeholder="Search models/providers/tags..."
          value={search}
          onChange={(event) => setSearch(event.target.value)}
        />
        <button type="button" onClick={() => setFilterOpen((open) => !open)}>
          Filters {filterCount > 0 ? `(${filterCount})` : ""}
        </button>
      </div>

      {filterOpen && (
        <div className="models-filter-panel panel">
          <section className="filter-section">
            <h3>Capabilities</h3>
            <div className="filter-options">
              {allCapabilities.map((tag) => (
                <label key={tag} className="filter-option">
                  <input
                    type="checkbox"
                    checked={capabilityFilter.includes(tag)}
                    onChange={(event) =>
                      setCapabilityFilter((current) =>
                        event.target.checked
                          ? [...current, tag]
                          : current.filter((item) => item !== tag),
                      )
                    }
                  />
                  {CAPABILITY_LABELS[tag] || tag}
                </label>
              ))}
            </div>
          </section>
          <section className="filter-section">
            <h3>Providers</h3>
            <div className="filter-options">
              {allProviders.map((provider) => (
                <label key={provider} className="filter-option">
                  <input
                    type="checkbox"
                    checked={providerFilter.includes(provider)}
                    onChange={(event) =>
                      setProviderFilter((current) =>
                        event.target.checked
                          ? [...current, provider]
                          : current.filter((item) => item !== provider),
                      )
                    }
                  />
                  {provider}
                </label>
              ))}
            </div>
          </section>
          <section className="filter-section">
            <h3>Context window</h3>
            <div className="filter-options">
              {CONTEXT_OPTIONS.map((option) => (
                <label key={option.value || "any"} className="filter-option">
                  <input
                    type="radio"
                    name="contextFilter"
                    checked={contextFilter === option.value}
                    onChange={() => setContextFilter(option.value)}
                  />
                  {option.label}
                </label>
              ))}
            </div>
          </section>
          <section className="filter-section">
            <h3>Routeability</h3>
            <div className="filter-options">
              {[
                { value: "", label: "All routes" },
                { value: "routeable", label: "Enabled only" },
                { value: "disabled", label: "Disabled/specialized" },
              ].map((option) => (
                <label key={option.value || "all"} className="filter-option">
                  <input
                    type="radio"
                    name="routeability"
                    checked={routeability === option.value}
                    onChange={() => setRouteability(option.value)}
                  />
                  {option.label}
                </label>
              ))}
            </div>
          </section>
          <button
            type="button"
            onClick={() => {
              setCapabilityFilter([]);
              setProviderFilter([]);
              setContextFilter("");
              setRouteability("");
            }}
          >
            Clear filters
          </button>
        </div>
      )}

      {catalog.isLoading && <EmptyState message="Loading model catalog..." />}
      {!catalog.isLoading && !visible.length && (
        <EmptyState message="No routes match the current filters." />
      )}

      <div className="models-grid">
        {visible.map((route) => (
          <details
            key={route.route_id}
            className={`model-card ${route.enabled ? "" : "disabled"}`}
            draggable
            onDragStart={() => setDraggedId(route.route_id)}
            onDragOver={(event) => event.preventDefault()}
            onDrop={() => handleDrop(route.route_id)}
          >
            <summary>
              <span className="rank">#{route.rank}</span>
              <span className="model-card-title">
                <strong>{route.display_name}</strong>
                <span className="muted"> {route.model_id}</span>
                <br />
                <span className="provider-tag">{route.provider_name}</span>
                {(route.tags || []).map((tag) => (
                  <span key={tag} className="pill muted">
                    {tag}
                  </span>
                ))}
                {route.health?.status && route.health.status !== "active" ? (
                  <StatusPill tone="warn">{healthLabel(route.health.status)}</StatusPill>
                ) : null}
              </span>
              <span className="summary-actions">
                <span className="state-label">{route.enabled ? "Enabled" : "Disabled"}</span>
                <button
                  type="button"
                  className={`toggle ${route.enabled ? "disable" : ""}`}
                  onClick={(event) => {
                    event.preventDefault();
                    toggleRouteEnabled(route.route_id);
                  }}
                >
                  {route.enabled ? "Disable" : "Enable"}
                </button>
              </span>
            </summary>
            <div className="model-card-body">
              <label className="field">
                <span>Rank</span>
                <input
                  type="number"
                  value={route.rank}
                  onChange={(event) => updateField(route.route_id, "rank", event.target.value)}
                  onBlur={() => saveMutation.mutate(routes)}
                />
              </label>
              <div className="meta-grid">
                <div>
                  <span className="meta-label">Display name</span>
                  <span>{route.display_name}</span>
                </div>
                <div>
                  <span className="meta-label">Context window</span>
                  <span>
                    {route.context_window ? route.context_window.toLocaleString() : "Unknown"}
                  </span>
                </div>
                <div>
                  <span className="meta-label">Quality</span>
                  <span>{route.quality}</span>
                </div>
                <div>
                  <span className="meta-label">Speed</span>
                  <span>{route.speed}</span>
                </div>
                <div>
                  <span className="meta-label">Cost</span>
                  <span>{route.cost}</span>
                </div>
                <div>
                  <span className="meta-label">Rank score</span>
                  <span>{route.rank_score ?? "Unknown"}</span>
                </div>
                <div>
                  <span className="meta-label">Notes</span>
                  <span>{route.notes || "None"}</span>
                </div>
              </div>
            </div>
          </details>
        ))}
      </div>

      {updateModalOpen && (
        <div
          className="modal-backdrop open"
          role="presentation"
          onClick={(event) => {
            if (event.target === event.currentTarget) setUpdateModalOpen(false);
          }}
        >
          <div className="modal" role="dialog" aria-labelledby="updates-title">
            <div className="modal-header">
              <h2 id="updates-title">Endpoint updates</h2>
              <button type="button" onClick={() => setUpdateModalOpen(false)}>
                Close
              </button>
            </div>
            <div className="modal-body">
              <p className="panel-copy">{updateSummary}</p>
              {endpointSuggestions.length ? (
                endpointSuggestions.map((item) => (
                  <label key={item.suggestion_id} className="suggestion">
                    <input
                      type="checkbox"
                      checked={selectedSuggestions.has(item.suggestion_id)}
                      onChange={(event) =>
                        setSelectedSuggestions((current) => {
                          const next = new Set(current);
                          if (event.target.checked) next.add(item.suggestion_id);
                          else next.delete(item.suggestion_id);
                          return next;
                        })
                      }
                    />
                    <span>
                      <span className="suggestion-title">{item.title}</span>
                      <StatusPill tone="muted">{suggestionActionLabel(item.action)}</StatusPill>
                      <div className="suggestion-meta">
                        {item.provider_name} / {item.model_id}
                      </div>
                      <div className="suggestion-meta">{item.details}</div>
                    </span>
                  </label>
                ))
              ) : (
                <div className="empty-updates">No pending endpoint updates.</div>
              )}
            </div>
            <div className="modal-actions">
              <button
                type="button"
                disabled={refreshSuggestions.isPending}
                onClick={() => refreshSuggestions.mutate()}
              >
                Refresh
              </button>
              <button
                type="button"
                onClick={() =>
                  setSelectedSuggestions(new Set(endpointSuggestions.map((s) => s.suggestion_id)))
                }
              >
                Select all
              </button>
              <button
                className="primary-action"
                type="button"
                disabled={applySuggestions.isPending}
                onClick={() => {
                  const ids = [...selectedSuggestions];
                  if (!ids.length) {
                    setUpdateSummary("Choose at least one suggestion to apply.");
                    return;
                  }
                  applySuggestions.mutate(ids);
                }}
              >
                Apply selected
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
