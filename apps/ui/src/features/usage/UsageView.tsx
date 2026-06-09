import React from "react";
import type { ModelRoute, ProviderStatus, RouteHealth, RouteUsage } from "../../api/types";
import { fmtCompact, fmtDate, fmtRelative } from "../../lib/format";
import {
  detailStats,
  embedSummaryLine,
  filterModels,
  formatSuccessFail,
  healthPillLabel,
  loadSortPreference,
  modelIdLeaf,
  saveSortPreference,
  sortIndicator,
  sortedModels,
  toggleSort,
} from "./usagePageLogic";
import "../../lib/section_embed_layout.css";
import "./usage.css";

const SORT_HINTS: Record<string, string> = {
  rank: "Sort by waterfall priority",
  model: "Sort by model name",
  health: "Sort by health state",
  successes: "Sort by successes",
  failures: "Sort by failures",
  tokens: "Sort by total tokens",
  prompt: "Sort by prompt tokens",
  completion: "Sort by completion tokens",
  last_used: "Sort by last used time",
};

type UsageViewProps = {
  providers: ProviderStatus[];
  loading: boolean;
  onReload: () => void;
};

export function UsageView({ providers, loading, onReload }: UsageViewProps) {
  const allModels = React.useMemo(
    () => providers.flatMap((provider) => provider.models || []),
    [providers],
  );
  const [search, setSearch] = React.useState("");
  const [providerFilter, setProviderFilter] = React.useState("");
  const [healthFilter, setHealthFilter] = React.useState("");
  const [sort, setSort] = React.useState(loadSortPreference);
  const [expanded, setExpanded] = React.useState<Set<string>>(() => new Set());

  React.useEffect(() => {
    saveSortPreference(sort);
  }, [sort]);

  const models = sortedModels(
    filterModels(allModels, search, providerFilter, healthFilter),
    sort,
  );
  const summary = loading
    ? "Loading..."
    : embedSummaryLine(models, providers);

  function setSortKey(key: string) {
    setSort((current) => toggleSort(current, key));
  }

  function toggleExpanded(routeId: string) {
    setExpanded((current) => {
      const next = new Set(current);
      if (next.has(routeId)) next.delete(routeId);
      else next.add(routeId);
      return next;
    });
  }

  return (
    <div className="section-embed-layout usage-page embed-mode">
      <main className="section-embed-main">
        <h2>Provider Usage</h2>
        <p className="muted">
          <span className="lead-full">
            Per-route usage from the local gateway database, with provider-level request and token
            totals.
          </span>
          <span className="lead-embed">Per-route usage and provider totals.</span>
        </p>
        <div className="toolbar">
          <div className="filters">
            <input
              id="search"
              className="filter-search"
              type="search"
              placeholder="Search models..."
              value={search}
              onChange={(event) => setSearch(event.target.value)}
            />
            <div className="filter-selects">
              <select
                id="providerFilter"
                value={providerFilter}
                onChange={(event) => setProviderFilter(event.target.value)}
              >
                <option value="">All providers</option>
                {providers.map((provider) => (
                  <option key={provider.name} value={provider.name}>
                    {provider.name}
                  </option>
                ))}
              </select>
              <select
                id="healthFilter"
                value={healthFilter}
                onChange={(event) => setHealthFilter(event.target.value)}
              >
                <option value="">All health states</option>
                <option value="active">Active</option>
                <option value="rate_limited">Rate limited</option>
                <option value="too_slow">Too slow</option>
                <option value="potentially_outdated">Potentially outdated</option>
              </select>
            </div>
          </div>
          <div className="toolbar-actions">
            <span id="summary" className="muted">
              {summary}
            </span>
            <button id="reload" type="button" onClick={onReload}>
              Reload
            </button>
          </div>
        </div>
        <div id="summaryCards" />
        <div id="tableRoot">
          {!loading && models.length === 0 ? (
            <div className="empty">No model usage matches the current filters.</div>
          ) : (
            <div className="usage-list">
              <div className="usage-list-head">
                <EmbedSortHeader sort={sort} sortKey="rank" label="Priority" className="usage-head-rank" onSort={setSortKey} />
                <EmbedSortHeader sort={sort} sortKey="model" label="Model route" className="usage-head-model" onSort={setSortKey} />
                <EmbedSortHeader sort={sort} sortKey="health" label="Health" className="usage-head-health" onSort={setSortKey} />
                <EmbedSortHeader sort={sort} sortKey="successes" label="Success / fail" className="usage-head-metric" onSort={setSortKey} />
                <EmbedSortHeader sort={sort} sortKey="tokens" label="Total tokens" className="usage-head-metric" onSort={setSortKey} />
                <EmbedSortHeader sort={sort} sortKey="prompt" label="Prompt / completion" className="usage-head-metric" onSort={setSortKey} />
                <EmbedSortHeader sort={sort} sortKey="last_used" label="Last used" className="usage-head-metric" onSort={setSortKey} />
                <span className="usage-head-action" aria-hidden="true" />
              </div>
              {models.map((model) => (
                <UsageListItem
                  key={model.route_id}
                  model={model}
                  expanded={expanded.has(model.route_id)}
                  providers={providers}
                  onToggle={() => toggleExpanded(model.route_id)}
                />
              ))}
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

function EmbedSortHeader(props: {
  sort: { key: string; dir: "asc" | "desc" };
  sortKey: string;
  label: string;
  className: string;
  onSort: (key: string) => void;
}) {
  const sorted = props.sort.key === props.sortKey ? " sorted" : "";
  const hint = SORT_HINTS[props.sortKey] || `Sort by ${props.label}`;
  return (
    <button
      type="button"
      className={`usage-sort${sorted} ${props.className}`}
      title={hint}
      onClick={() => props.onSort(props.sortKey)}
    >
      {props.label}
      <span className="sort-ind">{sortIndicator(props.sort, props.sortKey)}</span>
    </button>
  );
}

function UsageListItem(props: {
  model: ModelRoute;
  expanded: boolean;
  providers: ProviderStatus[];
  onToggle: () => void;
}) {
  const { model } = props;
  const usage: RouteUsage = model.usage || ({} as RouteUsage);
  const health: RouteHealth = model.health || ({} as RouteHealth);
  const displayName = model.display_name || model.model_id;
  const modelId = String(model.model_id || "");
  const leafId = modelIdLeaf(modelId);
  const successes = Number(usage.successes || 0);
  const failures = Number(usage.failures || 0);
  const promptTokens = Number(usage.prompt_tokens || 0);
  const completionTokens = Number(usage.completion_tokens || 0);
  const totalTokens = Number(usage.total_tokens || 0);
  const sf = formatSuccessFail(successes, failures);
  const status = health.status || "active";
  const pillClass =
    status === "active" ? "" : status === "rate_limited" ? "warning" : "error";

  return (
    <article className="usage-item">
      <div className="usage-item-main">
        <span className="usage-rank" title="Waterfall priority from Models page">
          #{model.rank ?? "—"}
        </span>
        <div className="usage-route">
          <div className="usage-route-title">
            <strong title={displayName}>{displayName}</strong>
          </div>
          <div className="usage-route-meta">
            <span className="provider-tag">{model.provider_name}</span>
            <span className="usage-route-id" title={modelId}>
              {leafId}
            </span>
          </div>
        </div>
        <div className="usage-health-cell">
          <span className={`pill${pillClass ? ` ${pillClass}` : ""}`}>
            {healthPillLabel(status)}
          </span>
        </div>
        <div
          className="usage-metric-cell"
          title={`${successes.toLocaleString()} successes · ${failures.toLocaleString()} failures`}
        >
          <span className="usage-sf usage-metric-value">
            <span className="usage-ok">{sf.ok}</span>
            <span className="usage-sep"> · </span>
            <span className={`usage-fail${sf.failWarn ? " usage-failures-warn" : ""}`}>{sf.fail}</span>
          </span>
        </div>
        <div className="usage-metric-cell" title={`${totalTokens.toLocaleString()} total tokens`}>
          <span className="usage-metric-value">{fmtCompact(totalTokens)}</span>
        </div>
        <div
          className="usage-metric-cell"
          title={`${promptTokens.toLocaleString()} prompt · ${completionTokens.toLocaleString()} completion`}
        >
          <span className="usage-token-split">
            <span>
              <span className="usage-token-label">in</span> {fmtCompact(promptTokens)}
            </span>
            <span>
              <span className="usage-token-label">out</span> {fmtCompact(completionTokens)}
            </span>
          </span>
        </div>
        <div className="usage-metric-cell" title={fmtDate(usage.last_used_at)}>
          <span className="usage-metric-value">{fmtRelative(usage.last_used_at)}</span>
        </div>
        <button
          type="button"
          className="expand usage-expand"
          title={props.expanded ? "Hide route details" : "Show route details"}
          aria-expanded={props.expanded}
          onClick={props.onToggle}
        >
          {props.expanded ? "Hide" : "Details"}
        </button>
      </div>
      {props.expanded ? (
        <div className="usage-details">
          <div className="details">
            {detailStats(model, props.providers).map(([label, value]) => (
              <div key={label} className="stat">
                <div className="label">{label}</div>
                <div className="value">{value}</div>
              </div>
            ))}
          </div>
        </div>
      ) : null}
    </article>
  );
}
