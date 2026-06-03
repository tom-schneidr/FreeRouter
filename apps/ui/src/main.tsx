import React from "react";
import ReactDOM from "react-dom/client";
import { QueryClient, QueryClientProvider, useMutation, useQuery } from "@tanstack/react-query";
import {
  Activity,
  Bot,
  Database,
  Download,
  Gauge,
  type LucideIcon,
  MessageSquareText,
  RefreshCw,
  Route,
  Save,
  Settings,
  ShieldCheck,
  Upload,
} from "lucide-react";
import "./styles.css";

type GatewayHealth = {
  status: string;
  service: string;
  version: string;
  database_path: string;
  model_catalog_path: string;
  providers: {
    total: number;
    configured: number;
    available: number;
  };
  routes: {
    total: number;
    enabled: number;
  };
  request_limits: {
    max_concurrent_requests: number;
    queue_timeout_seconds: number;
    max_waiting_requests: number | null;
  };
};

type ModelRoute = {
  route_id: string;
  display_name: string;
  provider_name: string;
  model_id: string;
  enabled: boolean;
  rank: number;
  tags?: string[];
  health?: {
    status?: string;
    status_reason?: string | null;
    consecutive_failures?: number;
  };
};

type ProviderStatus = {
  name: string;
  configured: boolean;
  available: boolean;
  unavailable_reason?: string;
  requests_today: number;
  tokens_used_today: number;
};

type LiveEvent = {
  request_id?: string;
  event_type?: string;
  method?: string;
  path?: string;
  status?: string;
  provider_name?: string;
  route_id?: string;
};

type ChatMessage = {
  role: "user" | "assistant";
  content: string;
};

type RouteEvent = {
  type: string;
  provider?: string;
  model_id?: string;
  route_id?: string;
  reason?: string;
  text?: string;
  content?: string;
  message?: string;
};

type EndpointSuggestion = {
  suggestion_id: string;
  action: string;
  provider_name: string;
  model_id: string;
  route_id: string;
  title: string;
  details: string;
};

type ProviderDiagnosis = {
  provider_name: string;
  configured: boolean;
  ok: boolean;
  discovered_model_count: number;
  new_route_suggestion_count: number;
  stale_route_suggestion_count: number;
  recovered_route_suggestion_count: number;
  error?: string | null;
};

type DiagnosisReport = {
  checked_at: number;
  providers: ProviderDiagnosis[];
  suggestions: EndpointSuggestion[];
};

type EndpointDiagnosisStatus = {
  enabled: boolean;
  auto_maintenance_enabled: boolean;
  last_auto_applied: EndpointSuggestion[];
  last_report: DiagnosisReport | null;
};

type DesktopField = {
  key: string;
  label: string;
  group: string;
  kind: string;
  secret: boolean;
  value: string;
  configured: boolean | null;
};

type DesktopSettingsPayload = {
  env_path: string;
  groups: string[];
  fields: DesktopField[];
};

type DesktopLogsPayload = {
  lines: string[];
};

type BackupExportPayload = {
  ok: boolean;
  path: string;
};

type BackupImportPayload = {
  ok: boolean;
  restored: string[];
};

const queryClient = new QueryClient();

async function fetchJson<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(path, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const message = typeof payload?.detail === "string" ? payload.detail : response.statusText;
    throw new Error(message || `Request failed: ${response.status}`);
  }
  return payload as T;
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <FreeRouterShell />
    </QueryClientProvider>
  );
}

function FreeRouterShell() {
  const [activeSection, setActiveSection] = React.useState("dashboard");
  const desktopToken = useDesktopToken();
  const health = useGatewayQuery<GatewayHealth>("gateway-health", "/v1/gateway/health.json", 5000);
  const models = useGatewayQuery<{ data: ModelRoute[] }>("gateway-models", "/v1/gateway/models", 10000);
  const providers = useGatewayQuery<{ data: ProviderStatus[] }>(
    "provider-status",
    "/v1/providers/status",
    10000,
  );
  const diagnosis = useGatewayQuery<EndpointDiagnosisStatus>(
    "endpoint-diagnosis",
    "/v1/gateway/endpoint-diagnosis",
    15000,
  );
  const live = useGatewayQuery<{ data: LiveEvent[] }>(
    "live-traffic",
    "/v1/gateway/live/snapshot",
    3000,
  );

  const routes = models.data?.data ?? [];
  const providerRows = providers.data?.data ?? [];
  const liveRows = live.data?.data ?? [];
  const configuredProviders = providerRows.filter((provider) => provider.configured).length;
  const availableProviders = providerRows.filter((provider) => provider.available).length;
  const flaggedRoutes = routes.filter(
    (route) => route.health?.status && route.health.status !== "active",
  ).length;

  return (
    <div className="app-shell">
      <aside className="sidebar" aria-label="Primary navigation">
        <div className="brand">
          <div className="brand-mark" aria-hidden="true">
            <Route size={20} />
          </div>
          <div>
            <strong>FreeRouter</strong>
            <span>Local AI gateway</span>
          </div>
        </div>
        <nav className="nav-list">
          {NAV_ITEMS.map((item) => (
            <button
              key={item.id}
              className={activeSection === item.id ? "active" : ""}
              type="button"
              onClick={() => setActiveSection(item.id)}
            >
              <item.icon size={18} />
              <span>{item.label}</span>
            </button>
          ))}
        </nav>
      </aside>

      <main className="main">
        <header className="topbar">
          <div className={`status-dot ${health.data?.status === "ok" ? "ok" : "warn"}`} />
          <div>
            <strong>{health.data?.status === "ok" ? "Gateway running" : "Checking gateway"}</strong>
            <span>
              {health.data
                ? `${health.data.providers.configured} configured providers, ${health.data.routes.enabled} enabled routes`
                : health.error?.message ?? "Loading local runtime state"}
            </span>
          </div>
          <button type="button" onClick={() => window.location.reload()}>
            <RefreshCw size={16} />
            Refresh
          </button>
        </header>

        <section className="content">
          {activeSection === "dashboard" && (
            <Dashboard
              health={health.data ?? null}
              configuredProviders={configuredProviders}
              availableProviders={availableProviders}
              routeCount={routes.length}
              flaggedRoutes={flaggedRoutes}
              providers={providerRows}
              live={liveRows}
            />
          )}
          {activeSection === "models" && <Models routes={routes} diagnosis={diagnosis.data ?? null} />}
          {activeSection === "providers" && <Providers providers={providerRows} />}
          {activeSection === "traffic" && <Traffic live={liveRows} />}
          {activeSection === "settings" && <SettingsPanel desktopToken={desktopToken} />}
          {activeSection === "backups" && <BackupsPanel desktopToken={desktopToken} />}
          {activeSection === "chat" && <ChatPanel />}
          {activeSection === "logs" && <LogsPanel desktopToken={desktopToken} />}
        </section>
      </main>
    </div>
  );
}

function useDesktopToken() {
  const [token] = React.useState(() => {
    const fromUrl = new URLSearchParams(window.location.search).get("desktop_token");
    const fromSession = window.sessionStorage.getItem("freerouterDesktopToken");
    const next = fromUrl || fromSession || "";
    if (next) window.sessionStorage.setItem("freerouterDesktopToken", next);
    return next;
  });
  return token;
}

function useGatewayQuery<T>(key: string, path: string, refetchInterval: number) {
  return useQuery({
    queryKey: [key],
    queryFn: () => fetchJson<T>(path),
    refetchInterval,
  });
}

function Dashboard(props: {
  health: GatewayHealth | null;
  configuredProviders: number;
  availableProviders: number;
  routeCount: number;
  flaggedRoutes: number;
  providers: ProviderStatus[];
  live: LiveEvent[];
}) {
  return (
    <div className="section-stack">
      <div className="section-heading">
        <div>
          <h1>Gateway Command Center</h1>
          <p>Desktop-first control plane for routing, provider health, traffic, and local state.</p>
        </div>
      </div>
      <div className="metrics">
        <Metric label="Gateway" value={props.health?.status ?? "Loading"} note={props.health?.version ?? ""} />
        <Metric
          label="Providers"
          value={`${props.configuredProviders}/${props.providers.length || props.health?.providers.total || 0}`}
          note={`${props.availableProviders} available now`}
        />
        <Metric
          label="Enabled Routes"
          value={props.health?.routes.enabled ?? 0}
          note={`${props.routeCount || props.health?.routes.total || 0} catalog routes`}
        />
        <Metric label="Route Flags" value={props.flaggedRoutes} note="Routes needing review" />
      </div>
      <div className="two-column">
        <Panel title="Provider Readiness" icon={ShieldCheck}>
          <ProviderTable providers={props.providers.slice(0, 8)} />
        </Panel>
        <Panel title="Recent Traffic" icon={Activity}>
          <TrafficList live={props.live.slice(0, 8)} />
        </Panel>
      </div>
    </div>
  );
}

function Models({
  routes,
  diagnosis,
}: {
  routes: ModelRoute[];
  diagnosis: EndpointDiagnosisStatus | null;
}) {
  const [message, setMessage] = React.useState<{ tone: "ok" | "bad"; text: string } | null>(null);
  const refreshModels = () => queryClient.invalidateQueries({ queryKey: ["gateway-models"] });
  const refreshDiagnosis = () => queryClient.invalidateQueries({ queryKey: ["endpoint-diagnosis"] });
  const toggleRoute = useMutation({
    mutationFn: async (route: ModelRoute) => {
      const action = route.enabled ? "disable" : "enable";
      return fetchJson<{ data: ModelRoute }>(
        `/v1/gateway/models/${encodeURIComponent(route.route_id)}/${action}`,
        { method: "POST" },
      );
    },
    onSuccess: (_, route) => {
      setMessage({
        tone: "ok",
        text: `${route.display_name} ${route.enabled ? "disabled" : "enabled"}.`,
      });
      refreshModels();
    },
    onError: (error) => setMessage({ tone: "bad", text: error.message }),
  });
  const resetHealth = useMutation({
    mutationFn: (route: ModelRoute) =>
      fetchJson<{ data: ModelRoute }>(
        `/v1/gateway/models/${encodeURIComponent(route.route_id)}/health/reset`,
        { method: "POST" },
      ),
    onSuccess: (_, route) => {
      setMessage({ tone: "ok", text: `${route.display_name} health reset.` });
      refreshModels();
    },
    onError: (error) => setMessage({ tone: "bad", text: error.message }),
  });
  const autoRank = useMutation({
    mutationFn: () => fetchJson<{ data: ModelRoute[] }>("/v1/gateway/models/auto-rank", { method: "POST" }),
    onSuccess: () => {
      setMessage({ tone: "ok", text: "Model routes auto-ranked." });
      refreshModels();
    },
    onError: (error) => setMessage({ tone: "bad", text: error.message }),
  });
  const runDiagnosis = useMutation({
    mutationFn: () => fetchJson<{ data: DiagnosisReport }>("/v1/gateway/endpoint-diagnosis/refresh", { method: "POST" }),
    onSuccess: (payload) => {
      const count = payload.data.suggestions.length;
      setMessage({ tone: "ok", text: `Endpoint diagnosis finished with ${count} suggestion${count === 1 ? "" : "s"}.` });
      refreshDiagnosis();
      refreshModels();
    },
    onError: (error) => setMessage({ tone: "bad", text: error.message }),
  });
  const applySuggestions = useMutation({
    mutationFn: (suggestions: EndpointSuggestion[]) =>
      fetchJson<{ data: EndpointSuggestion[] }>("/v1/gateway/endpoint-diagnosis/apply", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ suggestion_ids: suggestions.map((item) => item.suggestion_id) }),
      }),
    onSuccess: (payload) => {
      setMessage({ tone: "ok", text: `Applied ${payload.data.length} endpoint suggestion${payload.data.length === 1 ? "" : "s"}.` });
      refreshDiagnosis();
      refreshModels();
    },
    onError: (error) => setMessage({ tone: "bad", text: error.message }),
  });
  const resetCatalog = useMutation({
    mutationFn: () => fetchJson<{ data: ModelRoute[] }>("/v1/gateway/models/reset", { method: "POST" }),
    onSuccess: () => {
      setMessage({ tone: "ok", text: "Model catalog reset to defaults." });
      refreshModels();
    },
    onError: (error) => setMessage({ tone: "bad", text: error.message }),
  });

  return (
    <div className="section-stack">
      <div className="section-heading">
        <div>
          <h1>Model Routes</h1>
          <p>Control the ranked waterfall catalog used by new gateway requests.</p>
        </div>
        <div className="toolbar-actions">
          <button type="button" onClick={refreshModels}>
            <RefreshCw size={16} />
            Refresh
          </button>
          <button type="button" disabled={autoRank.isPending} onClick={() => autoRank.mutate()}>
            <Gauge size={16} />
            Auto-rank
          </button>
          <button className="danger-action" type="button" disabled={resetCatalog.isPending} onClick={() => resetCatalog.mutate()}>
            Reset
          </button>
        </div>
      </div>
      {message && <Notice tone={message.tone}>{message.text}</Notice>}
      <Panel title="Catalog" icon={Bot}>
        <div className="route-grid">
          {routes.slice(0, 60).map((route) => (
            <article className="route-card" key={route.route_id}>
              <div>
                <strong>{route.display_name}</strong>
                <span>{route.model_id}</span>
              </div>
              <div className="route-meta">
                <span>#{route.rank}</span>
                <span>{route.provider_name}</span>
                <StatusPill tone={healthTone(route.health?.status)}>{route.health?.status ?? "active"}</StatusPill>
                <StatusPill tone={route.enabled ? "ok" : "muted"}>{route.enabled ? "Enabled" : "Disabled"}</StatusPill>
                {route.health?.status && route.health.status !== "active" && (
                  <button
                    type="button"
                    disabled={resetHealth.isPending}
                    onClick={() => resetHealth.mutate(route)}
                  >
                    Reset health
                  </button>
                )}
                <button
                  className={route.enabled ? "small-danger" : "small-success"}
                  type="button"
                  disabled={toggleRoute.isPending}
                  onClick={() => toggleRoute.mutate(route)}
                >
                  {route.enabled ? "Disable" : "Enable"}
                </button>
              </div>
            </article>
          ))}
        </div>
      </Panel>
      <EndpointDiagnosisPanel
        diagnosis={diagnosis}
        onRefresh={() => runDiagnosis.mutate()}
        onApply={(suggestions) => applySuggestions.mutate(suggestions)}
        refreshing={runDiagnosis.isPending}
        applying={applySuggestions.isPending}
      />
    </div>
  );
}

function Providers({ providers }: { providers: ProviderStatus[] }) {
  return (
    <Panel title="Providers" icon={Database}>
      <ProviderTable providers={providers} />
    </Panel>
  );
}

function EndpointDiagnosisPanel({
  diagnosis,
  onRefresh,
  onApply,
  refreshing,
  applying,
}: {
  diagnosis: EndpointDiagnosisStatus | null;
  onRefresh: () => void;
  onApply: (suggestions: EndpointSuggestion[]) => void;
  refreshing: boolean;
  applying: boolean;
}) {
  const report = diagnosis?.last_report;
  const suggestions = report?.suggestions ?? [];
  return (
    <Panel title="Endpoint Diagnosis" icon={ShieldCheck}>
      <div className="panel-actions">
        <div>
          <p className="panel-copy">
            {diagnosis?.enabled
              ? `Automatic diagnosis is enabled${diagnosis.auto_maintenance_enabled ? " with safe maintenance" : ""}.`
              : "Automatic diagnosis is disabled."}
          </p>
          {report && (
            <p className="path-note">
              Last checked {new Date(report.checked_at * 1000).toLocaleString()} across {report.providers.length} providers.
            </p>
          )}
        </div>
        <div className="toolbar-actions">
          <button type="button" disabled={refreshing} onClick={onRefresh}>
            <RefreshCw size={16} />
            {refreshing ? "Running" : "Run diagnosis"}
          </button>
          <button
            className="primary-action"
            type="button"
            disabled={!suggestions.length || applying}
            onClick={() => onApply(suggestions)}
          >
            {applying ? "Applying" : "Apply suggestions"}
          </button>
        </div>
      </div>

      {!report ? (
        <EmptyState message="No diagnosis report yet." />
      ) : (
        <div className="diagnosis-grid">
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Provider</th>
                  <th>Status</th>
                  <th>Discovered</th>
                  <th>Suggestions</th>
                </tr>
              </thead>
              <tbody>
                {report.providers.map((provider) => (
                  <tr key={provider.provider_name}>
                    <td>
                      {provider.provider_name}
                      {provider.error && <span>{provider.error}</span>}
                    </td>
                    <td>
                      <StatusPill tone={provider.ok ? "ok" : provider.configured ? "warn" : "bad"}>
                        {provider.ok ? "ok" : provider.configured ? "attention" : "not configured"}
                      </StatusPill>
                    </td>
                    <td>{provider.discovered_model_count}</td>
                    <td>
                      {provider.new_route_suggestion_count +
                        provider.stale_route_suggestion_count +
                        provider.recovered_route_suggestion_count}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="suggestion-list">
            {suggestions.length ? (
              suggestions.slice(0, 8).map((suggestion) => (
                <article className="event-row" key={suggestion.suggestion_id}>
                  <div>
                    <strong>{suggestion.title}</strong>
                    <span>{suggestion.details}</span>
                  </div>
                  <StatusPill tone={suggestion.action === "add_route" ? "ok" : "warn"}>
                    {suggestion.action.replace(/_/g, " ")}
                  </StatusPill>
                </article>
              ))
            ) : (
              <EmptyState message="No endpoint changes suggested." />
            )}
          </div>
        </div>
      )}
    </Panel>
  );
}

function Traffic({ live }: { live: LiveEvent[] }) {
  return (
    <Panel title="Live Traffic" icon={Activity}>
      <TrafficList live={live} />
    </Panel>
  );
}

function SettingsPanel({ desktopToken }: { desktopToken: string }) {
  const settings = useDesktopQuery<DesktopSettingsPayload>(
    "desktop-settings",
    "/v1/desktop/settings",
    desktopToken,
  );
  const [values, setValues] = React.useState<Record<string, string>>({});
  const saveSettings = useMutation({
    mutationFn: async () =>
      fetchJson<{ settings: DesktopSettingsPayload; restart_required: boolean }>(
        "/v1/desktop/settings",
        {
          method: "POST",
          headers: desktopHeaders(desktopToken),
          body: JSON.stringify(values),
        },
      ),
    onSuccess: (payload) => {
      queryClient.setQueryData(["desktop-settings"], payload.settings);
    },
  });

  React.useEffect(() => {
    if (!settings.data) return;
    setValues(Object.fromEntries(settings.data.fields.map((field) => [field.key, field.value])));
  }, [settings.data]);

  if (!desktopToken) return <DesktopRequired title="Settings" />;
  if (settings.isLoading) return <Panel title="Settings" icon={Settings}><EmptyState message="Loading desktop settings..." /></Panel>;
  if (settings.error) {
    return (
      <Panel title="Settings" icon={Settings}>
        <EmptyState message={settings.error.message} />
      </Panel>
    );
  }
  const payload = settings.data;
  if (!payload) return null;

  return (
    <div className="section-stack">
      <div className="section-heading">
        <div>
          <h1>Settings</h1>
          <p>Provider keys, runtime limits, storage paths, and endpoint maintenance controls.</p>
        </div>
        <button
          className="primary-action"
          type="button"
          disabled={saveSettings.isPending}
          onClick={() => saveSettings.mutate()}
        >
          <Save size={16} />
          {saveSettings.isPending ? "Saving" : "Save settings"}
        </button>
      </div>
      {saveSettings.isSuccess && <Notice tone="ok">Settings saved. Restart the gateway to apply runtime changes.</Notice>}
      {saveSettings.error && <Notice tone="bad">{saveSettings.error.message}</Notice>}
      {payload.groups.map((group) => {
        const fields = payload.fields.filter((field) => field.group === group);
        return (
          <Panel title={group} icon={Settings} key={group}>
            <div className="form-grid">
              {fields.map((field) => (
                <label className="field" key={field.key}>
                  <span>{field.label}</span>
                  {field.kind === "bool" ? (
                    <select
                      value={values[field.key] ?? ""}
                      onChange={(event) =>
                        setValues((current) => ({ ...current, [field.key]: event.target.value }))
                      }
                    >
                      <option value="true">true</option>
                      <option value="false">false</option>
                    </select>
                  ) : (
                    <input
                      type={field.secret ? "password" : numericFieldKind(field.kind) ? "number" : "text"}
                      step={field.kind === "float" ? "0.1" : undefined}
                      value={values[field.key] ?? ""}
                      onChange={(event) =>
                        setValues((current) => ({ ...current, [field.key]: event.target.value }))
                      }
                    />
                  )}
                </label>
              ))}
            </div>
          </Panel>
        );
      })}
      <p className="path-note">Settings file: {payload.env_path}</p>
    </div>
  );
}

function LogsPanel({ desktopToken }: { desktopToken: string }) {
  const logs = useDesktopQuery<DesktopLogsPayload>("desktop-logs", "/v1/desktop/logs", desktopToken, 5000);
  if (!desktopToken) return <DesktopRequired title="Logs" />;
  return (
    <Panel title="Logs" icon={Activity}>
      {logs.isLoading && <EmptyState message="Loading desktop logs..." />}
      {logs.error && <EmptyState message={logs.error.message} />}
      {logs.data && (
        <pre className="log-box">{logs.data.lines.join("") || "No logs captured yet."}</pre>
      )}
    </Panel>
  );
}

function BackupsPanel({ desktopToken }: { desktopToken: string }) {
  const [path, setPath] = React.useState("");
  const [overwrite, setOverwrite] = React.useState(false);
  const [file, setFile] = React.useState<File | null>(null);
  const exportBackup = useMutation({
    mutationFn: () =>
      fetchJson<BackupExportPayload>("/v1/desktop/backups/export", {
        method: "POST",
        headers: desktopHeaders(desktopToken),
        body: JSON.stringify({}),
      }),
  });
  const importBackup = useMutation({
    mutationFn: async () => {
      if (file) {
        const form = new FormData();
        form.append("file", file);
        form.append("overwrite", overwrite ? "true" : "false");
        return fetchJson<BackupImportPayload>("/v1/desktop/backups/import-upload", {
          method: "POST",
          headers: { "X-FreeRouter-Desktop-Token": desktopToken },
          body: form,
        });
      }
      if (!path.trim()) throw new Error("Choose a backup zip file or enter a backup path.");
      return fetchJson<BackupImportPayload>("/v1/desktop/backups/import", {
        method: "POST",
        headers: desktopHeaders(desktopToken),
        body: JSON.stringify({ path: path.trim(), overwrite }),
      });
    },
    onSuccess: () => {
      setFile(null);
      setPath("");
    },
  });

  if (!desktopToken) return <DesktopRequired title="Backups" />;
  return (
    <div className="section-stack">
      <div className="section-heading">
        <div>
          <h1>Backups</h1>
          <p>Export and restore local model catalog, SQLite state, and non-secret local settings.</p>
        </div>
      </div>
      <div className="two-column">
        <Panel title="Export Local State" icon={Download}>
          <p className="panel-copy">
            Create a zip backup of editable local state. Provider secrets are intentionally excluded.
          </p>
          <button
            className="primary-action"
            type="button"
            disabled={exportBackup.isPending}
            onClick={() => exportBackup.mutate()}
          >
            <Download size={16} />
            {exportBackup.isPending ? "Exporting" : "Export backup"}
          </button>
          {exportBackup.data && <Notice tone="ok">Exported to {exportBackup.data.path}</Notice>}
          {exportBackup.error && <Notice tone="bad">{exportBackup.error.message}</Notice>}
        </Panel>
        <Panel title="Restore Local State" icon={Upload}>
          <div className="form-grid single">
            <label className="field">
              <span>Backup zip file</span>
              <input
                type="file"
                accept=".zip,application/zip"
                onChange={(event) => setFile(event.target.files?.[0] ?? null)}
              />
            </label>
            <label className="field">
              <span>Or restore from path</span>
              <input
                value={path}
                placeholder="C:\\path\\to\\freerouter-local-state.zip"
                onChange={(event) => setPath(event.target.value)}
              />
            </label>
            <label className="check-field">
              <input
                type="checkbox"
                checked={overwrite}
                onChange={(event) => setOverwrite(event.target.checked)}
              />
              <span>Overwrite existing local state</span>
            </label>
          </div>
          <button
            className="danger-action"
            type="button"
            disabled={importBackup.isPending}
            onClick={() => importBackup.mutate()}
          >
            <Upload size={16} />
            {importBackup.isPending ? "Restoring" : "Restore backup"}
          </button>
          {importBackup.data && (
            <Notice tone="ok">
              Restored {importBackup.data.restored.length} file(s). Restart the gateway to use the restored state.
            </Notice>
          )}
          {importBackup.error && <Notice tone="bad">{importBackup.error.message}</Notice>}
        </Panel>
      </div>
    </div>
  );
}

function ChatPanel() {
  const [input, setInput] = React.useState("");
  const [messages, setMessages] = React.useState<ChatMessage[]>([]);
  const [events, setEvents] = React.useState<RouteEvent[]>([]);
  const [isSending, setIsSending] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);

  async function sendMessage() {
    const text = input.trim();
    if (!text || isSending) return;
    setInput("");
    setError(null);
    setIsSending(true);
    const nextMessages: ChatMessage[] = [...messages, { role: "user", content: text }];
    setMessages(nextMessages);
    setEvents([]);

    try {
      const response = await fetch("/v1/chat/completions/stream-route", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: nextMessages, max_tokens: 4096 }),
      });
      if (!response.body) throw new Error("The gateway did not return a stream.");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let assistantText = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const blocks = buffer.split("\n\n");
        buffer = blocks.pop() ?? "";

        for (const block of blocks) {
          const raw = block
            .split("\n")
            .find((line) => line.startsWith("data: "))
            ?.slice(6)
            .trim();
          if (!raw || raw === "[DONE]") continue;
          const event = JSON.parse(raw) as RouteEvent;
          if (event.type === "content" && event.text) {
            assistantText += event.text;
            setMessages([...nextMessages, { role: "assistant", content: assistantText }]);
          } else if (event.type === "done") {
            if (event.content) assistantText = event.content;
            setMessages([...nextMessages, { role: "assistant", content: assistantText }]);
          } else if (event.type === "error") {
            throw new Error(event.message || "All providers exhausted.");
          } else {
            setEvents((current) => [event, ...current].slice(0, 80));
          }
        }
      }
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setIsSending(false);
    }
  }

  return (
    <div className="chat-layout">
      <section className="chat-main panel">
        <header>
          <MessageSquareText size={18} />
          <h2>Chat</h2>
        </header>
        <div className="messages">
          {!messages.length && <EmptyState message="Send a message to route through the local gateway." />}
          {messages.map((message, index) => (
            <article className={`message ${message.role}`} key={`${message.role}-${index}`}>
              <span>{message.role}</span>
              <p>{message.content}</p>
            </article>
          ))}
          {isSending && <div className="typing">Routing request...</div>}
          {error && <Notice tone="bad">{error}</Notice>}
        </div>
        <div className="composer">
          <textarea
            rows={3}
            value={input}
            placeholder="Type a message..."
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                sendMessage();
              }
            }}
          />
          <button className="primary-action" type="button" disabled={isSending} onClick={sendMessage}>
            {isSending ? "Sending" : "Send"}
          </button>
        </div>
      </section>
      <Panel title="Route Activity" icon={Route}>
        <div className="event-list">
          {!events.length && <EmptyState message="Route attempts will appear here for the current message." />}
          {events.map((event, index) => (
            <div className="event-row" key={`${event.type}-${index}`}>
              <div>
                <strong>{event.type.replace(/_/g, " ")}</strong>
                <span>
                  {[event.provider, event.model_id, event.reason].filter(Boolean).join(" / ")}
                </span>
              </div>
              <StatusPill tone={routeEventTone(event.type)}>{event.route_id || event.provider || "route"}</StatusPill>
            </div>
          ))}
        </div>
      </Panel>
    </div>
  );
}

function ProviderTable({ providers }: { providers: ProviderStatus[] }) {
  if (!providers.length) return <EmptyState message="No provider state loaded yet." />;
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Provider</th>
            <th>Status</th>
            <th>Requests</th>
            <th>Tokens</th>
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

function TrafficList({ live }: { live: LiveEvent[] }) {
  if (!live.length) return <EmptyState message="No traffic recorded in this session." />;
  return (
    <div className="event-list">
      {live.map((event, index) => (
        <div className="event-row" key={`${event.request_id ?? "event"}-${index}`}>
          <div>
            <strong>{event.method || event.event_type || "request"}</strong>
            <span>{event.path || event.request_id || ""}</span>
          </div>
          <StatusPill tone="muted">{event.status || event.event_type || "event"}</StatusPill>
        </div>
      ))}
    </div>
  );
}

function Placeholder({ title }: { title: string }) {
  return (
    <Panel title={title} icon={Settings}>
      <EmptyState message="This surface will move from the legacy desktop shell into the React app next." />
    </Panel>
  );
}

function DesktopRequired({ title }: { title: string }) {
  return (
    <Panel title={title} icon={Settings}>
      <EmptyState message="Open FreeRouter from the desktop shell to use this local-machine control." />
    </Panel>
  );
}

function Notice({ tone, children }: { tone: "ok" | "bad"; children: React.ReactNode }) {
  return <div className={`notice ${tone}`}>{children}</div>;
}

function numericFieldKind(kind: string) {
  return kind === "int" || kind === "optional_int" || kind === "float";
}

function routeEventTone(type: string): "ok" | "warn" | "bad" | "muted" {
  if (type === "route_selected") return "ok";
  if (type === "route_fail") return "bad";
  if (type === "route_skip" || type === "route_flagged") return "warn";
  return "muted";
}

function healthTone(status?: string | null): "ok" | "warn" | "bad" | "muted" {
  if (!status || status === "active") return "ok";
  if (status.includes("disabled") || status.includes("exhausted")) return "bad";
  return "warn";
}

function desktopHeaders(token: string) {
  return {
    "Content-Type": "application/json",
    "X-FreeRouter-Desktop-Token": token,
  };
}

function useDesktopQuery<T>(key: string, path: string, token: string, refetchInterval?: number) {
  return useQuery({
    queryKey: [key],
    queryFn: () => fetchJson<T>(path, { headers: desktopHeaders(token) }),
    enabled: Boolean(token),
    refetchInterval,
  });
}

function Metric({ label, value, note }: { label: string; value: React.ReactNode; note: string }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{note}</small>
    </div>
  );
}

function Panel({
  title,
  icon: Icon,
  children,
}: {
  title: string;
  icon: LucideIcon;
  children: React.ReactNode;
}) {
  return (
    <section className="panel">
      <header>
        <Icon size={18} />
        <h2>{title}</h2>
      </header>
      {children}
    </section>
  );
}

function StatusPill({ tone, children }: { tone: "ok" | "warn" | "bad" | "muted"; children: React.ReactNode }) {
  return <span className={`pill ${tone}`}>{children}</span>;
}

function EmptyState({ message }: { message: string }) {
  return <div className="empty">{message}</div>;
}

const NAV_ITEMS = [
  { id: "dashboard", label: "Dashboard", icon: Gauge },
  { id: "models", label: "Models", icon: Bot },
  { id: "providers", label: "Providers", icon: ShieldCheck },
  { id: "traffic", label: "Traffic", icon: Activity },
  { id: "settings", label: "Settings", icon: Settings },
  { id: "backups", label: "Backups", icon: Database },
  { id: "logs", label: "Logs", icon: Activity },
  { id: "chat", label: "Chat", icon: MessageSquareText },
];

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
