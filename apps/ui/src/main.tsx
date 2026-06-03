import React from "react";
import ReactDOM from "react-dom/client";
import { QueryClient, QueryClientProvider, useMutation, useQuery } from "@tanstack/react-query";
import {
  Activity,
  Bot,
  Database,
  Gauge,
  type LucideIcon,
  MessageSquareText,
  RefreshCw,
  Route,
  Save,
  Settings,
  ShieldCheck,
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
          {activeSection === "models" && <Models routes={routes} />}
          {activeSection === "providers" && <Providers providers={providerRows} />}
          {activeSection === "traffic" && <Traffic live={liveRows} />}
          {activeSection === "settings" && <SettingsPanel desktopToken={desktopToken} />}
          {activeSection === "backups" && <Placeholder title="Backups" />}
          {activeSection === "chat" && <Placeholder title="Chat" />}
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

function Models({ routes }: { routes: ModelRoute[] }) {
  return (
    <Panel title="Model Routes" icon={Bot}>
      <div className="route-grid">
        {routes.slice(0, 30).map((route) => (
          <article className="route-card" key={route.route_id}>
            <div>
              <strong>{route.display_name}</strong>
              <span>{route.model_id}</span>
            </div>
            <div className="route-meta">
              <span>#{route.rank}</span>
              <span>{route.provider_name}</span>
              <StatusPill tone={route.enabled ? "ok" : "muted"}>
                {route.enabled ? "Enabled" : "Disabled"}
              </StatusPill>
            </div>
          </article>
        ))}
      </div>
    </Panel>
  );
}

function Providers({ providers }: { providers: ProviderStatus[] }) {
  return (
    <Panel title="Providers" icon={Database}>
      <ProviderTable providers={providers} />
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
