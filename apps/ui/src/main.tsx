import React from "react";
import ReactDOM from "react-dom/client";
import { QueryClient, QueryClientProvider, useMutation, useQuery } from "@tanstack/react-query";
import {
  Activity,
  BarChart3,
  Bot,
  Database,
  Download,
  Gauge,
  HeartPulse,
  MessageSquareText,
  RefreshCw,
  Route,
  ScrollText,
  Settings,
  Upload,
} from "lucide-react";
import "./styles.css";
import "./theme.css";
import { applyTheme, getThemePreference, initTheme, type ThemePreference } from "./theme";

type GatewayHealth = {
  status: string;
  version: string;
  database_path: string;
  providers: { configured: number; total: number };
  routes: { enabled: number; total: number };
};

type ModelRoute = {
  route_id: string;
  enabled: boolean;
  health?: { status?: string };
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
};

type DesktopSettingsPayload = {
  env_path: string;
  groups: string[];
  fields: DesktopField[];
};

type DesktopCapabilities = {
  desktop: boolean;
  server?: { base_url?: string };
};

type DesktopLogsPayload = {
  lines: string[];
};

type BackupExportPayload = { ok: boolean; path: string };
type BackupImportPayload = { ok: boolean; restored: string[] };

const EMBED_SECTIONS: Record<string, string> = {
  chat: "/chat?embed=1",
  models: "/models?embed=1",
  usage: "/status?embed=1",
  health: "/health?embed=1",
  live: "/live?embed=1",
  docs: "/docs?embed=1&v=2",
};

const NAV_ITEMS = [
  { id: "dashboard", label: "Dashboard", icon: Gauge },
  { id: "chat", label: "Chat", icon: MessageSquareText },
  { id: "models", label: "Models", icon: Bot },
  { id: "usage", label: "Usage", icon: BarChart3 },
  { id: "health", label: "Route Health", icon: HeartPulse },
  { id: "live", label: "Live Traffic", icon: Activity },
  { id: "settings", label: "Settings", icon: Settings },
  { id: "backups", label: "Backups", icon: Database },
  { id: "logs", label: "Logs", icon: ScrollText },
] as const;

type SectionId = (typeof NAV_ITEMS)[number]["id"] | "docs";

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

function desktopHeaders(token: string) {
  return {
    "Content-Type": "application/json",
    "X-FreeRouter-Desktop-Token": token,
  };
}

function App() {
  React.useEffect(() => {
    initTheme();
  }, []);

  return (
    <QueryClientProvider client={queryClient}>
      <FreeRouterShell />
    </QueryClientProvider>
  );
}

function FreeRouterShell() {
  const desktopToken = useDesktopToken();
  const [activeSection, setActiveSection] = React.useState<SectionId>(() => initialSection());
  const [previousSection, setPreviousSection] = React.useState<SectionId>("dashboard");
  const [loadedEmbeds, setLoadedEmbeds] = React.useState<Set<string>>(() => new Set());

  React.useEffect(() => {
    if (EMBED_SECTIONS[activeSection]) {
      setLoadedEmbeds((current) => new Set(current).add(activeSection));
    }
  }, [activeSection]);

  const desktopReady = useDesktopReady(desktopToken);
  const health = useGatewayQuery<GatewayHealth>("gateway-health", "/v1/gateway/health.json", 5000);
  const models = useGatewayQuery<{ data: ModelRoute[] }>("gateway-models", "/v1/gateway/models", 10000);
  const providers = useGatewayQuery<{ data: ProviderStatus[] }>("provider-status", "/v1/providers/status", 10000);
  const live = useGatewayQuery<{ data: LiveEvent[] }>("live-traffic", "/v1/gateway/live/snapshot", 3000);

  const routes = models.data?.data ?? [];
  const providerRows = providers.data?.data ?? [];
  const liveRows = live.data?.data ?? [];
  const baseUrl =
    desktopReady.data?.server?.base_url ?? `${window.location.origin}/v1`;

  React.useEffect(() => {
    const onHashChange = () => setActiveSection(initialSection());
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  React.useEffect(() => {
    if (activeSection === "dashboard") {
      const timer = window.setInterval(() => {
        queryClient.invalidateQueries({ queryKey: ["live-traffic"] });
      }, 3000);
      return () => window.clearInterval(timer);
    }
    return undefined;
  }, [activeSection]);

  function selectSection(section: SectionId) {
    if (section !== "docs" && activeSection !== "docs") {
      setPreviousSection(activeSection);
    }
    setActiveSection(section);
    if (location.hash !== `#${section}`) {
      history.replaceState(null, "", `#${section}`);
    }
    if (EMBED_SECTIONS[section]) {
      setLoadedEmbeds((current) => new Set(current).add(section));
    }
  }

  function refreshAll() {
    queryClient.invalidateQueries();
  }

  async function restartServer() {
    if (!desktopToken) return;
    await fetchJson("/v1/desktop/restart", {
      method: "POST",
      headers: desktopHeaders(desktopToken),
      body: JSON.stringify({}),
    });
    window.setTimeout(refreshAll, 700);
  }

  function toggleDocs() {
    if (activeSection === "docs") {
      selectSection(previousSection);
      return;
    }
    setPreviousSection(activeSection);
    selectSection("docs");
  }

  const configuredProviders = providerRows.filter((provider) => provider.configured).length;
  const availableProviders = providerRows.filter((provider) => provider.available).length;
  const enabledRoutes = routes.filter((route) => route.enabled).length;
  const flaggedRoutes = routes.filter(
    (route) => route.health?.status && route.health.status !== "active",
  ).length;

  const statusTone =
    health.data?.status === "ok" ? "ok" : health.isLoading ? "warn" : health.error ? "error" : "warn";
  const statusTitle =
    health.data?.status === "ok" ? "Gateway running" : health.error ? "Gateway needs attention" : "Checking gateway";
  const statusDetail = health.data
    ? `${health.data.providers.configured} configured providers, ${health.data.routes.enabled} enabled routes.`
    : health.error?.message ?? "Loading local status...";

  const isEmbed = Boolean(EMBED_SECTIONS[activeSection]);

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
              onClick={() => selectSection(item.id)}
            >
              <item.icon size={18} />
              <span className="nav-label">{item.label}</span>
            </button>
          ))}
        </nav>
        <div className="sidebar-footer">
          <div className="base-url">
            <label>OpenAI base URL</label>
            <code>{baseUrl}</code>
            <button type="button" onClick={() => copyText(baseUrl)}>
              Copy URL
            </button>
          </div>
        </div>
      </aside>

      <main className="main">
        <header className="topbar">
          <div className={`status-dot ${statusTone}`} />
          <div className="topbar-title">
            <strong>{statusTitle}</strong>
            <span>{statusDetail}</span>
          </div>
          <div className="topbar-actions">
            <button type="button" onClick={toggleDocs}>
              {activeSection === "docs" ? "Back" : "API Docs"}
            </button>
            <button type="button" onClick={refreshAll}>
              <RefreshCw size={16} />
              Refresh
            </button>
            <button type="button" disabled={!desktopReady.data?.desktop} onClick={restartServer}>
              Restart
            </button>
          </div>
        </header>

        <section className={`content ${isEmbed ? "content-embed" : ""}`}>
          {activeSection === "dashboard" && (
            <DashboardView
              health={health.data ?? null}
              configuredProviders={configuredProviders}
              availableProviders={availableProviders}
              enabledRoutes={enabledRoutes}
              routeCount={routes.length}
              flaggedRoutes={flaggedRoutes}
              providers={providerRows}
              live={liveRows}
            />
          )}
          {Array.from(loadedEmbeds).map((sectionId) => {
            const src = EMBED_SECTIONS[sectionId];
            if (!src) return null;
            return (
              <div
                key={sectionId}
                className={`embed-panel ${activeSection === sectionId ? "active" : ""}`}
                aria-hidden={activeSection !== sectionId}
              >
                <EmbedFrame title={sectionId} src={src} />
              </div>
            );
          })}
          {activeSection === "settings" && (
            <SettingsView desktopToken={desktopToken} desktopReady={desktopReady.data?.desktop ?? false} />
          )}
          {activeSection === "backups" && (
            <BackupsView desktopToken={desktopToken} desktopReady={desktopReady.data?.desktop ?? false} />
          )}
          {activeSection === "logs" && (
            <LogsView desktopToken={desktopToken} desktopReady={desktopReady.data?.desktop ?? false} />
          )}
        </section>
      </main>
    </div>
  );
}

function initialSection(): SectionId {
  const hash = (location.hash || "#dashboard").slice(1);
  if (hash === "docs" || NAV_ITEMS.some((item) => item.id === hash)) {
    return hash as SectionId;
  }
  return "dashboard";
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

function useDesktopReady(token: string) {
  return useQuery({
    queryKey: ["desktop-capabilities", token],
    queryFn: () =>
      fetchJson<DesktopCapabilities>("/v1/desktop/capabilities", {
        headers: desktopHeaders(token),
      }),
    enabled: Boolean(token),
    retry: false,
  });
}

function useGatewayQuery<T>(key: string, path: string, refetchInterval: number) {
  return useQuery({
    queryKey: [key],
    queryFn: () => fetchJson<T>(path),
    refetchInterval,
  });
}

function EmbedFrame({ title, src }: { title: string; src: string }) {
  const iframeRef = React.useRef<HTMLIFrameElement>(null);

  React.useEffect(() => {
    const frame = iframeRef.current;
    if (!frame) return;
    const onLoad = () => {
      applyTheme(getThemePreference(), { persist: false, broadcast: true });
    };
    frame.addEventListener("load", onLoad);
    return () => frame.removeEventListener("load", onLoad);
  }, [src]);

  return (
    <iframe
      ref={iframeRef}
      className="embed-frame"
      title={title}
      src={src}
      loading="lazy"
    />
  );
}

function DashboardView(props: {
  health: GatewayHealth | null;
  configuredProviders: number;
  availableProviders: number;
  enabledRoutes: number;
  routeCount: number;
  flaggedRoutes: number;
  providers: ProviderStatus[];
  live: LiveEvent[];
}) {
  return (
    <div className="section-stack">
      <div className="section-heading">
        <div>
          <h1>Dashboard</h1>
          <p>Local gateway status, provider readiness, route health, and recent traffic in one place.</p>
        </div>
      </div>
      <div className="metrics">
        <Metric label="Gateway" value={props.health?.status ?? "Unknown"} note={props.health?.database_path ?? "Waiting for server"} />
        <Metric
          label="Providers"
          value={`${props.configuredProviders}/${props.providers.length}`}
          note={`${props.availableProviders} available right now`}
        />
        <Metric label="Enabled routes" value={props.enabledRoutes} note={`${props.routeCount} routes in catalog`} />
        <Metric
          label="Route flags"
          value={props.flaggedRoutes}
          note={props.flaggedRoutes ? "Review route health" : "No active route flags"}
        />
      </div>
      <div className="two-column">
        <Panel title="Provider readiness">
          <ProviderTable providers={props.providers} />
        </Panel>
        <Panel title="Recent traffic">
          <TrafficTable live={props.live.slice(0, 6)} />
        </Panel>
      </div>
    </div>
  );
}

function AppearanceSettings() {
  const [preference, setPreference] = React.useState(getThemePreference);
  return (
    <Panel title="Appearance">
      <p className="panel-copy">
        Use Windows theme automatically, or keep FreeRouter pinned to light or dark mode.
      </p>
      <div className="fr-theme-segmented" role="group" aria-label="Theme preference">
        {(["system", "light", "dark"] as ThemePreference[]).map((option) => (
          <button
            key={option}
            className={`fr-theme-option ${preference === option ? "active" : ""}`}
            type="button"
            aria-pressed={preference === option}
            onClick={() => {
              applyTheme(option, { persist: true, broadcast: true });
              setPreference(option);
            }}
          >
            {option === "system" ? "System" : option === "light" ? "Light" : "Dark"}
          </button>
        ))}
      </div>
    </Panel>
  );
}

function SettingsView({ desktopToken, desktopReady }: { desktopToken: string; desktopReady: boolean }) {
  const settings = useQuery({
    queryKey: ["desktop-settings", desktopToken],
    queryFn: () =>
      fetchJson<DesktopSettingsPayload>("/v1/desktop/settings", {
        headers: desktopHeaders(desktopToken),
      }),
    enabled: desktopReady,
  });
  const [values, setValues] = React.useState<Record<string, string>>({});
  const saveSettings = useMutation({
    mutationFn: () =>
      fetchJson("/v1/desktop/settings", {
        method: "POST",
        headers: desktopHeaders(desktopToken),
        body: JSON.stringify(values),
      }),
    onSuccess: () => settings.refetch(),
  });

  React.useEffect(() => {
    if (!settings.data) return;
    setValues(Object.fromEntries(settings.data.fields.map((field) => [field.key, field.value])));
  }, [settings.data]);

  return (
    <div className="section-stack">
      <div className="section-heading">
        <div>
          <h1>Settings</h1>
          <p>Local API keys, runtime values, storage paths, and endpoint maintenance options.</p>
        </div>
        <button
          className="primary-action"
          type="button"
          disabled={!desktopReady || saveSettings.isPending}
          onClick={() => saveSettings.mutate()}
        >
          {saveSettings.isPending ? "Saving" : "Save settings"}
        </button>
      </div>
      <AppearanceSettings />
      {!desktopReady && <DesktopRequired />}
      {desktopReady && settings.isLoading && <EmptyState message="Loading desktop settings..." />}
      {desktopReady && settings.error && <Notice tone="bad">{settings.error.message}</Notice>}
      {desktopReady && settings.data && (
        <>
          {settings.data.groups.map((group) => {
            const fields = settings.data.fields.filter((field) => field.group === group);
            return (
              <Panel title={group} key={group}>
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
          <p className="path-note">Settings file: {settings.data.env_path}</p>
          {saveSettings.isSuccess && (
            <Notice tone="ok">Settings saved. Restart the server to apply runtime changes.</Notice>
          )}
        </>
      )}
    </div>
  );
}

function BackupsView({ desktopToken, desktopReady }: { desktopToken: string; desktopReady: boolean }) {
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

  if (!desktopReady) {
    return (
      <div className="section-stack">
        <SectionIntro title="Backups" copy="Export or restore local model catalog and SQLite state. Secrets are not included in backups." />
        <DesktopRequired />
      </div>
    );
  }

  return (
    <div className="section-stack">
      <SectionIntro title="Backups" copy="Export or restore local model catalog and SQLite state. Secrets are not included in backups." />
      <div className="two-column">
        <Panel title="Export local state">
          <p className="panel-copy">
            Creates a zip with the editable model catalog, SQLite state, and non-secret local settings.
          </p>
          <button className="primary-action" type="button" disabled={exportBackup.isPending} onClick={() => exportBackup.mutate()}>
            <Download size={16} />
            {exportBackup.isPending ? "Exporting" : "Export backup"}
          </button>
          {exportBackup.data && <Notice tone="ok">Exported to {exportBackup.data.path}</Notice>}
          {exportBackup.error && <Notice tone="bad">{exportBackup.error.message}</Notice>}
        </Panel>
        <Panel title="Restore local state">
          <div className="form-grid single">
            <label className="field">
              <span>Backup zip file</span>
              <input type="file" accept=".zip,application/zip" onChange={(event) => setFile(event.target.files?.[0] ?? null)} />
            </label>
            <label className="field">
              <span>Or enter a path</span>
              <input
                value={path}
                placeholder="C:\\path\\to\\freerouter-local-state.zip"
                onChange={(event) => setPath(event.target.value)}
              />
            </label>
            <label className="check-field">
              <input type="checkbox" checked={overwrite} onChange={(event) => setOverwrite(event.target.checked)} />
              <span>Overwrite existing local state</span>
            </label>
          </div>
          <button className="danger-action" type="button" disabled={importBackup.isPending} onClick={() => importBackup.mutate()}>
            <Upload size={16} />
            {importBackup.isPending ? "Restoring" : "Restore backup"}
          </button>
          {importBackup.data && (
            <Notice tone="ok">Restored {importBackup.data.restored.length} file(s). Restart the server.</Notice>
          )}
          {importBackup.error && <Notice tone="bad">{importBackup.error.message}</Notice>}
        </Panel>
      </div>
    </div>
  );
}

function LogsView({ desktopToken, desktopReady }: { desktopToken: string; desktopReady: boolean }) {
  const logs = useQuery({
    queryKey: ["desktop-logs", desktopToken],
    queryFn: () =>
      fetchJson<DesktopLogsPayload>("/v1/desktop/logs", {
        headers: desktopHeaders(desktopToken),
      }),
    enabled: desktopReady,
    refetchInterval: 5000,
  });

  return (
    <div className="section-stack">
      <div className="section-heading">
        <div>
          <h1>Logs</h1>
          <p>Server output captured by the desktop launcher.</p>
        </div>
        <button type="button" disabled={!desktopReady} onClick={() => logs.refetch()}>
          Refresh logs
        </button>
      </div>
      {!desktopReady && <DesktopRequired />}
      {desktopReady && logs.isLoading && <EmptyState message="Loading desktop logs..." />}
      {desktopReady && logs.error && <Notice tone="bad">{logs.error.message}</Notice>}
      {desktopReady && logs.data && (
        <pre className="log-box">{logs.data.lines.join("") || "No logs captured yet."}</pre>
      )}
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

function TrafficTable({ live }: { live: LiveEvent[] }) {
  if (!live.length) return <EmptyState message="No traffic recorded in this session." />;
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Event</th>
            <th>Status</th>
            <th>Route</th>
          </tr>
        </thead>
        <tbody>
          {live.map((event, index) => (
            <tr key={`${event.request_id ?? "event"}-${index}`}>
              <td>
                <strong>{event.method || event.path || event.event_type || "request"}</strong>
                <span>{event.path || event.request_id || ""}</span>
              </td>
              <td>
                <StatusPill tone="muted">{event.status || event.event_type || "event"}</StatusPill>
              </td>
              <td>{event.provider_name || event.route_id || ""}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function SectionIntro({ title, copy }: { title: string; copy: string }) {
  return (
    <div className="section-heading">
      <div>
        <h1>{title}</h1>
        <p>{copy}</p>
      </div>
    </div>
  );
}

function DesktopRequired() {
  return (
    <div className="disabled-box">
      <div>
        <strong>Desktop app required</strong>
        <span>
          Open FreeRouter from the desktop shortcut to use settings, backups, logs, restart, and tray controls.
          Normal gateway pages still work in a browser.
        </span>
      </div>
    </div>
  );
}

function Notice({ tone, children }: { tone: "ok" | "bad"; children: React.ReactNode }) {
  return <div className={`notice ${tone}`}>{children}</div>;
}

function numericFieldKind(kind: string) {
  return kind === "int" || kind === "optional_int" || kind === "float";
}

function copyText(text: string) {
  navigator.clipboard?.writeText(text).catch(() => {
    // Ignore clipboard failures.
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

function Panel({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="panel">
      <header>
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

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
