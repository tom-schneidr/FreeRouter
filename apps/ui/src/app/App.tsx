import React from "react";
import { QueryClientProvider } from "@tanstack/react-query";
import { RefreshCw } from "lucide-react";
import brandIcon from "../assets/brand/favicon.png";
import brandLogo from "../assets/brand/logo.png";
import { desktopHeaders, fetchJson, queryClient } from "../api/client";
import type { GatewayHealth, ModelRoute, ProviderStatus } from "../api/types";
import { DocsOverlay } from "../components/DocsOverlay";
import { EmbedFrame } from "../components/EmbedFrame";
import { BackupsView } from "../features/backups/BackupsView";
import { DashboardView } from "../features/dashboard/DashboardView";
import { LogsView } from "../features/logs/LogsView";
import { ModelsView } from "../features/models/ModelsView";
import { SettingsView } from "../features/settings/SettingsView";
import { useDesktopReady, useDesktopToken, useGatewayQuery } from "../hooks/useDesktop";
import { embedSrcWithReload } from "../lib/embedSrc";
import { copyText } from "../lib/format";
import { waitForGatewayHealth } from "../lib/gatewayHealth";
import {
  PRIMARY_NAV_ITEMS,
  SETTINGS_NAV_ITEM,
  initialSection,
  isFillSection,
  isLegacyEmbedSection,
  LEGACY_EMBED_SECTIONS,
  type SectionId,
} from "./routes";
import "../styles.css";
import "../theme.css";

// Re-export for tests if needed
export { initialSection };

function FreeRouterShell() {
  const desktopToken = useDesktopToken();
  const [activeSection, setActiveSection] = React.useState<SectionId>(() => initialSection());
  const [loadedEmbeds, setLoadedEmbeds] = React.useState<Set<string>>(() => new Set());
  const [embedReloadKey, setEmbedReloadKey] = React.useState(0);
  const [actionNotice, setActionNotice] = React.useState<string | null>(null);
  const [restarting, setRestarting] = React.useState(false);
  const [docsOpen, setDocsOpen] = React.useState(false);

  React.useEffect(() => {
    if (isLegacyEmbedSection(activeSection)) {
      setLoadedEmbeds((current) => new Set(current).add(activeSection));
    }
  }, [activeSection]);

  const desktopReady = useDesktopReady(desktopToken);
  const health = useGatewayQuery<GatewayHealth>("gateway-health", "/v1/gateway/health.json", 5000);
  const models = useGatewayQuery<{ data: ModelRoute[] }>("gateway-models", "/v1/gateway/models", 10000);
  const providers = useGatewayQuery<{ data: ProviderStatus[] }>("provider-status", "/v1/providers/status", 10000);

  const routes = models.data?.data ?? [];
  const providerRows = providers.data?.data ?? [];
  const baseUrl =
    desktopReady.data?.server?.base_url ?? `${window.location.origin}/v1`;

  React.useEffect(() => {
    const onHashChange = () => setActiveSection(initialSection());
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  function selectSection(section: SectionId) {
    setActiveSection(section);
    if (location.hash !== `#${section}`) {
      history.replaceState(null, "", `#${section}`);
    }
  }

  React.useEffect(() => {
    if (!actionNotice) return;
    const timer = window.setTimeout(() => setActionNotice(null), 4000);
    return () => window.clearTimeout(timer);
  }, [actionNotice]);

  function refreshAll() {
    void queryClient.invalidateQueries();
    setEmbedReloadKey((current) => current + 1);
    setActionNotice("Refreshed");
  }

  async function restartServer() {
    if (!desktopToken) {
      setActionNotice("Restart unavailable (open the desktop app, not /app in a browser tab).");
      return;
    }
    setRestarting(true);
    setActionNotice("Restarting server…");
    try {
      await fetchJson<{ ok?: boolean; status?: string }>("/v1/desktop/restart", {
        method: "POST",
        headers: desktopHeaders(desktopToken),
        body: JSON.stringify({}),
      });
      const healthy = await waitForGatewayHealth(45_000);
      if (!healthy) {
        setActionNotice("Restart timed out. Run npm run stop, then npm run desktop:dev.");
        return;
      }
      await queryClient.invalidateQueries();
      setEmbedReloadKey((current) => current + 1);
      setActionNotice("Server restarted");
    } catch (error) {
      const message = error instanceof Error ? error.message : "Restart failed";
      setActionNotice(message);
    } finally {
      setRestarting(false);
    }
  }

  function openDocs() {
    setDocsOpen(true);
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
    health.data?.status === "ok"
      ? "Gateway running"
      : health.error
        ? "Gateway needs attention"
        : "Checking gateway";
  const statusDetail = health.data
    ? `${health.data.providers.configured} configured providers, ${health.data.routes.enabled} enabled routes.`
    : health.error?.message ?? "Loading local status...";

  return (
    <div className="app-shell">
      <aside className="sidebar" aria-label="Primary navigation">
        <div className="brand">
          <img className="brand-icon" src={brandIcon} alt="FreeRouter" />
          <img className="brand-logo" src={brandLogo} alt="FreeRouter" />
          <span className="brand-subtitle">Local AI gateway</span>
        </div>
        <nav className="nav-list" aria-label="Primary navigation">
          {PRIMARY_NAV_ITEMS.map((item) => (
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
        <div className="sidebar-bottom">
          <nav className="nav-list nav-list-settings" aria-label="Settings">
            <button
              className={activeSection === SETTINGS_NAV_ITEM.id ? "active" : ""}
              type="button"
              onClick={() => selectSection(SETTINGS_NAV_ITEM.id)}
            >
              <SETTINGS_NAV_ITEM.icon size={18} />
              <span className="nav-label">{SETTINGS_NAV_ITEM.label}</span>
            </button>
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
            {actionNotice ? (
              <span className="topbar-notice" role="status">
                {actionNotice}
              </span>
            ) : null}
            <button type="button" onClick={openDocs} aria-pressed={docsOpen}>
              {docsOpen ? "Close docs" : "API Docs"}
            </button>
            <button type="button" onClick={refreshAll}>
              <RefreshCw size={16} />
              Refresh
            </button>
            <button
              type="button"
              disabled={!desktopToken || restarting}
              onClick={() => void restartServer()}
              title={
                desktopToken
                  ? "Restart the local gateway process"
                  : "Requires the FreeRouter desktop app"
              }
            >
              {restarting ? "Restarting…" : "Restart"}
            </button>
          </div>
        </header>

        <DocsOverlay open={docsOpen} onClose={() => setDocsOpen(false)} />

        <section className={`content${isFillSection(activeSection) ? " content-fill" : ""}`}>
          {activeSection === "dashboard" && (
            <DashboardView
              health={health.data ?? null}
              configuredProviders={configuredProviders}
              availableProviders={availableProviders}
              enabledRoutes={enabledRoutes}
              routeCount={routes.length}
              flaggedRoutes={flaggedRoutes}
              providers={providerRows}
            />
          )}
          {activeSection === "models" && <ModelsView />}
          {Array.from(loadedEmbeds).map((sectionId) => {
            const src = LEGACY_EMBED_SECTIONS[sectionId as SectionId];
            if (!src) return null;
            return (
              <div
                key={sectionId}
                className={`embed-panel ${activeSection === sectionId ? "active" : ""}`}
                aria-hidden={activeSection !== sectionId}
              >
                <EmbedFrame title={sectionId} src={embedSrcWithReload(src, embedReloadKey)} />
              </div>
            );
          })}
          {activeSection === "settings" && (
            <SettingsView
              desktopToken={desktopToken}
              desktopReady={desktopReady.data?.desktop ?? false}
            />
          )}
          {activeSection === "backups" && (
            <BackupsView
              desktopToken={desktopToken}
              desktopReady={desktopReady.data?.desktop ?? false}
            />
          )}
          {activeSection === "logs" && (
            <LogsView
              desktopToken={desktopToken}
              desktopReady={desktopReady.data?.desktop ?? false}
            />
          )}
        </section>
      </main>
    </div>
  );
}

export function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <FreeRouterShell />
    </QueryClientProvider>
  );
}
