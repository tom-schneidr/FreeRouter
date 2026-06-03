from __future__ import annotations

DESKTOP_APP_HTML = r"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>FreeRouter</title>
    <style>
      *, *::before, *::after { box-sizing: border-box; }
      :root {
        color-scheme: dark;
        --bg: #07111f;
        --bg-soft: #0b1424;
        --surface: #101b2e;
        --surface-2: #142238;
        --line: #24354d;
        --line-soft: rgba(148, 163, 184, 0.16);
        --text: #e5edf8;
        --muted: #91a4bd;
        --subtle: #667892;
        --accent: #4f8cff;
        --accent-2: #22c55e;
        --warn: #f59e0b;
        --danger: #ef4444;
        --radius: 8px;
        --shadow: 0 20px 60px rgba(0, 0, 0, .28);
        --font: "Segoe UI", Inter, system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
      }
      html, body {
        height: var(--app-height, 100%);
        max-height: var(--app-height, 100%);
        margin: 0;
      }
      body {
        overflow: hidden;
        background: var(--bg);
        color: var(--text);
        font-family: var(--font);
        font-size: 14px;
        letter-spacing: 0;
      }
      button, input, textarea, select {
        font: inherit;
        letter-spacing: 0;
      }
      button {
        border: 1px solid var(--line);
        border-radius: 7px;
        background: var(--surface-2);
        color: var(--text);
        min-height: 34px;
        padding: 0 12px;
        cursor: pointer;
      }
      button:hover:not(:disabled) { border-color: #3a5578; background: #1a2c47; }
      button.primary { border-color: #3971dd; background: #2563eb; color: #fff; }
      button.primary:hover:not(:disabled) { background: #1d4ed8; }
      button.danger { border-color: rgba(239,68,68,.55); color: #fecaca; background: rgba(239,68,68,.12); }
      button:disabled { cursor: not-allowed; opacity: .52; }
      input, textarea, select {
        width: 100%;
        border: 1px solid var(--line);
        border-radius: 7px;
        background: #08111f;
        color: var(--text);
        padding: 9px 10px;
        outline: none;
      }
      textarea { resize: vertical; min-height: 118px; line-height: 1.45; }
      input:focus, textarea:focus, select:focus { border-color: var(--accent); box-shadow: 0 0 0 3px rgba(79,140,255,.16); }
      a { color: #93c5fd; text-decoration: none; }
      .app {
        display: grid;
        grid-template-columns: 248px minmax(0, 1fr);
        height: var(--app-height, 100%);
        max-height: var(--app-height, 100%);
        min-height: 0;
      }
      .sidebar {
        display: flex;
        flex-direction: column;
        min-width: 0;
        min-height: 0;
        overflow: hidden;
        background: #091321;
        border-right: 1px solid var(--line);
      }
      .brand {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 18px 18px 16px;
        border-bottom: 1px solid var(--line-soft);
      }
      .brand-mark {
        width: 34px;
        height: 34px;
        border-radius: 9px;
        display: grid;
        place-items: center;
        background: #2563eb;
        box-shadow: 0 10px 28px rgba(37,99,235,.3);
      }
      .brand-mark::before {
        content: "";
        width: 17px;
        height: 14px;
        border-left: 4px solid #93c5fd;
        border-bottom: 4px solid #22c55e;
        transform: skewX(-12deg);
      }
      .brand h1 { margin: 0; font-size: 16px; line-height: 1.1; }
      .brand span { display: block; color: var(--muted); font-size: 12px; margin-top: 3px; }
      .nav {
        display: grid;
        gap: 4px;
        padding: 12px;
        flex: 1;
        min-height: 0;
        overflow-y: auto;
        align-content: start;
      }
      .nav button {
        display: flex;
        align-items: center;
        gap: 10px;
        width: 100%;
        justify-content: flex-start;
        min-height: 38px;
        padding: 0 10px;
        border-color: transparent;
        background: transparent;
        color: var(--muted);
      }
      .nav button:hover { color: var(--text); background: rgba(148,163,184,.08); }
      .nav button.active { color: #fff; background: rgba(79,140,255,.16); border-color: rgba(79,140,255,.28); }
      .nav-icon {
        width: 18px;
        height: 18px;
        display: inline-grid;
        place-items: center;
        border: 1px solid currentColor;
        border-radius: 5px;
        font-size: 10px;
        color: inherit;
        opacity: .9;
        flex: 0 0 auto;
      }
      .sidebar-footer {
        margin-top: auto;
        padding: 14px;
        border-top: 1px solid var(--line-soft);
        flex-shrink: 0;
      }
      .base-url {
        display: grid;
        gap: 8px;
        padding: 10px;
        border: 1px solid var(--line);
        border-radius: var(--radius);
        background: var(--bg-soft);
      }
      .base-url label { color: var(--muted); font-size: 11px; text-transform: uppercase; }
      .base-url code { color: #bfdbfe; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
      .main {
        min-width: 0;
        min-height: 0;
        display: grid;
        grid-template-rows: auto minmax(0, 1fr);
        background: radial-gradient(circle at 76% -10%, rgba(59,130,246,.12), transparent 34%), var(--bg);
        overflow: hidden;
      }
      .topbar {
        min-width: 0;
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 14px 18px;
        border-bottom: 1px solid var(--line);
        background: rgba(7, 17, 31, .86);
        backdrop-filter: blur(12px);
      }
      .status-dot {
        width: 9px;
        height: 9px;
        border-radius: 50%;
        background: var(--subtle);
        box-shadow: 0 0 0 4px rgba(148,163,184,.12);
      }
      .status-dot.ok { background: var(--accent-2); box-shadow: 0 0 0 4px rgba(34,197,94,.14); }
      .status-dot.warn { background: var(--warn); box-shadow: 0 0 0 4px rgba(245,158,11,.14); }
      .status-dot.error { background: var(--danger); box-shadow: 0 0 0 4px rgba(239,68,68,.14); }
      .topbar-title { min-width: 0; flex: 1; }
      .topbar-title strong { display: block; font-size: 14px; }
      .topbar-title span { display: block; color: var(--muted); font-size: 12px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
      .topbar-actions { display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }
      .content {
        min-width: 0;
        min-height: 0;
        overflow: hidden;
        padding: 0;
        display: flex;
        flex-direction: column;
      }
      .section { display: none; max-width: none; margin: 0; flex: 1; min-height: 0; overflow: auto; }
      .section.active { display: block; }
      .section.embed-section.active { display: flex; flex-direction: column; overflow: hidden; }
      .section:not(.embed-section) { padding: 18px; max-width: 1480px; width: 100%; margin: 0 auto; box-sizing: border-box; }
      .section-header {
        display: flex;
        align-items: flex-end;
        justify-content: space-between;
        gap: 18px;
        margin-bottom: 16px;
      }
      .section-header h2 { margin: 0 0 4px; font-size: 24px; line-height: 1.15; }
      .section-header p { margin: 0; color: var(--muted); line-height: 1.5; max-width: 780px; }
      .toolbar { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
      .grid { display: grid; gap: 12px; }
      .grid.cols-4 { grid-template-columns: repeat(4, minmax(0, 1fr)); }
      .grid.cols-3 { grid-template-columns: repeat(3, minmax(0, 1fr)); }
      .grid.cols-2 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .panel {
        border: 1px solid var(--line);
        border-radius: var(--radius);
        background: rgba(16, 27, 46, .9);
        box-shadow: var(--shadow);
        min-width: 0;
      }
      .panel.pad { padding: 14px; }
      .metric { padding: 14px; min-width: 0; }
      .metric label {
        display: block;
        color: var(--muted);
        font-size: 11px;
        text-transform: uppercase;
        margin-bottom: 8px;
      }
      .metric strong { display: block; font-size: 24px; line-height: 1; overflow-wrap: anywhere; }
      .metric span { display: block; color: var(--subtle); margin-top: 8px; font-size: 12px; overflow-wrap: anywhere; }
      .table-wrap { overflow: auto; border-radius: var(--radius); border: 1px solid var(--line); background: var(--surface); }
      table { width: 100%; border-collapse: collapse; min-width: 760px; }
      th, td { padding: 10px 12px; border-bottom: 1px solid var(--line-soft); text-align: left; vertical-align: middle; }
      th { position: sticky; top: 0; z-index: 1; color: var(--muted); background: #0d1728; font-size: 11px; text-transform: uppercase; }
      td { color: #d9e4f2; }
      tr:hover td { background: rgba(79,140,255,.06); }
      .cell-main { display: grid; gap: 3px; min-width: 0; }
      .cell-main strong, .truncate { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; min-width: 0; }
      .cell-main small { color: var(--muted); overflow-wrap: anywhere; }
      .pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        min-height: 22px;
        padding: 0 8px;
        border-radius: 999px;
        border: 1px solid var(--line);
        color: var(--muted);
        background: rgba(148,163,184,.08);
        font-size: 12px;
        white-space: nowrap;
      }
      .pill.ok { color: #bbf7d0; border-color: rgba(34,197,94,.4); background: rgba(34,197,94,.12); }
      .pill.warn { color: #fde68a; border-color: rgba(245,158,11,.44); background: rgba(245,158,11,.12); }
      .pill.error { color: #fecaca; border-color: rgba(239,68,68,.44); background: rgba(239,68,68,.12); }
      .muted { color: var(--muted); }
      .empty, .error-box, .disabled-box {
        display: grid;
        place-items: center;
        min-height: 170px;
        padding: 24px;
        color: var(--muted);
        text-align: center;
        border: 1px dashed var(--line);
        border-radius: var(--radius);
        background: rgba(8,17,31,.74);
      }
      .error-box { color: #fecaca; border-color: rgba(239,68,68,.5); background: rgba(239,68,68,.08); }
      .form-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
      .field { display: grid; gap: 6px; min-width: 0; }
      .field label { color: var(--muted); font-size: 12px; }
      .field.full { grid-column: 1 / -1; }
      .settings-group { display: grid; gap: 12px; margin-bottom: 14px; }
      .settings-group h3 { margin: 0; font-size: 14px; }
      .log-box {
        height: min(520px, 58vh);
        overflow: auto;
        white-space: pre-wrap;
        font-family: Consolas, "Cascadia Mono", monospace;
        font-size: 12px;
        line-height: 1.45;
        padding: 14px;
        color: #c7d2fe;
        background: #050b15;
        border-radius: var(--radius);
        border: 1px solid var(--line);
      }
      .chat-layout { display: grid; grid-template-columns: minmax(0, 1fr) 360px; gap: 12px; }
      .chat-output { min-height: 280px; max-height: 50vh; overflow: auto; padding: 14px; line-height: 1.6; }
      .route-list { max-height: 58vh; overflow: auto; display: grid; gap: 8px; }
      .route-item { display: grid; gap: 5px; padding: 10px; border: 1px solid var(--line-soft); border-radius: 7px; background: rgba(8,17,31,.5); }
      .row-actions { display: flex; gap: 6px; flex-wrap: wrap; }
      .qa-note { margin-top: 12px; color: var(--subtle); font-size: 12px; }
      .embed-section { padding: 0; max-width: none; overflow: hidden; }
      .embed-frame {
        flex: 1;
        width: 100%;
        min-height: 0;
        border: none;
        background: #0a0e1a;
        display: block;
      }
      @media (max-width: 1180px) {
        .app { grid-template-columns: 74px minmax(0, 1fr); }
        .brand div:last-child, .nav-label, .sidebar-footer { display: none; }
        .brand { justify-content: center; padding-left: 0; padding-right: 0; }
        .nav button { justify-content: center; padding: 0; }
        .grid.cols-4 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        .grid.cols-2 { grid-template-columns: 1fr; }
        .chat-layout { grid-template-columns: 1fr; }
      }
      @media (max-width: 760px) {
        html, body { overflow: hidden; }
        .app { display: grid; grid-template-columns: 74px minmax(0, 1fr); height: var(--app-height, 100%); min-height: 0; }
        .sidebar { position: relative; z-index: 4; min-height: 0; }
        .brand { justify-content: flex-start; padding: 12px; }
        .brand div:last-child { display: block; }
        .nav { display: flex; overflow: auto; padding: 8px; }
        .nav button { width: auto; min-width: 42px; }
        .main { display: grid; min-height: 0; overflow: hidden; }
        .topbar, .section-header { align-items: flex-start; flex-direction: column; }
        .content { overflow: hidden; padding: 0; min-height: 0; }
        .section:not(.embed-section) { padding: 12px; }
        .grid.cols-4, .grid.cols-3, .grid.cols-2, .form-grid { grid-template-columns: 1fr; }
      }
    </style>
  </head>
  <body>
    <div class="app">
      <aside class="sidebar">
        <div class="brand">
          <div class="brand-mark" aria-hidden="true"></div>
          <div>
            <h1>FreeRouter</h1>
            <span>Local AI gateway</span>
          </div>
        </div>
        <nav class="nav" aria-label="App sections">
          <button data-section="dashboard" class="active"><span class="nav-icon">D</span><span class="nav-label">Dashboard</span></button>
          <button data-section="chat"><span class="nav-icon">C</span><span class="nav-label">Chat</span></button>
          <button data-section="models"><span class="nav-icon">M</span><span class="nav-label">Models</span></button>
          <button data-section="usage"><span class="nav-icon">U</span><span class="nav-label">Usage</span></button>
          <button data-section="health"><span class="nav-icon">H</span><span class="nav-label">Route Health</span></button>
          <button data-section="live"><span class="nav-icon">L</span><span class="nav-label">Live Traffic</span></button>
          <button data-section="settings"><span class="nav-icon">S</span><span class="nav-label">Settings</span></button>
          <button data-section="backups"><span class="nav-icon">B</span><span class="nav-label">Backups</span></button>
          <button data-section="logs"><span class="nav-icon">G</span><span class="nav-label">Logs</span></button>
        </nav>
        <div class="sidebar-footer">
          <div class="base-url">
            <label>OpenAI base URL</label>
            <code id="baseUrl">http://127.0.0.1:8000/v1</code>
            <button id="copyBaseUrl" type="button">Copy URL</button>
          </div>
        </div>
      </aside>

      <main class="main">
        <header class="topbar">
          <span id="serverDot" class="status-dot"></span>
          <div class="topbar-title">
            <strong id="serverTitle">Checking gateway</strong>
            <span id="serverDetail">Loading local status...</span>
          </div>
          <div class="topbar-actions">
            <button id="openDocs" type="button">API Docs</button>
            <button id="refreshAll" type="button">Refresh</button>
            <button id="restartServer" type="button" disabled>Restart</button>
          </div>
        </header>

        <div class="content">
          <section id="section-dashboard" class="section active">
            <div class="section-header">
              <div>
                <h2>Dashboard</h2>
                <p>Local gateway status, provider readiness, route health, and recent traffic in one place.</p>
              </div>
            </div>
            <div id="dashboardMetrics" class="grid cols-4"></div>
            <div class="grid cols-2" style="margin-top:12px">
              <div class="panel pad">
                <div class="section-header" style="margin-bottom:10px">
                  <div>
                    <h2 style="font-size:16px">Provider readiness</h2>
                    <p>Configured providers and current local quota state.</p>
                  </div>
                </div>
                <div id="dashboardProviders"></div>
              </div>
              <div class="panel pad">
                <div class="section-header" style="margin-bottom:10px">
                  <div>
                    <h2 style="font-size:16px">Recent traffic</h2>
                    <p>Latest local requests observed by the gateway.</p>
                  </div>
                </div>
                <div id="dashboardTraffic"></div>
              </div>
            </div>
          </section>

          <section id="section-chat" class="section embed-section">
            <iframe id="frame-chat" class="embed-frame" title="Chat" loading="lazy"></iframe>
          </section>

          <section id="section-models" class="section embed-section">
            <iframe id="frame-models" class="embed-frame" title="Models" loading="lazy"></iframe>
          </section>

          <section id="section-usage" class="section embed-section">
            <iframe id="frame-usage" class="embed-frame" title="Provider Usage" loading="lazy"></iframe>
          </section>

          <section id="section-health" class="section embed-section">
            <iframe id="frame-health" class="embed-frame" title="Route Health" loading="lazy"></iframe>
          </section>

          <section id="section-live" class="section embed-section">
            <iframe id="frame-live" class="embed-frame" title="Live Traffic" loading="lazy"></iframe>
          </section>

          <section id="section-settings" class="section">
            <div class="section-header">
              <div>
                <h2>Settings</h2>
                <p>Local API keys, runtime values, storage paths, and endpoint maintenance options.</p>
              </div>
              <div class="toolbar">
                <button id="saveSettings" class="primary" type="button" disabled>Save settings</button>
              </div>
            </div>
            <div id="settingsPanel"></div>
          </section>

          <section id="section-backups" class="section">
            <div class="section-header">
              <div>
                <h2>Backups</h2>
                <p>Export or restore local model catalog and SQLite state. Secrets are not included in backups.</p>
              </div>
            </div>
            <div id="backupsPanel"></div>
          </section>

          <section id="section-logs" class="section">
            <div class="section-header">
              <div>
                <h2>Logs</h2>
                <p>Server output captured by the desktop launcher.</p>
              </div>
              <div class="toolbar">
                <button id="refreshLogs" type="button" disabled>Refresh logs</button>
              </div>
            </div>
            <div id="logsPanel"></div>
          </section>
        </div>
      </main>
    </div>

    <script>
      const $ = (id) => document.getElementById(id);
      const appState = {
        desktop: false,
        bridgeReady: false,
        desktopToken: new URLSearchParams(location.search).get('desktop_token') || sessionStorage.getItem('freerouterDesktopToken') || '',
        health: null,
        models: [],
        providers: [],
        live: [],
        settings: null,
        activeSection: 'dashboard'
      };

      const navButtons = [...document.querySelectorAll('.nav button[data-section]')];
      const EMBED_SECTIONS = {
        chat: '/chat?embed=1',
        models: '/models?embed=1',
        usage: '/status?embed=1',
        health: '/health?embed=1',
        live: '/live?embed=1',
      };
      const loadedEmbeds = new Set();

      function syncViewportHeight() {
        const height = window.innerHeight;
        document.documentElement.style.setProperty('--app-height', `${height}px`);
      }

      function escapeHtml(value) {
        return String(value ?? '').replace(/[&<>"']/g, (char) => ({
          '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
        }[char]));
      }

      async function fetchJson(path, options = {}) {
        const response = await fetch(path, options);
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) {
          throw new Error(payload.detail || response.statusText || `HTTP ${response.status}`);
        }
        return payload;
      }

      async function bridgeCall(name, ...args) {
        if (!appState.desktopToken) {
          if (window.pywebview?.api?.[name]) {
            return window.pywebview.api[name](...args);
          }
          throw new Error('This action is available in the desktop app only.');
        }
        const endpointMap = {
          get_capabilities: ['GET', '/v1/desktop/capabilities'],
          get_settings: ['GET', '/v1/desktop/settings'],
          save_settings: ['POST', '/v1/desktop/settings'],
          export_backup: ['POST', '/v1/desktop/backups/export'],
          import_backup: ['POST', '/v1/desktop/backups/import'],
          import_backup_upload: ['POST', '/v1/desktop/backups/import-upload'],
          get_logs: ['GET', '/v1/desktop/logs'],
          restart_server: ['POST', '/v1/desktop/restart'],
        };
        const target = endpointMap[name];
        if (!target) throw new Error('This action is available in the desktop app only.');
        const [method, path] = target;
        const options = {
          method,
          headers: { 'X-FreeRouter-Desktop-Token': appState.desktopToken },
        };
        if (method !== 'GET') {
          options.headers['Content-Type'] = 'application/json';
          if (name === 'save_settings') options.body = JSON.stringify(args[0] || {});
          else if (name === 'import_backup') options.body = JSON.stringify({ path: args[0], overwrite: args[1] });
          else if (name !== 'import_backup_upload') options.body = JSON.stringify({});
        }
        if (name === 'import_backup_upload') {
          throw new Error('Use uploadBackupFile() for backup uploads.');
        }
        return fetchJson(path, options);
      }

      async function uploadBackupFile(file, overwrite) {
        if (!appState.desktopToken) throw new Error('This action is available in the desktop app only.');
        const form = new FormData();
        form.append('file', file);
        form.append('overwrite', overwrite ? 'true' : 'false');
        const response = await fetch('/v1/desktop/backups/import-upload', {
          method: 'POST',
          headers: { 'X-FreeRouter-Desktop-Token': appState.desktopToken },
          body: form,
        });
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) {
          throw new Error(payload.detail || response.statusText || `HTTP ${response.status}`);
        }
        return payload;
      }

      function ensureEmbedFrame(section) {
        const path = EMBED_SECTIONS[section];
        if (!path) return;
        const frame = $(`frame-${section}`);
        if (!frame) return;
        if (!loadedEmbeds.has(section)) {
          frame.src = path;
          loadedEmbeds.add(section);
        }
      }

      function setServerChrome(status, detail) {
        const dot = $('serverDot');
        dot.className = 'status-dot';
        if (status === 'ok' || status === 'running') dot.classList.add('ok');
        else if (status === 'starting' || status === 'degraded') dot.classList.add('warn');
        else dot.classList.add('error');
        $('serverTitle').textContent = status === 'ok' || status === 'running' ? 'Gateway running' : 'Gateway needs attention';
        $('serverDetail').textContent = detail || 'Local status ready.';
      }

      function metric(label, value, note = '') {
        return `<div class="panel metric"><label>${escapeHtml(label)}</label><strong>${escapeHtml(value)}</strong><span>${escapeHtml(note)}</span></div>`;
      }

      function pill(text, type = '') {
        return `<span class="pill ${type}">${escapeHtml(text)}</span>`;
      }

      async function detectDesktopBridge() {
        if (appState.desktopToken) {
          sessionStorage.setItem('freerouterDesktopToken', appState.desktopToken);
        }
        const maxAttempts = 30;
        for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
          if (window.pywebview?.api?.get_capabilities || appState.desktopToken) break;
          await new Promise((resolve) => setTimeout(resolve, 150));
        }
        try {
          const capabilities = await bridgeCall('get_capabilities');
          appState.desktop = true;
          appState.bridgeReady = true;
          if (capabilities.server?.base_url) $('baseUrl').textContent = capabilities.server.base_url;
          $('restartServer').disabled = false;
          $('refreshLogs').disabled = false;
          $('saveSettings').disabled = false;
          renderDesktopOnlyPanels();
        } catch {
          appState.desktop = false;
          appState.bridgeReady = true;
          renderDesktopOnlyPanels();
        }
      }

      async function refreshAll() {
        setServerChrome('starting', 'Refreshing local gateway state...');
        const [health, models, providers, live] = await Promise.allSettled([
          fetchJson('/v1/gateway/health.json'),
          fetchJson('/v1/gateway/models'),
          fetchJson('/v1/providers/status'),
          fetchJson('/v1/gateway/live/snapshot'),
        ]);
        if (health.status === 'fulfilled') {
          appState.health = health.value;
          if (!appState.desktop) $('baseUrl').textContent = `${location.origin}/v1`;
          setServerChrome('ok', `${health.value.providers.configured} configured providers, ${health.value.routes.enabled} enabled routes.`);
        } else {
          setServerChrome('error', health.reason.message);
        }
        appState.models = models.status === 'fulfilled' ? models.value.data || [] : [];
        appState.providers = providers.status === 'fulfilled' ? providers.value.data || [] : [];
        appState.live = live.status === 'fulfilled' ? live.value.data || [] : [];
        renderAll();
      }

      function renderAll() {
        renderDashboard();
        renderDesktopOnlyPanels();
      }

      function renderDashboard() {
        const h = appState.health;
        const providers = appState.providers;
        const configured = providers.filter((provider) => provider.configured).length;
        const available = providers.filter((provider) => provider.available).length;
        const activeRoutes = appState.models.filter((route) => route.enabled).length;
        const unhealthy = appState.models.filter((route) => route.health?.status && route.health.status !== 'active').length;
        $('dashboardMetrics').innerHTML = [
          metric('Gateway', h?.status || 'Unknown', h?.database_path || 'Waiting for server'),
          metric('Providers', `${configured}/${providers.length}`, `${available} available right now`),
          metric('Enabled routes', activeRoutes, `${appState.models.length} routes in catalog`),
          metric('Route flags', unhealthy, unhealthy ? 'Review route health' : 'No active route flags'),
        ].join('');

        $('dashboardProviders').innerHTML = providers.length ? table(
          ['Provider', 'Status', 'Requests today', 'Tokens today'],
          providers.map((provider) => [
            `<div class="cell-main"><strong>${escapeHtml(provider.name)}</strong><small>${provider.configured ? 'API key configured' : 'Missing API key'}</small></div>`,
            provider.configured ? pill(provider.available ? 'Available' : provider.unavailable_reason || 'Unavailable', provider.available ? 'ok' : 'warn') : pill('Not configured', 'error'),
            provider.requests_today,
            provider.tokens_used_today,
          ])
        ) : empty('No provider state loaded yet.');

        const rows = appState.live.slice(0, 6).map((event) => [
          `<div class="cell-main"><strong>${escapeHtml(event.method || event.path || event.event_type || 'request')}</strong><small>${escapeHtml(event.path || event.request_id || '')}</small></div>`,
          pill(event.status || event.event_type || 'event'),
          escapeHtml(event.provider_name || event.route_id || ''),
        ]);
        $('dashboardTraffic').innerHTML = rows.length ? table(['Event', 'Status', 'Route'], rows) : empty('No traffic recorded in this session.');
      }

      function table(headers, rows) {
        return `<div class="table-wrap"><table><thead><tr>${headers.map((header) => `<th>${escapeHtml(header)}</th>`).join('')}</tr></thead><tbody>${rows.map((row) => `<tr>${row.map((cell) => `<td>${cell}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
      }

      function empty(message) {
        return `<div class="empty">${escapeHtml(message)}</div>`;
      }

      function renderDesktopOnlyPanels() {
        if (!appState.bridgeReady) {
          $('settingsPanel').innerHTML = empty('Checking desktop capabilities...');
          $('backupsPanel').innerHTML = empty('Checking desktop capabilities...');
          $('logsPanel').innerHTML = empty('Checking desktop capabilities...');
          return;
        }
        if (!appState.desktop) {
          const disabled = `<div class="disabled-box"><div><strong>Desktop app required</strong><br><span>Open FreeRouter from the desktop shortcut to use settings, backups, logs, restart, and tray controls. Normal gateway pages still work in a browser.</span></div></div>`;
          $('settingsPanel').innerHTML = disabled;
          $('backupsPanel').innerHTML = disabled;
          $('logsPanel').innerHTML = disabled;
          return;
        }
        loadSettings();
        renderBackups();
        loadLogs();
      }

      async function loadSettings() {
        try {
          const payload = await bridgeCall('get_settings');
          appState.settings = payload;
          const groups = [...new Set(payload.fields.map((field) => field.group))];
          $('settingsPanel').innerHTML = groups.map((group) => {
            const fields = payload.fields.filter((field) => field.group === group);
            return `<div class="panel pad settings-group"><h3>${escapeHtml(group)}</h3><div class="form-grid">${fields.map(settingField).join('')}</div></div>`;
          }).join('') + `<div class="qa-note">Settings file: ${escapeHtml(payload.env_path)}</div>`;
        } catch (error) {
          $('settingsPanel').innerHTML = `<div class="error-box">${escapeHtml(error.message)}</div>`;
        }
      }

      function settingField(field) {
        if (field.kind === 'bool') {
          return `<div class="field"><label for="setting-${escapeHtml(field.key)}">${escapeHtml(field.label)}</label><select id="setting-${escapeHtml(field.key)}" data-setting="${escapeHtml(field.key)}"><option value="true" ${field.value === 'true' ? 'selected' : ''}>true</option><option value="false" ${field.value === 'false' ? 'selected' : ''}>false</option></select></div>`;
        }
        const type = field.secret ? 'password' : (field.kind === 'int' || field.kind === 'optional_int' || field.kind === 'float' ? 'number' : 'text');
        const step = field.kind === 'float' ? ' step="0.1"' : '';
        return `<div class="field"><label for="setting-${escapeHtml(field.key)}">${escapeHtml(field.label)}</label><input id="setting-${escapeHtml(field.key)}" data-setting="${escapeHtml(field.key)}" type="${type}"${step} value="${escapeHtml(field.value)}"></div>`;
      }

      function collectSettings() {
        const values = {};
        document.querySelectorAll('[data-setting]').forEach((input) => {
          values[input.dataset.setting] = input.value;
        });
        return values;
      }

      function renderBackups() {
        $('backupsPanel').innerHTML = `
          <div class="grid cols-2">
            <div class="panel pad">
              <h3 style="margin:0 0 8px">Export local state</h3>
              <p class="muted">Creates a zip with the editable model catalog, SQLite state, and non-secret local settings.</p>
              <button id="exportBackup" class="primary" type="button">Export backup</button>
              <div id="backupExportResult" class="qa-note"></div>
            </div>
            <div class="panel pad">
              <h3 style="margin:0 0 8px">Restore local state</h3>
              <p class="muted">Choose a FreeRouter backup zip. Restart the server after restore.</p>
              <div class="field"><label for="backupFile">Backup zip file</label><input id="backupFile" type="file" accept=".zip,application/zip"></div>
              <div class="field"><label for="backupPath">Or enter a path</label><input id="backupPath" placeholder="C:\\path\\to\\freerouter-local-state.zip"></div>
              <div class="toolbar" style="margin-top:10px">
                <button id="importBackup" class="danger" type="button">Restore backup</button>
              </div>
              <label style="display:flex; gap:8px; align-items:center; margin-top:10px; color:var(--muted)"><input id="backupOverwrite" type="checkbox" style="width:auto">Overwrite existing local state</label>
              <div id="backupImportResult" class="qa-note"></div>
            </div>
          </div>
        `;
      }

      async function loadLogs() {
        try {
          const payload = await bridgeCall('get_logs', 600);
          $('logsPanel').innerHTML = `<pre class="log-box">${escapeHtml((payload.lines || []).join('') || 'No logs captured yet.')}</pre>`;
        } catch (error) {
          $('logsPanel').innerHTML = `<div class="error-box">${escapeHtml(error.message)}</div>`;
        }
      }

      function selectSection(section) {
        appState.activeSection = section;
        navButtons.forEach((button) => button.classList.toggle('active', button.dataset.section === section));
        document.querySelectorAll('.section').forEach((el) => el.classList.toggle('active', el.id === `section-${section}`));
        if (location.hash !== `#${section}`) history.replaceState(null, '', `#${section}`);
        ensureEmbedFrame(section);
      }

      document.addEventListener('click', async (event) => {
        const nav = event.target.closest('.nav button[data-section]');
        if (nav) selectSection(nav.dataset.section);

        if (event.target.id === 'exportBackup') {
          const result = await bridgeCall('export_backup');
          $('backupExportResult').textContent = result.ok ? `Exported to ${result.path}` : result.error || 'Export failed';
        }
        if (event.target.id === 'importBackup') {
          const overwrite = $('backupOverwrite').checked;
          const fileInput = $('backupFile');
          try {
            if (fileInput?.files?.length) {
              const result = await uploadBackupFile(fileInput.files[0], overwrite);
              $('backupImportResult').textContent = `Restored ${result.restored.length} file(s). Restart the server.`;
              fileInput.value = '';
            } else {
              const path = $('backupPath').value.trim();
              if (!path) {
                $('backupImportResult').textContent = 'Choose a backup zip file or enter a path first.';
                return;
              }
              const result = await bridgeCall('import_backup', path, overwrite);
              $('backupImportResult').textContent = result.ok ? `Restored ${result.restored.length} file(s). Restart the server.` : result.error || 'Restore failed';
            }
          } catch (error) {
            $('backupImportResult').textContent = error.message;
          }
        }
      });

      $('refreshAll').addEventListener('click', refreshAll);
      $('openDocs').addEventListener('click', () => { window.location.href = '/docs'; });
      $('copyBaseUrl').addEventListener('click', async () => {
        const text = $('baseUrl').textContent;
        try {
          await navigator.clipboard.writeText(text);
        } catch {
          if (appState.desktop) await bridgeCall('copy_base_url');
        }
      });
      $('restartServer').addEventListener('click', async () => {
        $('restartServer').disabled = true;
        try {
          const status = await bridgeCall('restart_server');
          setServerChrome(status.status, status.detail || 'Server restarted.');
          setTimeout(refreshAll, 700);
        } finally {
          $('restartServer').disabled = false;
        }
      });
      $('saveSettings').addEventListener('click', async () => {
        $('saveSettings').disabled = true;
        try {
          await bridgeCall('save_settings', collectSettings());
          await loadSettings();
          setServerChrome('running', 'Settings saved. Restart the server to apply runtime changes.');
        } catch (error) {
          $('settingsPanel').insertAdjacentHTML('afterbegin', `<div class="error-box" style="margin-bottom:12px">${escapeHtml(error.message)}</div>`);
        } finally {
          $('saveSettings').disabled = false;
        }
      });
      $('refreshLogs').addEventListener('click', loadLogs);

      window.addEventListener('hashchange', () => selectSection((location.hash || '#dashboard').slice(1)));
      window.addEventListener('resize', syncViewportHeight);
      window.addEventListener('pywebviewready', detectDesktopBridge);

      syncViewportHeight();
      selectSection((location.hash || '#dashboard').slice(1));
      detectDesktopBridge();
      refreshAll();
      setInterval(() => {
        if (appState.activeSection === 'dashboard') {
          fetchJson('/v1/gateway/live/snapshot').then((payload) => {
            appState.live = payload.data || [];
            renderDashboard();
          }).catch(() => {});
        }
      }, 3000);
    </script>
  </body>
</html>
"""
