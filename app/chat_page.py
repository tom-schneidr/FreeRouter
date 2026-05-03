"""Chat page HTML served at /chat."""

CHAT_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Chat - FreeRouter</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    :root {
      --bg-primary: #0a0e1a; --bg-secondary: #111827; --bg-tertiary: #1e293b;
      --border: #2d3a4f; --text: #e2e8f0; --text-muted: #94a3b8;
      --accent: #3b82f6; --accent-glow: rgba(59,130,246,0.15);
      --green: #22c55e; --red: #ef4444; --amber: #f59e0b; --purple: #a78bfa;
      --radius: 12px; --font: 'Inter', system-ui, sans-serif;
    }
    html, body { height: 100%; }
    body { font-family: var(--font); background: var(--bg-primary); color: var(--text); display: flex; flex-direction: column; }

    /* === NAV === */
    nav { display: flex; align-items: center; gap: 1rem; padding: 0.75rem 1.5rem;
      background: var(--bg-secondary); border-bottom: 1px solid var(--border); flex-shrink: 0; }
    nav h1 { font-size: 1rem; font-weight: 700; background: linear-gradient(135deg, #60a5fa, #a78bfa);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    nav a { color: var(--text-muted); text-decoration: none; font-size: 0.85rem; transition: color 0.2s; }
    nav a:hover { color: var(--text); }
    .nav-spacer { flex: 1; }
    .model-badge { font-size: 0.75rem; padding: 0.25rem 0.6rem; border-radius: 999px;
      background: var(--accent-glow); border: 1px solid var(--accent); color: #93c5fd; }

    /* === LAYOUT === */
    .app { display: flex; flex: 1; overflow: hidden; }
    .chat-panel { flex: 1; display: flex; flex-direction: column; min-width: 0; }
    .route-panel { width: 380px; height: calc(100vh - 49px); flex-shrink: 0; min-height: 0; background: var(--bg-secondary);
      border-left: 1px solid var(--border); display: flex; flex-direction: column; overflow: hidden; }
    .route-panel-header { padding: 1rem 1.25rem; border-bottom: 1px solid var(--border);
      font-weight: 600; font-size: 0.9rem; display: flex; align-items: center; gap: 0.5rem; }
    .route-panel-header .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--green);
      animation: pulse 2s infinite; }
    @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }
    .route-log { flex: 1; min-height: 0; overflow-y: scroll; overscroll-behavior: contain; padding: 0.75rem; display: flex; flex-direction: column; gap: 0.5rem; }

    /* === MESSAGES === */
    .messages { flex: 1; overflow-y: auto; padding: 1.5rem; display: flex; flex-direction: column; gap: 1rem; }
    .msg { max-width: 75%; padding: 0.85rem 1.1rem; border-radius: var(--radius); line-height: 1.6;
      font-size: 0.92rem; word-wrap: break-word; white-space: pre-wrap; animation: fadeIn 0.3s ease; }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }
    .msg.user { align-self: flex-end; background: var(--accent); color: #fff; border-bottom-right-radius: 4px; }
    .msg.assistant { align-self: flex-start; background: var(--bg-tertiary); border: 1px solid var(--border);
      border-bottom-left-radius: 4px; }
    .msg.assistant .content-text { white-space: normal; }
    .msg.assistant .content-text > * + * { margin-top: 0.7rem; }
    .msg.assistant p { margin: 0; }
    .msg.assistant ul, .msg.assistant ol { padding-left: 1.25rem; }
    .msg.assistant li + li { margin-top: 0.25rem; }
    .msg.assistant pre { overflow-x: auto; padding: 0.75rem; border-radius: 8px; background: var(--bg-primary); border: 1px solid var(--border); }
    .msg.assistant code { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 0.85em; background: var(--bg-primary); border: 1px solid var(--border); border-radius: 0.35rem; padding: 0.1rem 0.3rem; }
    .msg.assistant pre code { background: transparent; border: none; padding: 0; }
    .msg.assistant blockquote { border-left: 3px solid var(--accent); padding-left: 0.75rem; color: var(--text-muted); }
    .msg.assistant a { color: #93c5fd; }
    .msg.assistant .md-table-wrap { overflow-x: auto; max-width: 100%; margin: 0.5rem 0; -webkit-overflow-scrolling: touch; }
    .msg.assistant table.md-table { border-collapse: collapse; font-size: 0.9em; width: max-content; max-width: 100%; }
    .msg.assistant table.md-table th,
    .msg.assistant table.md-table td { border: 1px solid var(--border); padding: 0.4rem 0.55rem; vertical-align: top; }
    .msg.assistant table.md-table th { background: rgba(30, 41, 59, 0.65); font-weight: 600; color: var(--text); }
    .msg-meta { font-size: 0.72rem; color: var(--text-muted); margin-top: 0.4rem; }
    .msg.assistant .msg-meta { color: var(--purple); }
    .msg.system-error { align-self: center; background: rgba(239,68,68,0.1); border: 1px solid var(--red);
      color: var(--red); font-size: 0.85rem; max-width: 90%; text-align: center; }
    .typing { align-self: flex-start; padding: 1rem 1.2rem; background: var(--bg-tertiary);
      border: 1px solid var(--border); border-radius: var(--radius); display: none; }
    .typing span { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
      background: var(--text-muted); animation: bounce 1.4s infinite ease-in-out; margin: 0 2px; }
    .typing span:nth-child(2) { animation-delay: 0.16s; }
    .typing span:nth-child(3) { animation-delay: 0.32s; }
    @keyframes bounce { 0%,80%,100% { transform: scale(0.6); } 40% { transform: scale(1); } }

    /* === INPUT === */
    .input-bar { padding: 1rem 1.5rem; border-top: 1px solid var(--border); background: var(--bg-secondary);
      display: flex; gap: 0.75rem; align-items: flex-end; flex-shrink: 0; }
    .input-bar textarea { flex: 1; resize: none; border: 1px solid var(--border); border-radius: var(--radius);
      background: var(--bg-primary); color: var(--text); padding: 0.75rem 1rem; font: inherit;
      font-size: 0.92rem; line-height: 1.5; max-height: 150px; outline: none; transition: border-color 0.2s; }
    .input-bar textarea:focus { border-color: var(--accent); }
    .input-options { display: flex; align-items: center; gap: 0.45rem; color: var(--text-muted);
      font-size: 0.82rem; white-space: nowrap; user-select: none; padding-bottom: 0.65rem; }
    .input-options input { accent-color: var(--accent); }
    .send-btn { width: 44px; height: 44px; border-radius: 50%; border: none; background: var(--accent);
      color: #fff; cursor: pointer; display: flex; align-items: center; justify-content: center;
      transition: transform 0.15s, background 0.2s; flex-shrink: 0; }
    .send-btn:hover { background: #2563eb; transform: scale(1.05); }
    .send-btn:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }
    .send-btn svg { width: 20px; height: 20px; }

    /* === ROUTE EVENT CARDS === */
    .route-group { flex-shrink: 0; background: var(--bg-tertiary); border: 1px solid var(--border); border-radius: 10px;
      overflow: hidden; animation: fadeIn 0.3s ease; }
    .route-group-header { padding: 0.6rem 0.85rem; font-size: 0.78rem; font-weight: 600;
      color: var(--text-muted); border-bottom: 1px solid var(--border);
      display: flex; justify-content: space-between; }
    .route-group-header .req-id { color: var(--purple); font-weight: 500; }
    .route-event { padding: 0.5rem 0.85rem; font-size: 0.78rem; display: flex; align-items: center;
      gap: 0.5rem; border-bottom: 1px solid rgba(45,58,79,0.5); }
    .route-event.flagged { margin: 0.35rem; border: 1px solid rgba(245,158,11,0.45); border-radius: 10px;
      background: linear-gradient(135deg, rgba(245,158,11,0.16), rgba(167,139,250,0.08)); }
    .route-event:last-child { border-bottom: none; }
    .route-event .icon { width: 18px; height: 18px; border-radius: 50%; display: flex;
      align-items: center; justify-content: center; font-size: 0.65rem; flex-shrink: 0; font-weight: 700; }
    .icon.trying { background: rgba(59,130,246,0.2); color: var(--accent); border: 1px solid var(--accent);
      animation: pulse 1s infinite; }
    .icon.ok { background: rgba(34,197,94,0.2); color: var(--green); border: 1px solid var(--green); }
    .icon.fail { background: rgba(239,68,68,0.15); color: var(--red); border: 1px solid var(--red); }
    .icon.skip { background: rgba(245,158,11,0.15); color: var(--amber); border: 1px solid var(--amber); }
    .icon.flagged-skip { background: rgba(245,158,11,0.22); color: #fcd34d; border: 1px solid var(--amber); }
    .icon.flagged { background: rgba(245,158,11,0.25); color: #fcd34d; border: 1px solid var(--amber); }
    .route-event .info { flex: 1; min-width: 0; }
    .route-event .provider-name { font-weight: 600; color: var(--text); }
    .route-event .model-name { color: var(--text-muted); font-size: 0.72rem; display: block;
      overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .route-event .reason { color: var(--text-muted); font-size: 0.72rem; margin-left: auto; flex-shrink: 0; }
    .route-event .duration { color: var(--text-muted); font-size: 0.68rem; margin-left: 0.5rem; }
    .route-disable { border: 1px solid rgba(239,68,68,0.35); border-radius: 999px; background: rgba(239,68,68,0.1);
      color: #fecaca; padding: 0.2rem 0.45rem; font: inherit; font-size: 0.68rem; cursor: pointer; flex-shrink: 0; }
    .route-disable:hover { background: rgba(239,68,68,0.18); }
    .route-disable.enable { border-color: rgba(34,197,94,0.35); background: rgba(34,197,94,0.12); color: #bbf7d0; }
    .route-disable.enable:hover { background: rgba(34,197,94,0.2); }

    .empty-state { text-align: center; color: var(--text-muted); padding: 3rem 1rem; font-size: 0.85rem; }
    .empty-state .icon-lg { font-size: 2.5rem; margin-bottom: 0.5rem; display: block; }

    /* scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

    @media (max-width: 800px) {
      .route-panel { display: none; }
    }
  </style>
</head>
<body>
  <nav>
    <h1>FreeRouter</h1>
    <span class="model-badge" id="active-model">Ready</span>
    <span class="nav-spacer"></span>
    <a href="/">Home</a>
    <a href="/chat">Chat</a>
    <a href="/models">Models</a>
    <a href="/health">Health</a>
    <a href="/status">Provider Usage</a>
    <a href="/live">Live Traffic</a>
  </nav>

  <div class="app">
    <div class="chat-panel">
      <div class="messages" id="messages">
        <div class="empty-state">
          <span class="icon-lg">💬</span>
          Send a message to start chatting.<br>
          The gateway will route to the best available model.
        </div>
      </div>
      <div class="typing" id="typing"><span></span><span></span><span></span></div>
      <div class="input-bar">
        <textarea id="input" rows="1" placeholder="Type a message..." autofocus></textarea>
        <label class="input-options" title="Route this message through /v1/chat/completions/web-search">
          <input type="checkbox" id="web-search-toggle">
          Web search
        </label>
        <button class="send-btn" id="send-btn" title="Send">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
        </button>
      </div>
    </div>

    <div class="route-panel">
      <div class="route-panel-header"><span class="dot"></span> Routing Activity</div>
      <div class="route-log" id="route-log">
        <div class="empty-state"><span class="icon-lg">📡</span>Routing events will appear here in real time.</div>
      </div>
    </div>
  </div>

<script>
const messagesEl = document.getElementById('messages');
const inputEl = document.getElementById('input');
const sendBtn = document.getElementById('send-btn');
const typingEl = document.getElementById('typing');
const routeLog = document.getElementById('route-log');
const activeBadge = document.getElementById('active-model');
const webSearchToggle = document.getElementById('web-search-toggle');

let chatHistory = [];
let msgCounter = 0;
let isFirstMsg = true;

function esc(s) {
  const d = document.createElement('div'); d.textContent = s; return d.innerHTML;
}

function renderInlineMarkdown(text) {
  return esc(text)
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/\*([^*]+)\*/g, '<em>$1</em>')
    .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>');
}

function parseTableRow(line) {
  const t = line.trim();
  if (!t.includes('|')) return null;
  const parts = t.split('|');
  if (parts.length < 3) return null;
  return parts.slice(1, -1).map((c) => c.trim());
}

function isSeparatorRow(cells) {
  return cells.length > 0 && cells.every((c) => /^:?-{3,}:?$/.test(c.trim()));
}

function cellAlign(cell) {
  const c = cell.trim();
  if (/^:-+$/.test(c)) return 'left';
  if (/^-+:$/.test(c)) return 'right';
  if (/^:-+:$/.test(c)) return 'center';
  return 'left';
}

function renderTableHtml(header, align, bodyRows) {
  const ths = header.map((h, idx) => {
    const a = align[idx] || 'left';
    return `<th style="text-align:${a}">${renderInlineMarkdown(h)}</th>`;
  });
  const trs = bodyRows.map((row) =>
    `<tr>${row.map((cell, idx) => {
      const a = align[idx] || 'left';
      return `<td style="text-align:${a}">${renderInlineMarkdown(cell)}</td>`;
    }).join('')}</tr>`
  );
  return `<div class="md-table-wrap"><table class="md-table"><thead><tr>${ths.join('')}</tr></thead><tbody>${trs.join('')}</tbody></table></div>`;
}

function detectGFMTable(lines, start) {
  if (start + 1 >= lines.length) return null;
  const row0 = lines[start].trim();
  const row1 = lines[start + 1].trim();
  if (!row0 || !row1) return null;
  const header = parseTableRow(row0);
  const sepCells = parseTableRow(row1);
  if (!header || !sepCells || header.length !== sepCells.length) return null;
  if (!isSeparatorRow(sepCells)) return null;
  const align = sepCells.map(cellAlign);
  const body = [];
  let j = start + 2;
  while (j < lines.length) {
    const tr = lines[j].trim();
    if (!tr) break;
    const row = parseTableRow(tr);
    if (!row || row.length !== header.length) break;
    body.push(row);
    j++;
  }
  return { html: renderTableHtml(header, align, body), nextIndex: j };
}

function renderMarkdown(markdown) {
  const codeBlocks = [];
  const protectedText = String(markdown || '').replace(/```(\w+)?\n?([\s\S]*?)```/g, (_, lang, code) => {
    const token = `\u0000CODE${codeBlocks.length}\u0000`;
    codeBlocks.push(`<pre><code>${esc(code.replace(/\n$/, ''))}</code></pre>`);
    return token;
  });
  const lines = protectedText.split('\n');
  const blocks = [];
  let paragraph = [];
  let listItems = [];
  let orderedItems = [];

  function flushParagraph() {
    if (!paragraph.length) return;
    blocks.push(`<p>${paragraph.map(renderInlineMarkdown).join('<br>')}</p>`);
    paragraph = [];
  }
  function flushLists() {
    if (listItems.length) {
      blocks.push(`<ul>${listItems.map((item) => `<li>${renderInlineMarkdown(item)}</li>`).join('')}</ul>`);
      listItems = [];
    }
    if (orderedItems.length) {
      blocks.push(`<ol>${orderedItems.map((item) => `<li>${renderInlineMarkdown(item)}</li>`).join('')}</ol>`);
      orderedItems = [];
    }
  }
  function flushAll() {
    flushParagraph();
    flushLists();
  }

  let i = 0;
  while (i < lines.length) {
    const rawLine = lines[i];
    const line = rawLine.trimEnd();
    const trimmed = line.trim();
    const codeMatch = trimmed.match(/^\u0000CODE(\d+)\u0000$/);
    if (codeMatch) {
      flushAll();
      blocks.push(codeBlocks[Number(codeMatch[1])]);
      i++;
      continue;
    }
    if (!trimmed) {
      flushAll();
      i++;
      continue;
    }

    const tbl = detectGFMTable(lines, i);
    if (tbl) {
      flushAll();
      blocks.push(tbl.html);
      i = tbl.nextIndex;
      continue;
    }

    const heading = trimmed.match(/^(#{1,6})\s+(.+)$/);
    if (heading) {
      flushAll();
      const level = Math.min(6, heading[1].length);
      blocks.push(`<h${level}>${renderInlineMarkdown(heading[2])}</h${level}>`);
      i++;
      continue;
    }

    const bullet = trimmed.match(/^[-*]\s+(.+)$/);
    if (bullet) {
      flushParagraph();
      orderedItems = [];
      listItems.push(bullet[1]);
      i++;
      continue;
    }
    const ordered = trimmed.match(/^\d+\.\s+(.+)$/);
    if (ordered) {
      flushParagraph();
      listItems = [];
      orderedItems.push(ordered[1]);
      i++;
      continue;
    }
    flushLists();
    paragraph.push(trimmed);
    i++;
  }

  flushAll();
  return blocks.join('');
}

function extractAssistantText(body) {
  if (!body) return '';
  if (typeof body.content === 'string') return body.content;
  if (typeof body.text === 'string') return body.text;
  if (typeof body.message?.content === 'string') return body.message.content;
  const message = body.choices?.[0]?.message?.content;
  if (typeof message === 'string') return message;
  if (Array.isArray(message)) {
    return message
      .map((item) => typeof item === 'string' ? item : (item?.text || item?.content || ''))
      .filter(Boolean)
      .join('\n');
  }
  const choiceText = body.choices?.[0]?.text;
  if (typeof choiceText === 'string') return choiceText;
  return '';
}

function autoResize() {
  inputEl.style.height = 'auto';
  inputEl.style.height = Math.min(inputEl.scrollHeight, 150) + 'px';
}
inputEl.addEventListener('input', autoResize);

function addMessage(role, content, meta) {
  if (isFirstMsg) { messagesEl.innerHTML = ''; isFirstMsg = false; }
  const div = document.createElement('div');
  div.className = `msg ${role}`;
  const span = document.createElement('span');
  span.className = 'content-text';
  if (role === 'assistant') {
    div.dataset.markdown = content || '';
    span.innerHTML = renderMarkdown(content || '');
  } else {
    span.textContent = content;
  }
  div.appendChild(span);
  if (meta) {
    const m = document.createElement('div');
    m.className = 'msg-meta'; m.textContent = meta;
    div.appendChild(m);
  }
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return div;
}

function addErrorMessage(text) {
  if (isFirstMsg) { messagesEl.innerHTML = ''; isFirstMsg = false; }
  const div = document.createElement('div');
  div.className = 'msg system-error';
  div.textContent = text;
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

// --- Route panel ---
let currentGroup = null;
let reqNum = 0;

function startRouteGroup() {
  reqNum++;
  // Clear empty state on first use
  if (reqNum === 1) routeLog.innerHTML = '';
  const group = document.createElement('div');
  group.className = 'route-group';
  group.innerHTML = `<div class="route-group-header"><span>Request #${reqNum}</span><span class="req-id" id="rg-status-${reqNum}">routing...</span></div>`;
  routeLog.prepend(group);
  currentGroup = group;
  return group;
}

function addRouteEvent(group, status, providerName, modelId, reason, durationMs, routeId) {
  let iconClass, iconText;
  switch (status) {
    case 'trying':  iconClass = 'trying'; iconText = '⟳'; break;
    case 'selected':iconClass = 'ok';     iconText = '✓'; break;
    case 'skipped': iconClass = isFlaggedSkip(reason) ? 'flagged-skip' : 'skip'; iconText = '⏭'; break;
    case 'flagged': iconClass = 'flagged'; iconText = '!'; break;
    default:        iconClass = 'fail';   iconText = '✗'; break;
  }
  const showDisable = routeId && status !== 'flagged';
  const ev = document.createElement('div');
  ev.className = `route-event ${status === 'flagged' ? 'flagged' : ''}`;
  ev.innerHTML = `
    <div class="icon ${iconClass}">${iconText}</div>
    <div class="info">
      <span class="provider-name">${esc(providerName)}</span>
      <span class="model-name">${esc(modelId || '')}</span>
    </div>
    ${reason ? `<span class="reason">${esc(reason)}</span>` : ''}
    ${durationMs != null ? `<span class="duration">${durationMs}ms</span>` : ''}
    ${showDisable ? `<button class="route-disable" data-route-id="${esc(routeId)}">Disable</button>` : ''}
  `;
  const toggleBtn = ev.querySelector('.route-disable');
  if (toggleBtn) toggleBtn.addEventListener('click', () => toggleRouteEnabled(toggleBtn.dataset.routeId, toggleBtn));
  group.appendChild(ev);
  routeLog.scrollTop = 0;
  return ev;
}

function isFlaggedSkip(reason) {
  return ['potentially_outdated', 'route_rate_limited', 'route_too_slow'].includes(reason);
}

async function toggleRouteEnabled(routeId, button) {
  if (!routeId) return;
  const currentEnabled = button.dataset.enabled !== 'false';
  const nextEnabled = !currentEnabled;
  if (!nextEnabled && !confirm('Disable this model route?')) return;
  button.disabled = true;
  button.textContent = nextEnabled ? 'Enabling...' : 'Disabling...';
  const action = nextEnabled ? 'enable' : 'disable';
  const response = await fetch(`/v1/gateway/models/${encodeURIComponent(routeId)}/${action}`, { method: 'POST' });
  button.disabled = false;
  if (!response.ok) {
    button.textContent = 'Failed';
    setTimeout(() => {
      button.textContent = currentEnabled ? 'Disable' : 'Enable';
      button.classList.toggle('enable', !currentEnabled);
    }, 1500);
    return;
  }
  button.dataset.enabled = String(nextEnabled);
  button.textContent = nextEnabled ? 'Disable' : 'Enable';
  button.classList.toggle('enable', !nextEnabled);
}

function updateTryingEvent(ev, status, reason, durationMs) {
  const icon = ev.querySelector('.icon');
  icon.className = 'icon ' + (status === 'selected' ? 'ok' : status === 'skipped' ? (isFlaggedSkip(reason) ? 'flagged-skip' : 'skip') : 'fail');
  icon.textContent = status === 'selected' ? '✓' : status === 'skipped' ? '⏭' : '✗';
  if (reason) {
    let span = ev.querySelector('.reason');
    if (!span) { span = document.createElement('span'); span.className = 'reason'; ev.appendChild(span); }
    span.textContent = reason;
  }
  if (durationMs != null) {
    let span = ev.querySelector('.duration');
    if (!span) { span = document.createElement('span'); span.className = 'duration'; ev.appendChild(span); }
    span.textContent = durationMs + 'ms';
  }
}

// --- Send logic via SSE ---
async function sendMessage() {
  const text = inputEl.value.trim();
  if (!text) return;
  inputEl.value = '';
  autoResize();
  sendBtn.disabled = true;
  addMessage('user', text);
  chatHistory.push({ role: 'user', content: text });

  typingEl.style.display = 'block';
  messagesEl.scrollTop = messagesEl.scrollHeight;

  const group = startRouteGroup();
  const statusEl = document.getElementById(`rg-status-${reqNum}`);
  const tryingEvents = {};
  let assistantDiv = null;
  let finalProvider = '';
  let finalModel = '';
  const t0 = performance.now();

  try {
    if (webSearchToggle.checked) {
      await sendWebSearchMessage(group, statusEl, t0);
      return;
    }

    const resp = await fetch('/v1/chat/completions/stream-route', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: chatHistory, max_tokens: 4096 })
    });

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const SEP = String.fromCharCode(10) + String.fromCharCode(10);
      while ((idx = buffer.indexOf(SEP)) !== -1) {
        const chunk = buffer.slice(0, idx).trim();
        buffer = buffer.slice(idx + 2);
        if (!chunk.startsWith('data: ')) continue;
        const raw = chunk.slice(6);
        if (raw === '[DONE]') continue;

        let evt;
        try { evt = JSON.parse(raw); } catch { continue; }

        if (evt.type === 'route_trying') {
          const ev = addRouteEvent(group, 'trying', evt.provider, evt.model_id, null, null, evt.route_id);
          tryingEvents[evt.provider + '/' + evt.model_id] = { el: ev, t: performance.now() };
        } else if (evt.type === 'route_skip' || evt.type === 'route_fail') {
          const key = evt.provider + '/' + evt.model_id;
          const te = tryingEvents[key];
          const dur = te ? Math.round(performance.now() - te.t) : null;
          if (te) { updateTryingEvent(te.el, evt.type === 'route_skip' ? 'skipped' : 'failed', evt.reason, dur); }
          else { addRouteEvent(group, evt.type === 'route_skip' ? 'skipped' : 'failed', evt.provider, evt.model_id, evt.reason, dur, evt.route_id); }
        } else if (evt.type === 'route_flagged') {
          addRouteEvent(group, 'flagged', evt.provider, evt.model_id, `Automatically flagged: ${evt.reason}`, null, evt.route_id);
        } else if (evt.type === 'route_selected') {
          const key = evt.provider + '/' + evt.model_id;
          const te = tryingEvents[key];
          const dur = te ? Math.round(performance.now() - te.t) : null;
          if (te) updateTryingEvent(te.el, 'selected', null, dur);
          else addRouteEvent(group, 'selected', evt.provider, evt.model_id, null, dur, evt.route_id);
          finalProvider = evt.provider;
          finalModel = evt.model_id;
          activeBadge.textContent = evt.provider + ' / ' + evt.model_id;
        } else if (evt.type === 'content') {
          typingEl.style.display = 'none';
          if (!assistantDiv) assistantDiv = addMessage('assistant', '');
          const ct = assistantDiv.querySelector('.content-text');
          assistantDiv.dataset.markdown = (assistantDiv.dataset.markdown || '') + evt.text;
          if (ct) ct.innerHTML = renderMarkdown(assistantDiv.dataset.markdown || '');
          messagesEl.scrollTop = messagesEl.scrollHeight;
        } else if (evt.type === 'done') {
          const totalMs = Math.round(performance.now() - t0);
          if (assistantDiv) {
            const metaDiv = assistantDiv.querySelector('.msg-meta') || document.createElement('div');
            metaDiv.className = 'msg-meta';
            metaDiv.textContent = `${finalProvider} · ${finalModel} · ${totalMs}ms`;
            if (!assistantDiv.querySelector('.msg-meta')) assistantDiv.appendChild(metaDiv);
          }
          if (evt.content) chatHistory.push({ role: 'assistant', content: evt.content });
          statusEl.textContent = `✓ ${totalMs}ms`;
          statusEl.style.color = '#22c55e';
        } else if (evt.type === 'error') {
          typingEl.style.display = 'none';
          addErrorMessage(evt.message || 'All providers exhausted.');
          statusEl.textContent = '✗ failed';
          statusEl.style.color = '#ef4444';
        }
      }
    }
  } catch (err) {
    typingEl.style.display = 'none';
    addErrorMessage('Network error: ' + err.message);
    statusEl.textContent = '✗ error';
    statusEl.style.color = '#ef4444';
  } finally {
    typingEl.style.display = 'none';
    sendBtn.disabled = false;
    inputEl.focus();
  }
}

async function sendWebSearchMessage(group, statusEl, t0) {
  statusEl.textContent = 'searching web...';
  const resp = await fetch('/v1/chat/completions/web-search', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ messages: chatHistory, max_tokens: 4096 })
  });
  const totalMs = Math.round(performance.now() - t0);
  const provider = resp.headers.get('X-Gateway-Provider') || '';
  const model = resp.headers.get('X-Gateway-Model') || '';
  const route = resp.headers.get('X-Gateway-Route') || '';
  let payload = {};
  try { payload = await resp.json(); } catch {}

  if (!resp.ok) {
    const message = payload?.error?.message || `Web search failed with HTTP ${resp.status}`;
    addErrorMessage(message);
    statusEl.textContent = '✗ failed';
    statusEl.style.color = '#ef4444';
    return;
  }

  addRouteEvent(group, 'selected', provider || 'web-search', model || 'web-search model', null, totalMs, route);
  activeBadge.textContent = provider && model ? `${provider} / ${model}` : 'Web search';

  const content = extractAssistantText(payload);
  if (content) {
    addMessage('assistant', content, `${provider} · ${model} · web search · ${totalMs}ms`);
    chatHistory.push({ role: 'assistant', content });
  } else {
    addErrorMessage('Web search returned successfully, but the provider sent an empty final assistant message.');
  }
  statusEl.textContent = `✓ ${totalMs}ms`;
  statusEl.style.color = '#22c55e';
}

sendBtn.addEventListener('click', sendMessage);
inputEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});
</script>
</body>
</html>
"""
