/**
 * Shared markdown renderer for legacy chat/live pages and the React UI.
 */

function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>"']/g, (char) => {
      const map = {
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;",
      };
      return map[char] ?? char;
    });
  }

  function protectEscapes(text) {
    const escapes = [];
    const protectedText = text.replace(/\\([\\`*_~[\]()#+.!-])/g, (_, ch) => {
      const index = escapes.length;
      escapes.push(ch);
      return `\u0000ESC${index}\u0000`;
    });
    return { protectedText, escapes };
  }

  function restoreEscapes(text, escapes) {
    return text.replace(/\u0000ESC(\d+)\u0000/g, (_, index) => escapes[Number(index)] ?? "");
  }

  function safeUrl(url) {
    const trimmed = String(url || "").trim();
    if (/^https?:\/\//i.test(trimmed) || /^mailto:/i.test(trimmed)) return trimmed;
    return "";
  }

  const PAIRED_INLINE_HTML_TAGS = "span|strong|em|b|i|u|sub|sup|mark|small|del|ins|abbr|kbd";
  const ALLOWED_HTML_ATTRS = new Set(["class", "style", "title", "id"]);
  const SAFE_STYLE_PROPS = new Set([
    "color",
    "background-color",
    "font-weight",
    "font-style",
    "text-decoration",
    "text-align",
  ]);

  function sanitizeHtmlStyle(style) {
    return String(style || "")
      .split(";")
      .map((part) => part.trim())
      .filter(Boolean)
      .map((part) => {
        const colon = part.indexOf(":");
        if (colon === -1) return "";
        const name = part.slice(0, colon).trim().toLowerCase();
        const value = part.slice(colon + 1).trim();
        if (!name || !value || !SAFE_STYLE_PROPS.has(name)) return "";
        if (/url\s*\(|expression\s*\(|javascript:/i.test(value)) return "";
        return `${name}: ${value}`;
      })
      .filter(Boolean)
      .join("; ");
  }

  function sanitizeHtmlAttrs(attrsRaw) {
    const attrs = [];
    const attrRe = /([a-zA-Z_:][-a-zA-Z0-9_:.]*)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))/g;
    let match;
    while ((match = attrRe.exec(String(attrsRaw || "")))) {
      const name = match[1].toLowerCase();
      if (!ALLOWED_HTML_ATTRS.has(name)) continue;
      let value = match[2] ?? match[3] ?? match[4] ?? "";
      if (name === "style") value = sanitizeHtmlStyle(value);
      if (name === "id" && !/^[a-zA-Z][\w:-]*$/.test(value)) continue;
      if (name === "class" && !/^[\w\s-]+$/.test(value)) continue;
      if (!value && name !== "class") continue;
      attrs.push(`${name}="${escapeHtml(value)}"`);
    }
    return attrs.length ? ` ${attrs.join(" ")}` : "";
  }

  function sanitizeHtmlInner(inner, stash) {
    const processed = stashSafeInlineHtml(inner, stash);
    return processed
      .split(/(\u0000TK\d+\u0000)/)
      .map((part) => (/^\u0000TK\d+\u0000$/.test(part) ? part : escapeHtml(part)))
      .join("");
  }

  function stashSafeInlineHtml(text, stash) {
    let out = String(text ?? "");
    let changed = true;
    while (changed) {
      changed = false;
      const paired = new RegExp(`<(${PAIRED_INLINE_HTML_TAGS})\\b([^>]*)>([\\s\\S]*?)<\\/\\1>`, "gi");
      out = out.replace(paired, (full, tag, attrs, inner) => {
        changed = true;
        const tagName = tag.toLowerCase();
        return stash(
          `<${tagName}${sanitizeHtmlAttrs(attrs)}>${sanitizeHtmlInner(inner, stash)}</${tagName}>`,
        );
      });
    }
    out = out.replace(/<(br)\b([^>]*)\s*\/?>/gi, (full, tag, attrs) => {
      return stash(`<${tag.toLowerCase()}${sanitizeHtmlAttrs(attrs)} />`);
    });
    return out;
  }

  function stashFootnoteRef(id, referenced, stash) {
    referenced.add(id);
    return stash(
      `<sup class="footnote-ref"><a href="#fn-${escapeHtml(id)}" id="fnref-${escapeHtml(id)}">${escapeHtml(id)}</a></sup>`,
    );
  }

  function renderFootnotesSection(footnotes, referenced) {
    const ids = [...referenced].filter((id) => footnotes.has(id));
    if (!ids.length) return "";
    const items = ids
      .map((id) => {
        const body = renderInlineMarkdown(footnotes.get(id) || "", { footnotes, referenced, skipFootnoteRefs: true });
        return `<li id="fn-${escapeHtml(id)}">${body} <a href="#fnref-${escapeHtml(id)}" class="footnote-backref" aria-label="Back to reference">↩</a></li>`;
      })
      .join("");
    return `<section class="footnotes"><ol>${items}</ol></section>`;
  }

  function renderInlineMarkdown(text, ctx) {
    const footnotes = ctx?.footnotes || new Map();
    const referenced = ctx?.referenced || new Set();
    const skipFootnoteRefs = Boolean(ctx?.skipFootnoteRefs);
    const tokens = [];

    const { protectedText, escapes } = protectEscapes(String(text ?? ""));
    let raw = protectedText;

    function stash(html) {
      const token = `\u0000TK${tokens.length}\u0000`;
      tokens.push(html);
      return token;
    }

    raw = stashSafeInlineHtml(raw, stash);

    if (!skipFootnoteRefs) {
      raw = raw.replace(/\[\^([^\]]+)\]/g, (_, id) => stashFootnoteRef(id, referenced, stash));
      raw = raw.replace(/\[(\d+)\](?!\()/g, (_, id) => stashFootnoteRef(id, referenced, stash));
    }

    raw = raw.replace(/`([^`\n]+)`/g, (_, code) => stash(`<code>${escapeHtml(code)}</code>`));
    raw = raw.replace(/!\[([^\]]*)\]\(([^\s")]+)(?:\s+"([^"]*)")?\)/g, (_, alt, url, title) => {
      const safe = safeUrl(url);
      if (!safe) return `![${alt}](${url})`;
      const titleAttr = title ? ` title="${escapeHtml(title)}"` : "";
      return stash(`<img src="${safe}" alt="${escapeHtml(alt)}"${titleAttr} loading="lazy">`);
    });
    raw = raw.replace(/\[([^\]]+)\]\(([^\s")]+)(?:\s+"([^"]*)")?\)/g, (_, label, url, title) => {
      const safe = safeUrl(url);
      if (!safe) return `[${label}](${url})`;
      const titleAttr = title ? ` title="${escapeHtml(title)}"` : "";
      return stash(
        `<a href="${safe}" target="_blank" rel="noreferrer"${titleAttr}>${escapeHtml(label)}</a>`,
      );
    });
    raw = raw.replace(/<((?:https?:\/\/|mailto:)[^>\s]+)>/gi, (_, url) => {
      const safe = safeUrl(url);
      return safe
        ? stash(`<a href="${safe}" target="_blank" rel="noreferrer">${escapeHtml(url)}</a>`)
        : url;
    });
    raw = raw.replace(/<([^>\s]+@[^>\s]+)>/g, (_, email) => {
      const safe = safeUrl(`mailto:${email}`);
      return safe
        ? stash(`<a href="${safe}">${escapeHtml(email)}</a>`)
        : email;
    });

    let html = escapeHtml(raw);
    html = html.replace(/\*\*\*([^*]+)\*\*\*/g, "<strong><em>$1</em></strong>");
    html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    html = html.replace(/__([^_]+)__/g, "<strong>$1</strong>");
    html = html.replace(/~~([^~]+)~~/g, "<del>$1</del>");
    html = html.replace(/(?<!\*)\*([^*\n]+)\*(?!\*)/g, "<em>$1</em>");
    html = html.replace(/(?<![_\w])_([^_\n]+?)_(?![_\w])/g, "<em>$1</em>");
    html = restoreEscapes(html, escapes);
    html = html.replace(/\u0000TK(\d+)\u0000/g, (_, index) => tokens[Number(index)] ?? "");
    return html;
  }

  function parseTableRow(line) {
    const t = line.trim();
    if (!t.includes("|")) return null;
    const parts = t.split("|");
    if (parts.length < 3) return null;
    return parts.slice(1, -1).map((c) => c.trim());
  }

  function isSeparatorRow(cells) {
    return cells.length > 0 && cells.every((c) => /^:?-{3,}:?$/.test(c.trim()));
  }

  function cellAlign(cell) {
    const c = cell.trim();
    if (/^:-+$/.test(c)) return "left";
    if (/^-+:$/.test(c)) return "right";
    if (/^:-+:$/.test(c)) return "center";
    return "left";
  }

  function renderTableHtml(header, align, bodyRows, ctx) {
    const ths = header.map((h, idx) => {
      const a = align[idx] || "left";
      return `<th style="text-align:${a}">${renderInlineMarkdown(h, ctx)}</th>`;
    });
    const trs = bodyRows.map(
      (row) =>
        `<tr>${row
          .map((cell, idx) => {
            const a = align[idx] || "left";
            return `<td style="text-align:${a}">${renderInlineMarkdown(cell, ctx)}</td>`;
          })
          .join("")}</tr>`,
    );
    return `<div class="md-table-wrap"><table class="md-table"><thead><tr>${ths.join("")}</tr></thead><tbody>${trs.join("")}</tbody></table></div>`;
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
    return { header, align, body, nextIndex: j };
  }

  function detectTabTable(lines, start) {
    const row0 = lines[start];
    if (!row0 || !row0.includes("\t")) return null;
    const header = row0.split("\t").map((c) => c.trim());
    if (header.length < 2) return null;
    const body = [];
    let j = start + 1;
    while (j < lines.length) {
      const row = lines[j];
      if (!row || !row.includes("\t")) break;
      const cells = row.split("\t").map((c) => c.trim());
      if (cells.length !== header.length) break;
      if (cells.every((cell) => /^:?-{3,}:?$/.test(cell))) {
        j++;
        continue;
      }
      body.push(cells);
      j++;
    }
    if (!body.length) return null;
    return { header, align: header.map(() => "left"), body, nextIndex: j };
  }

  function extractFootnoteDefinitions(markdown) {
    const lines = String(markdown || "").split("\n");
    const footnotes = new Map();
    const body = [];
    let i = 0;
    while (i < lines.length) {
      const match =
        lines[i].match(/^\[\^([^\]]+)\]:\s*(.*)$/) || lines[i].match(/^\[(\d+)\]:\s*(.*)$/);
      if (match) {
        let content = match[2];
        i++;
        while (i < lines.length && /^ {4}|\t/.test(lines[i])) {
          content += `\n${lines[i].trim()}`;
          i++;
        }
        footnotes.set(match[1], content);
        continue;
      }
      body.push(lines[i]);
      i++;
    }
    return { text: body.join("\n"), footnotes };
  }

  function parseListLine(line) {
    return line.match(/^(\s*)([-*+]|\d+\.)\s+(?:\[([ xX])\]\s+)?(.*)$/);
  }

  function parseStandaloneTaskLine(line) {
    return line.match(/^(\s*)\[([ xX])\]\s+(.+)$/);
  }

  function parseTaskBlock(lines, start) {
    const first = parseStandaloneTaskLine(lines[start]);
    if (!first) return null;
    const items = [];
    let i = start;
    while (i < lines.length) {
      const line = lines[i];
      if (!line.trim()) break;
      const match = parseStandaloneTaskLine(line);
      if (!match) break;
      items.push({
        indent: match[1].length,
        marker: "-",
        task: match[2],
        text: match[3],
      });
      i++;
    }
    if (!items.length) return null;
    return { items, nextIndex: i, ordered: false };
  }

  function parseIndentedCodeBlock(lines, start) {
    if (!/^ {4}/.test(lines[start] || "")) return null;
    const body = [];
    let i = start;
    while (i < lines.length) {
      const line = lines[i];
      if (!line.trim()) break;
      if (!/^ {4}/.test(line)) break;
      body.push(line.slice(4));
      i++;
    }
    if (!body.length) return null;
    return { code: body.join("\n"), nextIndex: i };
  }

  function detectSpacedTable(lines, start) {
    const row0 = lines[start];
    if (!row0 || row0.includes("|") || row0.includes("\t")) return null;
    const header = row0.trim().split(/\s{2,}/).map((c) => c.trim()).filter(Boolean);
    if (header.length < 2) return null;
    const body = [];
    let j = start + 1;
    while (j < lines.length) {
      const row = lines[j];
      if (!row || !row.trim()) break;
      if (row.includes("|") || row.includes("\t")) break;
      const cells = row.trim().split(/\s{2,}/).map((c) => c.trim()).filter(Boolean);
      if (cells.length !== header.length) break;
      if (cells.every((cell) => /^:?-{3,}:?$/.test(cell))) {
        j++;
        continue;
      }
      body.push(cells);
      j++;
    }
    if (!body.length) return null;
    return { header, align: header.map(() => "left"), body, nextIndex: j };
  }

  function findNestedListEnd(items, start, parentIndent) {
    let i = start;
    while (i < items.length && items[i].indent > parentIndent) i++;
    return i;
  }

  function renderListTree(items, start, end, baseIndent, ordered, ctx) {
    const tag = ordered ? "ol" : "ul";
    let html = `<${tag}>`;
    let i = start;
    while (i < end) {
      const item = items[i];
      if (item.indent < baseIndent) break;
      if (item.indent > baseIndent) {
        i++;
        continue;
      }
      let li = "<li";
      if (item.task != null) {
        const checked = String(item.task).toLowerCase() === "x";
        li += ` class="task-item${checked ? " is-checked" : ""}"`;
      }
      li += ">";
      if (item.task != null) {
        const checked = String(item.task).toLowerCase() === "x";
        li += `<span class="task-checkbox" aria-hidden="true">${checked ? "☑" : "☐"}</span> `;
      }
      li += renderInlineMarkdown(item.text, ctx);
      i++;
      if (i < end && items[i].indent > item.indent) {
        const childOrdered = /^\d+\.$/.test(items[i].marker);
        const childEnd = findNestedListEnd(items, i, item.indent);
        li += renderListTree(items, i, childEnd, items[i].indent, childOrdered, ctx);
        i = childEnd;
      }
      li += "</li>";
      html += li;
    }
    html += `</${tag}>`;
    return html;
  }

  function parseListBlock(lines, start) {
    const first = parseListLine(lines[start]);
    if (!first) return null;
    const ordered = /^\d+\.$/.test(first[2]);
    const items = [];
    let i = start;
    while (i < lines.length) {
      const line = lines[i];
      if (!line.trim()) break;
      const match = parseListLine(line);
      if (!match) break;
      if (/^\d+\.$/.test(match[2]) !== ordered) break;
      items.push({
        indent: match[1].length,
        marker: match[2],
        task: match[3] ?? null,
        text: match[4],
      });
      i++;
    }
    return { items, nextIndex: i, ordered };
  }

  function parseBlockquote(lines, start) {
    const entries = [];
    let i = start;
    while (i < lines.length) {
      const raw = lines[i];
      const trimmed = raw.trim();
      if (!trimmed) {
        if (i + 1 < lines.length && /^\s*>/.test(lines[i + 1])) {
          i++;
          continue;
        }
        break;
      }
      const match = raw.match(/^(\s*)((?:>\s*)+)(.*)$/);
      if (!match) break;
      const depth = (match[2].match(/>/g) || []).length;
      entries.push({ depth, text: match[3] });
      i++;
    }
    if (!entries.length) return null;
    return { entries, nextIndex: i };
  }

  function renderBlockquoteEntries(entries, ctx) {
    function renderLevel(start, depth) {
      let html = "<blockquote>";
      let i = start;
      while (i < entries.length && entries[i].depth >= depth) {
        if (entries[i].depth === depth) {
          if (entries[i].text.trim()) {
            html += `<p>${renderInlineMarkdown(entries[i].text, ctx)}</p>`;
          }
          i++;
        } else if (entries[i].depth > depth) {
          const nested = renderLevel(i, depth + 1);
          html += nested.html;
          i = nested.index;
        } else {
          break;
        }
      }
      html += "</blockquote>";
      return { html, index: i };
    }
    return renderLevel(0, entries[0].depth).html;
  }

  function parseDefinitionList(lines, start) {
    const term = lines[start]?.trim();
    if (!term || term.startsWith(":")) return null;
    let i = start + 1;
    const defs = [];
    while (i < lines.length && lines[i].trim().startsWith(":")) {
      defs.push(lines[i].trim().slice(1).trim());
      i++;
    }
    if (!defs.length) return null;
    return { term, defs, nextIndex: i };
  }

  function renderDefinitionList(term, defs, ctx) {
    const dd = defs.map((def) => `<dd>${renderInlineMarkdown(def, ctx)}</dd>`).join("");
    return `<dl><dt>${renderInlineMarkdown(term, ctx)}</dt>${dd}</dl>`;
  }

  function renderMarkdown(markdown) {
    const normalized = String(markdown ?? "")
      .replace(/\r\n/g, "\n")
      .replace(/\r/g, "\n");
    const extracted = extractFootnoteDefinitions(normalized);
    const footnotes = extracted.footnotes;
    const referenced = new Set();
    const ctx = { footnotes, referenced };

    const codeBlocks = [];
    const protectedText = String(extracted.text || "").replace(/```(\w+)?\n?([\s\S]*?)```/g, (_, lang, code) => {
      const token = `\u0000CODE${codeBlocks.length}\u0000`;
      const langClass = lang ? ` class="language-${escapeHtml(lang)}"` : "";
      codeBlocks.push(`<pre><code${langClass}>${escapeHtml(code.replace(/\n$/, ""))}</code></pre>`);
      return token;
    });

    const lines = protectedText.split("\n");
    const blocks = [];
    let paragraph = [];

    function flushParagraph() {
      if (!paragraph.length) return;
      const parts = [];
      for (let j = 0; j < paragraph.length; j++) {
        parts.push(renderInlineMarkdown(paragraph[j].text, ctx));
        if (j < paragraph.length - 1) {
          parts.push(paragraph[j].hardBreak ? "<br>" : " ");
        }
      }
      blocks.push(`<p>${parts.join("")}</p>`);
      paragraph = [];
    }

    let i = 0;
    while (i < lines.length) {
      const rawLine = lines[i];
      const line = rawLine.trimEnd();
      const trimmed = line.trim();
      const codeMatch = trimmed.match(/^\u0000CODE(\d+)\u0000$/);
      if (codeMatch) {
        flushParagraph();
        blocks.push(codeBlocks[Number(codeMatch[1])]);
        i++;
        continue;
      }
      if (!trimmed) {
        flushParagraph();
        i++;
        continue;
      }
      if (/^=+$/.test(trimmed) && paragraph.length === 1) {
        const text = paragraph[0].text;
        paragraph = [];
        blocks.push(`<h1>${renderInlineMarkdown(text, ctx)}</h1>`);
        i++;
        continue;
      }
      if (/^-+$/.test(trimmed) && paragraph.length === 1) {
        const text = paragraph[0].text;
        paragraph = [];
        blocks.push(`<h2>${renderInlineMarkdown(text, ctx)}</h2>`);
        i++;
        continue;
      }
      if (/^(-{3,}|\*{3,}|_{3,})\s*$/.test(trimmed)) {
        flushParagraph();
        blocks.push("<hr>");
        i++;
        continue;
      }

      const gfmTable = detectGFMTable(lines, i);
      if (gfmTable) {
        flushParagraph();
        blocks.push(renderTableHtml(gfmTable.header, gfmTable.align, gfmTable.body, ctx));
        i = gfmTable.nextIndex;
        continue;
      }

      const tabTable = detectTabTable(lines, i);
      if (tabTable) {
        flushParagraph();
        blocks.push(renderTableHtml(tabTable.header, tabTable.align, tabTable.body, ctx));
        i = tabTable.nextIndex;
        continue;
      }

      const spacedTable = detectSpacedTable(lines, i);
      if (spacedTable) {
        flushParagraph();
        blocks.push(renderTableHtml(spacedTable.header, spacedTable.align, spacedTable.body, ctx));
        i = spacedTable.nextIndex;
        continue;
      }

      const indentedCode = parseIndentedCodeBlock(lines, i);
      if (indentedCode) {
        flushParagraph();
        blocks.push(`<pre><code>${escapeHtml(indentedCode.code)}</code></pre>`);
        i = indentedCode.nextIndex;
        continue;
      }

      const blockquote = parseBlockquote(lines, i);
      if (blockquote) {
        flushParagraph();
        blocks.push(renderBlockquoteEntries(blockquote.entries, ctx));
        i = blockquote.nextIndex;
        continue;
      }

      const defList = parseDefinitionList(lines, i);
      if (defList) {
        flushParagraph();
        blocks.push(renderDefinitionList(defList.term, defList.defs, ctx));
        i = defList.nextIndex;
        continue;
      }

      const listBlock = parseListBlock(lines, i);
      if (listBlock) {
        flushParagraph();
        blocks.push(renderListTree(listBlock.items, 0, listBlock.items.length, listBlock.items[0].indent, listBlock.ordered, ctx));
        i = listBlock.nextIndex;
        continue;
      }

      const taskBlock = parseTaskBlock(lines, i);
      if (taskBlock) {
        flushParagraph();
        blocks.push(
          renderListTree(taskBlock.items, 0, taskBlock.items.length, taskBlock.items[0].indent, false, ctx),
        );
        i = taskBlock.nextIndex;
        continue;
      }

      const heading = trimmed.match(/^(#{1,6})\s+(.+)$/);
      if (heading) {
        flushParagraph();
        const level = Math.min(6, heading[1].length);
        blocks.push(`<h${level}>${renderInlineMarkdown(heading[2], ctx)}</h${level}>`);
        i++;
        continue;
      }

      paragraph.push({ text: trimmed, hardBreak: / {2,}$/.test(rawLine) });
      i++;
    }

    flushParagraph();
    const footnotesHtml = renderFootnotesSection(footnotes, referenced);
    if (footnotesHtml) blocks.push(footnotesHtml);
    return blocks.join("");
}

globalThis.escapeHtml = escapeHtml;
globalThis.renderInlineMarkdown = renderInlineMarkdown;
globalThis.renderMarkdown = renderMarkdown;
