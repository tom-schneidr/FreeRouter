import { escapeHtml } from "./format";

function renderInlineMarkdown(text: string): string {
  return escapeHtml(text)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*]+)\*/g, "<em>$1</em>")
    .replace(
      /\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g,
      '<a href="$2" target="_blank" rel="noreferrer">$1</a>',
    );
}

function parseTableRow(line: string): string[] | null {
  const t = line.trim();
  if (!t.includes("|")) return null;
  const parts = t.split("|");
  if (parts.length < 3) return null;
  return parts.slice(1, -1).map((c) => c.trim());
}

function isSeparatorRow(cells: string[]): boolean {
  return cells.length > 0 && cells.every((c) => /^:?-{3,}:?$/.test(c.trim()));
}

function cellAlign(cell: string): "left" | "right" | "center" {
  const c = cell.trim();
  if (/^:-+$/.test(c)) return "left";
  if (/^-+:$/.test(c)) return "right";
  if (/^:-+:$/.test(c)) return "center";
  return "left";
}

function renderTableHtml(header: string[], align: string[], bodyRows: string[][]): string {
  const ths = header.map((h, idx) => {
    const a = align[idx] || "left";
    return `<th style="text-align:${a}">${renderInlineMarkdown(h)}</th>`;
  });
  const trs = bodyRows.map(
    (row) =>
      `<tr>${row
        .map((cell, idx) => {
          const a = align[idx] || "left";
          return `<td style="text-align:${a}">${renderInlineMarkdown(cell)}</td>`;
        })
        .join("")}</tr>`,
  );
  return `<div class="md-table-wrap"><table class="md-table"><thead><tr>${ths.join("")}</tr></thead><tbody>${trs.join("")}</tbody></table></div>`;
}

function detectGFMTable(
  lines: string[],
  start: number,
): { html: string; nextIndex: number } | null {
  if (start + 1 >= lines.length) return null;
  const row0 = lines[start].trim();
  const row1 = lines[start + 1].trim();
  if (!row0 || !row1) return null;
  const header = parseTableRow(row0);
  const sepCells = parseTableRow(row1);
  if (!header || !sepCells || header.length !== sepCells.length) return null;
  if (!isSeparatorRow(sepCells)) return null;
  const align = sepCells.map(cellAlign);
  const body: string[][] = [];
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

/** Port of `renderMarkdown` from app/chat_page.py */
export function renderMarkdown(markdown: string): string {
  const codeBlocks: string[] = [];
  const protectedText = String(markdown || "").replace(
    /```(\w+)?\n?([\s\S]*?)```/g,
    (_, _lang: string, code: string) => {
      const token = `\u0000CODE${codeBlocks.length}\u0000`;
      codeBlocks.push(`<pre><code>${escapeHtml(code.replace(/\n$/, ""))}</code></pre>`);
      return token;
    },
  );
  const lines = protectedText.split("\n");
  const blocks: string[] = [];
  let paragraph: string[] = [];
  let listItems: string[] = [];
  let orderedItems: string[] = [];

  function flushParagraph() {
    if (!paragraph.length) return;
    blocks.push(`<p>${paragraph.map(renderInlineMarkdown).join("<br>")}</p>`);
    paragraph = [];
  }

  function flushLists() {
    if (listItems.length) {
      blocks.push(
        `<ul>${listItems.map((item) => `<li>${renderInlineMarkdown(item)}</li>`).join("")}</ul>`,
      );
      listItems = [];
    }
    if (orderedItems.length) {
      blocks.push(
        `<ol>${orderedItems.map((item) => `<li>${renderInlineMarkdown(item)}</li>`).join("")}</ol>`,
      );
      orderedItems = [];
    }
  }

  function flushAll() {
    flushParagraph();
    flushLists();
  }

  let i = 0;
  while (i < lines.length) {
    const line = lines[i].trimEnd();
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
  return blocks.join("");
}

export function extractAssistantText(body: Record<string, unknown> | null | undefined): string {
  if (!body) return "";
  if (typeof body.content === "string") return body.content;
  if (typeof body.text === "string") return body.text;
  const message = body.message as { content?: unknown } | undefined;
  if (typeof message?.content === "string") return message.content;
  const choices = body.choices as Array<{ message?: { content?: unknown }; text?: string }> | undefined;
  const choiceContent = choices?.[0]?.message?.content;
  if (typeof choiceContent === "string") return choiceContent;
  if (Array.isArray(choiceContent)) {
    return choiceContent
      .map((item) =>
        typeof item === "string" ? item : (item as { text?: string; content?: string })?.text || (item as { content?: string })?.content || "",
      )
      .filter(Boolean)
      .join("\n");
  }
  if (typeof choices?.[0]?.text === "string") return choices[0].text;
  return "";
}
