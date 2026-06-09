import { readFileSync } from "node:fs";
import vm from "node:vm";

vm.runInThisContext(readFileSync(new URL("../app/ui/markdown_renderer.js", import.meta.url), "utf8"));
const { renderMarkdown } = globalThis;

const src = `You can also use __bold__ and _italic_. Here is ~~strikethrough~~.

> This is a blockquote.
>
> > This is a nested blockquote.

+ Yet another style

[This is a link with a title](https://www.example.com "Example Website")

<https://www.google.com>

![Placeholder Image](https://via.placeholder.com/150 "Image Title")

Here is footnote[^1].

[^1]: This is the footnote content.

Left-aligned\tCenter-aligned\tRight-aligned
Cell 1\tCell 2\tCell 3

[x] Completed task
[ ] Incomplete task

Term 1
: Definition 1

\\*not italic\\*.`;

const html = renderMarkdown(src);
const htmlSample = renderMarkdown(
  `Here is a <span style="color:red;">HTML span tag</span> and footnote[^1].\n\n[^1]: Footnote body.`,
);
const checks = [
  ["__bold__", !html.includes("__bold__") && html.includes("<strong>bold</strong>")],
  ["_italic_", !html.includes("_italic_") && html.includes("<em>italic</em>")],
  ["strikethrough", html.includes("<del>")],
  ["blockquote", html.includes("<blockquote>") && !html.includes("> This")],
  ["plus list", html.includes("Yet another") && !html.includes("+ Yet")],
  ["titled link", html.includes('title="Example Website"')],
  ["autolink", html.includes('href="https://www.google.com"')],
  ["image", html.includes("<img")],
  ["footnote", html.includes("footnote-ref")],
  ["tab table", html.includes("<table")],
  ["task", html.includes("task-item")],
  ["def list", html.includes("<dl>")],
  ["escape", html.includes("*not italic*") && !html.includes("<em>not italic</em>")],
  [
    "raw html",
    htmlSample.includes('<span style="color: red">') && !htmlSample.includes("&lt;span"),
  ],
];
for (const [name, ok] of checks) console.log(name, ok ? "OK" : "FAIL");
if (checks.some(([, ok]) => !ok)) process.exit(1);
