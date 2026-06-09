import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import vm from "node:vm";

const rendererPath = join(
  dirname(fileURLToPath(import.meta.url)),
  "../../../../app/ui/markdown_renderer.js",
);

if (typeof globalThis.renderMarkdown !== "function") {
  vm.runInThisContext(readFileSync(rendererPath, "utf8"), {
    filename: rendererPath,
  });
}
