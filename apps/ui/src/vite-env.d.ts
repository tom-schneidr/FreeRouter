/// <reference types="vite/client" />

declare global {
  // eslint-disable-next-line no-var
  var escapeHtml: (value: unknown) => string;
  // eslint-disable-next-line no-var
  var renderInlineMarkdown: (text: string, ctx?: unknown) => string;
  // eslint-disable-next-line no-var
  var renderMarkdown: (markdown: string) => string;
}

export {};
