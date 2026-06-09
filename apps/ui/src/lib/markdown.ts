import "./markdown_setup";

export const escapeHtml = globalThis.escapeHtml as (value: unknown) => string;
export const renderInlineMarkdown = globalThis.renderInlineMarkdown as (
  text: string,
  ctx?: unknown,
) => string;
export const renderMarkdown = globalThis.renderMarkdown as (markdown: string) => string;

export { extractAssistantText } from "./markdown_extract";
