/** Append a cache-busting query param so embedded legacy pages reload on Refresh. */
export function embedSrcWithReload(src: string, reloadKey: number): string {
  if (reloadKey <= 0) {
    return src;
  }
  const separator = src.includes("?") ? "&" : "?";
  return `${src}${separator}_=${reloadKey}`;
}
