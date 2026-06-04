export function escapeHtml(value: unknown): string {
  return String(value ?? "").replace(/[&<>"']/g, (char) => {
    const map: Record<string, string> = {
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    };
    return map[char] ?? char;
  });
}

export function fmtNumber(value: unknown): string {
  return value == null ? "Unknown" : Number(value).toLocaleString();
}

export function fmtDate(value: number | null | undefined): string {
  return value ? new Date(Number(value) * 1000).toLocaleString() : "Never";
}

export function fmtCompact(value: unknown): string {
  const amount = Number(value || 0);
  if (amount >= 1_000_000) return `${(amount / 1_000_000).toFixed(1)}M`;
  if (amount >= 10_000) return `${Math.round(amount / 1000)}k`;
  if (amount >= 1000) return `${(amount / 1000).toFixed(1)}k`;
  return String(amount);
}

export function fmtRelative(value: number | null | undefined): string {
  if (!value) return "Never";
  const date = new Date(Number(value) * 1000);
  const ageMs = Date.now() - date.getTime();
  if (ageMs < 60_000) return "Just now";
  const minutes = Math.floor(ageMs / 60_000);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  if (days < 7) return `${days}d ago`;
  return date.toLocaleDateString([], { month: "short", day: "numeric" });
}

export function modelIdLeaf(modelId: string): string {
  const raw = String(modelId || "").trim();
  if (!raw) return "";
  return raw.includes("/") ? (raw.split("/").filter(Boolean).pop() ?? raw) : raw;
}

export function healthLabel(value: string | undefined): string {
  return String(value || "active").replace(/_/g, " ");
}

export function copyText(text: string) {
  navigator.clipboard?.writeText(text).catch(() => {
    // Ignore clipboard failures.
  });
}
