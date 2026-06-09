import type { ModelRoute, ProviderStatus, RouteHealth, RouteUsage } from "../../api/types";
import { escapeHtml, fmtCompact, fmtDate, fmtNumber, fmtRelative, healthLabel, modelIdLeaf } from "../../lib/format";

export const SORT_STORAGE_KEY = "freerouter.usage.sort";

const HEALTH_ORDER: Record<string, number> = {
  active: 0,
  rate_limited: 1,
  too_slow: 2,
  potentially_outdated: 3,
};

export type SortState = { key: string; dir: "asc" | "desc" };

export function loadSortPreference(): SortState {
  try {
    const saved = JSON.parse(localStorage.getItem(SORT_STORAGE_KEY) || "null") as SortState | null;
    if (saved && typeof saved.key === "string" && (saved.dir === "asc" || saved.dir === "desc")) {
      return saved;
    }
  } catch {
    /* ignore */
  }
  return { key: "rank", dir: "asc" };
}

export function saveSortPreference(sort: SortState) {
  try {
    localStorage.setItem(SORT_STORAGE_KEY, JSON.stringify(sort));
  } catch {
    /* ignore */
  }
}

export function toggleSort(sort: SortState, key: string): SortState {
  if (sort.key === key) {
    return { key, dir: sort.dir === "asc" ? "desc" : "asc" };
  }
  return { key, dir: key === "rank" ? "asc" : "desc" };
}

export function sortIndicator(sort: SortState, key: string): string {
  if (sort.key !== key) return "↕";
  return sort.dir === "asc" ? "↑" : "↓";
}

function compareText(a: unknown, b: unknown) {
  return String(a || "").localeCompare(String(b || ""), undefined, { sensitivity: "base" });
}

function compareNumber(a: unknown, b: unknown) {
  return Number(a || 0) - Number(b || 0);
}

export function compareModels(a: ModelRoute, b: ModelRoute, sort: SortState): number {
  let cmp = 0;
  const usageA: RouteUsage = a.usage || ({} as RouteUsage);
  const usageB: RouteUsage = b.usage || ({} as RouteUsage);
  const healthA = a.health?.status || "active";
  const healthB = b.health?.status || "active";
  switch (sort.key) {
    case "rank":
      cmp = compareNumber(a.rank ?? 9999, b.rank ?? 9999);
      break;
    case "model":
      cmp = compareText(a.display_name || a.model_id, b.display_name || b.model_id);
      break;
    case "provider":
      cmp = compareText(a.provider_name, b.provider_name);
      break;
    case "health":
      cmp =
        compareNumber(HEALTH_ORDER[healthA] ?? 99, HEALTH_ORDER[healthB] ?? 99) ||
        compareText(healthA, healthB);
      break;
    case "successes":
      cmp = compareNumber(usageA.successes, usageB.successes);
      break;
    case "failures":
      cmp = compareNumber(usageA.failures, usageB.failures);
      break;
    case "tokens":
      cmp = compareNumber(usageA.total_tokens, usageB.total_tokens);
      break;
    case "prompt":
      cmp = compareNumber(usageA.prompt_tokens, usageB.prompt_tokens);
      break;
    case "completion":
      cmp = compareNumber(usageA.completion_tokens, usageB.completion_tokens);
      break;
    case "last_used":
      cmp = compareNumber(usageA.last_used_at, usageB.last_used_at);
      break;
    default:
      cmp = compareNumber(a.rank ?? 9999, b.rank ?? 9999);
  }
  if (cmp === 0) cmp = compareNumber(a.rank ?? 9999, b.rank ?? 9999);
  if (cmp === 0) cmp = compareText(a.route_id, b.route_id);
  return sort.dir === "desc" ? -cmp : cmp;
}

export function sortedModels(models: ModelRoute[], sort: SortState): ModelRoute[] {
  return [...models].sort((a, b) => compareModels(a, b, sort));
}

export function healthPillClass(status: string): "ok" | "warning" | "error" {
  if (status === "active") return "ok";
  if (status === "rate_limited") return "warning";
  return "error";
}

export function healthPillLabel(status: string): string {
  if (status === "active") return "Active";
  if (status === "rate_limited") return "Rate limited";
  if (status === "too_slow") return "Too slow";
  if (status === "potentially_outdated") return "Outdated";
  return healthLabel(status);
}

export function formatSuccessFail(successes: number, failures: number): { ok: string; fail: string; failWarn: boolean } {
  const failWarn = failures > 0;
  return {
    ok: `${fmtCompact(successes)} ok`,
    fail: `${fmtCompact(failures)} fail`,
    failWarn,
  };
}

export function providerStatus(provider: ProviderStatus | undefined): string {
  if (!provider?.configured) return "Not configured";
  if (!provider.available) return provider.unavailable_reason || "Limited";
  return "Available";
}

export function embedSummaryLine(models: ModelRoute[], providers: ProviderStatus[]): string {
  const totalTokens = models.reduce((sum, model) => sum + Number(model.usage?.total_tokens || 0), 0);
  const providerRequests = providers.reduce((sum, provider) => sum + Number(provider.requests_today || 0), 0);
  return `${fmtCompact(totalTokens)} tokens · ${fmtCompact(providerRequests)} requests`;
}

export function flattenModels(providers: ProviderStatus[]): ModelRoute[] {
  return providers.flatMap((provider) => provider.models || []);
}

export function filterModels(
  allModels: ModelRoute[],
  query: string,
  provider: string,
  health: string,
): ModelRoute[] {
  const q = query.trim().toLowerCase();
  return allModels.filter((model) => {
    const haystack = `${model.display_name} ${model.model_id} ${model.provider_name}`.toLowerCase();
    return (
      (!q || haystack.includes(q)) &&
      (!provider || model.provider_name === provider) &&
      (!health || (model.health?.status || "active") === health)
    );
  });
}

export function detailStats(model: ModelRoute, providers: ProviderStatus[]) {
  const usage: RouteUsage = model.usage || ({} as RouteUsage);
  const health: RouteHealth = model.health || ({} as RouteHealth);
  const provider = providers.find((item) => item.name === model.provider_name);
  return [
    ["Route ID", model.route_id],
    ["Enabled", model.enabled ? "Yes" : "No"],
    ["Context window", model.context_window ? fmtNumber(model.context_window) : "Unknown"],
    ["Prompt tokens", fmtNumber(usage.prompt_tokens || 0)],
    ["Completion tokens", fmtNumber(usage.completion_tokens || 0)],
    ["Rate limits", fmtNumber(usage.rate_limits || 0)],
    ["Timeouts", fmtNumber(usage.timeouts || 0)],
    ["Not found errors", fmtNumber(usage.not_found || 0)],
    ["Consecutive failures", fmtNumber(health.consecutive_failures || 0)],
    ["Last event", fmtDate(usage.last_used_at)],
    ["Last status code", usage.last_status_code ?? "None"],
    ["Provider state", providerStatus(provider)],
    ["Provider requests today", fmtNumber(provider?.requests_today || 0)],
    ["Provider cooldown until", fmtDate(provider?.cooldown_until)],
  ] as const;
}

export { escapeHtml, fmtCompact, fmtRelative, modelIdLeaf };
