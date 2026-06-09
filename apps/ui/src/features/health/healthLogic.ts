import type { ModelRoute } from "../../api/types";

export function formatHealthTime(value: number | undefined | null): string {
  return value ? new Date(value * 1000).toLocaleString() : "not scheduled";
}

export function limitedRoutes(routes: ModelRoute[]): ModelRoute[] {
  return routes.filter((route) => route.health && route.health.status !== "active");
}

export function healthSummaryText(count: number): string {
  return `${count} automatically limited route${count === 1 ? "" : "s"}`;
}

export function healthStatusLabel(status: string): string {
  return status.replace(/_/g, " ");
}
