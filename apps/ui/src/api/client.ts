import { QueryClient } from "@tanstack/react-query";

export const queryClient = new QueryClient();

export async function fetchJson<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(path, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const message =
      typeof (payload as { detail?: unknown })?.detail === "string"
        ? (payload as { detail: string }).detail
        : response.statusText;
    throw new Error(message || `Request failed: ${response.status}`);
  }
  return payload as T;
}

export function desktopHeaders(token: string): Record<string, string> {
  return {
    "Content-Type": "application/json",
    "X-FreeRouter-Desktop-Token": token,
  };
}
