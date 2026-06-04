import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import type { DesktopCapabilities } from "../api/types";
import { desktopHeaders, fetchJson } from "../api/client";

export function useDesktopToken() {
  const [token] = useState(() => {
    const fromUrl = new URLSearchParams(window.location.search).get("desktop_token");
    const fromSession = window.sessionStorage.getItem("freerouterDesktopToken");
    const next = fromUrl || fromSession || "";
    if (next) window.sessionStorage.setItem("freerouterDesktopToken", next);
    return next;
  });
  return token;
}

export function useDesktopReady(token: string) {
  return useQuery({
    queryKey: ["desktop-capabilities", token],
    queryFn: () =>
      fetchJson<DesktopCapabilities>("/v1/desktop/capabilities", {
        headers: desktopHeaders(token),
      }),
    enabled: Boolean(token),
    retry: false,
  });
}

export function useGatewayQuery<T>(key: string, path: string, refetchInterval: number) {
  return useQuery({
    queryKey: [key],
    queryFn: () => fetchJson<T>(path),
    refetchInterval,
  });
}
