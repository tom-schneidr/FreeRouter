import React from "react";
import { fetchJson } from "../../api/client";
import type { LiveRequestEvent, LiveSnapshotResponse } from "../../api/types";
import {
  type LiveRow,
  normalizeLiveEvent,
  trimLiveRequests,
} from "./livePageLogic";

const MAX_ROWS = 250;

export function useLiveTraffic() {
  const requestsRef = React.useRef(new Map<string, LiveRow>());
  const seenEventIdsRef = React.useRef(new Set<number>());
  const [rows, setRows] = React.useState<LiveRow[]>([]);
  const [liveBadge, setLiveBadge] = React.useState("Live");
  const [liveBadgeClass, setLiveBadgeClass] = React.useState("badge");
  const [countBadge, setCountBadge] = React.useState("0 requests");

  const publish = React.useCallback(() => {
    const list = [...requestsRef.current.values()].sort((a, b) => (b.last_ts || 0) - (a.last_ts || 0));
    const activeCount = list.filter((row) => row.phase === "in_progress" || row.phase === "routing").length;
    setRows(list);
    setCountBadge(`${list.length} requests`);
    setLiveBadge(activeCount > 0 ? `Live (${activeCount} active)` : "Live");
  }, []);

  const addEvent = React.useCallback(
    (evt: LiveRequestEvent) => {
      const row = normalizeLiveEvent(evt, requestsRef.current, seenEventIdsRef.current);
      if (!row) return;
      trimLiveRequests(requestsRef.current, MAX_ROWS);
      publish();
    },
    [publish],
  );

  React.useEffect(() => {
    let closed = false;
    let eventSource: EventSource | null = null;

    async function loadSnapshot() {
      try {
        const payload = await fetchJson<LiveSnapshotResponse>("/v1/gateway/live/snapshot");
        for (const evt of payload.data || []) {
          if (!closed) addEvent(evt);
        }
      } catch {
        /* ignore */
      }
    }

    function connect() {
      if (eventSource) eventSource.close();
      eventSource = new EventSource("/v1/gateway/live/events");
      eventSource.onopen = () => {
        if (!closed) setLiveBadgeClass("badge live");
      };
      eventSource.onmessage = (msg) => {
        try {
          addEvent(JSON.parse(msg.data) as LiveRequestEvent);
        } catch {
          /* ignore */
        }
      };
      eventSource.onerror = () => {
        if (!closed) {
          setLiveBadge("Reconnecting...");
          setLiveBadgeClass("badge");
        }
      };
    }

    void loadSnapshot();
    connect();

    return () => {
      closed = true;
      eventSource?.close();
    };
  }, [addEvent]);

  return { rows, liveBadge, liveBadgeClass, countBadge };
}
