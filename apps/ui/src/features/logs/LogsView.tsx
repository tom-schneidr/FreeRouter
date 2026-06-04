import { useQuery } from "@tanstack/react-query";
import { desktopHeaders, fetchJson } from "../../api/client";
import type { DesktopLogsPayload } from "../../api/types";
import { DesktopRequired, EmptyState, Notice } from "../../components/ui";

export function LogsView({
  desktopToken,
  desktopReady,
}: {
  desktopToken: string;
  desktopReady: boolean;
}) {
  const logs = useQuery({
    queryKey: ["desktop-logs", desktopToken],
    queryFn: () =>
      fetchJson<DesktopLogsPayload>("/v1/desktop/logs", {
        headers: desktopHeaders(desktopToken),
      }),
    enabled: desktopReady,
    refetchInterval: 5000,
  });

  return (
    <div className="section-stack">
      <div className="section-heading">
        <div>
          <h1>Logs</h1>
          <p>Server output captured by the desktop launcher.</p>
        </div>
        <button type="button" disabled={!desktopReady} onClick={() => logs.refetch()}>
          Refresh logs
        </button>
      </div>
      {!desktopReady && <DesktopRequired />}
      {desktopReady && logs.isLoading && <EmptyState message="Loading desktop logs..." />}
      {desktopReady && logs.error && <Notice tone="bad">{logs.error.message}</Notice>}
      {desktopReady && logs.data && (
        <pre className="log-box">{logs.data.lines.join("") || "No logs captured yet."}</pre>
      )}
    </div>
  );
}
