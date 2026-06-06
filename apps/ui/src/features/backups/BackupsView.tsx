import React from "react";
import { useMutation } from "@tanstack/react-query";
import { Download, Upload } from "lucide-react";
import { desktopHeaders, fetchJson } from "../../api/client";
import type { BackupExportPayload, BackupImportPayload } from "../../api/types";
import { DesktopRequired, Notice, Panel, SectionIntro } from "../../components/ui";

export function BackupsView({
  desktopToken,
  desktopReady,
}: {
  desktopToken: string;
  desktopReady: boolean;
}) {
  const [path, setPath] = React.useState("");
  const [overwrite, setOverwrite] = React.useState(false);
  const [file, setFile] = React.useState<File | null>(null);
  const exportBackup = useMutation({
    mutationFn: () =>
      fetchJson<BackupExportPayload>("/v1/desktop/backups/export", {
        method: "POST",
        headers: desktopHeaders(desktopToken),
        body: JSON.stringify({}),
      }),
  });
  const importBackup = useMutation({
    mutationFn: async () => {
      if (file) {
        const form = new FormData();
        form.append("file", file);
        form.append("overwrite", overwrite ? "true" : "false");
        return fetchJson<BackupImportPayload>("/v1/desktop/backups/import-upload", {
          method: "POST",
          headers: { "X-FreeRouter-Desktop-Token": desktopToken },
          body: form,
        });
      }
      if (!path.trim()) throw new Error("Choose a backup zip file or enter a backup path.");
      return fetchJson<BackupImportPayload>("/v1/desktop/backups/import", {
        method: "POST",
        headers: desktopHeaders(desktopToken),
        body: JSON.stringify({ path: path.trim(), overwrite }),
      });
    },
    onSuccess: () => {
      setFile(null);
      setPath("");
    },
  });

  if (!desktopReady) {
    return (
      <div className="section-stack">
        <SectionIntro
          title="Backups"
          copy="Export or restore local model catalog and SQLite state. Secrets are not included in backups."
        />
        <DesktopRequired />
      </div>
    );
  }

  return (
    <div className="section-stack">
      <SectionIntro
        title="Backups"
        copy="Export or restore local model catalog and SQLite state. Secrets are not included in backups."
      />
      <div className="two-column">
        <Panel title="Export local state">
          <p className="panel-copy">
            Creates a zip with the editable model catalog, SQLite state, and non-secret local settings.
          </p>
          <button
            className="primary-action"
            type="button"
            disabled={exportBackup.isPending}
            onClick={() => exportBackup.mutate()}
          >
            <Download size={16} />
            {exportBackup.isPending ? "Exporting" : "Export backup"}
          </button>
          {exportBackup.data && <Notice tone="ok">Exported to {exportBackup.data.path}</Notice>}
          {exportBackup.error && <Notice tone="bad">{exportBackup.error.message}</Notice>}
        </Panel>
        <Panel title="Restore local state">
          <div className="form-grid single">
            <label className="field">
              <span>Backup zip file</span>
              <input
                type="file"
                accept=".zip,application/zip"
                onChange={(event) => setFile(event.target.files?.[0] ?? null)}
              />
            </label>
            <label className="field">
              <span>Or enter a path</span>
              <input
                value={path}
                placeholder="C:\\path\\to\\freerouter-local-state.zip"
                onChange={(event) => setPath(event.target.value)}
              />
            </label>
            <label className="check-field">
              <input
                type="checkbox"
                checked={overwrite}
                onChange={(event) => setOverwrite(event.target.checked)}
              />
              <span>Overwrite existing local state</span>
            </label>
          </div>
          <button
            className="danger-action"
            type="button"
            disabled={importBackup.isPending}
            onClick={() => importBackup.mutate()}
          >
            <Upload size={16} />
            {importBackup.isPending ? "Restoring" : "Restore backup"}
          </button>
          {importBackup.data && (
            <Notice tone="ok">
              Restored {importBackup.data.restored.length} file(s). Restart the server.
            </Notice>
          )}
          {importBackup.error && <Notice tone="bad">{importBackup.error.message}</Notice>}
        </Panel>
      </div>
    </div>
  );
}
