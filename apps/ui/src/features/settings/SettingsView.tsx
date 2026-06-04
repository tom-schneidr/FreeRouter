import React from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { desktopHeaders, fetchJson } from "../../api/client";
import type { DesktopSettingsPayload } from "../../api/types";
import {
  DesktopRequired,
  EmptyState,
  Notice,
  Panel,
  numericFieldKind,
} from "../../components/ui";
import { applyTheme, getThemePreference, type ThemePreference } from "../../theme";

function AppearanceSettings() {
  const [preference, setPreference] = React.useState(getThemePreference);
  return (
    <Panel title="Appearance">
      <p className="panel-copy">
        Use Windows theme automatically, or keep FreeRouter pinned to light or dark mode.
      </p>
      <div className="fr-theme-segmented" role="group" aria-label="Theme preference">
        {(["system", "light", "dark"] as ThemePreference[]).map((option) => (
          <button
            key={option}
            className={`fr-theme-option ${preference === option ? "active" : ""}`}
            type="button"
            aria-pressed={preference === option}
            onClick={() => {
              applyTheme(option, { persist: true, broadcast: true });
              setPreference(option);
            }}
          >
            {option === "system" ? "System" : option === "light" ? "Light" : "Dark"}
          </button>
        ))}
      </div>
    </Panel>
  );
}

export function SettingsView({
  desktopToken,
  desktopReady,
}: {
  desktopToken: string;
  desktopReady: boolean;
}) {
  const settings = useQuery({
    queryKey: ["desktop-settings", desktopToken],
    queryFn: () =>
      fetchJson<DesktopSettingsPayload>("/v1/desktop/settings", {
        headers: desktopHeaders(desktopToken),
      }),
    enabled: desktopReady,
  });
  const [values, setValues] = React.useState<Record<string, string>>({});
  const saveSettings = useMutation({
    mutationFn: () =>
      fetchJson("/v1/desktop/settings", {
        method: "POST",
        headers: desktopHeaders(desktopToken),
        body: JSON.stringify(values),
      }),
    onSuccess: () => settings.refetch(),
  });

  React.useEffect(() => {
    if (!settings.data) return;
    setValues(Object.fromEntries(settings.data.fields.map((field) => [field.key, field.value])));
  }, [settings.data]);

  return (
    <div className="section-stack">
      <div className="section-heading">
        <div>
          <h1>Settings</h1>
          <p>Local API keys, runtime values, storage paths, and endpoint maintenance options.</p>
        </div>
        <button
          className="primary-action"
          type="button"
          disabled={!desktopReady || saveSettings.isPending}
          onClick={() => saveSettings.mutate()}
        >
          {saveSettings.isPending ? "Saving" : "Save settings"}
        </button>
      </div>
      <AppearanceSettings />
      {!desktopReady && <DesktopRequired />}
      {desktopReady && settings.isLoading && <EmptyState message="Loading desktop settings..." />}
      {desktopReady && settings.error && <Notice tone="bad">{settings.error.message}</Notice>}
      {desktopReady && settings.data && (
        <>
          {settings.data.groups.map((group) => {
            const fields = settings.data!.fields.filter((field) => field.group === group);
            return (
              <Panel title={group} key={group}>
                <div className="form-grid">
                  {fields.map((field) => (
                    <label className="field" key={field.key}>
                      <span>{field.label}</span>
                      {field.kind === "bool" ? (
                        <select
                          value={values[field.key] ?? ""}
                          onChange={(event) =>
                            setValues((current) => ({ ...current, [field.key]: event.target.value }))
                          }
                        >
                          <option value="true">true</option>
                          <option value="false">false</option>
                        </select>
                      ) : (
                        <input
                          type={field.secret ? "password" : numericFieldKind(field.kind) ? "number" : "text"}
                          step={field.kind === "float" ? "0.1" : undefined}
                          value={values[field.key] ?? ""}
                          onChange={(event) =>
                            setValues((current) => ({ ...current, [field.key]: event.target.value }))
                          }
                        />
                      )}
                    </label>
                  ))}
                </div>
              </Panel>
            );
          })}
          <p className="path-note">Settings file: {settings.data.env_path}</p>
          {saveSettings.isSuccess && (
            <Notice tone="ok">Settings saved. Restart the server to apply runtime changes.</Notice>
          )}
        </>
      )}
    </div>
  );
}
