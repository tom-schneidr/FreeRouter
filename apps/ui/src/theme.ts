export type ThemePreference = "system" | "light" | "dark";
export type ThemeMode = "light" | "dark";

const STORAGE_KEY = "freerouter.theme";

function normalizePreference(preference: string | null | undefined): ThemePreference {
  return preference === "light" || preference === "dark" ? preference : "system";
}

function resolveTheme(preference: ThemePreference): ThemeMode {
  if (preference === "light" || preference === "dark") {
    return preference;
  }
  if (typeof window !== "undefined" && window.matchMedia?.("(prefers-color-scheme: light)").matches) {
    return "light";
  }
  return "dark";
}

function postToFrames(preference: ThemePreference, theme: ThemeMode) {
  document.querySelectorAll("iframe").forEach((frame) => {
    try {
      frame.contentWindow?.postMessage(
        { source: "freerouter", type: "theme", preference, theme },
        "*",
      );
    } catch {
      // Ignore cross-origin frames.
    }
  });
}

export function getThemePreference(): ThemePreference {
  try {
    return normalizePreference(localStorage.getItem(STORAGE_KEY));
  } catch {
    return "system";
  }
}

export function applyTheme(
  preference: ThemePreference,
  options: { persist?: boolean; broadcast?: boolean } = {},
) {
  const normalized = normalizePreference(preference);
  const theme = resolveTheme(normalized);
  document.documentElement.dataset.theme = theme;
  document.documentElement.dataset.themePreference = normalized;
  document.documentElement.style.colorScheme = theme;
  if (options.persist !== false) {
    try {
      localStorage.setItem(STORAGE_KEY, normalized);
    } catch {
      // Ignore storage failures.
    }
  }
  if (options.broadcast) {
    postToFrames(normalized, theme);
  }
}

export function initTheme() {
  applyTheme(getThemePreference(), { persist: false, broadcast: true });
  const media = window.matchMedia?.("(prefers-color-scheme: light)");
  const onSystemThemeChange = () => {
    if (getThemePreference() === "system") {
      applyTheme("system", { persist: false, broadcast: true });
    }
  };
  media?.addEventListener?.("change", onSystemThemeChange);
  window.addEventListener("message", (event) => {
    const data = event.data ?? {};
    if (data.source === "freerouter" && data.type === "theme") {
      applyTheme(data.preference ?? data.theme, { persist: true, broadcast: false });
    }
  });
  window.addEventListener("storage", (event) => {
    if (event.key === STORAGE_KEY && event.newValue) {
      applyTheme(normalizePreference(event.newValue), { persist: false, broadcast: true });
    }
  });
}
