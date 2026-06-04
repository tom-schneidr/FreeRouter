import {
  Activity,
  BarChart3,
  Bot,
  Database,
  Gauge,
  HeartPulse,
  MessageSquareText,
  ScrollText,
  Settings,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";

export const PRIMARY_NAV_ITEMS = [
  { id: "dashboard", label: "Dashboard", icon: Gauge },
  { id: "chat", label: "Chat", icon: MessageSquareText },
  { id: "models", label: "Models", icon: Bot },
  { id: "usage", label: "Usage", icon: BarChart3 },
  { id: "health", label: "Route Health", icon: HeartPulse },
  { id: "live", label: "Live Traffic", icon: Activity },
  { id: "backups", label: "Backups", icon: Database },
  { id: "logs", label: "Logs", icon: ScrollText },
] as const;

export const SETTINGS_NAV_ITEM = { id: "settings", label: "Settings", icon: Settings } as const;

export const NAV_ITEMS = [...PRIMARY_NAV_ITEMS, SETTINGS_NAV_ITEM] as const;

export type SectionId = (typeof NAV_ITEMS)[number]["id"];

/** Sections that fill the main pane edge-to-edge. */
export const FILL_SECTIONS = new Set<SectionId>(["models"]);

/** Legacy HTML pages embedded for pixel-perfect parity. */
export const LEGACY_EMBED_SECTIONS: Partial<Record<SectionId, string>> = {
  chat: "/chat?embed=1",
  usage: "/status?embed=1",
  health: "/health?embed=1",
  live: "/live?embed=1",
};

export function isFillSection(section: SectionId): boolean {
  return FILL_SECTIONS.has(section) || section in LEGACY_EMBED_SECTIONS;
}

export function isLegacyEmbedSection(section: SectionId): boolean {
  return section in LEGACY_EMBED_SECTIONS;
}

export type NavItem = {
  id: SectionId;
  label: string;
  icon: LucideIcon;
};

export function initialSection(): SectionId {
  const hash = (location.hash || "#dashboard").slice(1);
  if (NAV_ITEMS.some((item) => item.id === hash)) {
    return hash as SectionId;
  }
  return "dashboard";
}
