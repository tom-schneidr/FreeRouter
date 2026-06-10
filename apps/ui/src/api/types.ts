export type GatewayHealth = {
  status: string;
  version: string;
  database_path: string;
  providers: { configured: number; total: number };
  routes: { enabled: number; total: number };
};

export type RouteHealth = {
  route_id: string;
  provider_name: string;
  model_id: string;
  status: string;
  status_reason: string | null;
  consecutive_failures: number;
  rate_limited_until: number;
  next_probe_at: number;
  updated_at: number;
};

export type RouteUsage = {
  route_id: string;
  total_events: number;
  successes: number;
  failures: number;
  rate_limits: number;
  timeouts: number;
  not_found: number;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  last_used_at: number | null;
  last_status_code: number | null;
};

export type CapabilityClaim = {
  tag: string;
  status: "unknown" | "supported" | "unsupported" | "inconclusive";
  source: "provider_metadata" | "registry" | "probe" | "runtime" | "manual";
  confidence: "high" | "medium" | "low";
  checked_at: number | null;
  evidence: string;
};

export type ModelRoute = {
  route_id: string;
  provider_name: string;
  model_id: string;
  display_name: string;
  rank: number;
  enabled: boolean;
  context_window: number | null;
  quality: string;
  speed: string;
  cost: string;
  tags: string[];
  capabilities?: Record<string, CapabilityClaim>;
  tag_locks?: string[];
  notes: string;
  source_url: string;
  rank_score: number | null;
  rank_reason: string;
  rank_source: string;
  health?: RouteHealth;
  usage?: RouteUsage;
};

export type ModelsListResponse = {
  object?: string;
  data: ModelRoute[];
  catalog_path?: string;
};

export type ProviderStatus = {
  name: string;
  configured: boolean;
  available: boolean;
  unavailable_reason?: string | null;
  retry_after_seconds?: number | null;
  tokens_used_today: number;
  requests_today: number;
  requests_this_minute?: number;
  cooldown_until?: number;
  max_context_tokens?: number;
  models: ModelRoute[];
};

export type ProvidersStatusResponse = {
  object?: string;
  data: ProviderStatus[];
};

export type DesktopField = {
  key: string;
  label: string;
  group: string;
  kind: string;
  secret: boolean;
  value: string;
};

export type DesktopSettingsPayload = {
  env_path: string;
  groups: string[];
  fields: DesktopField[];
};

export type DesktopCapabilities = {
  desktop: boolean;
  server?: { base_url?: string };
};

export type DesktopLogsPayload = {
  lines: string[];
};

export type BackupExportPayload = { ok: boolean; path: string };
export type BackupImportPayload = { ok: boolean; restored: string[] };

export type LiveRequestEvent = {
  event_id?: number;
  event_type: string;
  request_id: string;
  timestamp: number;
  payload: Record<string, unknown>;
};

export type LiveSnapshotResponse = {
  object?: string;
  data: LiveRequestEvent[];
};

export type LiveRequestRow = {
  request_id: string;
  first_ts: number;
  last_ts: number;
  path: string;
  model: string;
  tokens: string;
  status_code: number | null;
  provider_name: string;
  route_id: string;
  latency_ms: number | null;
  phase: string;
  reason: string;
};

export type EndpointSuggestion = {
  suggestion_id: string;
  action: string;
  provider_name: string;
  model_id: string;
  route_id: string;
  title: string;
  details: string;
  route?: Record<string, unknown> | null;
};

export type ProviderDiagnosis = {
  provider_name: string;
  configured: boolean;
  ok: boolean;
  discovered_model_count?: number;
  new_route_suggestion_count?: number;
  confirmed_route_count?: number;
  stale_route_suggestion_count?: number;
  recovered_route_suggestion_count?: number;
  capability_probes_run?: number;
  capability_updates?: number;
  error?: string | null;
};

export type DiagnosisReport = {
  checked_at: number;
  providers: ProviderDiagnosis[];
  suggestions: EndpointSuggestion[];
};

export type EndpointDiagnosisResponse = {
  enabled?: boolean;
  auto_maintenance_enabled?: boolean;
  last_auto_applied?: EndpointSuggestion[];
  last_report: DiagnosisReport | null;
};

export type EndpointDiagnosisRefreshResponse = {
  data: DiagnosisReport;
};

export type StreamRouteEvent =
  | { type: "route_trying"; provider: string; model_id: string; route_id?: string }
  | { type: "route_skip" | "route_fail"; provider: string; model_id: string; reason?: string; route_id?: string }
  | { type: "route_flagged"; provider: string; model_id: string; reason?: string; route_id?: string }
  | { type: "route_selected"; provider: string; model_id: string; route_id?: string }
  | { type: "content"; text: string }
  | { type: "done"; content?: string }
  | { type: "error"; message?: string };
