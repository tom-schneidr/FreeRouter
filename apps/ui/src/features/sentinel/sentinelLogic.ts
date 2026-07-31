export type SentinelProbe = {
  check_id: string;
  label: string;
  status: "pass" | "warn" | "fail" | "error";
  score: number;
  latency_ms: number;
  evidence: string;
  remediation: string;
};

export type SentinelEvaluation = {
  run_id: string;
  score: number;
  readiness: "ready" | "limited" | "blocked";
  completed_at: number;
  duration_ms: number;
  summary: string;
  probes: SentinelProbe[];
};

export type SentinelRoute = {
  route_id: string;
  provider_name: string;
  model_id: string;
  display_name: string;
  rank: number;
  enabled: boolean;
  configured: boolean;
  cost: string;
  zero_cost: boolean;
  speed: string;
  quality: string;
  evaluation: SentinelEvaluation | null;
};

export type SentinelProfile = {
  profile_id: string;
  name: string;
  description: string;
  minimum_score: number;
  required_checks: string[];
  zero_cost_only: boolean;
  consumer_id?: string | null;
  tool_policy: string;
  timeout_seconds: number;
  max_retries: number;
  status: "ready" | "untested" | "blocked";
  message: string;
  remediation: string;
  counts: {
    enabled: number;
    free: number;
    configured: number;
    evaluated: number;
    qualified: number;
  };
  qualified_route_ids: string[];
};

export type SentinelConsumer = {
  consumer_id: "semesteros" | "agentrange";
  name: string;
  product: string;
  profile_id: string;
  description: string;
  repository: string;
  accent: "violet" | "amber";
  env: string;
  preflight_url: string;
  doctor_url: string;
  contract: {
    version: string;
    zero_cost_only: boolean;
    tool_policy: string;
    timeout_seconds: number;
    max_retries: number;
    fallback_model: string;
  };
  readiness: SentinelProfile;
};

export type SentinelReceipt = {
  run_id: string;
  created_at: number;
  consumer_id: string | null;
  profile_id: string | null;
  status: "healthy" | "degraded" | "blocked";
  policy_verdict: string;
  provider_name: string;
  route_id: string;
  model_id: string;
  latency_ms: number;
  attempts: number;
  fallback_used: boolean;
  fallback_reason: string;
  stream: boolean;
  capabilities: string[];
  tool_policy: string;
  request_path: string;
};

export type SentinelPreflight = {
  ok: boolean;
  status: "healthy" | "degraded" | "blocked";
  profile: { profile_id: string; tool_policy: string };
  reason: string;
  next_action: string;
  qualified_route_ids: string[];
  checks: Array<{ id: string; status: "pass" | "warn" | "fail"; message: string }>;
};

export type SentinelSnapshot = {
  object: string;
  contract_version: string;
  zero_cost_guard: { enabled: boolean; message: string };
  summary: {
    routes: number;
    configured: number;
    evaluated: number;
    ready: number;
    blocked: number;
  };
  profiles: SentinelProfile[];
  consumers: SentinelConsumer[];
  receipts: SentinelReceipt[];
  routes: SentinelRoute[];
  opencode: {
    config: Record<string, unknown>;
    config_json: string;
    steps: string[];
    doctor_url: string;
  };
};

export function routeTrustLabel(route: SentinelRoute): string {
  if (!route.enabled) return "Disabled";
  if (!route.zero_cost) return "$0 guard blocked";
  if (!route.evaluation) return "Needs evidence";
  if (route.evaluation.readiness === "ready") return "Agent ready";
  if (route.evaluation.readiness === "limited") return "Use with limits";
  return "Blocked";
}

export function routeTrustTone(route: SentinelRoute): "ok" | "warn" | "bad" | "muted" {
  if (!route.enabled) return "muted";
  if (!route.zero_cost || route.evaluation?.readiness === "blocked") return "bad";
  if (route.evaluation?.readiness === "ready") return "ok";
  return "warn";
}

export function filterSentinelRoutes(
  routes: SentinelRoute[],
  search: string,
  filter: "all" | "ready" | "needs-evidence" | "blocked",
): SentinelRoute[] {
  const needle = search.trim().toLowerCase();
  return routes.filter((route) => {
    const matchesSearch =
      !needle ||
      `${route.display_name} ${route.model_id} ${route.provider_name}`
        .toLowerCase()
        .includes(needle);
    const matchesFilter =
      filter === "all" ||
      (filter === "ready" && route.evaluation?.readiness === "ready") ||
      (filter === "needs-evidence" && !route.evaluation) ||
      (filter === "blocked" &&
        (!route.zero_cost || route.evaluation?.readiness === "blocked"));
    return matchesSearch && matchesFilter;
  });
}

export function formatEvidenceAge(timestamp: number, now = Date.now() / 1000): string {
  const seconds = Math.max(0, now - timestamp);
  if (seconds < 60) return "just now";
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return `${Math.floor(seconds / 86400)}d ago`;
}


export function receiptStatusLabel(receipt: SentinelReceipt): string {
  if (receipt.status === "healthy") return "Healthy";
  if (receipt.status === "degraded") return "Fallback used";
  return "Blocked";
}

export function receiptStatusTone(receipt: SentinelReceipt): "ok" | "warn" | "bad" {
  if (receipt.status === "healthy") return "ok";
  if (receipt.status === "degraded") return "warn";
  return "bad";
}
