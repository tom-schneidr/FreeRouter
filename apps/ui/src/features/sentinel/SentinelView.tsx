import React from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  AlertTriangle,
  Check,
  ChevronRight,
  CircleDollarSign,
  Clipboard,
  Code2,
  FlaskConical,
  Gauge,
  LoaderCircle,
  LockKeyhole,
  Radar,
  Search,
  ShieldCheck,
  Sparkles,
  TerminalSquare,
  X,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { fetchJson } from "../../api/client";
import { copyText } from "../../lib/format";
import {
  filterSentinelRoutes,
  formatEvidenceAge,
  routeTrustLabel,
  routeTrustTone,
  type SentinelProfile,
  type SentinelRoute,
  type SentinelSnapshot,
} from "./sentinelLogic";
import "./sentinel.css";

type RouteFilter = "all" | "ready" | "needs-evidence" | "blocked";

export function SentinelView() {
  const queryClient = useQueryClient();
  const [profileId, setProfileId] = React.useState("safe-coding");
  const [search, setSearch] = React.useState("");
  const [filter, setFilter] = React.useState<RouteFilter>("all");
  const [notice, setNotice] = React.useState<string | null>(null);

  const snapshot = useQuery({
    queryKey: ["sentinel"],
    queryFn: () => fetchJson<SentinelSnapshot>("/v1/gateway/sentinel"),
    refetchInterval: 15_000,
  });

  const evaluation = useMutation({
    mutationFn: (routeId: string) =>
      fetchJson(`/v1/gateway/sentinel/routes/${encodeURIComponent(routeId)}/evaluate`, {
        method: "POST",
      }),
    onSuccess: async () => {
      setNotice("Fresh evidence saved");
      await queryClient.invalidateQueries({ queryKey: ["sentinel"] });
    },
    onError: (error) => {
      setNotice(error instanceof Error ? error.message : "Evaluation failed");
    },
  });

  React.useEffect(() => {
    if (!notice) return;
    const timer = window.setTimeout(() => setNotice(null), 5000);
    return () => window.clearTimeout(timer);
  }, [notice]);

  if (snapshot.isLoading) return <SentinelLoading />;
  if (snapshot.error) {
    return (
      <div className="sentinel-page sentinel-state-page">
        <div className="sentinel-error" role="alert">
          <Radar size={30} />
          <h1>Sentinel could not load</h1>
          <p>{snapshot.error.message}</p>
          <button type="button" onClick={() => void snapshot.refetch()}>
            Try again
          </button>
        </div>
      </div>
    );
  }
  if (!snapshot.data) return null;

  const data = snapshot.data;
  const selectedProfile =
    data.profiles.find((profile) => profile.profile_id === profileId) ?? data.profiles[0];
  const visibleRoutes = filterSentinelRoutes(data.routes, search, filter);

  return (
    <div className="sentinel-page">
      <header className="sentinel-hero">
        <div className="sentinel-eyebrow">
          <span className="sentinel-pulse" aria-hidden="true" />
          Local evidence · no paid probes
        </div>
        <div className="sentinel-hero-row">
          <div>
            <h1>Know which routes you can trust.</h1>
            <p>
              Sentinel tests the behaviors coding agents depend on, then gates virtual routes with
              current evidence and a hard zero-cost policy.
            </p>
          </div>
          <div className="sentinel-guard" title={data.zero_cost_guard.message}>
            <CircleDollarSign size={19} />
            <span>
              <strong>$0 guard</strong>
              <small>Fail-closed</small>
            </span>
          </div>
        </div>
      </header>

      {notice ? (
        <div className="sentinel-toast" role="status">
          {evaluation.isError ? <AlertTriangle size={16} /> : <Check size={16} />}
          {notice}
        </div>
      ) : null}

      <section className="sentinel-metrics" aria-label="Sentinel overview">
        <Metric icon={ShieldCheck} label="Agent ready" value={data.summary.ready} note="current evidence" />
        <Metric icon={FlaskConical} label="Evaluated" value={data.summary.evaluated} note={`of ${data.summary.routes} routes`} />
        <Metric icon={LockKeyhole} label="Configured" value={data.summary.configured} note="can be tested" />
        <Metric icon={Gauge} label="Needs attention" value={data.summary.blocked} note="failed evidence" />
      </section>

      <section className="sentinel-section" aria-labelledby="profiles-title">
        <div className="sentinel-section-heading">
          <div>
            <span className="section-kicker">Routing policy</span>
            <h2 id="profiles-title">Choose an agent profile</h2>
          </div>
          <span className="sentinel-helper">Use the profile ID as your API model</span>
        </div>
        <div className="sentinel-profiles">
          {data.profiles.map((profile) => (
            <ProfileCard
              key={profile.profile_id}
              profile={profile}
              selected={profile.profile_id === selectedProfile?.profile_id}
              onSelect={() => setProfileId(profile.profile_id)}
            />
          ))}
        </div>
        {selectedProfile ? <ProfileDiagnostic profile={selectedProfile} /> : null}
      </section>

      <section className="sentinel-section" aria-labelledby="routes-title">
        <div className="sentinel-section-heading sentinel-route-heading">
          <div>
            <span className="section-kicker">Evidence ledger</span>
            <h2 id="routes-title">Route readiness</h2>
          </div>
          <div className="sentinel-search">
            <Search size={16} aria-hidden="true" />
            <input
              aria-label="Search Sentinel routes"
              placeholder="Search routes"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
            />
          </div>
        </div>
        <div className="sentinel-filter-row" role="group" aria-label="Filter routes">
          {(["all", "ready", "needs-evidence", "blocked"] as const).map((value) => (
            <button
              key={value}
              type="button"
              className={filter === value ? "active" : ""}
              onClick={() => setFilter(value)}
            >
              {value === "all"
                ? "All routes"
                : value === "ready"
                  ? "Agent ready"
                  : value === "needs-evidence"
                    ? "Needs evidence"
                    : "Blocked"}
            </button>
          ))}
        </div>
        <div className="sentinel-cost-note">
          <FlaskConical size={15} />
          Each evaluation sends four short, deterministic prompts through the selected free-tier
          route. It uses provider quota but cannot select a paid route.
        </div>
        <div className="sentinel-route-list">
          {visibleRoutes.length ? (
            visibleRoutes.map((route) => (
              <RouteCard
                key={route.route_id}
                route={route}
                busy={evaluation.isPending && evaluation.variables === route.route_id}
                onEvaluate={() => evaluation.mutate(route.route_id)}
              />
            ))
          ) : (
            <div className="sentinel-empty">
              <Search size={24} />
              <strong>No routes match this view</strong>
              <span>Clear the search or choose another evidence filter.</span>
            </div>
          )}
        </div>
      </section>

      <OpenCodeSetup
        profileId={profileId}
        configJson={data.opencode.config_json.replace(
          "freerouter/safe-coding",
          `freerouter/${profileId}`,
        )}
        doctorUrl={data.opencode.doctor_url.replace("safe-coding", profileId)}
        onCopied={setNotice}
      />
    </div>
  );
}

function Metric(props: {
  icon: LucideIcon;
  label: string;
  value: number;
  note: string;
}) {
  return (
    <article className="sentinel-metric">
      <span className="sentinel-metric-icon"><props.icon size={18} /></span>
      <span>{props.label}</span>
      <strong>{props.value}</strong>
      <small>{props.note}</small>
    </article>
  );
}

function ProfileCard(props: {
  profile: SentinelProfile;
  selected: boolean;
  onSelect: () => void;
}) {
  const { profile } = props;
  return (
    <button
      type="button"
      className={`sentinel-profile ${props.selected ? "selected" : ""}`}
      onClick={props.onSelect}
      aria-pressed={props.selected}
    >
      <span className={`profile-status ${profile.status}`} aria-hidden="true">
        {profile.status === "ready" ? <Check size={15} /> : <Sparkles size={15} />}
      </span>
      <span className="profile-copy">
        <strong>{profile.name}</strong>
        <code>{profile.profile_id}</code>
        <small>{profile.description}</small>
      </span>
      <span className="profile-count">
        <strong>{profile.counts.qualified}</strong>
        <small>qualified</small>
      </span>
      <ChevronRight size={18} className="profile-arrow" />
    </button>
  );
}

function ProfileDiagnostic({ profile }: { profile: SentinelProfile }) {
  return (
    <div className={`sentinel-diagnostic ${profile.status}`} role="status">
      <span className="diagnostic-icon">
        {profile.status === "ready" ? <ShieldCheck size={20} /> : <AlertTriangle size={20} />}
      </span>
      <div>
        <strong>{profile.message}</strong>
        <span>{profile.remediation}</span>
      </div>
      <code>model: {profile.profile_id}</code>
    </div>
  );
}

function RouteCard(props: {
  route: SentinelRoute;
  busy: boolean;
  onEvaluate: () => void;
}) {
  const { route } = props;
  const evaluation = route.evaluation;
  const canEvaluate = route.enabled && route.configured && route.zero_cost;
  const disabledReason = !route.enabled
    ? "Enable this route in Models first"
    : !route.zero_cost
      ? "Blocked by the hard $0 guard"
      : !route.configured
        ? `Add a ${route.provider_name} API key in Settings`
        : "";
  return (
    <article className={`sentinel-route-card ${evaluation?.readiness ?? "untested"}`}>
      <div className="route-score" aria-label={evaluation ? `Readiness score ${evaluation.score}` : "Not evaluated"}>
        <strong>{evaluation?.score ?? "—"}</strong>
        <small>{evaluation ? "/100" : "untested"}</small>
      </div>
      <div className="sentinel-route-main">
        <div className="sentinel-route-title">
          <div>
            <span>{route.provider_name}</span>
            <h3>{route.display_name}</h3>
            <code>{route.model_id}</code>
          </div>
          <span className={`trust-pill ${routeTrustTone(route)}`}>
            {routeTrustLabel(route)}
          </span>
        </div>
        <div className="route-meta">
          <span><CircleDollarSign size={14} /> {route.cost}</span>
          <span><Gauge size={14} /> {route.speed}</span>
          <span><Sparkles size={14} /> {route.quality}</span>
          {evaluation ? (
            <span><Radar size={14} /> tested {formatEvidenceAge(evaluation.completed_at)}</span>
          ) : null}
        </div>
        {evaluation ? (
          <details className="evidence-details">
            <summary>
              <span>{evaluation.summary}</span>
              <span>Inspect evidence</span>
            </summary>
            <div className="probe-grid">
              {evaluation.probes.map((probe) => (
                <div className={`probe-row ${probe.status}`} key={probe.check_id}>
                  <span className="probe-icon">
                    {probe.status === "pass" ? <Check size={14} /> : <X size={14} />}
                  </span>
                  <div>
                    <strong>{probe.label}</strong>
                    <span>{probe.evidence}</span>
                    {probe.remediation ? <small>{probe.remediation}</small> : null}
                  </div>
                  <span className="probe-score">+{probe.score}</span>
                </div>
              ))}
            </div>
          </details>
        ) : (
          <p className="route-remediation">
            {disabledReason || "Run the four probes to create inspectable readiness evidence."}
          </p>
        )}
      </div>
      <div className="sentinel-route-action">
        <button
          type="button"
          className="sentinel-evaluate"
          disabled={!canEvaluate || props.busy}
          title={disabledReason || "Run four short readiness probes"}
          onClick={props.onEvaluate}
        >
          {props.busy ? <LoaderCircle className="spin" size={16} /> : <FlaskConical size={16} />}
          {props.busy ? "Testing…" : evaluation ? "Run again" : "Evaluate"}
        </button>
        {!canEvaluate ? <small>{disabledReason}</small> : null}
      </div>
    </article>
  );
}

function OpenCodeSetup(props: {
  profileId: string;
  configJson: string;
  doctorUrl: string;
  onCopied: (message: string) => void;
}) {
  return (
    <section className="sentinel-section opencode-section" aria-labelledby="opencode-title">
      <div className="sentinel-section-heading">
        <div>
          <span className="section-kicker">OpenCode integration</span>
          <h2 id="opencode-title">Connect without a fork</h2>
        </div>
        <span className="opencode-mark"><Code2 size={18} /> OpenAI-compatible</span>
      </div>
      <div className="opencode-grid">
        <div className="opencode-steps">
          <div><span>1</span><p><strong>Get a green profile</strong>Run Sentinel until at least one route qualifies.</p></div>
          <div><span>2</span><p><strong>Add project config</strong>Save the generated JSON as <code>opencode.json</code>.</p></div>
          <div><span>3</span><p><strong>Start coding</strong>OpenCode uses <code>freerouter/{props.profileId}</code>.</p></div>
          <a href={props.doctorUrl} target="_blank" rel="noreferrer">
            <TerminalSquare size={16} /> Open doctor JSON
          </a>
        </div>
        <div className="opencode-code">
          <div className="code-toolbar">
            <span>opencode.json</span>
            <button
              type="button"
              onClick={() => {
                void copyText(props.configJson);
                props.onCopied("OpenCode config copied");
              }}
            >
              <Clipboard size={14} /> Copy
            </button>
          </div>
          <pre><code>{props.configJson}</code></pre>
        </div>
      </div>
    </section>
  );
}

function SentinelLoading() {
  return (
    <div className="sentinel-page" aria-busy="true" aria-label="Loading Sentinel">
      <div className="sentinel-skeleton hero" />
      <div className="sentinel-skeleton-grid">
        {[1, 2, 3, 4].map((item) => <div className="sentinel-skeleton metric" key={item} />)}
      </div>
      <div className="sentinel-skeleton panel" />
      <div className="sentinel-skeleton panel" />
    </div>
  );
}
