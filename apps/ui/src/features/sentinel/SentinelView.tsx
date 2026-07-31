import React from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Activity,
  AlertTriangle,
  BookOpen,
  Check,
  ChevronRight,
  CircleDollarSign,
  Clipboard,
  Code2,
  ExternalLink,
  FlaskConical,
  Gauge,
  LoaderCircle,
  LockKeyhole,
  Radar,
  Search,
  ServerCog,
  ShieldCheck,
  Sparkles,
  TerminalSquare,
  TimerReset,
  Wrench,
  X,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { fetchJson } from "../../api/client";
import { copyText } from "../../lib/format";
import {
  filterSentinelRoutes,
  formatEvidenceAge,
  receiptStatusLabel,
  receiptStatusTone,
  routeTrustLabel,
  routeTrustTone,
  type SentinelConsumer,
  type SentinelPreflight,
  type SentinelProfile,
  type SentinelReceipt,
  type SentinelRoute,
  type SentinelSnapshot,
} from "./sentinelLogic";
import "./sentinel.css";

type RouteFilter = "all" | "ready" | "needs-evidence" | "blocked";

export function SentinelView() {
  const queryClient = useQueryClient();
  const [profileId, setProfileId] = React.useState("safe-study");
  const [search, setSearch] = React.useState("");
  const [filter, setFilter] = React.useState<RouteFilter>("all");
  const [notice, setNotice] = React.useState<string | null>(null);
  const [preflightResults, setPreflightResults] = React.useState<Record<string, SentinelPreflight>>({});

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
    onError: (error) => setNotice(error instanceof Error ? error.message : "Evaluation failed"),
  });

  const preflight = useMutation({
    mutationFn: async (consumer: SentinelConsumer) => {
      const target = new URL(consumer.preflight_url);
      return fetchJson<SentinelPreflight>(`${target.pathname}${target.search}`);
    },
    onSuccess: (result, consumer) => {
      setPreflightResults((current) => ({ ...current, [consumer.consumer_id]: result }));
      setNotice(`${consumer.name} preflight: ${result.status}`);
    },
    onError: (error) => setNotice(error instanceof Error ? error.message : "Preflight failed"),
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
          <button type="button" onClick={() => void snapshot.refetch()}>Try again</button>
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
      <header className="sentinel-hero control-plane-hero">
        <div className="sentinel-eyebrow">
          <span className="sentinel-pulse" aria-hidden="true" />
          Shared AI runtime · contract v{data.contract_version}
        </div>
        <div className="sentinel-hero-row">
          <div>
            <h1>One trusted runtime. Two protected products.</h1>
            <p>
              Sentinel gives SemesterOS and AgentRange evidence-backed routing, deterministic
              policy gates, preflight health, and correlation receipts—without paid routes or
              frontend secrets.
            </p>
          </div>
          <div className="sentinel-guard" title={data.zero_cost_guard.message}>
            <CircleDollarSign size={19} />
            <span><strong>$0 guard</strong><small>Fail-closed</small></span>
          </div>
        </div>
      </header>

      {notice ? (
        <div className="sentinel-toast" role="status">
          {evaluation.isError || preflight.isError ? <AlertTriangle size={16} /> : <Check size={16} />}
          {notice}
        </div>
      ) : null}

      <section className="sentinel-metrics" aria-label="Sentinel overview">
        <Metric icon={ShieldCheck} label="Agent ready" value={data.summary.ready} note="current evidence" />
        <Metric icon={ServerCog} label="Consumers" value={data.consumers.length} note="shared runtime" />
        <Metric icon={Activity} label="Recent receipts" value={data.receipts.length} note="content-free traces" />
        <Metric icon={Gauge} label="Needs attention" value={data.summary.blocked} note="failed evidence" />
      </section>

      <ConsumerControlPlane
        consumers={data.consumers}
        results={preflightResults}
        pendingId={preflight.isPending ? preflight.variables?.consumer_id : undefined}
        onPreflight={(consumer) => preflight.mutate(consumer)}
        onCopy={(text, message) => {
          void copyText(text);
          setNotice(message);
        }}
        onSelectProfile={setProfileId}
      />

      <ReceiptLedger receipts={data.receipts} />

      <section className="sentinel-section" aria-labelledby="profiles-title">
        <div className="sentinel-section-heading">
          <div>
            <span className="section-kicker">Routing contracts</span>
            <h2 id="profiles-title">Evidence-backed virtual models</h2>
          </div>
          <span className="sentinel-helper">Consumers may fall back to auto; profiles never bypass policy</span>
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
          <div><span className="section-kicker">Evidence ledger</span><h2 id="routes-title">Route readiness</h2></div>
          <div className="sentinel-search">
            <Search size={16} aria-hidden="true" />
            <input aria-label="Search Sentinel routes" placeholder="Search routes" value={search} onChange={(event) => setSearch(event.target.value)} />
          </div>
        </div>
        <div className="sentinel-filter-row" role="group" aria-label="Filter routes">
          {(["all", "ready", "needs-evidence", "blocked"] as const).map((value) => (
            <button key={value} type="button" className={filter === value ? "active" : ""} onClick={() => setFilter(value)}>
              {value === "all" ? "All routes" : value === "ready" ? "Agent ready" : value === "needs-evidence" ? "Needs evidence" : "Blocked"}
            </button>
          ))}
        </div>
        <div className="sentinel-cost-note"><FlaskConical size={15} />Each evaluation sends four short deterministic prompts only to an explicitly free-tier route.</div>
        <div className="sentinel-route-list">
          {visibleRoutes.length ? visibleRoutes.map((route) => (
            <RouteCard key={route.route_id} route={route} busy={evaluation.isPending && evaluation.variables === route.route_id} onEvaluate={() => evaluation.mutate(route.route_id)} />
          )) : (
            <div className="sentinel-empty"><Search size={24} /><strong>No routes match this view</strong><span>Clear the search or choose another evidence filter.</span></div>
          )}
        </div>
      </section>

      <OpenCodeSetup
        profileId={profileId}
        configJson={data.opencode.config_json.replace("freerouter/safe-coding", `freerouter/${profileId}`)}
        doctorUrl={data.opencode.doctor_url.replace("safe-coding", profileId)}
        onCopied={setNotice}
      />
    </div>
  );
}

function ConsumerControlPlane(props: {
  consumers: SentinelConsumer[];
  results: Record<string, SentinelPreflight>;
  pendingId?: string;
  onPreflight: (consumer: SentinelConsumer) => void;
  onCopy: (text: string, message: string) => void;
  onSelectProfile: (profileId: string) => void;
}) {
  return (
    <section className="sentinel-section consumer-section" aria-labelledby="consumers-title">
      <div className="sentinel-section-heading">
        <div><span className="section-kicker">Connected products</span><h2 id="consumers-title">Consumer control plane</h2></div>
        <span className="sentinel-helper"><LockKeyhole size={13} /> Keys stay server-side</span>
      </div>
      <div className="consumer-grid">
        {props.consumers.map((consumer) => {
          const result = props.results[consumer.consumer_id];
          const state = result?.status ?? (consumer.readiness.status === "ready" ? "healthy" : "blocked");
          const Icon = consumer.consumer_id === "semesteros" ? BookOpen : Wrench;
          return (
            <article className={`consumer-card ${consumer.accent}`} key={consumer.consumer_id}>
              <div className="consumer-card-head">
                <span className="consumer-icon"><Icon size={21} /></span>
                <div><span>{consumer.product}</span><h3>{consumer.name}</h3><code>{consumer.profile_id}</code></div>
                <span className={`consumer-health ${state}`}><span />{state}</span>
              </div>
              <p>{consumer.description}</p>
              <div className="contract-chips">
                <span><Check size={12} /> JSON</span><span><Check size={12} /> Stream</span>
                <span><LockKeyhole size={12} /> {consumer.contract.tool_policy}</span>
              </div>
              <dl className="consumer-stats">
                <div><dt>Qualified</dt><dd>{consumer.readiness.counts.qualified}</dd></div>
                <div><dt>Timeout</dt><dd>{consumer.contract.timeout_seconds}s</dd></div>
                <div><dt>Retries</dt><dd>{consumer.contract.max_retries}</dd></div>
                <div><dt>Fallback</dt><dd>{consumer.contract.fallback_model}</dd></div>
              </dl>
              {result ? <div className={`preflight-result ${result.status}`} role="status"><strong>{result.status}</strong><span>{result.reason}</span></div> : null}
              <div className="consumer-config"><span>Server environment</span><pre><code>{consumer.env}</code></pre></div>
              <div className="consumer-actions">
                <button type="button" onClick={() => props.onPreflight(consumer)} disabled={props.pendingId === consumer.consumer_id}>
                  {props.pendingId === consumer.consumer_id ? <LoaderCircle className="spin" size={15} /> : <Activity size={15} />} Run preflight
                </button>
                <button type="button" onClick={() => props.onCopy(consumer.env, `${consumer.name} config copied`)}><Clipboard size={15} /> Copy config</button>
                <button type="button" className="text-action" onClick={() => props.onSelectProfile(consumer.profile_id)}>Inspect profile <ChevronRight size={14} /></button>
              </div>
            </article>
          );
        })}
      </div>
    </section>
  );
}

function ReceiptLedger({ receipts }: { receipts: SentinelReceipt[] }) {
  return (
    <section className="sentinel-section receipts-section" aria-labelledby="receipts-title">
      <div className="sentinel-section-heading">
        <div><span className="section-kicker">AI receipts</span><h2 id="receipts-title">Recent trusted runs</h2></div>
        <span className="sentinel-helper">Metadata only · no prompts, responses, or keys</span>
      </div>
      {receipts.length ? <div className="receipt-list">
        {receipts.map((receipt) => (
          <article className="receipt-row" key={receipt.run_id}>
            <span className={`receipt-state ${receiptStatusTone(receipt)}`}>{receipt.status === "healthy" ? <Check size={14} /> : receipt.status === "degraded" ? <TimerReset size={14} /> : <X size={14} />}</span>
            <div className="receipt-primary"><strong>{receipt.consumer_id === "semesteros" ? "SemesterOS" : receipt.consumer_id === "agentrange" ? "AgentRange" : "Gateway"}</strong><code>{receipt.run_id}</code></div>
            <div className="receipt-route"><strong>{receipt.route_id || "Policy gate"}</strong><span>{receipt.provider_name && receipt.model_id ? `${receipt.provider_name} · ${receipt.model_id}` : receipt.fallback_reason || "No route selected"}</span></div>
            <div className="receipt-meta"><span className={`trust-pill ${receiptStatusTone(receipt)}`}>{receiptStatusLabel(receipt)}</span><small>{receipt.latency_ms}ms · {receipt.attempts} attempt{receipt.attempts === 1 ? "" : "s"}</small><small>{formatEvidenceAge(receipt.created_at)}</small></div>
          </article>
        ))}
      </div> : <div className="sentinel-empty"><Activity size={24} /><strong>No consumer receipts yet</strong><span>Run preflight or send a profile request to create safe metadata.</span></div>}
    </section>
  );
}

function Metric(props: { icon: LucideIcon; label: string; value: number; note: string }) {
  return <article className="sentinel-metric"><span className="sentinel-metric-icon"><props.icon size={18} /></span><span>{props.label}</span><strong>{props.value}</strong><small>{props.note}</small></article>;
}

function ProfileCard(props: { profile: SentinelProfile; selected: boolean; onSelect: () => void }) {
  const { profile } = props;
  return <button type="button" className={`sentinel-profile ${props.selected ? "selected" : ""}`} onClick={props.onSelect} aria-pressed={props.selected}>
    <span className={`profile-status ${profile.status}`} aria-hidden="true">{profile.status === "ready" ? <Check size={15} /> : <Sparkles size={15} />}</span>
    <span className="profile-copy"><strong>{profile.name}</strong><code>{profile.profile_id}</code><small>{profile.description}</small></span>
    <span className="profile-count"><strong>{profile.counts.qualified}</strong><small>qualified</small></span><ChevronRight size={18} className="profile-arrow" />
  </button>;
}

function ProfileDiagnostic({ profile }: { profile: SentinelProfile }) {
  return <div className={`sentinel-diagnostic ${profile.status}`} role="status"><span className="diagnostic-icon">{profile.status === "ready" ? <ShieldCheck size={20} /> : <AlertTriangle size={20} />}</span><div><strong>{profile.message}</strong><span>{profile.remediation}</span></div><code>model: {profile.profile_id}</code></div>;
}

function RouteCard(props: { route: SentinelRoute; busy: boolean; onEvaluate: () => void }) {
  const { route } = props;
  const evaluation = route.evaluation;
  const canEvaluate = route.enabled && route.configured && route.zero_cost;
  const disabledReason = !route.enabled ? "Enable this route in Models first" : !route.zero_cost ? "Blocked by the hard $0 guard" : !route.configured ? `Add a ${route.provider_name} API key in Settings` : "";
  return <article className={`sentinel-route-card ${evaluation?.readiness ?? "untested"}`}>
    <div className="route-score" aria-label={evaluation ? `Readiness score ${evaluation.score}` : "Not evaluated"}><strong>{evaluation?.score ?? "—"}</strong><small>{evaluation ? "/100" : "untested"}</small></div>
    <div className="sentinel-route-main"><div className="sentinel-route-title"><div><span>{route.provider_name}</span><h3>{route.display_name}</h3><code>{route.model_id}</code></div><span className={`trust-pill ${routeTrustTone(route)}`}>{routeTrustLabel(route)}</span></div>
      <div className="route-meta"><span><CircleDollarSign size={14} /> {route.cost}</span><span><Gauge size={14} /> {route.speed}</span><span><Sparkles size={14} /> {route.quality}</span>{evaluation ? <span><Radar size={14} /> tested {formatEvidenceAge(evaluation.completed_at)}</span> : null}</div>
      {evaluation ? <details className="evidence-details"><summary><span>{evaluation.summary}</span><span>Inspect evidence</span></summary><div className="probe-grid">{evaluation.probes.map((probe) => <div className={`probe-row ${probe.status}`} key={probe.check_id}><span className="probe-icon">{probe.status === "pass" ? <Check size={14} /> : <X size={14} />}</span><div><strong>{probe.label}</strong><span>{probe.evidence}</span>{probe.remediation ? <small>{probe.remediation}</small> : null}</div><span className="probe-score">+{probe.score}</span></div>)}</div></details> : <p className="route-remediation">{disabledReason || "Run the four probes to create inspectable readiness evidence."}</p>}
    </div>
    <div className="sentinel-route-action"><button type="button" className="sentinel-evaluate" disabled={!canEvaluate || props.busy} title={disabledReason || "Run four short readiness probes"} onClick={props.onEvaluate}>{props.busy ? <LoaderCircle className="spin" size={16} /> : <FlaskConical size={16} />}{props.busy ? "Testing…" : evaluation ? "Run again" : "Evaluate"}</button>{!canEvaluate ? <small>{disabledReason}</small> : null}</div>
  </article>;
}

function OpenCodeSetup(props: { profileId: string; configJson: string; doctorUrl: string; onCopied: (message: string) => void }) {
  return <section className="sentinel-section opencode-section" aria-labelledby="opencode-title"><div className="sentinel-section-heading"><div><span className="section-kicker">OpenCode integration</span><h2 id="opencode-title">Connect without a fork</h2></div><span className="opencode-mark"><Code2 size={18} /> OpenAI-compatible</span></div><div className="opencode-grid"><div className="opencode-steps"><div><span>1</span><p><strong>Get a green profile</strong>Run Sentinel until at least one route qualifies.</p></div><div><span>2</span><p><strong>Add project config</strong>Save the generated JSON as <code>opencode.json</code>.</p></div><div><span>3</span><p><strong>Start coding</strong>Use <code>freerouter/{props.profileId}</code>.</p></div><a href={props.doctorUrl} target="_blank" rel="noreferrer"><ExternalLink size={16} /> Open doctor JSON</a></div><div className="opencode-code"><div className="code-toolbar"><span>opencode.json</span><button type="button" onClick={() => { void copyText(props.configJson); props.onCopied("OpenCode config copied"); }}><Clipboard size={14} /> Copy</button></div><pre><code>{props.configJson}</code></pre></div></div></section>;
}

function SentinelLoading() {
  return <div className="sentinel-page" aria-busy="true" aria-label="Loading Sentinel"><div className="sentinel-skeleton hero" /><div className="sentinel-skeleton-grid">{[1, 2, 3, 4].map((item) => <div className="sentinel-skeleton metric" key={item} />)}</div><div className="sentinel-skeleton panel" /><div className="sentinel-skeleton panel" /></div>;
}
