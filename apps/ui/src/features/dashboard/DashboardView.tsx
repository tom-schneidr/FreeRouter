import type { GatewayHealth, ProviderStatus } from "../../api/types";
import { Metric, Panel, ProviderTable } from "../../components/ui";

export function DashboardView(props: {
  health: GatewayHealth | null;
  configuredProviders: number;
  availableProviders: number;
  enabledRoutes: number;
  routeCount: number;
  flaggedRoutes: number;
  providers: ProviderStatus[];
}) {
  return (
    <div className="section-stack">
      <div className="section-heading">
        <div>
          <h1>Dashboard</h1>
          <p>Local gateway status, provider readiness, and route health in one place.</p>
        </div>
      </div>
      <div className="metrics">
        <Metric
          label="Gateway"
          value={props.health?.status ?? "Unknown"}
          note={props.health?.database_path ?? "Waiting for server"}
        />
        <Metric
          label="Providers"
          value={`${props.configuredProviders}/${props.providers.length}`}
          note={`${props.availableProviders} available right now`}
        />
        <Metric
          label="Enabled routes"
          value={props.enabledRoutes}
          note={`${props.routeCount} routes in catalog`}
        />
        <Metric
          label="Route flags"
          value={props.flaggedRoutes}
          note={props.flaggedRoutes ? "Review route health" : "No active route flags"}
        />
      </div>
      <Panel title="Provider readiness">
        <ProviderTable providers={props.providers} />
      </Panel>
    </div>
  );
}
