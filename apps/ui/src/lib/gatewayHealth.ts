/** Poll until the gateway health endpoint responds or the deadline passes. */
export async function waitForGatewayHealth(timeoutMs = 45_000): Promise<boolean> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const response = await fetch("/v1/gateway/health.json", { cache: "no-store" });
      if (response.ok) {
        return true;
      }
    } catch {
      // backend still starting
    }
    await new Promise((resolve) => window.setTimeout(resolve, 500));
  }
  return false;
}
