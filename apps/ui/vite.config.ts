import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const gatewayPort = process.env.GATEWAY_PORT || "8000";
const gatewayTarget = process.env.FREEROUTER_API_TARGET || `http://127.0.0.1:${gatewayPort}`;

export default defineConfig({
  base: "/app-next/",
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      "/v1": gatewayTarget,
      "/internal": gatewayTarget
    }
  },
  preview: {
    port: 4173,
    strictPort: true
  }
});
