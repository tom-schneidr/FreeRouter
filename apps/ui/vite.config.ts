import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: "/app-next/",
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      "/v1": "http://127.0.0.1:8000",
      "/internal": "http://127.0.0.1:8000"
    }
  },
  preview: {
    port: 4173,
    strictPort: true
  }
});
