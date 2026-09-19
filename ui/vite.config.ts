import { fileURLToPath, URL } from "node:url";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

/** Where a dev server should forward API calls. */
const BACKEND = process.env.FRINK_BACKEND ?? "http://127.0.0.1:8383";

/**
 * Every prefix `frink-server` owns.
 *
 * In dev these are proxied so the browser sees one origin and CORS never
 * enters the picture — no server-side configuration, no preflights, and
 * the `Authorization` header behaves exactly as it will in production.
 * A deployment that serves this app from a different origin instead sets
 * `FRINK_CORS_ORIGINS` on the server to that exact origin (the wildcard
 * is rejected on purpose: `*` plus a bearer token is a credential-leak
 * shape) and points the app at the API with its base-URL setting on the
 * Connect screen.
 */
const API_PREFIXES = ["/v1", "/admin", "/health", "/metrics", "/cache"];

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  server: {
    proxy: Object.fromEntries(
      API_PREFIXES.map((prefix) => [
        prefix,
        { target: BACKEND, changeOrigin: true },
      ]),
    ),
  },
  preview: {
    proxy: Object.fromEntries(
      API_PREFIXES.map((prefix) => [
        prefix,
        { target: BACKEND, changeOrigin: true },
      ]),
    ),
  },
  build: {
    // `ui/dist/` is gitignored: the frontend is a standalone app now, so
    // nothing built here is committed and nothing ships inside the Rust
    // crate. Serve `dist/` with any static file server.
    outDir: "dist",
    target: "es2022",
    sourcemap: false,
    chunkSizeWarningLimit: 900,
  },
});
