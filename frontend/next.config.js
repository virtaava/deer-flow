/**
 * Run `build` or `dev` with `SKIP_ENV_VALIDATION` to skip env validation. This is especially useful
 * for Docker builds.
 */
import "./src/env.js";

/** @type {import("next").NextConfig} */
const config = {
  devIndicators: false,
  async rewrites() {
    const LANGGRAPH = process.env.LANGGRAPH_URL || "http://127.0.0.1:2024";
    const GATEWAY = process.env.GATEWAY_URL || "http://127.0.0.1:8011";
    return [
      // LangGraph API — strip /api/langgraph prefix
      { source: "/api/langgraph/:path*", destination: `${LANGGRAPH}/:path*` },
      // Gateway API routes
      { source: "/api/models/:path*", destination: `${GATEWAY}/api/models/:path*` },
      { source: "/api/memory/:path*", destination: `${GATEWAY}/api/memory/:path*` },
      { source: "/api/mcp/:path*", destination: `${GATEWAY}/api/mcp/:path*` },
      { source: "/api/skills/:path*", destination: `${GATEWAY}/api/skills/:path*` },
      { source: "/api/agents/:path*", destination: `${GATEWAY}/api/agents/:path*` },
      { source: "/api/threads/:path*", destination: `${GATEWAY}/api/threads/:path*` },
      { source: "/health", destination: `${GATEWAY}/health` },
    ];
  },
};

export default config;
