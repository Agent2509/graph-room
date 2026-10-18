import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Static export: the FastAPI backend serves the built files. No Node runtime in prod.
  output: "export",
  trailingSlash: true,
  images: { unoptimized: true },
};

export default nextConfig;
