import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

const currentDirectory = fileURLToPath(new URL(".", import.meta.url));

export default defineConfig({
  plugins: [react()],
  test: {
    environment: "jsdom",
    setupFiles: "./src/test/setup.ts",
    exclude: ["e2e/**", "node_modules/**"]
  },
  base: "/",
  build: {
    outDir: resolve(currentDirectory, "../src/salvi/web/static"),
    emptyOutDir: true,
    sourcemap: false
  },
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8765"
    }
  }
});
