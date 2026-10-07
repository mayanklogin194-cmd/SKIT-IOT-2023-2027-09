import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In dev, forward API calls to the Flask server on :5000.
const api = "http://localhost:5000";
export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/dashboard": api, "/predict": api, "/health": api } },
});
