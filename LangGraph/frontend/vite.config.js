import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // FastAPI 백엔드로 프록시
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8800",
        changeOrigin: true,
      },
    },
  },
});
