import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // Backend runs on :8000 (see README/run instructions) — proxied so the
    // frontend's same-origin fetch("/livekit/token") just works in dev.
    proxy: {
      "/livekit": "http://localhost:8000",
      "/health": "http://localhost:8000",
    },
  },
})
