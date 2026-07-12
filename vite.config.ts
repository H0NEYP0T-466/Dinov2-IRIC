import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // Proxy /api -> backend during dev so the frontend can call the FastAPI
    // server on :8000 without CORS friction. In production builds the
    // VITE_API_BASE env var (see src/services/api.ts) takes over.
    proxy: {
      '/api': {
        target: 'http://localhost:8011',
        changeOrigin: true,
      },
    },
  },
})
