import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The API runs on Laptop 1. Set API_TARGET to use a different address.
const apiTarget = process.env.API_TARGET || 'http://10.1.4.208:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: apiTarget,
        changeOrigin: true,
      },
    },
  },
})
