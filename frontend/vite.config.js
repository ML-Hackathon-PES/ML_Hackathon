import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Laptop 2 sets API_TARGET to the API address on Laptop 1.
const apiTarget = process.env.API_TARGET || 'http://localhost:8000'

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
