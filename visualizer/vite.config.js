import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    port: 5173,
    open: true,
    proxy: {
      '/live': { target: 'http://localhost:8765', changeOrigin: true }
    }
  },
  build: {
    outDir: 'dist'
  }
});
