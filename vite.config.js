import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// If you host the repo at https://<user>.github.io/final,
// the base should be set to '/final/'. Update if your repo name differs.
export default defineConfig({
  // Use relative base so built assets work from repository root or docs/
  base: './',
  plugins: [react()],
  build: {
    outDir: 'docs'
  }
})
