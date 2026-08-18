import { defineConfig } from 'vitest/config'

export default defineConfig({
  resolve: {
    tsconfigPaths: true,
  },
  test: {
    exclude: ['e2e/**', 'node_modules/**', '.next/**'],
  },
})
