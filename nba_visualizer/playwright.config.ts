import { defineConfig, devices } from '@playwright/test'

export default defineConfig({
  testDir: './e2e',
  fullyParallel: false,
  retries: 0,
  reporter: 'list',
  use: {
    baseURL: 'http://localhost:3001',
    trace: 'retain-on-failure',
    ...devices['Desktop Chrome'],
  },
  webServer: [
    {
      command: '.venv/bin/uvicorn app.main:app --port 8000',
      cwd: './backend',
      url: 'http://127.0.0.1:8000/api/v1/health',
      reuseExistingServer: true,
    },
    {
      command: 'npm run dev',
      url: 'http://localhost:3001',
      reuseExistingServer: true,
    },
  ],
})
