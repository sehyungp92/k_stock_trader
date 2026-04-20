import 'server-only'

import { Pool } from 'pg'

declare global {
  // eslint-disable-next-line no-var
  var __dashboardPool: Pool | undefined
}

export function getDashboardPool(): Pool {
  if (!process.env.DATABASE_URL) {
    throw new Error('DATABASE_URL is required for the DB-backed dashboard')
  }

  if (!global.__dashboardPool) {
    global.__dashboardPool = new Pool({
      connectionString: process.env.DATABASE_URL,
      application_name: 'dashboard',
      connectionTimeoutMillis: 5_000,
      max: 4,
      idleTimeoutMillis: 10_000,
      query_timeout: 5_000,
    })
  }

  return global.__dashboardPool
}
