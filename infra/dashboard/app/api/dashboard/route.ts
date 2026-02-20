import { NextResponse } from 'next/server'
import type { HealthResponse, AccountState, PositionInfo, DashboardData } from '@/lib/types'

const OMS_URL = process.env.OMS_URL ?? 'http://localhost:8000'
const TIMEOUT_MS = 5000

async function omsGet<T>(path: string): Promise<T> {
  const res = await fetch(`${OMS_URL}${path}`, {
    signal: AbortSignal.timeout(TIMEOUT_MS),
    headers: { Accept: 'application/json' },
    cache: 'no-store',
  })
  if (!res.ok) throw new Error(`OMS ${path} returned ${res.status}`)
  return res.json() as Promise<T>
}

export async function GET() {
  const [health, account, positions] = await Promise.allSettled([
    omsGet<HealthResponse>('/health'),
    omsGet<AccountState>('/api/v1/state/account'),
    omsGet<Record<string, PositionInfo>>('/api/v1/positions'),
  ])

  const data: DashboardData = {
    health: health.status === 'fulfilled' ? health.value : null,
    account: account.status === 'fulfilled' ? account.value : null,
    positions: positions.status === 'fulfilled' ? positions.value : null,
    is_paper: process.env.KIS_IS_PAPER !== 'false',
    fetchedAt: new Date().toISOString(),
  }

  return NextResponse.json(data)
}
