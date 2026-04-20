import { NextResponse } from 'next/server'

import { getDashboardPool } from '@/lib/db'
import type {
  DashboardData,
  DashboardOmsRow,
  DashboardPositionRow,
  DashboardServiceRow,
  DashboardStatus,
  DashboardSummary,
} from '@/lib/types'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'

type RawOmsStateRow = {
  oms_id: string
  safe_mode: boolean
  halt_new_entries: boolean
  flatten_in_progress: boolean
  equity_krw: string | number | null
  buyable_cash_krw: string | number | null
  daily_pnl_krw: string | number | null
  daily_pnl_pct: string | number | null
  kis_connected: boolean | null
  recon_status: string | null
  last_heartbeat_ts: string | null
  version: string | null
}

type RawRiskRow = {
  oms_id: string
  positions_count: string | number | null
}

type RawServiceRow = {
  service: string
  oms_id: string
  instance: string
  health: string
  seconds_since_heartbeat: string | number | null
  safe_mode: boolean | null
  kis_connected: boolean | null
  recon_status: string | null
  version: string | null
}

type RawPositionRow = {
  oms_id: string
  symbol: string
  strategy_id: string
  qty: string | number
  avg_price: string | number | null
  entry_ts: string | null
  soft_stop_px: string | number | null
  hard_stop_px: string | number | null
  frozen: boolean
  drift: string | number | null
}

const STATUS_RANK: Record<DashboardStatus, number> = {
  ok: 0,
  warn: 1,
  degraded: 2,
  error: 3,
}

function toNumber(value: string | number | null | undefined): number {
  if (value === null || value === undefined || value === '') {
    return 0
  }
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : 0
}

function combineStatuses(statuses: DashboardStatus[]): DashboardStatus {
  return statuses.reduce<DashboardStatus>((worst, current) => {
    return STATUS_RANK[current] > STATUS_RANK[worst] ? current : worst
  }, 'ok')
}

function healthToStatus(health: string | null | undefined): DashboardStatus {
  const normalized = String(health ?? '').toUpperCase()
  if (!normalized || normalized === 'UNKNOWN') {
    return 'degraded'
  }
  if (normalized === 'CRITICAL') {
    return 'error'
  }
  if (normalized === 'WARNING' || normalized === 'STOPPED') {
    return 'warn'
  }
  return normalized === 'HEALTHY' ? 'ok' : 'degraded'
}

function reconToStatus(reconStatus: string | null | undefined): DashboardStatus {
  const normalized = String(reconStatus ?? '').toLowerCase()
  if (!normalized || normalized === 'ok') {
    return 'ok'
  }
  if (normalized.includes('dead')) {
    return 'error'
  }
  if (normalized.includes('warn') || normalized.includes('persist_fail')) {
    return 'warn'
  }
  return 'degraded'
}

export async function GET() {
  try {
    const pool = getDashboardPool()
    const [omsStateRes, riskRes, serviceRes, positionsRes] = await Promise.all([
      pool.query<RawOmsStateRow>(
        `
        SELECT
          oms_id,
          safe_mode,
          halt_new_entries,
          flatten_in_progress,
          equity_krw,
          buyable_cash_krw,
          daily_pnl_krw,
          daily_pnl_pct,
          kis_connected,
          recon_status,
          last_heartbeat_ts,
          version
        FROM oms_state
        ORDER BY oms_id
        `
      ),
      pool.query<RawRiskRow>(
        `
        SELECT
          oms_id,
          positions_count
        FROM v_today_risk
        WHERE entity = 'PORTFOLIO'
        ORDER BY oms_id
        `
      ),
      pool.query<RawServiceRow>(
        `
        SELECT
          service,
          oms_id,
          instance,
          health,
          seconds_since_heartbeat,
          safe_mode,
          kis_connected,
          recon_status,
          version
        FROM v_service_health
        ORDER BY service, oms_id, instance
        `
      ),
      pool.query<RawPositionRow>(
        `
        SELECT
          oms_id,
          symbol,
          strategy_id,
          qty,
          avg_price,
          entry_ts,
          soft_stop_px,
          hard_stop_px,
          frozen,
          drift
        FROM v_live_allocations
        ORDER BY oms_id, symbol, strategy_id
        `
      ),
    ])

    const riskByOms = new Map(
      riskRes.rows.map((row) => [row.oms_id, toNumber(row.positions_count)])
    )
    const omsServiceByOms = new Map(
      serviceRes.rows
        .filter((row) => row.service === 'OMS')
        .map((row) => [row.oms_id, row])
    )

    const oms: DashboardOmsRow[] = omsStateRes.rows.map((row) => {
      const serviceRow = omsServiceByOms.get(row.oms_id)
      const status = combineStatuses([
        serviceRow ? healthToStatus(serviceRow.health) : 'degraded',
        reconToStatus(row.recon_status),
        row.kis_connected === false ? 'degraded' : 'ok',
        row.safe_mode || row.halt_new_entries || row.flatten_in_progress ? 'warn' : 'ok',
      ])

      return {
        oms_id: row.oms_id,
        status,
        service_health: serviceRow?.health ?? 'UNKNOWN',
        recon_status: row.recon_status,
        safe_mode: Boolean(row.safe_mode),
        halt_new_entries: Boolean(row.halt_new_entries),
        flatten_in_progress: Boolean(row.flatten_in_progress),
        kis_connected: row.kis_connected !== false,
        equity: toNumber(row.equity_krw),
        buyable_cash: toNumber(row.buyable_cash_krw),
        daily_pnl: toNumber(row.daily_pnl_krw),
        daily_pnl_pct: toNumber(row.daily_pnl_pct),
        tracked_positions: riskByOms.get(row.oms_id) ?? 0,
        last_heartbeat_ts: row.last_heartbeat_ts,
        version: row.version,
      }
    })

    const services: DashboardServiceRow[] = serviceRes.rows.map((row) => ({
      service: row.service,
      oms_id: row.oms_id,
      instance: row.instance,
      health: row.health,
      status: combineStatuses([
        healthToStatus(row.health),
        Boolean(row.safe_mode) ? 'warn' : 'ok',
        row.kis_connected === false ? 'degraded' : 'ok',
        reconToStatus(row.recon_status),
      ]),
      seconds_since_heartbeat:
        row.seconds_since_heartbeat === null ? null : toNumber(row.seconds_since_heartbeat),
      safe_mode: Boolean(row.safe_mode),
      kis_connected: row.kis_connected !== false,
      recon_status: row.recon_status,
      version: row.version,
    }))

    const positions: DashboardPositionRow[] = positionsRes.rows.map((row) => ({
      oms_id: row.oms_id,
      symbol: row.symbol,
      strategy_id: row.strategy_id,
      qty: toNumber(row.qty),
      avg_price: toNumber(row.avg_price),
      entry_ts: row.entry_ts,
      soft_stop_px: row.soft_stop_px === null ? null : toNumber(row.soft_stop_px),
      hard_stop_px: row.hard_stop_px === null ? null : toNumber(row.hard_stop_px),
      frozen: Boolean(row.frozen),
      drift: toNumber(row.drift),
    }))

    const summary: DashboardSummary = {
      status:
        oms.length === 0 && services.length === 0
          ? 'error'
          : combineStatuses([
              ...oms.map((row) => row.status),
              ...services.map((row) => row.status),
            ]),
      total_equity: oms.reduce((total, row) => total + row.equity, 0),
      total_cash: oms.reduce((total, row) => total + row.buyable_cash, 0),
      total_daily_pnl: oms.reduce((total, row) => total + row.daily_pnl, 0),
      total_daily_pnl_pct: 0,
      total_open_allocations: positions.length,
      oms_count: oms.length,
    }
    summary.total_daily_pnl_pct =
      summary.total_equity > 0 ? summary.total_daily_pnl / summary.total_equity : 0

    const data: DashboardData = {
      summary,
      oms,
      services,
      positions,
      is_paper: process.env.KIS_IS_PAPER !== 'false',
      fetchedAt: new Date().toISOString(),
    }

    return NextResponse.json(data)
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Failed to load dashboard data'
    return NextResponse.json({ error: message }, { status: 500 })
  }
}
