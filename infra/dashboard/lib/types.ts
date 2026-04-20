export type DashboardStatus = 'ok' | 'warn' | 'degraded' | 'error'

export interface DashboardSummary {
  status: DashboardStatus
  total_equity: number
  total_cash: number
  total_daily_pnl: number
  total_daily_pnl_pct: number
  total_open_allocations: number
  oms_count: number
}

export interface DashboardOmsRow {
  oms_id: string
  status: DashboardStatus
  service_health: string
  recon_status: string | null
  safe_mode: boolean
  halt_new_entries: boolean
  flatten_in_progress: boolean
  kis_connected: boolean
  equity: number
  buyable_cash: number
  daily_pnl: number
  daily_pnl_pct: number
  tracked_positions: number
  last_heartbeat_ts: string | null
  version: string | null
}

export interface DashboardServiceRow {
  service: string
  oms_id: string
  instance: string
  health: string
  status: DashboardStatus
  seconds_since_heartbeat: number | null
  safe_mode: boolean
  kis_connected: boolean
  recon_status: string | null
  version: string | null
}

export interface DashboardPositionRow {
  oms_id: string
  symbol: string
  strategy_id: string
  qty: number
  avg_price: number
  entry_ts: string | null
  soft_stop_px: number | null
  hard_stop_px: number | null
  frozen: boolean
  drift: number
}

export interface DashboardData {
  summary: DashboardSummary
  oms: DashboardOmsRow[]
  services: DashboardServiceRow[]
  positions: DashboardPositionRow[]
  is_paper: boolean
  fetchedAt: string
}
