export interface HealthResponse {
  status: 'ok' | 'warn' | 'degraded' | 'error'
  uptime_sec: number
  positions_count: number
  kis_circuit_breaker: string
  recon_status: string
}

export interface AccountState {
  equity: number
  buyable_cash: number
  daily_pnl: number
  daily_pnl_pct: number
  safe_mode: boolean
  halt_new_entries: boolean
  flatten_in_progress: boolean
}

export interface StrategyAllocation {
  strategy_id: string
  qty: number
  cost_basis: number
  entry_ts: string | null
  soft_stop_px: number | null
  time_stop_ts: string | null
}

export interface PositionInfo {
  real_qty: number
  avg_price: number
  allocations: Record<string, StrategyAllocation>
  hard_stop_px: number | null
  entry_lock_owner: string | null
  frozen: boolean
  working_order_count: number
}

export interface DashboardData {
  health: HealthResponse | null
  account: AccountState | null
  positions: Record<string, PositionInfo> | null
  is_paper: boolean
  fetchedAt: string
}
