'use client'

import useSWR from 'swr'
import { useState, useEffect } from 'react'
import { AreaChart, Area, ResponsiveContainer, Tooltip, YAxis } from 'recharts'
import { cn, formatKRW, formatUptime } from '@/lib/utils'
import type { DashboardData, PositionInfo } from '@/lib/types'

const fetcher = (url: string) => fetch(url).then((r) => r.json())
const MAX_PNL_POINTS = 60

const STRATEGIES = ['KMP', 'NULRIMOK', 'KPR', 'PCIM'] as const
type Strategy = (typeof STRATEGIES)[number]

const STRATEGY_BORDER: Record<Strategy, string> = {
  KMP: 'border-blue-500/30',
  NULRIMOK: 'border-purple-500/30',
  KPR: 'border-orange-500/30',
  PCIM: 'border-emerald-500/30',
}

const STRATEGY_TEXT: Record<Strategy, string> = {
  KMP: 'text-blue-400',
  NULRIMOK: 'text-purple-400',
  KPR: 'text-orange-400',
  PCIM: 'text-emerald-400',
}

// ─── Sub-components ───────────────────────────────────────────────────────────

function KSTClock() {
  const [time, setTime] = useState('')
  useEffect(() => {
    const tick = () => {
      const kst = new Date().toLocaleString('en-US', { timeZone: 'Asia/Seoul' })
      setTime(
        new Date(kst).toLocaleTimeString('en-GB', {
          hour: '2-digit',
          minute: '2-digit',
          second: '2-digit',
          hour12: false,
        })
      )
    }
    tick()
    const id = setInterval(tick, 1000)
    return () => clearInterval(id)
  }, [])
  return <span className="font-mono text-zinc-400 text-sm">{time} KST</span>
}

function StatusBadge({ status }: { status: string }) {
  const colors: Record<string, string> = {
    ok: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
    warn: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
    degraded: 'bg-orange-500/20 text-orange-400 border-orange-500/30',
    error: 'bg-red-500/20 text-red-400 border-red-500/30',
  }
  return (
    <span
      className={cn(
        'inline-flex items-center px-2 py-0.5 rounded text-xs font-semibold border',
        colors[status] ?? colors.error
      )}
    >
      {status.toUpperCase()}
    </span>
  )
}

function StatCard({
  label,
  value,
  sub,
  valueClassName,
}: {
  label: string
  value: string
  sub?: string
  valueClassName?: string
}) {
  return (
    <div className="bg-zinc-900 rounded-lg p-4 border border-zinc-800">
      <p className="text-xs text-zinc-500 uppercase tracking-wider">{label}</p>
      <p className={cn('text-2xl font-bold mt-1 truncate', valueClassName)}>{value}</p>
      {sub && <p className="text-xs text-zinc-500 mt-1">{sub}</p>}
    </div>
  )
}

interface PnlPoint {
  time: string
  value: number
}

function PnlSparkline({ points, positive }: { points: PnlPoint[]; positive: boolean }) {
  const color = positive ? '#10b981' : '#ef4444'
  const gradientId = `pnl-grad`
  return (
    <div className="bg-zinc-900 rounded-lg p-4 border border-zinc-800">
      <p className="text-xs text-zinc-500 uppercase tracking-wider mb-3">P&L History (Session)</p>
      <ResponsiveContainer width="100%" height={80}>
        <AreaChart data={points} margin={{ top: 5, right: 0, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={color} stopOpacity={0.3} />
              <stop offset="95%" stopColor={color} stopOpacity={0} />
            </linearGradient>
          </defs>
          <YAxis domain={['auto', 'auto']} hide />
          <Tooltip
            contentStyle={{
              background: '#18181b',
              border: '1px solid #3f3f46',
              borderRadius: 6,
              fontSize: 12,
            }}
            formatter={(v: number) => [formatKRW(v), 'P&L']}
            labelFormatter={() => ''}
            labelStyle={{ color: '#a1a1aa' }}
          />
          <Area
            type="monotone"
            dataKey="value"
            stroke={color}
            fill={`url(#${gradientId})`}
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}

function StrategyCard({
  strategy,
  positions,
}: {
  strategy: Strategy
  positions: Record<string, PositionInfo> | null
}) {
  const myPositions = positions
    ? Object.entries(positions).filter(([, pos]) => strategy in pos.allocations)
    : []

  return (
    <div className={cn('bg-zinc-900 rounded-lg p-4 border', STRATEGY_BORDER[strategy])}>
      <div className="flex items-center justify-between mb-3">
        <span className={cn('font-semibold text-sm', STRATEGY_TEXT[strategy])}>{strategy}</span>
        <span className="text-xs text-zinc-500">{myPositions.length} pos</span>
      </div>
      {myPositions.length === 0 ? (
        <p className="text-xs text-zinc-600">—</p>
      ) : (
        <div className="space-y-1">
          {myPositions.map(([symbol, pos]) => {
            const alloc = pos.allocations[strategy]
            return (
              <div key={symbol} className="flex justify-between text-xs">
                <span className="font-mono text-zinc-300">{symbol}</span>
                <span className="text-zinc-500">{alloc.qty}</span>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

function PositionsTable({ positions }: { positions: Record<string, PositionInfo> | null }) {
  if (!positions || Object.keys(positions).length === 0) {
    return (
      <div className="bg-zinc-900 rounded-lg p-4 border border-zinc-800">
        <p className="text-xs text-zinc-500 uppercase tracking-wider mb-3">Positions</p>
        <p className="text-sm text-zinc-600">No open positions</p>
      </div>
    )
  }

  type Row = {
    symbol: string
    strategy: string
    qty: number
    avg_price: number
    entry_ts: string | null
    soft_stop_px: number | null
    hard_stop_px: number | null
    frozen: boolean
  }

  const rows: Row[] = []
  for (const [symbol, pos] of Object.entries(positions)) {
    for (const [stratId, alloc] of Object.entries(pos.allocations)) {
      rows.push({
        symbol,
        strategy: stratId,
        qty: alloc.qty,
        avg_price: pos.avg_price,
        entry_ts: alloc.entry_ts,
        soft_stop_px: alloc.soft_stop_px,
        hard_stop_px: pos.hard_stop_px,
        frozen: pos.frozen,
      })
    }
  }

  return (
    <div className="bg-zinc-900 rounded-lg border border-zinc-800 overflow-hidden">
      <div className="p-4 border-b border-zinc-800">
        <p className="text-xs text-zinc-500 uppercase tracking-wider">Positions</p>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-zinc-800">
              <th className="text-left p-3 text-zinc-500 font-medium">Symbol</th>
              <th className="text-left p-3 text-zinc-500 font-medium">Strategy</th>
              <th className="text-right p-3 text-zinc-500 font-medium">Qty</th>
              <th className="text-right p-3 text-zinc-500 font-medium">Avg Price</th>
              <th className="text-left p-3 text-zinc-500 font-medium">Entry</th>
              <th className="text-right p-3 text-zinc-500 font-medium">Soft Stop</th>
              <th className="text-right p-3 text-zinc-500 font-medium">Hard Stop</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr
                key={`${r.symbol}-${r.strategy}-${i}`}
                className={cn(
                  'border-b border-zinc-800/50 hover:bg-zinc-800/30 transition-colors',
                  r.frozen && 'bg-red-950/20'
                )}
              >
                <td className="p-3 font-mono text-zinc-200">
                  {r.symbol}
                  {r.frozen && <span className="ml-1 text-red-400 text-xs">⚠</span>}
                </td>
                <td className="p-3">
                  <span
                    className={cn(
                      'font-medium',
                      STRATEGY_TEXT[r.strategy as Strategy] ?? 'text-zinc-400'
                    )}
                  >
                    {r.strategy}
                  </span>
                </td>
                <td className="p-3 text-right text-zinc-300">{r.qty}</td>
                <td className="p-3 text-right font-mono text-zinc-300">
                  {formatKRW(r.avg_price)}
                </td>
                <td className="p-3 text-zinc-500">
                  {r.entry_ts
                    ? new Date(r.entry_ts).toLocaleTimeString('en-GB', {
                        hour: '2-digit',
                        minute: '2-digit',
                        timeZone: 'Asia/Seoul',
                      })
                    : '—'}
                </td>
                <td className="p-3 text-right font-mono text-zinc-400">
                  {r.soft_stop_px ? formatKRW(r.soft_stop_px) : '—'}
                </td>
                <td className="p-3 text-right font-mono text-zinc-400">
                  {r.hard_stop_px ? formatKRW(r.hard_stop_px) : '—'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// ─── Main Dashboard ───────────────────────────────────────────────────────────

export default function Dashboard() {
  const { data, error, isLoading } = useSWR<DashboardData>('/api/dashboard', fetcher, {
    refreshInterval: 10_000,
    revalidateOnFocus: false,
  })

  const [pnlHistory, setPnlHistory] = useState<PnlPoint[]>([])

  useEffect(() => {
    if (data?.account?.daily_pnl === undefined) return
    setPnlHistory((prev) => {
      const last = prev.at(-1)
      if (last && last.value === data.account!.daily_pnl) return prev
      return [
        ...prev.slice(-(MAX_PNL_POINTS - 1)),
        { time: data.fetchedAt, value: data.account!.daily_pnl },
      ]
    })
  }, [data])

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <p className="text-zinc-600 text-sm animate-pulse">Loading dashboard…</p>
      </div>
    )
  }

  if (error || !data) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <p className="text-red-400 text-sm">Failed to load dashboard data</p>
      </div>
    )
  }

  const { health, account, positions, is_paper } = data
  const pnlPositive = (account?.daily_pnl ?? 0) >= 0
  const openCount = positions ? Object.keys(positions).length : 0

  return (
    <div className="min-h-screen p-4 max-w-7xl mx-auto space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          {health ? <StatusBadge status={health.status} /> : <StatusBadge status="error" />}
          {is_paper ? (
            <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-semibold border bg-yellow-500/20 text-yellow-400 border-yellow-500/30">
              PAPER
            </span>
          ) : (
            <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-semibold border bg-emerald-500/20 text-emerald-400 border-emerald-500/30">
              LIVE
            </span>
          )}
          <h1 className="text-lg font-bold text-zinc-100">Trading Dashboard</h1>
        </div>
        <div className="flex items-center gap-4">
          <KSTClock />
          {health && (
            <span className="text-zinc-600 text-xs hidden sm:inline">
              up {formatUptime(health.uptime_sec)}
            </span>
          )}
        </div>
      </div>

      {/* Alert Banners */}
      {account?.flatten_in_progress && (
        <div className="px-4 py-2 rounded-lg bg-red-500/10 border border-red-500/30 text-red-400 text-sm font-medium">
          FLATTEN IN PROGRESS — Liquidating all positions
        </div>
      )}
      {account?.safe_mode && (
        <div className="px-4 py-2 rounded-lg bg-yellow-500/10 border border-yellow-500/30 text-yellow-400 text-sm font-medium">
          SAFE MODE ACTIVE — New entries suspended
        </div>
      )}
      {account?.halt_new_entries && !account.safe_mode && (
        <div className="px-4 py-2 rounded-lg bg-orange-500/10 border border-orange-500/30 text-orange-400 text-sm font-medium">
          HALT NEW ENTRIES
        </div>
      )}

      {/* Stat Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <StatCard label="Equity" value={account ? formatKRW(account.equity) : '—'} />
        <StatCard
          label="Daily P&L"
          value={account ? `${account.daily_pnl >= 0 ? '+' : ''}${formatKRW(account.daily_pnl)}` : '—'}
          sub={
            account
              ? `${account.daily_pnl_pct >= 0 ? '+' : ''}${(account.daily_pnl_pct * 100).toFixed(2)}%`
              : undefined
          }
          valueClassName={account ? (pnlPositive ? 'text-emerald-400' : 'text-red-400') : ''}
        />
        <StatCard label="Cash" value={account ? formatKRW(account.buyable_cash) : '—'} />
        <StatCard
          label="Open Positions"
          value={String(openCount)}
          sub={health ? `OMS: ${health.positions_count} tracked` : undefined}
        />
      </div>

      {/* P&L Sparkline — only once we have at least 2 points */}
      {pnlHistory.length >= 2 && (
        <PnlSparkline points={pnlHistory} positive={pnlPositive} />
      )}

      {/* Strategy Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {STRATEGIES.map((s) => (
          <StrategyCard key={s} strategy={s} positions={positions} />
        ))}
      </div>

      {/* Positions Table */}
      <PositionsTable positions={positions} />

      {/* Circuit Breaker / Recon Alerts */}
      {health?.kis_circuit_breaker && health.kis_circuit_breaker !== 'NORMAL' && (
        <div className="px-4 py-2 rounded-lg bg-red-500/10 border border-red-500/30 text-red-400 text-sm">
          KIS Circuit Breaker: <span className="font-semibold">{health.kis_circuit_breaker}</span>
        </div>
      )}
      {health?.recon_status && health.recon_status !== 'ok' && (
        <div className="px-4 py-2 rounded-lg bg-yellow-500/10 border border-yellow-500/30 text-yellow-400 text-sm">
          Reconciliation Alert: <span className="font-semibold">{health.recon_status}</span>
        </div>
      )}

      {/* Footer */}
      <p className="text-center text-xs text-zinc-700 pb-2">
        Last updated:{' '}
        {new Date(data.fetchedAt).toLocaleTimeString('en-GB', {
          timeZone: 'Asia/Seoul',
          hour: '2-digit',
          minute: '2-digit',
          second: '2-digit',
        })}{' '}
        KST · refreshes every 10s
      </p>
    </div>
  )
}
