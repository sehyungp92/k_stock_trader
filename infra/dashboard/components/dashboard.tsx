'use client'

import { useEffect, useState } from 'react'

import {
  Area,
  AreaChart,
  ResponsiveContainer,
  Tooltip,
  YAxis,
} from 'recharts'
import useSWR from 'swr'

import type {
  DashboardData,
  DashboardOmsRow,
  DashboardPositionRow,
  DashboardServiceRow,
  DashboardStatus,
} from '@/lib/types'
import { cn, formatKRW } from '@/lib/utils'

const STRATEGIES = ['PCIM'] as const
type Strategy = (typeof STRATEGIES)[number]

const STRATEGY_BORDER: Record<Strategy, string> = {
  PCIM: 'border-emerald-500/30',
}

const STRATEGY_TEXT: Record<Strategy, string> = {
  PCIM: 'text-emerald-400',
}

const MAX_PNL_POINTS = 60

async function fetcher(url: string): Promise<DashboardData> {
  const response = await fetch(url)
  if (!response.ok) {
    throw new Error('Failed to load dashboard data')
  }
  return response.json()
}

function formatPercent(value: number): string {
  const sign = value >= 0 ? '+' : ''
  return `${sign}${(value * 100).toFixed(2)}%`
}

function formatAge(seconds: number | null): string {
  if (seconds === null || seconds < 0) {
    return '--'
  }
  if (seconds < 60) {
    return `${seconds}s`
  }
  if (seconds < 3600) {
    return `${Math.floor(seconds / 60)}m`
  }
  return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`
}

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
    const timer = setInterval(tick, 1000)
    return () => clearInterval(timer)
  }, [])

  return <span className="font-mono text-sm text-zinc-400">{time} KST</span>
}

function StatusBadge({ status }: { status: DashboardStatus }) {
  const colors: Record<DashboardStatus, string> = {
    ok: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
    warn: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
    degraded: 'bg-orange-500/20 text-orange-400 border-orange-500/30',
    error: 'bg-red-500/20 text-red-400 border-red-500/30',
  }

  return (
    <span
      className={cn(
        'inline-flex items-center rounded border px-2 py-0.5 text-xs font-semibold uppercase',
        colors[status]
      )}
    >
      {status}
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
    <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4">
      <p className="text-xs uppercase tracking-wider text-zinc-500">{label}</p>
      <p className={cn('mt-1 truncate text-2xl font-bold', valueClassName)}>{value}</p>
      {sub ? <p className="mt-1 text-xs text-zinc-500">{sub}</p> : null}
    </div>
  )
}

function OmsCard({ row }: { row: DashboardOmsRow }) {
  const pnlPositive = row.daily_pnl >= 0
  const flags = [
    row.safe_mode ? 'SAFE MODE' : null,
    row.halt_new_entries ? 'HALT ENTRIES' : null,
    row.flatten_in_progress ? 'FLATTENING' : null,
    !row.kis_connected ? 'KIS DISCONNECTED' : null,
  ].filter(Boolean)

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4">
      <div className="mb-3 flex items-start justify-between gap-3">
        <div>
          <p className="text-xs uppercase tracking-wider text-zinc-500">OMS</p>
          <h2 className="font-mono text-lg font-semibold text-zinc-100">{row.oms_id}</h2>
        </div>
        <StatusBadge status={row.status} />
      </div>

      <div className="grid grid-cols-2 gap-3 text-sm">
        <div>
          <p className="text-zinc-500">Equity</p>
          <p className="font-mono text-zinc-200">{formatKRW(row.equity)}</p>
        </div>
        <div>
          <p className="text-zinc-500">Cash</p>
          <p className="font-mono text-zinc-200">{formatKRW(row.buyable_cash)}</p>
        </div>
        <div>
          <p className="text-zinc-500">Daily P&L</p>
          <p className={cn('font-mono', pnlPositive ? 'text-emerald-400' : 'text-red-400')}>
            {row.daily_pnl >= 0 ? '+' : ''}
            {formatKRW(row.daily_pnl)}
          </p>
        </div>
        <div>
          <p className="text-zinc-500">Tracked Positions</p>
          <p className="font-mono text-zinc-200">{row.tracked_positions}</p>
        </div>
      </div>

      <div className="mt-3 space-y-1 text-xs text-zinc-500">
        <p>Service Health: {row.service_health}</p>
        <p>Recon: {row.recon_status ?? 'ok'}</p>
        <p>Version: {row.version ?? 'unknown'}</p>
      </div>

      {flags.length > 0 ? (
        <div className="mt-3 flex flex-wrap gap-2">
          {flags.map((flag) => (
            <span
              key={flag}
              className="rounded border border-yellow-500/30 bg-yellow-500/10 px-2 py-1 text-xs font-medium text-yellow-400"
            >
              {flag}
            </span>
          ))}
        </div>
      ) : null}
    </div>
  )
}

function StrategyCard({
  strategy,
  positions,
}: {
  strategy: Strategy
  positions: DashboardPositionRow[]
}) {
  const myPositions = positions.filter((position) => position.strategy_id === strategy)
  const totalQty = myPositions.reduce((sum, position) => sum + position.qty, 0)

  return (
    <div className={cn('rounded-lg border bg-zinc-900 p-4', STRATEGY_BORDER[strategy])}>
      <div className="mb-3 flex items-center justify-between">
        <span className={cn('text-sm font-semibold', STRATEGY_TEXT[strategy])}>{strategy}</span>
        <span className="text-xs text-zinc-500">{myPositions.length} alloc</span>
      </div>

      <div className="space-y-1 text-xs">
        <div className="flex justify-between">
          <span className="text-zinc-500">Total Qty</span>
          <span className="font-mono text-zinc-300">{totalQty}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-zinc-500">OMS Spread</span>
          <span className="font-mono text-zinc-300">
            {new Set(myPositions.map((position) => position.oms_id)).size}
          </span>
        </div>
      </div>
    </div>
  )
}

function ServicesTable({ services }: { services: DashboardServiceRow[] }) {
  return (
    <div className="overflow-hidden rounded-lg border border-zinc-800 bg-zinc-900">
      <div className="border-b border-zinc-800 p-4">
        <p className="text-xs uppercase tracking-wider text-zinc-500">Service Health</p>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-zinc-800">
              <th className="p-3 text-left font-medium text-zinc-500">Service</th>
              <th className="p-3 text-left font-medium text-zinc-500">Instance</th>
              <th className="p-3 text-left font-medium text-zinc-500">OMS</th>
              <th className="p-3 text-left font-medium text-zinc-500">Status</th>
              <th className="p-3 text-right font-medium text-zinc-500">Heartbeat Age</th>
              <th className="p-3 text-left font-medium text-zinc-500">Details</th>
            </tr>
          </thead>
          <tbody>
            {services.map((row) => (
              <tr key={`${row.service}-${row.oms_id}-${row.instance}`} className="border-b border-zinc-800/50">
                <td className="p-3 text-zinc-300">{row.service}</td>
                <td className="p-3 font-mono text-zinc-200">{row.instance}</td>
                <td className="p-3 font-mono text-zinc-400">{row.oms_id}</td>
                <td className="p-3">
                  <div className="flex items-center gap-2">
                    <StatusBadge status={row.status} />
                    <span className="text-zinc-500">{row.health}</span>
                  </div>
                </td>
                <td className="p-3 text-right font-mono text-zinc-400">
                  {formatAge(row.seconds_since_heartbeat)}
                </td>
                <td className="p-3 text-zinc-500">
                  {row.recon_status ? `Recon ${row.recon_status}` : row.safe_mode ? 'Paused' : 'Normal'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function PositionsTable({ positions }: { positions: DashboardPositionRow[] }) {
  if (positions.length === 0) {
    return (
      <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4">
        <p className="mb-3 text-xs uppercase tracking-wider text-zinc-500">Live Allocations</p>
        <p className="text-sm text-zinc-600">No open allocations</p>
      </div>
    )
  }

  return (
    <div className="overflow-hidden rounded-lg border border-zinc-800 bg-zinc-900">
      <div className="border-b border-zinc-800 p-4">
        <p className="text-xs uppercase tracking-wider text-zinc-500">Live Allocations</p>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-zinc-800">
              <th className="p-3 text-left font-medium text-zinc-500">OMS</th>
              <th className="p-3 text-left font-medium text-zinc-500">Symbol</th>
              <th className="p-3 text-left font-medium text-zinc-500">Strategy</th>
              <th className="p-3 text-right font-medium text-zinc-500">Qty</th>
              <th className="p-3 text-right font-medium text-zinc-500">Avg Price</th>
              <th className="p-3 text-left font-medium text-zinc-500">Entry</th>
              <th className="p-3 text-right font-medium text-zinc-500">Soft Stop</th>
              <th className="p-3 text-right font-medium text-zinc-500">Hard Stop</th>
              <th className="p-3 text-right font-medium text-zinc-500">Drift</th>
            </tr>
          </thead>
          <tbody>
            {positions.map((row) => (
              <tr
                key={`${row.oms_id}-${row.symbol}-${row.strategy_id}`}
                className={cn('border-b border-zinc-800/50', row.frozen && 'bg-red-950/20')}
              >
                <td className="p-3 font-mono text-zinc-400">{row.oms_id}</td>
                <td className="p-3 font-mono text-zinc-200">{row.symbol}</td>
                <td className={cn('p-3 font-medium', STRATEGY_TEXT[row.strategy_id as Strategy] ?? 'text-zinc-300')}>
                  {row.strategy_id}
                </td>
                <td className="p-3 text-right text-zinc-300">{row.qty}</td>
                <td className="p-3 text-right font-mono text-zinc-300">{formatKRW(row.avg_price)}</td>
                <td className="p-3 text-zinc-500">
                  {row.entry_ts
                    ? new Date(row.entry_ts).toLocaleTimeString('en-GB', {
                        hour: '2-digit',
                        minute: '2-digit',
                        timeZone: 'Asia/Seoul',
                      })
                    : '--'}
                </td>
                <td className="p-3 text-right font-mono text-zinc-400">
                  {row.soft_stop_px === null ? '--' : formatKRW(row.soft_stop_px)}
                </td>
                <td className="p-3 text-right font-mono text-zinc-400">
                  {row.hard_stop_px === null ? '--' : formatKRW(row.hard_stop_px)}
                </td>
                <td className={cn('p-3 text-right font-mono', row.drift === 0 ? 'text-zinc-500' : 'text-yellow-400')}>
                  {row.drift}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

interface PnlPoint {
  time: string
  value: number
}

function PnlSparkline({ points, positive }: { points: PnlPoint[]; positive: boolean }) {
  const color = positive ? '#10b981' : '#ef4444'

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4">
      <p className="mb-3 text-xs uppercase tracking-wider text-zinc-500">Portfolio P&L History</p>
      <ResponsiveContainer width="100%" height={80}>
        <AreaChart data={points} margin={{ top: 5, right: 0, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="pnl-grad" x1="0" y1="0" x2="0" y2="1">
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
            formatter={(value: number) => [formatKRW(value), 'P&L']}
            labelFormatter={() => ''}
            labelStyle={{ color: '#a1a1aa' }}
          />
          <Area
            type="monotone"
            dataKey="value"
            stroke={color}
            fill="url(#pnl-grad)"
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}

export default function Dashboard() {
  const { data, error, isLoading } = useSWR<DashboardData>('/api/dashboard', fetcher, {
    refreshInterval: 10_000,
    revalidateOnFocus: false,
  })
  const [pnlHistory, setPnlHistory] = useState<PnlPoint[]>([])

  useEffect(() => {
    if (data === undefined) {
      return
    }

    setPnlHistory((previous) => {
      const last = previous[previous.length - 1]
      if (last && last.value === data.summary.total_daily_pnl) {
        return previous
      }

      return [
        ...previous.slice(-(MAX_PNL_POINTS - 1)),
        { time: data.fetchedAt, value: data.summary.total_daily_pnl },
      ]
    })
  }, [data])

  if (isLoading) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <p className="text-sm text-zinc-600 animate-pulse">Loading dashboard...</p>
      </div>
    )
  }

  if (error || !data) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <p className="text-sm text-red-400">Failed to load dashboard data</p>
      </div>
    )
  }

  const anyFlatten = data.oms.some((row) => row.flatten_in_progress)
  const anySafeMode = data.oms.some((row) => row.safe_mode)
  const anyHalt = data.oms.some((row) => row.halt_new_entries && !row.safe_mode)
  const pnlPositive = data.summary.total_daily_pnl >= 0

  return (
    <div className="mx-auto min-h-screen max-w-7xl space-y-4 p-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <StatusBadge status={data.summary.status} />
          <span
            className={cn(
              'inline-flex items-center rounded border px-2 py-0.5 text-xs font-semibold',
              data.is_paper
                ? 'border-yellow-500/30 bg-yellow-500/20 text-yellow-400'
                : 'border-emerald-500/30 bg-emerald-500/20 text-emerald-400'
            )}
          >
            {data.is_paper ? 'PAPER' : 'LIVE'}
          </span>
          <h1 className="text-lg font-bold text-zinc-100">Unified Trading Dashboard</h1>
        </div>
        <KSTClock />
      </div>

      {anyFlatten ? (
        <div className="rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-2 text-sm font-medium text-red-400">
          Flatten in progress on at least one OMS instance.
        </div>
      ) : null}
      {anySafeMode ? (
        <div className="rounded-lg border border-yellow-500/30 bg-yellow-500/10 px-4 py-2 text-sm font-medium text-yellow-400">
          Safe mode is active on at least one OMS instance.
        </div>
      ) : null}
      {anyHalt ? (
        <div className="rounded-lg border border-orange-500/30 bg-orange-500/10 px-4 py-2 text-sm font-medium text-orange-400">
          New entries are halted on at least one OMS instance.
        </div>
      ) : null}

      <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
        <StatCard label="Total Equity" value={formatKRW(data.summary.total_equity)} />
        <StatCard
          label="Total Daily P&L"
          value={`${data.summary.total_daily_pnl >= 0 ? '+' : ''}${formatKRW(data.summary.total_daily_pnl)}`}
          sub={formatPercent(data.summary.total_daily_pnl_pct)}
          valueClassName={pnlPositive ? 'text-emerald-400' : 'text-red-400'}
        />
        <StatCard label="Total Cash" value={formatKRW(data.summary.total_cash)} />
        <StatCard label="Open Allocations" value={String(data.summary.total_open_allocations)} />
        <StatCard label="OMS Instances" value={String(data.summary.oms_count)} />
      </div>

      {pnlHistory.length >= 2 ? <PnlSparkline points={pnlHistory} positive={pnlPositive} /> : null}

      <div className="grid gap-3 md:grid-cols-2">
        {data.oms.map((row) => (
          <OmsCard key={row.oms_id} row={row} />
        ))}
      </div>

      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        {STRATEGIES.map((strategy) => (
          <StrategyCard key={strategy} strategy={strategy} positions={data.positions} />
        ))}
      </div>

      <ServicesTable services={data.services} />
      <PositionsTable positions={data.positions} />

      <p className="pb-2 text-center text-xs text-zinc-700">
        Last updated:{' '}
        {new Date(data.fetchedAt).toLocaleTimeString('en-GB', {
          timeZone: 'Asia/Seoul',
          hour: '2-digit',
          minute: '2-digit',
          second: '2-digit',
        })}{' '}
        KST
      </p>
    </div>
  )
}
