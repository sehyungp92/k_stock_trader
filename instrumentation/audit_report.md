# Instrumentation Audit Report

## Bot Identity
- **Bot ID**: `k_stock_trader`
- **Strategy types**: Multi-strategy (KMP momentum breakout, KPR mean-reversion, PCIM event-driven influencer, NULRIMOK swing dip-buying)
- **Exchange(s)**: KRX (Korea Exchange) via KIS (Korea Investment & Securities) API
- **Pairs traded**: Korean equities (dynamic universe per strategy, e.g. Samsung, SK Hynix, etc.)
- **Architecture**: Hybrid (event-driven WS ticks + polling loops per strategy, all routing through centralized OMS)

## Entry Logic

### KMP — Intraday Momentum Breakout (v2.3.4)
- **Signal generation**: `strategy_kmp/core/fsm.py:210-216` `alpha_step()` — Opening range break above OR high + 1 tick, price > VWAP
- **Signal strength**: YES — Quality score 0-100 in `strategy_kmp/core/sizing.py:63-105` (surge, RVOL, tick imbalance, spread, acceptance, regime breadth, chop)
- **Filters**:
  - Regime gate: `strategy_kmp/main.py:452` / `fsm.py:93-130` — breadth >= min AND not chop
  - Risk-off: `strategy_kmp/main.py:449` — KOSPI drawdown ≤ -1.0%
  - Entry cutoff: `strategy_kmp/core/fsm.py:133-137` — after (10,30)
  - OR range: `strategy_kmp/core/fsm.py:141-147` — 1.2% ≤ range ≤ 7.0%
  - Spread gate: `strategy_kmp/core/fsm.py:202-206` — spread_pct > 0.4%
  - Surge decay: `strategy_kmp/core/fsm.py:164-173` — surge < threshold(minutes)
  - RVol hard gate: `strategy_kmp/core/fsm.py:187-189` — rvol_1m ≥ 2.0 (switch)
  - VI wall/cooldown: `strategy_kmp/core/gates.py:107-126` — near VI ceiling or in cooldown
  - Sector cap: `strategy_kmp/core/fsm.py:268-272` — per-sector allocation limit
  - Quality threshold: `strategy_kmp/core/sizing.py:89` — score < 30 = skip
- **Order placement**: `strategy_kmp/core/fsm.py:283-302` → Intent via OMSClient
- **Entry confirmation**: OMS position sync `strategy_kmp/main.py:202-213` — polls OMS allocations

### KPR — Intraday Mean-Reversion (v4.3.1)
- **Signal generation**: `strategy_kpr/core/setup_detection.py:59-92` — Panic flush (3% drop <15min) or Drift (2% drop >60min) + VWAP band check
- **Signal strength**: YES — 3-pillar confidence (investor flow, micro pressure, program): RED/YELLOW/GREEN in `strategy_kpr/core/fsm.py:92-148`
- **Filters**:
  - Time window: `strategy_kpr/core/fsm.py:173-174` — 09:10-14:00
  - Lunch block: `strategy_kpr/main.py:23-51` — 11:20-13:10 (switch, default off)
  - OMS halt: `strategy_kpr/core/fsm.py:174` — halt_new_entries from OMS
  - Stop breach invalidation: `strategy_kpr/core/fsm.py:183-185` — low ≤ stop_level
  - Sector cap: `strategy_kpr/core/fsm.py:245-248` — MAX_SECTOR_POSITIONS=2
  - RED confidence: `strategy_kpr/core/fsm.py:121-134` — any pillar DISTRIBUTE
  - Drift block: `strategy_kpr/main.py:159-161` — global_trade_block from drift monitor
- **Order placement**: `strategy_kpr/core/fsm.py:278-284` → Intent via OMSClient
- **Entry confirmation**: OMS allocation polling `strategy_kpr/main.py:385-402`

### PCIM — Event-Driven Influencer (v1.3.2)
- **Signal generation**: `strategy_pcim/main.py:256-307` — YouTube influencer signals via Gemini LLM extraction, conviction score 0-1
- **Signal strength**: YES — LLM conviction score + multi-influencer consolidation boost
- **Filters**:
  - Conviction threshold: 0.7 minimum
  - Trend gate: `strategy_pcim/pipeline/trend_gate.py:9-30` — close > SMA20
  - ADTV minimum: `strategy_pcim/pipeline/filters.py:11-34` — ≥5B KRW
  - Market cap range: 30B-50T KRW
  - Earnings window: no earnings within 5 days
  - Gap reversal: `strategy_pcim/pipeline/gap_reversal.py:19-66` — rate ≤ 65%
  - Gap bucketing: `strategy_pcim/premarket/bucketing.py:19-57` — A(0-3%), B(3-7%), D(reject)
  - Execution vetoes: `strategy_pcim/execution/vetoes.py:10-62` — VI, upper limit proximity, spread
  - Regime exposure caps: CRISIS 20%, WEAK 50%, NORMAL 80%, STRONG 100%
- **Order placement**: `strategy_pcim/execution/orders.py:9-43` → Intent via OMSClient
- **Entry confirmation**: OMS allocation polling `strategy_pcim/main.py:509`

### NULRIMOK — Swing Dip-Buying (v1.0.1)
- **Signal generation**: `strategy_nulrimok/iepe/entry.py:44-61` — AVWAP band touch + dip below SMA5 + volume dry-up (<60% avg)
- **Signal strength**: Partial — rank percentile from DSE (flow score + RS + sector + AVWAP proximity) drives sizing multiplier
- **Filters**:
  - Regime tier: `strategy_nulrimok/dse/regime.py:47-98` — Tier A/B/C based on KOSPI, breadth, vol, FX
  - Flow persistence: `strategy_nulrimok/dse/flow.py:41-119` — % positive smart_money days ≥ 55%
  - Leader RS percentile: `strategy_nulrimok/dse/engine.py:153-177` — Tier A ≥50th, Tier B ≥60th
  - Trend check: close > SMA50 AND slope ≥ 0
  - AVWAP acceptance: at least 1 bar accepts AVWAP reference
  - Exposure headroom: `strategy_nulrimok/iepe/entry.py:108-115` — headroom > 0.5%
  - Daily risk budget: 4% equity cap on open risk
  - Sector cap: 30% per sector
- **Order placement**: `strategy_nulrimok/iepe/entry.py:152-159` → Intent via OMSClient
- **Entry confirmation**: OMS allocation polling `strategy_nulrimok/main.py:394-424`

## Exit Logic

### KMP Exits (`strategy_kmp/core/exits.py:65-103`)
- **STOP_LOSS (hard)**: price ≤ entry - 1.2×ATR → exits.py:83
- **STOP_LOSS (acceptance failure)**: within 15min, price < OR high AND < VWAP → exits.py:88-90
- **SIGNAL (stall scratch)**: after 8min, R < 0.5 → exits.py:93-96
- **TRAILING**: adaptive trailing stop → exits.py:100-101 + exits.py:46-62
- **TIMEOUT (flatten)**: 14:30 → main.py:409
- **RISK_OFF**: KOSPI drawdown ≤ -1.0% → main.py:449

### KPR Exits (`strategy_kpr/core/exits.py:35-97`)
- **STOP_LOSS (hard)**: price ≤ stop_level → exits.py:55-57
- **TRAILING**: starts at 1.0R, 50% of gain → exits.py:61-77
- **TAKE_PROFIT (partial)**: 1.5R target, 50% exit → exits.py:79-84
- **TAKE_PROFIT (full)**: adaptive R (1.5-2.5 based on vol) → exits.py:86-89
- **TIMEOUT**: 45 min with R < 1.0 → exits.py:91-95

### PCIM Exits
- **STOP_LOSS**: price ≤ entry - 1.5×ATR → `strategy_pcim/positions/stops.py:8-13`
- **TAKE_PROFIT**: +2.5 ATR, sell 60% → `strategy_pcim/positions/profit_taking.py:9-26`
- **TRAILING**: EOD update, close - 1.5×ATR (ratcheting) → `strategy_pcim/positions/trailing.py:9-22`
- **TIMEOUT**: 15 trading days → `strategy_pcim/positions/time_exit.py:33-52`
- **PARTIAL_FILL_EXIT**: fill < 30% of intended → main.py:619-625

### NULRIMOK Exits (`strategy_nulrimok/iepe/exits.py:100-195`)
- **STOP_LOSS (AVWAP breakdown)**: close < AVWAP×0.993 + vol surge → exits.py:117-120
- **SIGNAL (failure to reclaim)**: 5 bars below AVWAP after breakdown → exits.py:122-130
- **TRAILING (momentum)**: SMA10 trail → exits.py:137-140
- **TAKE_PROFIT (mean rev partial)**: 1.5×ATR gain, 70% exit → exits.py:141-147
- **TRAILING (mean rev remaining)**: max(entry+0.5×ATR, SMA5) → exits.py:148-156
- **SIGNAL (flow grind AVWAP failure)**: 2 bars below AVWAP → exits.py:157-164
- **TIMEOUT (intraday)**: break-even or loss at close → exits.py:170-171
- **TIMEOUT (multi-day)**: 2+ sessions, PnL < 1% → exits.py:172-174
- **FLOW_REVERSAL**: 2 days negative smart_money → exits.py:198-251

## Position Sizing
- **KMP**: `strategy_kmp/core/sizing.py:18-60` — Risk parity 0.5% × quality × time × program regime. Caps: 5% of 5m volume, 20% NAV
- **KPR**: `strategy_kpr/core/fsm.py:260-268` — Risk parity 0.5% × confidence (GREEN 1.0/YELLOW 0.65) × TOD × stale penalty (0.85)
- **PCIM**: `strategy_pcim/premarket/sizing.py:9-85` — Risk 0.5% / (1.5×ATR) × conviction × soft_mult × tier_mult. Caps: 15% single name, TV_5m participation
- **NULRIMOK**: `strategy_nulrimok/iepe/entry.py:117-141` — Risk 0.5% × regime_mult × rank_percentile_mult × vol_bonus. Cap: 0.825% max
- **OMS Risk**: `oms/risk.py:120-184` — Gross exposure 80%, per-symbol 15%, sector 30%, regime caps, strategy budgets

## Data Sources
- **Price**: KIS REST API (`kis_core/kis_client.py`) + WebSocket (`kis_core/ws_client.py`) for real-time ticks
- **Bid/Ask**: WebSocket AskBidMessage (KMP), REST quote (PCIM vetoes)
- **OHLCV**: `api.get_daily_bars()`, `api.get_minute_bars()`, `api.get_daily_ohlcv()` — daily and intraday bars
- **Funding rate**: N/A (equity, not derivatives)
- **Open interest**: N/A
- **Index (KOSPI)**: `api.get_index_realtime()`, `api.get_index_daily()` — for regime/risk-off
- **Investor flow**: `api.get_investor_trend()` — foreign + institutional net (KPR, NULRIMOK)
- **External**: YouTube + Gemini LLM for influencer signals (PCIM only)

## Existing Logging
- **Format**: Structured text via loguru `{time} | {level} | {message}`
- **Location**: `/app/data/logs/{strategy}_{date}.log` — daily rotation, 30-day retention, gzip
- **Trade logging**: INFO level for entries/exits with prices, quantities, reasons
- **Error logging**: WARNING/ERROR via loguru, exception tracebacks
- **Signal logging (blocked)**: Partial — gate blocks logged (DEBUG/INFO), would-block tracking in switches for permissive modes
- **Detail level**: Moderate — prices, quantities, reasons logged but no structured JSON event schema

## Configuration
- **Strategy configs**: Per-strategy `config/constants.py` + `config/switches.py` (dataclass with permissive/conservative modes)
- **OMS config**: `oms_config.yaml` (risk limits, strategy budgets, regime caps)
- **KIS config**: Environment variables (KIS_APP_KEY, KIS_IS_PAPER, etc.)
- **Conservative mode**: `CONSERVATIVE_MODE=true` env var loads conservative.yaml overrides
- **Configurable params**: All indicator periods, TP/SL levels, filters, sizing multipliers, time windows

## State Management
- **Position tracking**: OMS in-memory `OMSState` + Postgres persistence. Strategies track local FSM + reconcile with OMS.
- **Restart recovery**: OMS loads from Postgres (positions, allocations, working orders, flags). Strategies recover from OMS allocations at startup.
- **Heartbeat**: Every 30s per strategy → OMS `/api/v1/strategies/{id}/heartbeat`

## Dependencies
- **Python**: 3.11+
- **Key packages**: loguru, pandas, requests, aiohttp, asyncpg, fastapi, uvicorn, pyyaml, pydantic

## Integration Plan

### Hook Points
1. **Pre-entry hook (all strategies)**: Where intent is submitted to OMS — KMP `fsm.py:283`, KPR `fsm.py:278`, PCIM `main.py:555`, NULRIMOK `entry.py:152`
2. **Post-entry hook (fill confirmation)**: KMP `main.py:214`, KPR `main.py:288`, PCIM `main.py:509`, NULRIMOK `main.py:415`
3. **Pre-exit hook (exit decision)**: KMP `main.py:559`, KPR `main.py:361`, PCIM `main.py:635`, NULRIMOK `exits.py:100`
4. **Post-exit hook (fill confirmation)**: KMP `main.py:225`, KPR `main.py:392`, PCIM (OMS allocation check), NULRIMOK (OMS allocation check)
5. **Signal generation hook**: KMP `fsm.py:210`, KPR `setup_detection.py:75`, PCIM `main.py:296`, NULRIMOK `entry.py:44`
6. **Filter hooks**: Each strategy has multiple gates/filters — see Filters sections above
7. **Error hook**: All strategies wrap OMS calls in try/except with loguru
8. **Main loop hook**: KMP `main.py:397`, KPR `main.py:276`, PCIM `main.py:515`, NULRIMOK `main.py:380`

### Missing Data (must be added via instrumentation)
- [x] Signal strength (exists via quality scores / confidence / conviction)
- [ ] Structured JSON trade events (currently plain text logs)
- [ ] Market snapshots at trade time (bid/ask/spread/ATR captured ad-hoc, not stored)
- [ ] Missed opportunity tracking (would-block logged but not structured/backfilled)
- [ ] Process quality scoring (no scoring framework)
- [ ] Daily aggregate snapshots (no rollup computation)
- [ ] Regime classification at trade time (regime computed but not tagged to events)
- [ ] Event forwarding to central relay (no sidecar)

### Risks
- **Rate limiting**: Paper trading has 5 req/sec limit. Snapshot service must not exhaust budget.
- **Performance**: 30-second main loops (NULRIMOK) and 2-second reconciliation (KPR/KMP) — instrumentation must be non-blocking.
- **State coupling**: Strategies maintain complex FSM state. Instrumentation wrappers must not alter FSM transitions.
- **Multi-strategy interaction**: OMS arbitration/locking. Instrumentation hooks in strategies should not affect OMS intent flow.
- **Disk I/O**: JSONL writes on hot path. Use buffered/async writes to avoid blocking.
