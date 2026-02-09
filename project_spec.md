## Project Spec

---

# 1) KMP

## Session Parameters

* OR window: 09:00–09:15 KST
* Entry window: 09:16–10:00
* Forced flatten: 14:30
* Max positions: 4
* Max per sector/theme: 1
* Halt new entries at −2% day PnL
* Flatten all at −3% or `risk_off`

---

# 0. Practical Constraints Addressed

### WS Subscription Budget (assume 20 combined regs)

Implement **Subscription Budget Manager**:

* Priority 1: `H0STCNT0` (ticks + VI ref) for up to 20 candidates
* Priority 2: `H0STASP0` (bid/ask) only for:

  * ARMED
  * IN_POSITION
  * WAIT_ACCEPTANCE within X ticks of trigger

Aggressively unsubscribe DONE symbols.

If `H0STCNT0` includes bid/ask reliably → disable `H0STASP0`.

---

### No H0STPGM0

Replace with:

1. **Market-wide cumulative program flow (REST)**
2. **Tick-derived per-ticker aggressor imbalance**

Program becomes **overlay only**, never a hard gate.

---

# 1. Required Feeds

### WebSocket

* `H0STCNT0`: price, tick volume, cum vol/value, **정적VI발동기준가**
* `H0STASP0` (optional, budgeted): best bid/ask

### REST

* Daily bars
* Intraday 1m bars (or tick aggregation)
* Orders / account / positions
* `/inquire-program-trend` (market-wide cumulative program flow)

---

# 2. Pinned Constants

| Parameter           | Value                  |
| ------------------- | ---------------------- |
| RVol_min            | 2.0                    |
| OR range            | 1.2%–5.5%              |
| Acceptance timeout  | 5 min                  |
| VI wall             | 10 ticks               |
| VI cooldown         | 10 min                 |
| Base risk           | 0.5% NAV               |
| Liquidity cap       | ≤5% last 5m value      |
| Stall scratch       | < +0.5R after 8 min    |
| Cushion             | max(3 ticks, 2×spread) |
| Market program poll | 60s                    |
| EWMA alpha          | 0.35                   |

---

# 3. Premarket Setup

Universe:

* common shares only
* exclude ETF/ETN/SPAC/preferred
* exclude 관리종목 / halts

Cache:

* SMA20 / SMA60
* `trend_ok = close>SMA20 AND SMA20≥SMA60 AND SMA20 slope ≥0`
* 14D 09:00–09:15 value baseline
* 20D 1m volume baseline (09:00–10:00 pooled)

---

# 4. Hard Filters

### Gap Skip

```
if (open − prior_close)/prior_close ≥ 5%:
    DONE
```

---

# 5. Market Regime Engine (15s)

Index:

* risk_off
* chop

Leader breadth:

```
leader_count =
    count(symbols where surge≥3 AND rvol≥1.5 AND price≥VWAP)

leader_breadth_ok = leader_count ≥ 8
```

Final:

```
regime_ok = not risk_off AND not chop AND leader_breadth_ok
```

---

# 6. Market-Wide Program Regime (Cumulative REST)

REST returns **cumulative-from-open** values.

We derive **interval delta + EWMA**.

### Robust Implementation

```python
class MarketProgramRegime:
    def __init__(self, alpha=0.35):
        self.prev = {}          # market -> last cumulative
        self.ewma = {}          # market -> smoothed delta
        self.alpha = alpha

    def update(self, market, cumulative):
        # reset detection (new day or feed reset)
        if market not in self.prev or cumulative < self.prev[market]:
            self.prev[market] = cumulative
            self.ewma[market] = 0
            return

        delta = cumulative - self.prev[market]
        self.prev[market] = cumulative

        if market not in self.ewma:
            self.ewma[market] = delta
        else:
            self.ewma[market] = (
                self.alpha * delta +
                (1 - self.alpha) * self.ewma[market]
            )

    def regime(self):
        k1 = self.ewma.get("KOSPI", 0)
        k2 = self.ewma.get("KOSDAQ", 0)

        if k1 > 0 and k2 > 0:
            return "strong_inflow"
        elif k1 < 0 and k2 < 0:
            return "outflow"
        else:
            return "mixed"
```

### Usage

Sizing overlay only:

* strong_inflow → ×1.10
* mixed → ×1.00
* outflow → ×0.85

Never blocks entry.

If feed unstable / missing → force `"mixed"`.

---

# 7. 09:15 Scanner

Compute:

* value15
* surge = value15 / baseline

Filter:

* trend_ok
* surge ≥ 3

Rank top 20 → CANDIDATE

---

# 8. Opening Range Filter

```
or_pct = (or_high − or_low)/or_mid
if not 0.012 ≤ or_pct ≤ 0.055:
    DONE
```

---

# 9. Linear Time Decay

Minutes since 09:16 = m:

* `min_surge = 3.0 + 0.04*m`
* `size_time_mult = max(0.45, 1 − 0.012*m)`

---

# 10. Velocity

1m Relative Volume:

```
rvol = curr_1m_vol / avg_1m_vol
```

Gate:

```
rvol ≥ 2.0
```

---

# 11. Per-Ticker Sponsorship (Primary Replacement for Program)

Tick-imbalance over last 60–120s:

* classify ticks via uptick/downtick
* compute:

```
imbalance = (buy_value − sell_value)/total_value
```

Used for:

* QualityScore (0–15 pts)
* exit tightening when flips negative

---

# 12. VI Protection (Mandatory)

From H0STCNT0:

```
static_up = round(vi_ref * 1.02)
```

Block if:

```
entry ≥ static_up − 10 ticks
```

Cooldown:

```
if now − last_vi < 10 min → block
```

If vi_ref missing → DONE.

---

# 13. Acceptance (Binary + Timeout)

Break:

* price > OR_high + tick
* price > VWAP

Acceptance within 5 min:

* pulled_back
* held_support ≥ min(VWAP, OR_high)*0.998
* reclaimed

Else DONE.

---

# 14. Entry

Stop:

```
stop = OR_high + 1 tick
```

Spread sanity ≤0.40%.

Limit:

```
limit = stop + max(3 ticks, 2×spread)
```

Stop-limit buy.

Cancel after ~30s if unfilled → log missed fill.

---

# 15. Position Sizing

Stops:

* structure stop = retest_low buffered
* hard stop = entry − 1.2×ATR1m

Risk parity:

```
Qty_base = (Equity*0.005)/(Entry − Stop)
```

Apply caps:

* ≤5% last 5m traded value
* ≤20% NAV

QualityScore (0–100):

* surge
* rvol
* tick imbalance
* acceptance cleanliness
* regime_ok
* spread/liquidity
* sector saturation penalty

Multiplier:

* <40 skip
* 40–60: 0.5×
* 60–80: 1.0×
* 80–100: 1.5×

Final:

```
Qty = Qty_base * quality_mult * size_time_mult * program_regime_mult
```

---

# 16. Exits

Hard exits:

* hard stop
* acceptance failure early (<15 min below OR & VWAP)
* risk_off / −3% day
* 14:30 flatten

Stall scratch:

```
if held ≥ 8 min and R < +0.5:
    exit
```

Adaptive trail:

Let gain = max_fav − entry.

Retracement factor:

```
if minutes ≤ 15: f=0.5
else: f = 0.5 + min(0.25, (minutes−15)*0.0167)
```

Tighten:

* if program regime == outflow → f=max(f,0.7)
* if tick imbalance negative → f=max(f,0.7)

Trail:

```
trail = entry + gain*f
trail = max(trail, structure_stop)
```

Exit if price ≤ trail.

---

# 17. Logging / Monitoring

* WS registrations + churn
* regime + leader breadth
* VI blocks
* tick imbalance
* market program EWMA
* missed fills
* exit reasons

Alerts:

* WS disconnect
* REST failures
* repeated VI blocks
* high missed-fill rate


---

# 2) KPR

# KPR v4.3 — Final Spec (REST-first, Tiered Universe, Alpha-preserving, ProgramSignal “AUTO” fallback)

This version rewrites the full spec and **adds the practical resolution**: **ProgramSignal is supported when possible, but the system automatically degrades to a two-pillar model when per-stock program data isn’t available for the user’s KIS product/venue**. (KIS documents program-trading summary endpoints and real-time program streams in its API portal, but availability can vary by market/plan/account.) ([Korea Investment API Portal][1])

It also includes the four required adjustments:

* **VolumeSurge confirmation = reclaim-quality** (no default `close > VWAP`)
* **Invalidation threshold = stop-aligned**
* **VWAP = canonical cumulative VWAP** (no tier discontinuity)
* **KRX micro-windows** with tighter staleness/polling governed by **RateBudget**

---

## 0) Objectives

### Preserve original KPR alpha

* Setup: **Panic Flush / Drift Exhaustion** from HOD
* Context: **VWAP depth band** (2–5% below VWAP)
* Trigger: **Reclaim + acceptance closes** FSM
* Confirmation pillars:

  * **ProgramSignal** (when available)
  * **InvestorSignal** (per-stock)
  * **MicroPressure** (tick-pressure when HOT, proxies otherwise)
* Risk/Exits: **structural stop**, **partial/full R targets**, **time stop**, **flow deterioration tightening**

### Expand universe under KIS constraints

* **REST-first** for broad monitoring (50+ symbols)
* **WS ticks only for HOT** (positions + hot setups)
* **No continuous per-stock program WS**; use **on-demand** program provider (AUTO)

### Deterministic API compliance

* All REST calls through **RateBudget**
* AlphaEngine never blocks on REST

---

## 1) Architecture

### Components

* **UniverseManager**: HOT/WARM/COLD tiering, WS slot arbitration
* **MarketStream (WS)**: tick stream (`H0STCNT0` or applicable) for HOT symbols only
* **PricePoller (REST)**: quote/bars polling for WARM/COLD on a budget
* **BarAggregator**: canonical 1m bars from ticks or REST bars
* **VWAPLedger (Canonical)**: continuous cumulative VWAP per symbol (no tier discontinuity)
* **FlowPoller (REST)**:

  * investor flow snapshots (per-stock)
  * market regime snapshots (index / optional market-wide program)
* **ProgramProvider (AUTO)**: resolves ProgramSignal using best available source (WS/REST/none)
* **SnapshotCache**: caches (program/investor/market-regime) with staleness + in-flight dedup
* **AlphaEngine**: FSM + entry/exit decisions (pure, non-blocking)
* **RiskManager**: sizing, sector caps, time-of-day multipliers, kill switches
* **RateBudget**: token-bucket quotas, jitter, endpoint cooldowns
* **MicroWindowPolicy**: KRX micro-window overrides (polling & staleness)

---

## 2) ProgramSignal Practical Resolution (“AUTO”)

### Why

Per-stock program data availability can vary; the strategy must not assume a specific TR ID is available. KIS documents program-trading endpoints and real-time program streams, but you must handle “not accessible” gracefully. ([Korea Investment API Portal][1])

### ProgramProvider modes

```yaml
program_signal:
  mode: "AUTO"          # AUTO | FORCE_WS | FORCE_REST | DISABLED
  stale_sec: 120
  required_for_green: false   # if true, GREEN requires program signal when available
```

### Provider selection logic (AUTO)

At startup (and per venue if relevant), ProgramProvider runs a **capability probe**:

1. Try **WS program stream** (only for HOT symbols; never for all symbols)
2. Else try **REST per-stock program endpoint** (if available in the user’s product scope)
3. Else set `program_signal = UNAVAILABLE` and operate in **two-pillar** mode

> Note: KIS program “summary” endpoints exist (e.g., program trading status pages), but whether they provide *per-stock* program trend and are enabled for your API keys must be tested by the probe. ([Korea Investment API Portal][1])

### Output contract

`ProgramSignal` ∈ {ACCUMULATE, DISTRIBUTE, NEUTRAL, STALE, UNAVAILABLE}

* **UNAVAILABLE** is distinct from STALE:

  * STALE: endpoint exists but data too old
  * UNAVAILABLE: endpoint not supported / not authorized / not provided

---

## 3) Tiered Universe (HOT/WARM/COLD)

### 3.1 Tiers

**HOT (WS)**

* In position OR FSM ∈ {SETUP_DETECTED, ACCEPTING}
* OR drop_from_open ≤ -1.5%
* OR inside VWAP depth band (2–5% below VWAP)
* OR volatility/range expansion flagged
* Max HOT: `hot_max` (default 38–40)

**WARM (REST fast)**

* drop_from_open ≤ -0.7% OR range expansion OR approaching VWAP band
* Poll 15–30s (budgeted; warm_max limited)

**COLD (REST slow)**

* Poll 120–180s

### 3.2 Promotion/demotion

* COLD→WARM: cross -0.7% or range expansion
* WARM→HOT: cross -1.5% OR enters VWAP band OR setup-likelihood rises
* HOT→WARM: not in position, FSM=IDLE, stable N minutes
* WARM→COLD: stable N minutes, far from triggers

### 3.3 WS slot arbitration

If HOT > hot_max: evict IDLE/non-position first; never evict positions or SETUP/ACCEPTING.

---

## 4) Canonical Bars + Canonical VWAP (discontinuity-safe)

### 4.1 1m bars

AlphaEngine consumes completed 1m bars with:

* O/H/L/C, volume, timestamp

Source:

* HOT: ticks → synthetic bars
* WARM/COLD: REST bars → canonical bars

### 4.2 VWAPLedger (canonical cumulative)

Per symbol, per session:

* `cum_vol`, `cum_pv`, `vwap = cum_pv/cum_vol`

Updates:

* HOT: from ticks (if they include price/volume)
* WARM/COLD: from REST bars using a **consistent price mode** (e.g., typical price or close)

No reset on tier switching.

```yaml
vwap_mode: canonical_cumulative
vwap_update_source_priority: ["ticks_if_hot", "rest_bars"]
vwap_bar_price_mode: "typical_price"  # or "close", but must be constant
vwap_switching: "continuous_no_reset"
```

---

## 5) Setup Detection (unchanged alpha)

Run on every completed 1m bar.

**Panic Flush**

* drop_from_HOD ≥ 3%
* time_since_HOD ≤ 15m

**Drift Exhaustion**

* drop_from_HOD ≥ 2%
* time_since_HOD ≥ 60m

**VWAP depth band (required)**

* depth below VWAP in [2%, 5%]

If (panic OR drift) AND vwap band:

* FSM = SETUP_DETECTED
* Lock:

  * `setup_low = current LOD`
  * `reclaim_level = setup_low * (1 + reclaim_offset)` (default 0.3%)
  * `stop_level = setup_low * (1 - stop_buffer)` (default 0.3%)

---

## 6) FSM: Reclaim + Acceptance (alpha-preserving)

States: IDLE → SETUP_DETECTED → ACCEPTING → IN_POSITION → INVALIDATED

### 6.1 SETUP_DETECTED → ACCEPTING

Trigger:

* `bar.high >= reclaim_level`

### 6.2 Acceptance closes

Count bars where `bar.close >= reclaim_level`.

Required closes:

* base = 2
* +1 if MicroPressure uses proxy (no true ticks yet)
* +1 if investor flow stale/missing
* +1 if late timing window
* +1 if regime unfavorable
* +1 if market-wide program selling regime (optional)

### 6.3 Invalidation (stop-aligned)

**Fix incorporated:** invalidation threshold == stop level.

* If `bar.low <= stop_level` → INVALIDATED
  (optionally use close for less sensitivity)

```yaml
invalidation_mode: stop_aligned
```

---

## 7) Signals: Investor, MicroPressure, Program (AUTO)

### 7.1 InvestorSignal (per-stock REST)

InvestorSignal ∈ {STRONG, DISTRIBUTE, CONFLICT, NEUTRAL, STALE}

Stale if older than `investor_stale_sec` (default 120; overridden by micro-windows).

### 7.2 MicroPressure (hybrid)

**A) TickPressure (HOT ticks)**

* Original uptick/downtick pressure logic (windowed)

**B) VolumeSurge proxy (REST-first)**
Compute:

* `surge = actual_volume_pct / expected_volume_pct`

**Fix incorporated:** confirmation uses **reclaim-quality**, not `close > VWAP`.

Reclaim-quality:

* `bar.close >= reclaim_level`
* `CPR >= 0.6`, CPR = (close-low)/(high-low+ε)
* recommended: `bar.close > bar.open`

ACCUMULATE if surge≥1.3 AND reclaim-quality holds.

**C) BarStrength proxy**

* CPR ≥ 0.75 AND close>open AND vol/median20 ≥ 1.3 → ACCUMULATE

Resolution:

* Prefer TickPressure if available; otherwise use proxy ensemble.

### 7.3 ProgramSignal (AUTO)

ProgramSignal ∈ {ACCUMULATE, DISTRIBUTE, NEUTRAL, STALE, UNAVAILABLE}

* Used only if provider returns non-UNAVAILABLE.
* Provider is queried **only for HOT symbols and/or symbols in SETUP/ACCEPTING** (never for the full cold universe).

---

## 8) Confidence Logic (three-pillar with automatic two-pillar fallback)

### 8.1 Base rules (always)

**RED**: any strong negative signal blocks entry:

* InvestorSignal == DISTRIBUTE
* MicroPressure == DISTRIBUTE (true ticks only; proxies rarely emit DISTRIBUTE)
* ProgramSignal == DISTRIBUTE (if available)

### 8.2 GREEN/YELLOW rules depend on Program availability

#### Mode A: Three-pillar (Program available)

* GREEN: ≥2-of-3 positive (Investor STRONG, MicroPressure ACCUMULATE, Program ACCUMULATE) with no conflict
* YELLOW: otherwise (including staleness)

#### Mode B: Two-pillar (Program UNAVAILABLE)

* GREEN: Investor STRONG AND MicroPressure ACCUMULATE
* YELLOW: one positive, none negative (or stale)

Sizing multipliers:

```yaml
confidence_sizing:
  green: 1.0
  yellow: 0.65
  stale_penalty: 0.85   # multiplies green/yellow if stale inputs were used
```

Acceptance adders automatically compensate for reduced redundancy in two-pillar mode:

* if Program UNAVAILABLE: optionally +1 acceptance close (configurable)

---

## 9) Entry, Stops, Exits (alpha preserved)

### 9.1 Entry

Allowed when:

* FSM == ACCEPTING
* acceptance closes met
* confidence != RED
* timing gate allows
* sector cap allows
* RateBudget not violated (orders always prioritized)

Order type:

* marketable limit (existing logic)

### 9.2 Stop

* structural stop at `stop_level`
* optional volatility-adjusted stop only if size reduced to keep R constant

### 9.3 Profit targets

* partial: 50% at 1.5R
* full: adaptive by volatility bucket (optional):

  * high vol: 2.5R
  * normal: 2.0R
  * low: 1.5R

### 9.4 Time stop

* 45 minutes

### 9.5 Flow deterioration

* if investor flow flips materially negative post-entry:

  * tighten stop near breakeven
  * optionally reduce size

---

## 10) Timing Gates + Time-of-Day Sizing

Hard entry blocks:

* lunch: 11:20–13:10
* after entry_end (e.g., 14:00+)

Sizing multipliers (example):

* 09:30–10:30: 1.0
* 10:30–11:20: 0.8
* 13:10–14:00: 0.9
* 14:00–14:30: 0.5

---

## 11) Sector Correlation Limits

* Max 2 open positions per sector (default)
* Block additional entries if exceeded

---

## 12) KRX Micro-Windows (explicit; RateBudget-governed)

**Fix incorporated:** temporary tighter staleness and polling cadence.

```yaml
krx_micro_windows:
  - time: "09:30"
    duration_sec: 180
  - time: "10:00"
    duration_sec: 180
  - time: "11:30"
    duration_sec: 180
  - time: "13:00"
    duration_sec: 180
  - time: "14:30"
    duration_sec: 180

default_flow_stale_sec: 120
micro_flow_stale_sec: 30

default_warm_poll_sec: 30
micro_warm_poll_sec: 15
```

Enforcement:

* MicroWindowPolicy proposes overrides
* RateBudget decides if calls can be made
* If constrained: degrade gracefully (treat as stale; add acceptance closes; reduce sizing)

---

## 13) REST Scheduling and Budgets (template)

* HOT: WS ticks + minimal REST (flow snapshots + ops)
* WARM: poll at warm_poll_sec limited by warm_max and RateBudget
* COLD: slow poll
* ProgramProvider:

  * only queries for HOT or SETUP/ACCEPTING
  * never bulk-scans

All REST calls:

* token bucket per endpoint class
* jitter + cooldown
* in-flight dedup via SnapshotCache

---

## 14) Instrumentation (mandatory)

Log with reason codes:

* tiering promotions/demotions
* setup type + VWAP depth
* acceptance closes and adders
* confidence inputs (including ProgramProvider mode + UNAVAILABLE events)
* blocked entries (timing/sector/budget/confidence)
* exits (target/stop/time/flow deterioration)
* VWAPLedger continuity checks

---

## 15) Rollout Plan (safe)

1. Add RateBudget + non-blocking cache pattern everywhere
2. Add VWAPLedger canonical cumulative VWAP
3. Implement tiering + WS arbitration
4. Implement ProgramProvider AUTO probe + fallback logic
5. Add VolumeSurge reclaim-quality + bar-strength proxies
6. Apply stop-aligned invalidation
7. Add micro-windows policy
8. Add sector caps + time-of-day sizing + adaptive targets (optional)


---

# 3) Nulrimok

## 0) Objective and Edge

**Edge thesis:** In Korea, the highest-expectancy dip buys occur when:

* market regime is supportive (breadth/vol/FX),
* smart money accumulation is persistent and accelerating,
* the stock is a sector leader,
* pullbacks mean-revert into institutional cost basis (Anchored VWAP) on volume contraction,
* exits are optimized with invalidation + setup-aware profit management + time-stops.

---

## 1) System Architecture (KIS + Local Research Store)

### 1.1 Local Research Store (LRS) — Source of Truth

A local DB (SQLite/Postgres/Parquet) updated nightly (post-close) with:

**Daily price history (≥ 2 years)**

* Universe stocks OHLCV
* KOSPI/KOSDAQ index OHLCV
* KOSPI200 constituents OHLCV (or constituent list + OHLCV)

**Daily flow history (≥ 1 year)**

* Foreign + Institutional net buying per stock (daily)

**Other**

* KRW/USD daily close series
* Sector mapping file (static, updated monthly)
* Earnings/event calendar (external source)
* Monthly “quality blacklist” flags (minimal, mechanical)

**Derived/precomputed nightly**

* regime components + tier
* sector score components (flow trend, breadth, participation)
* per-stock flow_score components (persistence/intensity/accel)
* per-stock RS metrics and sector percentiles
* trend states (SMA50, slope)
* anchor candidates and chosen anchor_date
* daily ATR estimates
* ranked watchlist candidates and overflow ranks (optional)

### 1.2 Daily Selection Engine (DSE) — 08:00–08:30 KST

Reads LRS and writes the **Watchlist Artifact** (the only input the intraday engine needs).

### 1.3 Intraday Execution + Position Engine (IEPE) — 09:10–15:10 KST

Uses KIS:

* WebSocket quotes for a limited **Active Monitoring Set** (≤ 41 subs; target K ≤ 20)
* REST for orders, positions, and **scheduled** 1m candle pulls

---

## 2) Universe Definition (Daily)

From LRS:

* Start: top N stocks by 20D ADV (N = 300–800)
* Exclude:

  * earnings within next 3 sessions (calendar)
  * monthly quality blacklist
  * halted/관리/거래정지 if available

Universe exists only for DSE; IEPE sees only a small subset.

---

## 3) Composite Market Regime (DSE from LRS)

### 3.1 Inputs (daily)

* Index close vs MA50
* Breadth: % of KOSPI200 above their own 20D SMA
* Index 20D realized vol vs trailing 1Y distribution
* KRW/USD 5D % change

### 3.2 Components (binary)

```
price_ok   = index_close > MA50
breadth_ok = breadth_20D > 55%
vol_ok     = vol_20D < 90th percentile of trailing 1Y
fx_ok      = KRWUSD_5D_change < +2.0%
```

### 3.3 Regime score and tiers

```
regime_score =
0.25*price_ok +
0.30*breadth_ok +
0.25*vol_ok +
0.20*fx_ok
```

* **Tier A (Full risk)**: score > 0.65 → risk_multiplier = 1.0
* **Tier B (Reduced risk)**: 0.40–0.65 → risk_multiplier = 0.5
* **Tier C (No new entries)**: < 0.40 → risk_multiplier = 0.0

---

## 4) Sector Layer (Soft Boost Only, DSE)

No gating by sector rank.

### 4.1 Sector baskets

Top 15 liquid stocks per sector (from mapping).

### 4.2 Sector score

```
sector_score =
0.45*z(foreign_flow_trend_20D) +
0.30*z(sector_breadth_20D) +
0.25*z(sector_participation)
```

Where:

* breadth = % basket above own 20D SMA
* participation = % basket with flow_score > basket median (or persistence ≥ 0.6)

### 4.3 Sector rank weight (boost)

* rank1=1.0, rank2=0.8, rank3=0.6, others=0.3

Used only in ranking and sizing.

---

## 5) Stock Signals (Eligibility, DSE)

A stock becomes a **Candidate** if it passes Flow + Leader + Trend + Event filters.

### 5.1 Refined Smart Money Flow (hard gate)

Smart money = foreign + institutional net buying (daily).

Components:

* **Persistence (10D)**: `persistence = count(net_buy>0)/10`
* **Intensity**: `(5D net_buy_won) / (20D ADV_won)` → z-score within sector
* **Acceleration**: `MA5(flow) - MA20(flow)` → z-score within sector

Score:

```
flow_score = 0.40*persistence + 0.35*intensity_z + 0.25*accel_z
```

Flow pass:

* persistence ≥ 0.6
* flow_score > sector median

### 5.2 Leader filter (tiered, intentional)

RS = stock 20–60D return − sector basket 20–60D return.

* **Tier A:** RS percentile ≥ 60 (top 40%)
* **Tier B:** RS percentile ≥ 70 (top 30%)
* **Tier C:** N/A (no entries)

**Rationale (documented):** In marginal regimes (Tier B), only the strongest leaders are worth taking.

### 5.3 Trend filter (simplified, hard gate)

* price > SMA50
* SMA50 slope ≥ 0

### 5.4 Lightweight fundamentals (middle ground)

* Earnings within 3 sessions → skip (hard)
* Monthly quality blacklist → exclude (hard)

---

## 6) Anchored VWAP (DSE) — v1 Daily Approximation + Upgrade Path

### 6.1 Anchor selection (deterministic)

Candidates in last 40 sessions:

1. Start of ≥5-day smart-money net buy streak
2. Most recent impulse day: volume > 2× 20D avg AND strong close

Choose the **more recent** candidate that passes acceptance; else choose the other.

### 6.2 Acceptance criterion (explicit)

An anchor passes acceptance if, since anchor_date:

* price has **traded within AVWAP ± 1.0% at least once** (using daily high/low vs AVWAP)

### 6.3 AVWAP computation choice (explicit decision)

**v1 (recommended rollout): Anchored Daily Approximation**
Compute an anchored “daily AVWAP approximation” from LRS daily bars:

For each day d from anchor_date to T-1:

* `typical_d = (High_d + Low_d + Close_d) / 3`
* `vwap_num += typical_d * Volume_d`
* `vwap_den += Volume_d`
* `avwap_daily_approx = vwap_num / vwap_den`

DSE outputs:

* anchor_date
* `avwap_ref = avwap_daily_approx`
* `band = avwap_ref ± 0.5%`

**v2 (upgrade once stable): True Intraday Anchored VWAP**
IEPE updates anchored VWAP using KIS 1m bars from anchor_date forward (or from anchor_date day’s intraday) for higher precision.

---

## 7) Ranking + Allocation (DSE)

### 7.1 Stage 1: Ranked Watchlist (no hard cap)

All Candidates are ranked; none are discarded due to monitoring constraints.

Daily rank:

```
daily_rank =
0.40*norm(flow_score) +
0.20*norm(leader_RS_percentile) +
0.20*sector_rank_weight +
0.20*AVWAP_proximity_score
```

**Tradable set thresholds:**

* Tier A: top 30% of Candidates marked `tradable=True`
* Tier B: top 20% marked `tradable=True`
* Tier C: none

### 7.2 Stage 2: Active Monitoring Set (K-limited)

IEPE monitors only top K by daily_rank among `tradable=True`:

* K target: 10–20 (must be ≤ 41)

### 7.3 Deterministic rotation policy

Maintain:

* `ranked_overflow` = remaining tradable tickers not in Active Set.

Rotate/Promote next-ranked from overflow when a slot frees due to:

1. a ticker enters a position AND you choose to keep max monitored tickers fixed (slot freed by demoting a non-armed ticker)
2. a monitored ticker becomes “inactive”:

   * not within (band ± 1.0%) for last 2 hours, AND not armed
3. scheduled refresh every 90 minutes:

   * demote the lowest-ranked monitored ticker that is not armed
   * promote next-ranked overflow

This keeps optionality while preventing churn.

### 7.4 Conviction sizing

Base conviction:

```
conviction =
0.35*flow_percentile +
0.25*leader_percentile +
0.20*sector_rank_weight +
0.20*trend_strength
```

Buckets:

* top 20%: 1.5× base risk R
* 20–50%: 1.0× R
* 50–80%: 0.7× R
* bottom: skip

**Intraday bounded bonus (no overfit):**

* if volume < 40% of 20-bar avg at trigger → size × 1.10
* else → unchanged

### 7.5 Portfolio constraints

* total open risk ≤ risk_multiplier × daily_risk_budget
* sector cap ≤ 35% of total risk

If new order breaches sector cap:

* reduce size to remaining sector risk budget
* if remaining < 30% of normal → skip
* never replace existing positions

---

## 8) Intraday Data Handling (KIS-Optimized)

### 8.1 Candle building (explicit choice)

Use **KIS 1m candle pulls** and aggregate to 30m.

At each 30m boundary (e.g., 09:30, 10:00, …):

* for each ticker in Active Monitoring Set:

  * REST call: fetch last 30 1m candles (1 call)
  * aggregate into one 30m bar
  * compute 5SMA(30m), volume averages, etc.

This matches strategy cadence and is robust.

### 8.2 WebSocket usage

Use WebSocket for real-time price to:

* detect when price enters the AVWAP band (arming)
* avoid excessive REST polling

---

## 9) Entry Rules (IEPE, 30m evaluation)

### 9.1 Preconditions

* regime tier != C
* ticker is `tradable=True`
* ticker is in Active Monitoring Set (or promoted before evaluation window)
* portfolio risk & sector cap allow an entry

### 9.2 Entry trigger (must all hold on a 30m close)

1. **Location:** price traded within `avwap_ref ± 0.5%` during the bar
2. **Dip structure:** 30m close < 5SMA(30m)
3. **Volume dry-up:** 30m volume < 60% of 20-bar avg

### 9.3 Confirmation (explicit, must occur within next 2 candles)

Within the next 1–2 30m bars, require either:

**A) Reclaim**

* 30m close > avwap_ref

**OR B) Acceptance higher-low**

* current 30m low > prior 30m low
* current close ≤ (band_upper + 0.3%)
* during confirmation window: no 30m close < (band_lower − 0.2%)

If confirmation fails by end of 2nd bar → disarm, cancel any pending order.

### 9.4 Execution

* Single limit order placed near `avwap_ref` (inside band)
* If conditions invalidate before fill → cancel
* No split entries, no averaging down, no order-book delta

---

## 10) Position Management (IEPE + DSE staged exits)

### 10.1 Always-on hard invalidation (intraday, #8)

**AVWAP breakdown invalidation**

* if 30m close < avwap_ref − 0.7% AND volume > 1.5× avg → exit immediately

**Failure-to-reclaim invalidation**

* if avwap_ref not reclaimed within 3–5 bars post-fill AND a lower low forms → exit 100%

### 10.2 Two-tier time stop

**Intraday**

* if by market close PnL ≤ 0% → exit 100%

**Multi-day**

* if after 2 full sessions PnL < +1% → exit 100%

### 10.3 Setup classification and exits (#11)

Trades start UNCLASSIFIED; tag once criteria met:

**Momentum continuation**

* breaks prior swing high within 3 sessions
  Exit: trail with 10SMA(30m)

**Mean reversion bounce**

* bounce but stalls below swing high + volume fades
  Exit: sell 70% at +1.5×ATR; trail rest tight

**Flow-driven grind**

* slow climb + persistent flow
  Exit: flow reversal flag (below) OR AVWAP failure and no reclaim in 2 bars

**Failed**

* no progress after 2 days
  Exit: 100% at close

### 10.4 Flow reversal exit timing (staged in DSE)

Because daily flow finalizes after close, flow reversal is computed **pre-market**:

DSE computes:

* `flow_reversal_flag = True` if smart money net flow < 0 on **T-1 and T-2**

DSE writes this into the artifact for held positions.

IEPE enforces:

* if flag true at open → exit at market open (or first liquid window)

### 10.5 Baseline stop

Initial stop:

* max(structural swing low, avwap_ref − 1.2×ATR30m)

---

## 11) Watchlist Artifact (DSE Output Schema)

### Per ticker (candidates and tradables)

* date
* regime_score, tier, risk_multiplier
* sector_score, sector_rank_weight
* flow_score + components (persistence/intensity_z/accel_z)
* RS percentile (tiered thresholds)
* trend state (SMA50, slope, trend_strength)
* earnings_risk_flag, blacklist_flag
* anchor_date, acceptance_pass
* avwap_ref (anchored daily approx), band_lower/upper
* ATR30m_est
* daily_rank, tradable_flag
* conviction_bucket, recommended_risk
* overflow_rank (for rotation)

### Per open position

* position_id, ticker, entry_time, avg_price, size
* setup_tag, stop, timers
* flow_reversal_flag (from DSE)
* exit directives (if any)

---

## 12) KIS Call Budget Rules (Operational)

* WebSocket: Active Monitoring Set only (K ≤ 20 recommended; ≤ 41 hard)
* REST:

  * 1m candle pull: once per 30m per monitored ticker (scheduled)
  * orders: event-driven only
  * positions/balance: periodic (e.g., every 1–5 minutes) or event-driven
  * optional program trading proxy: at most once per 30m (market-level preferred)

On REST failures:

* continue monitoring via WebSocket
* skip evaluation for affected ticker for that bar (deterministic “no action” fail-safe)

---

## 13) Deployment Plan (explicit)

* **v1:** anchored daily AVWAP approximation + scheduled 1m pulls + full exit stack
* **v2:** upgrade to true intraday anchored VWAP (only after stable live operation)


---

# 4) PCIM

# Project Spec: PCIM-Alpha v1.3.1

**Korean Influencer Post-Close Momentum (Day-1 Only, Risk-Normalized, Operationally Deterministic)**

---

## 1) Purpose and Scope

### Goal

Automate a semi-systematic trading workflow that:

1. Detects new YouTube videos from 2–3 whitelisted Korean stock influencers (post-market)
2. Extracts mentioned tickers + basic conviction from transcripts
3. Produces a ranked next-day watchlist
4. Classifies candidates into gap buckets using Korean pre-market expected open (동시호가)
5. Executes bucket-specific entries via KIS API with liquidity-aware controls
6. Manages positions with ATR-based stops, profit-taking, and a simple attention-decay time exit

### Out of Scope (v1.3.1)

* Whisper transcription / audio parsing
* Fully autonomous “no human check” pipeline (manual sanity check retained)
* Short-selling
* Multi-day pullback entries (Bucket C removed)
* Sector exposure cap enforcement (removed in v1)
* Intraday (tick-by-tick) trailing stops (EOD trailing only)

---

## 2) Strategy Definition

---

### 2.1 Gemini Extraction Output (decision-relevant only)

Per video, Gemini returns:

* `ticker_or_name`
* `conviction` = HIGH | MEDIUM
  Optional: `video_summary` (UI only)

**Removed from contract:** `signal_type`, `entry_hint`, `levels`

---

### 2.2 Eligibility Gate (binary trend gate + conviction)

**20DMA trend gate (binary):**

* If `prior_close <= 20DMA(prior_close series)` → **Reject**
* Else proceed

Conviction multiplier (used for ranking/sizing):

* HIGH → 1.0
* MEDIUM → 0.7

Minimum conviction gate:

* If conviction not in {HIGH, MEDIUM} → Reject

---

### 2.3 Hard Filters (reject)

Reject if any:

* **ADTV(20d) < ₩10B**
* **Market cap < ₩50B or > ₩5T**
* **Earnings within 5 trading days**
* **Gap reversal rate > 60%** (formal definition below)

---

### 2.4 Gap Reversal Rate (FORMAL)

**Lookback:** last **60 trading days** (exclude today)

**Gap-up event:**
`gap_up_pct = (open - prev_close) / prev_close`
Event if `gap_up_pct >= +1.0%`

**Reversal:**
Event is reversal if `close < open`

**Rate:**

```text
gap_reversal_rate =
reversal_count / event_count
```

**Minimum sample size rule:**

* If `event_count < 10` → set `insufficient_sample = true` and **do not apply** this filter
* Else if `gap_reversal_rate > 0.60` → Reject

**Logging required:** event_count, reversal_count, rate, insufficient_sample, lookback dates.

---

### 2.5 Tradability Tiers (execution + sizing constraints)

Assigned during premarket classification.

| Tier   | ADTV (20d) |
| ------ | ---------- |
| Tier 1 | ≥ ₩30B     |
| Tier 2 | ₩15B–₩30B  |
| Tier 3 | ₩10B–₩15B  |

**Tier sizing multipliers**

* Tier 1: ×1.0
* Tier 2: ×0.8
* Tier 3: ×0.5

**Tier execution constraints**

* Tier 1: marketable-limit allowed
* Tier 2: limit / marketable-limit only (tighter band)
* Tier 3: limit only and **no Bucket A entries**

**Opening liquidity participation cap**
Let `TV_5m` = traded value from 09:00–09:05 (KRW). Constrain order notional:

* Tier 1: ≤ 15% of TV_5m
* Tier 2: ≤ 12% of TV_5m
* Tier 3: ≤ 8% of TV_5m

If TV_5m unavailable at decision time, use proxy:

* `TV_5m_proxy = ADTV_20d / 78`

---

### 2.6 Soft Filters (simple, non-redundant)

Applied after hard filters.

* ₩10B ≤ ADTV < ₩15B → ×0.5
* Price up >20% last 5 trading days → ×0.5

Removed: 52w-high penalty, ATR% penalty, KOSPI<20DMA sizing multiplier.

---

### 2.7 Gap Bucketing (08:40–09:00 KST)

Using expected open from 동시호가:

* **Bucket A:** 0% to +3%
* **Bucket B:** +3% to +7%
* **Bucket D:** > +7% or negative gap → **No trade**

(Bucket C removed; merged into D.)

---

## 3) Execution Rules (09:00–10:00 KST)

### 3.1 Universal Rules

* No trades in first **60 seconds**
* Cancel unfilled entry orders at **10:00**
* **Bucket B trigger path is single and unambiguous:** VWAP touch+reclaim only (see below)
* Prefer **marketable limit**; avoid pure market orders
* Reject entry if:

  * instrument is in **VI** at decision time
  * within **2 ticks** of upper price limit (configurable)
  * spread at decision time > **0.6%** (configurable)

### 3.2 Bucket A (0% to +3%) — Tier 1–2 only

Signal candle: 09:00–09:03 (first 3-min bar)

Enter if:

* Close in **top 30%** of range
* Volume ≥ **120%** of typical 3-min volume baseline

**Typical 3-min volume baseline**

* Baseline = **median** 09:00–09:03 volume over last **20 sessions**
* **Monitoring requirement (first month):** log baseline + actual volume + ratio for every candidate.

  * If qualification hit-rate > 80% → raise threshold (e.g., 140%)
  * If < 30% → lower (e.g., 110%)

Execution:

* Place one marketable-limit buy for full intended size:

  * `limit_price = last_price × (1 + slip_band)`
  * slip_band:

    * Tier 1: 0.20%
    * Tier 2: 0.12%
* If not filled within 30 seconds: cancel and skip (v1 simplicity)

### 3.3 Bucket B (+3% to +7%) — Tier 1–3

No open buys.

Entry window: **09:10–10:00**

**Trigger: VWAP touch + reclaim only** (Section 4)

Execution:

* Tier 1: marketable-limit (band 0.12%)
* Tier 2: marketable-limit (band 0.08%)
* Tier 3: limit only

Size:

* Max **80%** of computed final position size

---

## 4) VWAP Computation + Touch/Reclaim Definition

### 4.1 VWAP computation (minute bars)

Compute from 1-minute OHLCV bars from 09:00 onward.

For each minute `i`:

* `typical_price_i = (high_i + low_i + close_i) / 3` (fallback: close_i)
* `value_i = typical_price_i × volume_i`

```text
VWAP_t = sum(value_i from 09:00..t) / sum(volume_i from 09:00..t)
```

### 4.2 Touch tolerance

Set `tol = 0.10%` (0.001), configurable.

Touch at minute t if:

* `low_t <= VWAP_t × (1 + tol)` AND
* `high_t >= VWAP_t × (1 - tol)`

### 4.3 Reclaim definition

Reclaim at minute t if:

* `close_{t-1} < VWAP_{t-1}` AND `close_t > VWAP_t`
* and `close_t >= VWAP_t × (1 + 0.05%)` (noise buffer)

### 4.4 Bucket B trigger rule

Trigger if:

* a touch occurred at time `k` and a reclaim occurred at time `t` where `t ∈ [k, k+2]`

All VWAP decisions must log: VWAP, tol, OHLC, volume, and trigger timestamps.

---

## 5) Regime-Adaptive Exposure + Intraday Halt

### 5.1 Regime computation (08:30 KST)

```python
kospi_regime = (KOSPI - SMA50) / ATR50
```

Max gross exposure:

* Crisis (< -2): **20%**
* Weak (-2 to 0): **50%**
* Normal (0 to 2): **80%**
* Strong (> 2): **100%**

Behavior:

* Crisis: disable Bucket A; Bucket B only
* Weak: disable Bucket A (recommended); Bucket B allowed
* Normal/Strong: A and B enabled

### 5.2 Intraday market halt

If KOSPI drawdown from prior close exceeds **-1.5%** at any time after 09:00:

* halt all new entries for the day
* manage existing positions normally

Note: Regime is applied via exposure caps only (no regime_mult in sizing).

---

## 6) Volatility-Parity Position Sizing

### 6.1 Core sizing

```python
target_risk_per_trade = 0.5% of equity
stop_distance = 1.5 * ATR20
raw_qty = target_risk_per_trade / stop_distance
```

### 6.2 Apply multipliers

```text
final_qty =
raw_qty
× conviction_mult
× soft_mult
× tier_mult
```

Caps:

* single name notional ≤ 15% equity
* portfolio exposure ≤ regime cap
* order notional ≤ TV_5m participation cap

### 6.3 Sizing floor (reject tiny trades)

If:

```text
final_qty < 0.20 × raw_qty
```

→ Reject trade

Log raw_qty, final_qty, each multiplier, and which cap/floor triggered.

---

## 7) Portfolio Construction Controls (NEW: deterministic, required)

### 7.1 Max concurrent positions (required)

* `max_open_positions = 8` (hard cap)

### 7.2 Priority selection when caps bind (required)

If exposure cap or position cap prevents taking all qualifying trades, select using:

**Priority tuple (ascending is better):**

* bucket_rank: A=0, B=1
* conviction_rank: HIGH=0, MED=1
* gap_pct: ascending

This favors:

* Bucket A over Bucket B
* HIGH over MED
* smaller gaps first within the same bucket/conviction

If still tied, break ties by higher ADTV.

---

## 8) Order Management Edge Cases (required)

### 8.1 Partial fills at 10:00

At 10:00:

* cancel any remaining unfilled entry quantity

If `filled_qty < 30% of intended_qty`:

* immediately exit filled portion using a marketable-limit sell
* mark as `ABORTED_SMALL_FILL`

Else:

* keep position and manage normally

Log intended_qty, filled_qty, fill_pct, and decision.

### 8.2 Same-ticker conflict

If ticker is already held (OPEN position) at decision time:

* skip new entry signals for that ticker
* log `SKIP_ALREADY_HOLDING` with influencer/video references

(No add-ons/pyramiding in v1.3.1.)

---

## 9) Exits (simplified, momentum-aligned)

### 9.1 Stops

* Initial stop: `entry − 1.5 × ATR20`
* No breakeven stop

### 9.2 Profit-taking

* At `+2.5 × ATR` from entry: sell **60%**
* Remaining **40%**: trailing stop at `1.5 × ATR`

### 9.3 Trailing stop implementation (EOD-only)

To avoid intraday wick-outs and complexity:

* Update trailing stop **once per day after close**.
* For v1.3.1, define:

  * `trail_level = max(previous_trail_level, close_today − 1.5×ATR20_today)`
* Stop orders can be placed/updated at next session open (depending on broker support), or managed locally with deterministic checks.

Log daily trail updates and the inputs used.

### 9.4 Time exit (attention decay)

* **Day 15:** unconditional exit all remaining shares

---

## 10) EV Table (analytics)

For analytics and optional sizing tweaks later:

* key = (influencer, bucket)

No large matrix.

---

## 11) Data Sources and Required APIs

### 11.1 YouTube

Prefer YouTube Data API v3.

Store:

* video_id, channel_id, publish_time (KST), title, url

### 11.2 Transcript extraction (yt-dlp)

Primary:

* captions via `--write-auto-sub`/`--write-sub`, `--sub-lang ko`, `--skip-download`

Fallback:

* if no captions: status `NO_TRANSCRIPT`, skip

### 11.3 Signal extraction (Gemini) — strict JSON

```json
{
  "video_summary": "string",
  "recommendations": [
    { "ticker_or_name": "삼성전자 or 005930", "conviction": "HIGH | MEDIUM" }
  ]
}
```

### 11.4 Market data + execution (KIS API)

Required:

* Daily OHLCV (ATR20, ADTV20, 20DMA, gap reversal inputs)
* Pre-market expected open (동시호가)
* Intraday 1m + 3m bars
* Orders: limit + marketable-limit equivalent, cancel/replace
* Account equity, positions

### 11.5 Earnings calendar

Must support “earnings within next 5 trading days” check.

---

## 12) System Components

* video_watcher
* transcript_fetcher
* signal_extractor (Gemini + schema validation)
* signal_ranker (20DMA gate + hard filters + watchlist ranking)
* human_sanity_check (approve/reject watchlist)
* premarket_classifier (expected open, bucket, tier, sizing inputs)
* execution_engine (A/B rules + idempotent orders)
* position_manager (stops, partials, EOD trailing, Day 15 exit)
* metrics_and_audit (decision snapshots, slippage, MFE/MAE, influencer stats)

---

## 13) Data Storage (DB Schema Notes)

Add / ensure fields for:

* trend_gate_pass (bool)
* gap_reversal: event_count, reversal_count, rate, insufficient_sample
* VWAP decision logs: VWAP, tol, touch/reclaim timestamps, bar OHLCV
* sizing breakdown and floor/cap triggers
* position_limit/exposure_limit selection outcomes (priority rank)
* partial fill handling outcome
* same-ticker skip reason

---

## 14) Scheduling and Runtime (KST)

Night:

* 20:00–23:59: poll for new videos
* immediate: transcript → Gemini extraction
* 00:00–06:00: refresh market stats + filters + watchlist by 06:00

Morning:

* 08:00–08:30: human sanity check → approve
* 08:40–09:00: expected open + bucket + tier + sizing + regime caps
* 09:01–10:00: execute entries (A/B only)
* 10:00+: manage-only mode
* EOD: update trailing stops and logs

Weekly:

* Friday after close: influencer scorecard update (analytics)

---

## 15) Risk Controls (Portfolio Level)

* Single position cap: ≤ 15% equity notional
* Regime exposure caps: 20/50/80/100%
* Intraday halt on KOSPI drawdown > 1.5%
* Participation caps vs opening liquidity (TV_5m)
* Execution vetoes: VI, near limit, excessive spread
* **Max concurrent positions: 8 (required)**

---

## 16) Deliverables (Definition of Done)

Must-have:

* End-to-end pipeline: video → next-day orders (manual approval step)
* Deterministic A/B-only bucketing and execution (single trigger path per bucket)
* Formal definitions: gap reversal + VWAP touch/reclaim + volume baseline
* Vol-parity sizing with reject floor
* Portfolio construction logic with max positions + deterministic priority
* Explicit partial fill + same-ticker conflict handling
* Position management: ATR stop, profit-taking, **EOD-only** trailing, Day 15 exit
* Persistent logging of decisions, snapshots, orders, fills
* Basic dashboards/logs: watchlist, fills, open positions, PnL, influencer performance
