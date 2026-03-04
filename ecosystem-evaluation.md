# Trading Ecosystem Evaluation

> Comprehensive assessment of the trading bot repositories (k_stock_trader, momentum_trader, swing_trader) and the trading assistant orchestrator — evaluating data capture, integration quality, and structural improvement capability.
>
> **Date:** 2026-03-04

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [k_stock_trader Evaluation](#2-k_stock_trader-evaluation)
3. [momentum_trader Evaluation](#3-momentum_trader-evaluation)
4. [swing_trader Evaluation](#4-swing_trader-evaluation)
5. [Trading Assistant Orchestrator Evaluation](#5-trading-assistant-orchestrator-evaluation)
6. [Integration Layer Assessment](#6-integration-layer-assessment)
7. [Structural Improvement Capability](#7-structural-improvement-capability)
8. [Prioritized Recommendations](#8-prioritized-recommendations)

---

## 1. Executive Summary

### Overall Verdict

The trading ecosystem has **strong foundational instrumentation** — all three bots emit structured events with deterministic IDs, process quality scoring, and root cause taxonomy. The relay-to-orchestrator pipeline is solid with HMAC auth, watermark-based delivery, and idempotent dedup at every layer.

However, the system has a **fundamental ceiling on the value it can deliver**. It excels at detecting problems (alpha decay, filter cost, regime mismatch) and optimizing parameters (WFO grid search), but **cannot propose or validate structural strategy improvements**. The gap is not in Claude's analytical capability — it's that Claude receives pre-classified, pre-aggregated data with no computational tools to test hypotheses. It can say "consider adding a regime gate" but cannot simulate what that gate would have done to historical performance.

### Key Findings

| Area | Grade | Summary |
|------|-------|---------|
| **Bot Instrumentation** | B+ | Strong baseline across all three; momentum_trader leads, k_stock_trader lags |
| **Event Pipeline** | A- | Reliable, idempotent, authenticated; minor gaps in batch processing |
| **Data Reduction** | B | DailyMetricsBuilder produces good analysis packages; missing several high-value metrics |
| **Parameter Optimization** | B+ | WFO works with safety checks; grid search is limiting; cost model disconnected from real data |
| **Structural Analysis** | D | System detects structural issues but has zero capability to model or validate structural changes |
| **Feedback Loop** | B+ | Corrections, rejected suggestions, and failure log all feed back into prompts; no outcome measurement automation |

### The Core Gap

The system is designed around this loop:

```
Data → Classify → Aggregate → Present to Claude → Claude suggests → Human approves → Deploy
```

What's missing is:

```
Data → Classify → Aggregate → [SIMULATE ALTERNATIVES] → Present evidence → Claude interprets → Human approves
```

Without simulation, every "structural improvement" suggestion is qualitative. Claude can hypothesize but cannot quantify. This makes the system excellent at **monitoring and parameter tuning** but limited at **strategy evolution**.

---

## 2. k_stock_trader Evaluation

### What It Is

Korean equity (KRX) trading bot. LONG only (no retail shorting on KRX). Trades via KIS REST API. Multiple strategy variants (kmp, kpr, pcim, nulrimok, conservative) with per-strategy configuration.

### Instrumentation Strengths

- **Core event emission is complete.** TradeEvent with entry/exit, signal_id, signal_strength, market_regime, filter tracking, process quality score (0-100), root causes from controlled taxonomy, entry/exit slippage in bps.
- **Missed opportunity tracking with backfill.** Logs blocked signals with simulation assumptions, backfills 24-hour hypothetical outcomes (entire KRX session), includes confidence scores.
- **Multi-timeframe regime classification.** primary_regime + higher_tf_regime from 50-MA/200-MA + ADX indicators. Deterministic, repeatable.
- **Sidecar integration is solid.** Watermark-based JSONL shipping with HMAC auth, exponential backoff retry, idempotent event IDs.
- **Process quality scoring.** Rule-based 0-100 scoring with regime fit, signal strength, entry/exit latency, slippage checks. Classification into good_process/neutral/bad_process with result tags (normal_win, exceptional_win, etc.).
- **Extended root cause taxonomy.** Adds equity-specific tags: high_entry_slippage, liquidity_gap, spread_blow_out, adverse_news beyond the core 21 tags.
- **Per-strategy configuration.** Strategy parameters exposed in YAML config files, making WFO parameter space definition straightforward.

### Instrumentation Gaps

These are the specific data points that k_stock_trader does NOT emit but the trading assistant needs for high-value analysis:

#### Critical (Directly Limits Analysis Quality)

1. **No post-exit price tracking.** momentum_trader and swing_trader both backfill 1h/4h prices after exit. k_stock_trader doesn't. Without this, the system cannot:
   - Measure exit efficiency (did price keep going favorably after exit?)
   - Detect premature exits systematically
   - Calibrate trailing stop parameters
   - The strategy engine's `detect_exit_timing_issues` detector is completely blind for k_stock_trader trades

2. **Sparse signal_factors.** Signal metadata is limited to signal_id (string) and signal_strength (0-1.0). No breakdown of contributing factors with individual values, thresholds, or contribution weights. This means:
   - Cannot decompose what drives winners vs losers at the signal level
   - Cannot identify which signal components are decaying
   - Cannot propose specific signal recalibration (only "signal may need recalibration")
   - momentum_trader provides `[{factor_name, factor_value, threshold, contribution}]` — k_stock_trader should match this

3. **No filter_decision margin_pct.** Filter decisions log pass/fail and threshold/actual_value, but momentum_trader additionally logs `margin_pct` — how close each filter was to blocking. Without margin:
   - Cannot assess filter sensitivity (a filter that passes by 0.1% vs 50% tells very different stories)
   - Cannot recommend fine-grained threshold adjustments
   - Filter analysis is binary (blocked/passed) rather than continuous

#### Important (Limits Optimization Scope)

4. **No sizing_inputs detail.** Only captures `sizing_context: {sizing_model, target_risk_pct, account_equity, volatility_basis}`. momentum_trader captures 10 fields including unit_risk_usd, setup_size_mult, session_mult, hour_mult, dow_mult, dd_mult. Without this:
   - Cannot analyze whether position sizing multipliers are helping or hurting
   - Cannot detect over/under-sizing in specific conditions
   - WFO cannot optimize sizing parameters

5. **No portfolio_state_at_entry.** swing_trader logs `{total_exposure, num_positions, correlated_pairs}` at entry time. k_stock_trader doesn't track:
   - How many other positions were open at entry
   - Total portfolio exposure at decision time
   - Whether entry decisions account for existing risk
   - This limits the portfolio risk computer's accuracy

6. **No drawdown state tracking.** momentum_trader tracks `drawdown_pct`, `drawdown_tier` (full/half/quarter/halt), and `drawdown_size_mult` (position throttle). k_stock_trader is missing:
   - Account drawdown percentage at entry
   - Whether sizing was reduced due to drawdown
   - Drawdown tier classification
   - This prevents analyzing whether the bot correctly throttles risk during adverse periods

7. **No concurrent_positions_at_entry.** momentum_trader tracks how many other positions were open. Without this:
   - Cannot correlate performance with portfolio load
   - Cannot detect "adding positions in drawdown" patterns
   - Heat cap analysis is incomplete

#### Nice-to-Have (Would Enable Future Analysis)

8. **No session_type distinction.** momentum_trader separates RTH (Regular Trading Hours) from ETH (Extended). For KRX this is less critical since the exchange has single session hours, but if the bot trades during pre-market or after-hours, this would be valuable.

9. **No strategy_params_at_entry snapshot.** Currently parameters are in config YAML, but the exact parameter values active when a trade was taken aren't recorded in the TradeEvent. If parameters change mid-day, historical trades can't be attributed to the correct parameter set.

10. **Sector_regime placeholder.** Multi-TF regime includes sector_regime but it's not yet implemented. Sector context (KRX sector indices, foreign investor flow) would help regime classification.

### Summary Assessment: k_stock_trader

| Dimension | Score | Key Issue |
|-----------|-------|-----------|
| Trade lifecycle capture | A- | Complete entry/exit with fees/slippage |
| Signal metadata | C | Missing factor decomposition and contribution scores |
| Filter transparency | B- | Has threshold/actual but no margin or sensitivity |
| Risk context | C- | No drawdown state, no portfolio context, basic sizing |
| Post-trade analysis | D | No post-exit price tracking at all |
| Missed opportunity backfill | A | 24-hour backfill with confidence scores |
| Sidecar/relay integration | A | Solid watermark + HMAC + retry |
| Process quality scoring | A | Comprehensive rule-based 0-100 with taxonomy |

**Bottom line:** k_stock_trader has the weakest instrumentation of the three bots. The most impactful improvement would be adding post-exit price backfill and signal factor decomposition — these two changes alone would unlock exit timing analysis and signal attribution for the entire analysis pipeline.

---

## 3. momentum_trader Evaluation

### What It Is

CME/GLOBEX futures trading (NQ/MNQ — Nasdaq 100 E-mini). LONG and SHORT. Three concurrent strategies (Helix, NQDTC, VdubusNQ) via Interactive Brokers API. Most mature instrumentation of the three bots.

### Instrumentation Strengths

This is the **reference implementation** for what all three bots should look like:

1. **Complete signal factor decomposition.** Each trade includes `signal_factors: [{factor_name, factor_value, threshold, contribution}]` with factors ranked by impact. This enables:
   - Per-factor win rate analysis
   - Factor importance ranking over time
   - Signal decay detection at the component level
   - Specific factor threshold optimization recommendations

2. **Post-exit price tracking.** `post_exit_1h_price` and `post_exit_4h_price` backfilled after exit with `post_exit_backfill_status` tracking. This enables:
   - Exit efficiency measurement (captured % of available move)
   - Premature exit detection
   - Trailing stop calibration
   - "Should you have stayed in?" analysis

3. **Comprehensive drawdown state.** `drawdown_pct`, `drawdown_tier` (full/half/quarter/halt), `drawdown_size_mult` (position throttle applied). This enables:
   - Drawdown-conditional performance analysis
   - Size throttle effectiveness measurement
   - "Does the bot trade worse when losing?" analysis

4. **Rich sizing inputs.** 10+ fields: target_risk_pct, account_equity, volatility_basis, sizing_model, unit_risk_usd, setup_size_mult, session_mult, hour_mult, dow_mult, dd_mult. This enables:
   - Per-multiplier attribution (is the session multiplier helping?)
   - Size optimization per condition
   - WFO on sizing parameters

5. **Session-type tracking.** RTH vs ETH distinction critical for futures (different liquidity, spread, volatility characteristics). Enables time-of-session performance analysis.

6. **Concurrent position tracking.** `concurrent_positions_at_entry` enables portfolio load analysis.

7. **Contract month tracking.** `contract_month` for roll management. Enables detecting performance anomalies around contract roll dates.

8. **Margin utilization.** `margin_used_pct` for leverage monitoring.

9. **Detailed exit reasons.** TAKE_PROFIT, STOP_LOSS, TRAILING, MFE_RATCHET, EARLY_ADVERSE, STALE, CATASTROPHIC, TIMEOUT, MANUAL — rich exit reason taxonomy enables precise exit strategy analysis.

10. **Audit-quality documentation.** 12,000+ line audit report documenting all three strategies' entry/exit/sizing logic. This is invaluable for:
    - The trading assistant understanding strategy intent
    - WFO knowing which parameters to optimize
    - Bug triage understanding expected vs actual behavior

### Instrumentation Gaps

Even momentum_trader — the best-instrumented bot — has notable gaps:

#### Important

1. **No MFE/MAE tracking.** Maximum Favorable Excursion (highest unrealized profit during trade) and Maximum Adverse Excursion (deepest unrealized loss) are not captured. Without these:
   - Cannot determine if stops are optimally placed (MAE analysis shows if stop was hit unnecessarily)
   - Cannot determine if take-profits are optimally placed (MFE shows how much was left on the table)
   - Exit efficiency calculation from post-exit prices is a proxy but not as precise
   - **This is the single most valuable metric missing from all three bots.**

2. **No order book context at entry.** Only spread_at_entry via MarketSnapshotService, no depth (bid/ask volume at best 5 levels). Without this:
   - Cannot model market impact for larger orders
   - Cannot distinguish "spread widened because of volatility" from "spread widened because book is thin"
   - Cost model cannot accurately simulate large-order slippage

3. **No per-order fill detail.** TradeEvent captures entry_price and exit_price but not the fill sequence (partial fills, re-quotes, iceberg behavior). For futures this matters less (highly liquid), but for accurate cost modeling:
   - Fill price vs theoretical price at signal time
   - Number of fills to complete order
   - Time from order submission to complete fill

4. **No strategy_params_at_entry.** Same gap as k_stock_trader. If Helix parameters are changed mid-session, cannot attribute historical trades to specific parameter sets.

5. **No portfolio_state_at_entry.** concurrent_positions_at_entry exists but no total_exposure, no correlated_positions tracking (swing_trader has this).

#### Nice-to-Have

6. **No experiment_id / A/B test flags.** If running strategy variants in parallel, no structured way to tag which variant produced which trade. This limits controlled experimentation.

7. **No funding rate tracking at entry.** For NQ/MNQ futures this is less relevant (no funding rate), but if the bot ever trades perpetual swaps, this field would matter.

8. **No market condition snapshot.** VIX level, market breadth, put/call ratio at entry time would enrich regime analysis beyond simple trend/range classification.

### Summary Assessment: momentum_trader

| Dimension | Score | Key Issue |
|-----------|-------|-----------|
| Trade lifecycle capture | A | Complete with post-exit tracking |
| Signal metadata | A | Factor decomposition with contribution scores |
| Filter transparency | A- | Full decisions with margin_pct |
| Risk context | A- | Drawdown state + concurrent positions + sizing detail |
| Post-trade analysis | B+ | 1h/4h backfill but no MFE/MAE |
| Missed opportunity backfill | A | 4h+ backfill with confidence scoring |
| Sidecar/relay integration | A | Solid watermark + HMAC + retry |
| Process quality scoring | A | Comprehensive with extended evidence refs |

**Bottom line:** momentum_trader is the gold standard. The two highest-impact additions would be MFE/MAE tracking (unlocks precise stop/TP optimization) and strategy_params_at_entry (enables WFO parameter attribution). The audit documentation is excellent and should be the model for the other bots.

---

## 4. swing_trader Evaluation

### What It Is

US equities + crypto trading. LONG and SHORT. Multiple strategy variants (ATRSS, AKC, SWING_BREAKOUT_V3, etc.) via IBKR and other APIs. Middle-tier instrumentation between k_stock_trader and momentum_trader.

### Instrumentation Strengths

1. **strategy_id per trade.** Each trade tagged with the specific strategy variant that produced it. This is critical for:
   - Per-strategy performance attribution
   - Strategy-specific WFO optimization
   - Identifying which strategies to keep/modify/retire
   - **k_stock_trader and momentum_trader should adopt this field**

2. **Portfolio state at entry.** `portfolio_state_at_entry: {total_exposure, num_positions, correlated_pairs}`. Enables:
   - Portfolio-aware performance analysis
   - Correlation risk assessment at entry time
   - "Did we add risk at the worst time?" analysis

3. **Post-exit percentage tracking.** `post_exit_1h_pct` and `post_exit_4h_pct` as percentage moves (not raw prices). This normalizes across different price levels, making cross-symbol comparison easier.

4. **Enriched signal_factors.** Factor list with contribution scores, similar to momentum_trader.

5. **Multi-strategy variant support.** With strategy_id, the system can:
   - Run WFO per strategy variant
   - Detect which variants are decaying
   - Propose variant retirement with quantified impact

6. **Rich exit reasons.** SIGNAL, STOP_LOSS, TAKE_PROFIT, TRAILING, TIMEOUT, MANUAL, STALE, CATASTROPHIC, BIAS_FLIP — includes BIAS_FLIP which is unique to swing trading (regime reversal trigger).

### Instrumentation Gaps

#### Important

1. **No drawdown tier tracking.** momentum_trader captures drawdown_pct and drawdown_tier with size throttle. swing_trader doesn't track:
   - Account drawdown state at entry
   - Whether position size was throttled
   - Drawdown recovery dynamics
   - Given that swing trades can be multi-day with larger positions, this is particularly important

2. **No MFE/MAE tracking.** Same gap as other bots. For multi-day swing trades, MFE/MAE is even more valuable because:
   - Intraday drawdowns on overnight positions create stop-loss optimization opportunities
   - Maximum favorable excursion over multi-day holds reveals optimal holding period
   - MAE distribution across regimes shows which regimes produce false adverse moves

3. **No session/time-of-day context.** Unlike momentum_trader's RTH/ETH distinction, swing_trader doesn't tag when entries/exits occur relative to market sessions. For US equities:
   - Pre-market entries (4:00-9:30 ET) have different characteristics than regular hours
   - After-hours exits may have wider spreads
   - "Do opening range breakout entries outperform afternoon entries?" is unanswerable

4. **No concurrent_positions_at_entry count.** Has portfolio_state which includes num_positions, but this is portfolio state, not concurrent position count per strategy. Subtle difference that matters for per-strategy heat cap analysis.

5. **No strategy_params_at_entry.** Same gap as all three bots. For swing trading with multiple concurrent strategies, this is critical because parameter changes affect all pending orders.

#### Nice-to-Have

6. **No order book depth.** Same as other bots. For swing trading in less liquid names, order book context is more important than for NQ futures.

7. **No correlated_pairs detail.** portfolio_state has correlated_pairs count but no detail on which pairs. Knowing "5 correlated pairs" is less useful than knowing "3 positions are all tech sector longs."

8. **No overnight gap tracking.** For multi-day swing trades, overnight gaps (open vs previous close) are a significant P&L driver. Currently no specific field captures gap size or gap direction.

### Summary Assessment: swing_trader

| Dimension | Score | Key Issue |
|-----------|-------|-----------|
| Trade lifecycle capture | A- | Complete with strategy_id tagging |
| Signal metadata | A- | Good factor decomposition |
| Filter transparency | B+ | Full decisions, some margin context |
| Risk context | B | Portfolio state but no drawdown tier |
| Post-trade analysis | B | 1h/4h pct backfill but no MFE/MAE |
| Missed opportunity backfill | A | Backfill with confidence scoring |
| Sidecar/relay integration | A | Solid watermark + HMAC + retry |
| Process quality scoring | A | Comprehensive rule-based scoring |

**Bottom line:** swing_trader has good mid-tier instrumentation. The most impactful additions would be drawdown tier tracking (already proven in momentum_trader), MFE/MAE (critical for multi-day holding period analysis), and overnight gap tracking (unique to multi-day strategies).

---

## 5. Trading Assistant Orchestrator Evaluation

### Architecture Assessment

The trading assistant is well-architected around the principle of **deterministic classification, human synthesis**. Every layer up to Claude invocation is pure computation — no randomness, no ML, no LLM. Claude enters only at the interpretation layer, where it receives pre-classified, pre-aggregated, quality-gated data.

This is a sound design choice. It means:
- Results are reproducible (same data → same classification → same prompt)
- Failures are debuggable (you can inspect the exact PromptPackage Claude received)
- Cost is proportional to analysis volume, not data volume (Claude only runs N times per day/week)
- Safety is enforced by deterministic permission gates, not LLM judgment

### What It Does Well

#### 1. Deterministic Strategy Detection (10 Rules)

The strategy engine (`analysis/strategy_engine.py`) has 10 pattern detectors across 4 tiers:

| Detector | Tier | What It Catches | Quality |
|----------|------|-----------------|---------|
| Tight stop detection | Parameter | avg_loss < 30% of avg_win | Good — clear metric, actionable |
| Filter cost analysis | Filter | filter_net_impact_pnl < threshold | Good — quantifies filter cost |
| Regime fit analysis | Strategy Variant | 3+ losing weeks in specific regime | Good — catches regime mismatch |
| Alpha decay | Hypothesis | 30d Sharpe < 70% of 90d Sharpe | Good — early warning of edge loss |
| Signal quality decay | Hypothesis | Signal→outcome correlation drop > 20% | Good — detects signal degradation |
| Exit timing issues | Strategy Variant | Exit efficiency < 50% OR premature > 40% | **Problem: no data feeds this detector** |
| Correlation breakdown | Strategy Variant | Cross-bot correlation > 0.7 | Good — systemic risk detection |
| Time-of-day patterns | Filter | Losing hours with win_rate < 35% | Good — actionable time gates |
| Drawdown patterns | Strategy Variant | Largest loss > 3× avg loss | Good — concentration risk |
| Position sizing issues | Strategy Variant | avg_loss > 1.5× avg_win despite + win rate | Good — R:R asymmetry |

**Key limitation:** All detectors output suggestions with `requires_human_judgment=True`. None can model the expected impact of implementing the suggestion. This makes them diagnostic tools, not prescriptive ones.

#### 2. Walk-Forward Optimization Pipeline

WFO (`skills/run_wfo.py`) is the most sophisticated component:

- **8-stage pipeline:** Fold generation → Parameter optimization → Backtesting → Robustness testing → Leakage detection → Cost sensitivity → Safety flags → Recommendation
- **Leakage detection:** Checks feature computation timestamps vs label timestamps. Prevents forward-looking bias.
- **Robustness testing:** Parameter neighborhood stability (±1 step), regime-split performance, Sharpe threshold checks.
- **Cost sensitivity:** Tests at 1×, 1.5×, 2× cost multipliers to detect fragile parameters.
- **Safety flags:** Automatically flags overfitting, regime dependency, cost sensitivity, data leakage.
- **Recommendation system:** ADOPT / TEST_FURTHER / REJECT with reasoning.

**This is genuinely well-built.** The safety checks prevent the most common WFO failure modes (overfitting, cost sensitivity, regime dependency).

#### 3. Feedback Loop Architecture

The correction → memory → prompt injection loop is well-designed:

- **Corrections** (`corrections.jsonl`): Human feedback on past analyses (trade reclassification, regime overrides, positive reinforcement). Loaded into every future prompt.
- **Rejected suggestions** (`suggestions.jsonl`): Tracks proposed → rejected suggestions with reasons. Prevents Claude from re-suggesting the same thing.
- **Failure log** (`failure-log.jsonl`): Records triage outcomes and PR rejection reasons. Teaches the system what fixes were rejected and why.
- **Context injection**: Every PromptPackage includes recent corrections, rejected suggestions, and failure log entries. Claude sees this before generating new analysis.

**This creates genuine learning over time** — not ML learning, but institutional memory that shapes future analysis.

#### 4. Permission Gates

Three-tier permissions enforced deterministically:
- **AUTO:** Config changes, analysis scripts, test additions
- **REQUIRES_APPROVAL:** Signal logic, filter logic, position sizing
- **REQUIRES_DOUBLE_APPROVAL:** Risk limits, kill switch, permission gates themselves

Permission checks happen at the PR review stage (TriagePRReviewChecker), preventing unauthorized changes from reaching the bots. This is essential for a system that can generate code changes.

#### 5. Portfolio Risk Monitoring

The portfolio risk computer (`skills/compute_portfolio_risk.py`) provides:
- Per-symbol and per-direction exposure
- Herfindahl-Hirschman Index (HHI) for concentration scoring (0-100)
- Crowding alerts: correlation > 0.7, same-side crowding, total exposure limits

Proactive scanner (`skills/proactive_scanner.py`) adds real-time detection:
- Unusual losses (2σ+ deviation from mean)
- Repeated errors (pattern detection)
- Missing heartbeats (bot health monitoring)

### Critical Gaps

#### Gap A: No Counterfactual Simulation Engine

**Impact: HIGH — This is the single biggest limitation of the entire system.**

The system can detect "filter X blocked 42 trades, 8 of which would have been winners." But it cannot answer: "If we changed filter X's threshold from 2.0 to 1.5, what would have happened?"

Why this matters:
- Strategy suggestions are directional ("relax this filter") not quantified ("change to this value, expected +$420/month, +0.3% drawdown")
- The user's soul.md explicitly demands quantified suggestions: "Each must be specific, testable, and quantified"
- Without counterfactual simulation, Claude can only hypothesize about impact, not calculate it
- WFO can optimize parameters within a grid, but cannot simulate structural changes (adding/removing filters, changing exit logic, adding regime gates)

What would be needed:
- A trade replay engine that takes historical trades + a modified strategy specification and simulates outcomes
- Filter perturbation: "replay all trades with filter X threshold = Y" → compute P&L, drawdown, win rate
- Exit perturbation: "replay all trades with trailing stop instead of fixed stop" → compute comparative metrics
- Regime gate simulation: "replay all trades but skip entries in regime Z" → compute impact

This is architecturally feasible because all trade events already include the decision context (signal, filters, regime, sizing). The simulation would replay decisions with modified rules.

#### Gap B: No Exit Strategy Analysis Pipeline

**Impact: HIGH — Exit optimization is one of the highest-leverage improvements in trading.**

The situation:
- momentum_trader and swing_trader emit `post_exit_1h_price` and `post_exit_4h_price`
- The strategy engine has `detect_exit_timing_issues` expecting `avg_exit_efficiency` as input
- **Nothing computes exit efficiency.** DailyMetricsBuilder doesn't process post-exit prices. The data exists in the bots but is never consumed by the analysis pipeline.

What's lost:
- Cannot measure "price moved 3% further after our exit — we left money on the table"
- Cannot compare stop-loss triggered exits vs time-based exits vs signal-based exits
- Cannot optimize trailing stop parameters based on actual post-exit price behavior
- Cannot answer "should we use ATR-based stops instead of fixed percentage?" with data
- Exit optimization is widely recognized as higher-leverage than entry optimization in systematic trading

What would be needed:
- DailyMetricsBuilder should compute exit_efficiency per trade: `(actual_capture / max_favorable_excursion)`
- If MFE tracking is added to bots (see bot recommendations), this becomes precise
- Without MFE, approximate from post_exit_1h/4h: `1.0 - (post_exit_move / trade_pnl)` for favorable continuation
- Weekly aggregation of exit efficiency by strategy, regime, signal strength, time-of-day

#### Gap C: No Signal Factor Attribution

**Impact: HIGH — This is where the highest-value structural improvements would come from.**

The situation:
- momentum_trader and swing_trader emit `signal_factors: [{factor_name, factor_value, threshold, contribution}]`
- The analysis pipeline never decomposes these factors
- DailyMetricsBuilder ignores signal_factors entirely
- No aggregation of "factor X has 65% win rate, factor Y has 40% win rate"

What's lost:
- Cannot identify which signal components drive winners vs losers
- Cannot detect individual factor decay (overall alpha decay detected, but not which factor is decaying)
- Cannot propose "factor X should use threshold 0.7 instead of 0.5" with supporting data
- Cannot identify factor interactions ("factor A + factor B together have 80% win rate, but separately only 50%")
- This is the bridge between parameter optimization (which WFO does) and signal architecture improvement (which nothing does)

What would be needed:
- Signal factor aggregation in DailyMetricsBuilder: per-factor win rate, avg PnL, Sharpe contribution
- Weekly factor importance ranking: which factors are contributing most to edge?
- Factor decay tracking: compare 30-day vs 90-day per-factor metrics
- Factor interaction analysis: pairwise factor co-occurrence in winners vs losers
- Output: factor_attribution.json in curated data, consumed by prompt assemblers

#### Gap D: Strategy Engine Detects But Cannot Prescribe Structural Changes

**Impact: MEDIUM-HIGH — Reduces the value of every detection.**

All 10 strategy engine detectors produce suggestions like:
- "Consider widening stop by 0.5× ATR" — but doesn't model what this would do
- "Consider adding a regime gate" — but doesn't simulate the gate's impact
- "Filter X costs more than it saves" — but doesn't suggest the optimal threshold

This is a natural consequence of Gap A (no counterfactual simulator), but the strategy engine could still be improved:
- When detecting filter cost: compute the optimal threshold by finding the breakeven point in the filter analysis data
- When detecting regime mismatch: compute what the P&L would have been with trades in that regime excluded
- When detecting exit timing: compute the optimal exit lag from post-exit price data

Some of these are computationally simple and don't need a full simulation engine. For example, "remove all trades in regime X" is a filter operation on existing curated data.

#### Gap E: Cost Model Disconnected from Real Data

**Impact: MEDIUM — WFO runs with inaccurate assumptions.**

The situation:
- `SlippageAnalyzer` computes per-symbol, per-hour slippage distributions from actual trade data
- `CostModel` has an `empirical` mode that reads from a JSON file
- **These two are never connected.** SlippageAnalyzer writes to curated data; CostModel reads from a separate file that nothing writes.
- WFO therefore runs with either fixed_bps or spread-proportional slippage, both of which are approximations

What's lost:
- WFO results may be optimistic (if real slippage is higher than modeled)
- Parameter changes that look profitable in WFO may be unprofitable in live trading
- Time-of-day dependent slippage (wider spreads at market open/close) is ignored
- Symbol-specific liquidity differences are ignored

What would be needed:
- SlippageAnalyzer output → CostModel input pipeline
- Per-symbol slippage_bps from empirical distribution (p50 or p75)
- Per-hour adjustment factor (if 3AM spread is 2× noon spread, costs should reflect this)
- This is a straightforward wiring task, not a design challenge

#### Gap F: Quality Gate Prevents Partial Analysis

**Impact: MEDIUM — Causes silent analysis failures.**

The quality gate (`analysis/quality_gate.py`) validates:
- All expected bots reported (directory exists)
- All 8 curated files present for each bot
- Portfolio risk card computed

If ANY check fails, the entire daily report is blocked. This means:
- If one bot's sidecar goes down, all bots lose their daily analysis
- If one curated file is malformed, nothing gets analyzed
- The user may not realize analysis was skipped (depends on alerting)

Better approach: degrade gracefully. Analyze available bots, flag missing ones prominently in the report header, note reduced confidence due to incomplete data.

#### Gap G: No Market Microstructure Data

**Impact: MEDIUM — Limits cost and execution analysis accuracy.**

None of the bots capture:
- Order book depth (bid/ask volume at best N levels)
- Volume profile (distribution of volume across price levels)
- Market impact estimate (how much price moved due to order)
- Tick-level trade flow (aggressive buys vs sells)

For k_stock_trader (equity) and swing_trader (equity/crypto), this matters because:
- Position sizes may be large relative to book depth
- Slippage is a function of order size, not just spread
- Market impact is non-linear (2× size ≠ 2× slippage)

For momentum_trader (NQ/MNQ futures), this matters less — these are among the most liquid instruments globally.

#### Gap H: Claude Has No Computational Tools During Analysis

**Impact: MEDIUM-HIGH — Limits the quality of structural improvement proposals.**

When Claude receives the PromptPackage and generates a weekly report, it operates as a language model interpreting pre-computed data. It cannot:
- Run a backtest to validate a hypothesis
- Query historical data to check a specific claim
- Simulate a parameter change to quantify impact
- Compute a statistic that wasn't pre-computed

This means Claude's structural improvement suggestions are inherently qualitative. It can say "this pattern suggests alpha decay" but cannot say "backtesting with updated parameters shows a 12% Calmar improvement."

The fix isn't to give Claude arbitrary code execution — that would violate the "deterministic classification, human synthesis" architecture. Instead:
- Pre-compute more analysis artifacts (factor attribution, exit efficiency, filter sensitivity curves)
- Run lightweight counterfactual simulations before Claude sees the data
- Let Claude request specific follow-up analyses (not implemented, but architecturally feasible with the agent runner)

#### Gap I: No Automated Outcome Measurement

**Impact: MEDIUM — Feedback loop works but requires human effort.**

The suggestion tracking system (`skills/suggestion_tracker.py`) has lifecycle tracking:
- Suggested → Rejected/Implemented → Outcome measured (7d/30d)

But outcome measurement is manual — someone has to call `record_outcome(pnl_delta, win_rate_delta, drawdown_delta)`. There's no automation that:
- Detects when a suggestion was implemented (parameter change deployed)
- Automatically computes before/after performance metrics
- Generates a "was this suggestion actually good?" report

Without this, the feedback loop is incomplete. The system learns what was rejected but not what worked.

---

## 6. Integration Layer Assessment

### Data Flow: Strong

```
VPS Bot → Sidecar (JSONL) → Relay (HMAC + dedup) → VPSReceiver (watermark) → EventQueue (SQLite)
    → Brain (deterministic routing) → Worker (dispatch) → Handler (pipeline) → Claude → Notification
```

This is robust:
- **Triple deduplication:** event_id enforced at sidecar, relay, and event queue layers
- **Watermark recovery:** if gateway restarts, relay holds events until acked
- **HMAC authentication:** prevents spoofing; per-bot shared secrets
- **Rate limiting:** per-bot sliding window prevents flood (60/min, 1000/hour)
- **Dead-letter queue:** failed events moved to dead_letter after max retries

### Integration Gaps

#### 1. QUEUE_FOR_DAILY / QUEUE_FOR_WEEKLY Events Are Acked But Not Batch-Processed

When the brain routes a trade event to `QUEUE_FOR_DAILY`, the event is acked in the queue but **no mechanism batches yesterday's queued events into a single daily analysis task**. The daily analysis is triggered by a cron job (`22:30 daily`), which runs DailyMetricsBuilder on the curated data directory — not on queued events.

This means the queue routing for trades is effectively a no-op. Events are processed by the curation pipeline (build_daily_metrics.py) independently of the event queue. The queue is used for routing errors (triage) and triggers (daily/weekly/wfo cron), but not for data aggregation.

**This isn't necessarily wrong** — the curated data pipeline reads from raw JSONL files, not the event queue. But it means the event queue doesn't provide the primary data flow for analysis, which may confuse anyone reading the code.

#### 2. Brain Doesn't Escalate Recurring Errors

If the same error fires 10 times in 1 minute, the brain creates 10 separate SPAWN_TRIAGE actions. The error_rate_tracker exists in the triage runner but not in the brain itself. This means:
- 10 concurrent triage tasks for the same error type
- Resource waste (10 Claude invocations for the same problem)
- No "error storm" detection at the routing layer

The brain should count error frequency and escalate: 1 error → normal triage, 3+ in 1 hour → single consolidated triage with urgency flag.

#### 3. Dead-Letter Queue Has No Recovery or Alerting

Events moved to dead_letter status after max retries sit there permanently. No mechanism:
- Alerts when events enter dead_letter
- Periodically retries dead_letter events
- Escalates if dead_letter count grows

For a single-user system this is manageable (manual inspection), but it's a silent failure mode.

#### 4. No Observability Endpoints

The orchestrator has no `/metrics` or `/health` endpoint exposing:
- Event queue depth (how many events pending?)
- Agent invocation count and latency (how long are analyses taking?)
- Error rate (how many events failing?)
- Cost tracking (how many Claude tokens used?)
- Dead-letter count

For operational monitoring, this data is essential. The proactive scanner handles some bot-level monitoring, but orchestrator self-monitoring is absent.

#### 5. VPS Receiver Poll Interval Creates Latency

Events are pulled from relay every 5 minutes. This means:
- Worst case: 5 minutes of latency between bot event and orchestrator processing
- For CRITICAL errors, this delay could matter
- No push mechanism (webhook) to accelerate urgent events

The relay could support a webhook callback for CRITICAL events, or the poll interval could be dynamic (shorter when recent events detected, longer during quiet periods).

---

## 7. Structural Improvement Capability

### The Fundamental Question

Can this system propose meaningful structural improvements to trading strategies, or is it limited to parameter optimization?

### Current State: Parameter Optimization Only

| Capability | Can Do? | How |
|-----------|---------|-----|
| Optimize TP/SL percentages | Yes | WFO grid search |
| Optimize filter thresholds | Partially | Filter cost detection + directional suggestion |
| Optimize signal thresholds | Partially | WFO if params in grid |
| Optimize position sizing params | Partially | WFO if params in grid |
| Add regime-conditional gates | Detect only | StrategyEngine flags, Claude suggests, human implements |
| Add time-of-day gates | Detect only | HourlyAnalyzer flags, Claude suggests, human implements |
| Change exit strategy type | No | No exit simulation capability |
| Modify signal architecture | No | No signal attribution or simulation |
| Add/remove filters | No | No counterfactual filter simulation |
| Change position sizing model | No | Permission gates + no simulation |
| Propose new parameters | No | WFO grid is pre-defined by human |

### What Would Move This from "Parameter" to "Structural"

#### Level 1: Enhanced Detection (Achievable with current architecture)

Build more analysis artifacts that give Claude (and the user) visibility into structural issues:

- **Signal factor attribution report:** Per-factor win rate, PnL contribution, decay tracking. Claude can then identify *which* factors are working and which are degrading, instead of just detecting overall signal decay.
- **Exit efficiency report:** Per-trade exit efficiency (actual capture / available move), aggregated by strategy, regime, exit reason. Claude can then recommend specific exit strategy changes with supporting data.
- **Filter sensitivity curves:** For each filter, compute performance at threshold ±10%, ±20%, ±30%. Shows whether the current threshold is optimal, too conservative, or too aggressive.
- **Regime-conditional P&L decomposition:** Instead of just "you lose in ranging markets," compute the exact P&L, drawdown, and Sharpe with trades in that regime excluded. This is a simple filter operation on curated data.

#### Level 2: Lightweight Counterfactual Simulation (Requires new component)

A trade replay engine that takes historical trades and simulates modified outcomes:

- **Filter perturbation simulator:** Replay all trades with a filter threshold changed. Compute net P&L, max drawdown, win rate at the new threshold. Output: "Changing volume filter from 2.0× to 1.5× would have produced +$420/month net PnL with max drawdown increasing from 4.2% to 4.8%."
- **Exit strategy simulator:** Replay all trades with different exit logic (trailing stop with parameter X, ATR-based stop with multiplier Y). Requires post-exit price data at minimum, MFE/MAE data for precision.
- **Regime gate simulator:** Exclude trades by regime and compute portfolio impact. Simple but powerful.

This is architecturally feasible because trade events already contain the full decision context. The simulation doesn't need to re-run the bot — it filters/modifies the existing trade outcomes.

#### Level 3: Full Strategy Simulation (Major architecture extension)

A forward-testing simulation engine that takes raw market data and runs modified strategy logic:

- Requires access to historical price data (not just trade events)
- Requires a strategy specification format (declarative rules → simulated execution)
- Requires realistic order simulation (fill modeling, partial fills, slippage)
- **This is a backtesting engine** and is a significant engineering effort

The trading bots already have backtesting capabilities internally. The question is whether the trading assistant should replicate this or invoke the bots' existing backtesting infrastructure.

### Recommendation

**Level 1 is the immediate priority.** It requires only data pipeline work (compute more analysis artifacts from existing data) and significantly improves Claude's ability to make specific, quantified suggestions. The soul.md requirement for "specific, testable, quantified" suggestions is currently unmet for structural changes — Level 1 would largely address this.

**Level 2 should follow.** A lightweight counterfactual simulator that replays trade decisions with modified rules would unlock filter optimization, regime gating simulation, and basic exit strategy comparison. This doesn't require raw market data — it works with the existing trade event stream.

**Level 3 is not recommended at this stage.** The bots themselves have backtesting capability. Rather than duplicating this in the trading assistant, the better approach is to make the trading assistant's structural suggestions specific enough that the user can run backtests on the bots directly.

---

## 8. Prioritized Recommendations

### Bot-Side Changes (Required for Optimal Data Supply)

#### Tier 1: High Value, Low Effort (< 1 day each)

| # | Bot | Change | Impact |
|---|-----|--------|--------|
| B1 | k_stock_trader | **Add post-exit 1h/4h price backfill.** Pattern already exists in momentum_trader. Implement BackfillService that checks prices 1h and 4h after exit, updates TradeEvent. | Unlocks exit timing analysis for KRX trades. Feeds detect_exit_timing_issues detector. |
| B2 | k_stock_trader | **Enrich signal_factors with contribution scores.** Change from `{signal_id, signal_strength}` to `[{factor_name, factor_value, threshold, contribution}]`. Pattern from momentum_trader. | Enables signal factor attribution. Unlocks "which factors drive winners?" analysis. |
| B3 | All bots | **Add strategy_params_at_entry to TradeEvent.** Snapshot the active parameter set when trade is opened. Enables WFO to attribute historical trades to specific parameter configurations. | Critical for WFO accuracy. Without this, parameter changes mid-day corrupt attribution. |
| B4 | k_stock_trader | **Add filter_decision margin_pct.** For each filter, log how close the actual value was to the blocking threshold. Pattern from momentum_trader. | Enables filter sensitivity analysis and precise threshold optimization. |

#### Tier 2: High Value, Medium Effort (1-3 days each)

| # | Bot | Change | Impact |
|---|-----|--------|--------|
| B5 | All bots | **Add MFE/MAE tracking.** During trade lifetime, track Maximum Favorable Excursion (peak unrealized profit) and Maximum Adverse Excursion (peak unrealized loss). Record in TradeEvent as `mfe_pct`, `mae_pct`, `mfe_price`, `mae_price`. | **Single most valuable metric addition.** Enables precise stop/TP optimization. MAE distribution reveals optimal stop placement. MFE distribution reveals optimal take-profit placement. |
| B6 | k_stock_trader | **Add drawdown state to TradeEvent.** Track `drawdown_pct`, `drawdown_tier`, `drawdown_size_mult` at entry time. Pattern from momentum_trader. | Enables drawdown-conditional performance analysis. Validates whether risk throttling works. |
| B7 | k_stock_trader | **Add portfolio_state_at_entry.** Track `{total_exposure, num_positions, correlated_pairs}` at entry time. Pattern from swing_trader. | Enables portfolio-aware analysis and correlation risk assessment. |
| B8 | swing_trader | **Add drawdown tier tracking.** Adopt momentum_trader's drawdown_pct/tier/size_mult pattern. | Validates risk throttling for multi-day positions. |

#### Tier 3: Medium Value, Low-Medium Effort

| # | Bot | Change | Impact |
|---|-----|--------|--------|
| B9 | All bots | **Add exit_efficiency metric.** Compute `actual_pnl / mfe` at exit time (requires B5). Records what percentage of the available move was captured. | Aggregated exit efficiency enables systematic exit strategy optimization. |
| B10 | swing_trader | **Add overnight gap tracking.** For multi-day holds, record `overnight_gap_pct` (open vs previous close) for each day held. | Enables gap risk analysis and gap-conditional strategy modifications. |
| B11 | All bots | **Add experiment_id to TradeEvent.** Optional field for A/B testing. When running strategy variants in parallel, tag which variant produced each trade. | Enables controlled experimentation and statistically valid strategy comparison. |

### Assistant-Side Changes (Required for Optimal Analysis)

#### Tier 1: High Value, Medium Effort (2-5 days each)

| # | Component | Change | Impact |
|---|-----------|--------|--------|
| A1 | DailyMetricsBuilder | **Add signal factor attribution.** Aggregate signal_factors across daily trades: per-factor win rate, avg PnL, Sharpe contribution. Output: `factor_attribution.json` in curated data. | Enables Claude to identify specific factor decay, propose factor threshold changes. Directly addresses Gap C. |
| A2 | DailyMetricsBuilder | **Add exit efficiency computation.** From post_exit_1h/4h prices (or MFE/MAE when available): compute per-trade exit efficiency, aggregate by strategy/regime/exit_reason. Output: `exit_efficiency.json`. | Feeds detect_exit_timing_issues detector. Enables quantified exit strategy recommendations. Addresses Gap B. |
| A3 | CostModel ← SlippageAnalyzer | **Wire empirical slippage into WFO cost model.** SlippageAnalyzer outputs per-symbol/per-hour distributions. CostModel should read these automatically. | WFO runs with accurate costs. Prevents optimistic parameter selections. Addresses Gap E. |
| A4 | New: FilterSensitivityAnalyzer | **Build filter sensitivity curve computation.** For each filter: compute performance at threshold ±10/20/30%. Output: sensitivity curve data showing optimal threshold and breakeven point. | Transforms filter suggestions from "relax this filter" to "change threshold to X for +$Y/month with Z% drawdown impact." |

#### Tier 2: High Value, High Effort (1-2 weeks each)

| # | Component | Change | Impact |
|---|-----------|--------|--------|
| A5 | New: CounterfactualSimulator | **Build lightweight trade replay engine.** Takes historical trades + modified strategy specification → replays decisions → computes P&L/drawdown/win_rate. Supports: filter threshold changes, regime gate addition/removal, trade exclusion. | Transforms all structural suggestions from qualitative to quantitative. Addresses Gap A. |
| A6 | New: ExitStrategySimulator | **Build exit strategy comparison engine.** Takes historical trades + post-exit prices + alternative exit logic → computes what would have happened with different exits. | Enables "trailing stop vs fixed stop vs ATR-based stop" comparison with data. Addresses Gap B. |
| A7 | New: AutoOutcomeMeasurer | **Automate suggestion outcome tracking.** Detect when a suggestion is implemented (parameter change deployed), compute before/after metrics automatically. | Closes the feedback loop completely. Currently manual. Addresses Gap I. |

#### Tier 3: Medium Value, Medium Effort

| # | Component | Change | Impact |
|---|-----------|--------|--------|
| A8 | QualityGate | **Degrade gracefully instead of fail.** Analyze available bots, flag missing ones. Compute confidence score based on data completeness. | Prevents silent analysis failures when one bot is down. Addresses Gap F. |
| A9 | OrchestratorBrain | **Add error frequency tracking.** Count errors per type per time window. Consolidate 10 identical errors into 1 triage with urgency flag. | Prevents resource waste on error storms. |
| A10 | Worker | **Add cross-event batching for daily/weekly.** QUEUE_FOR_DAILY events should be batched and fed into daily analysis. | Makes the event queue meaningful for data aggregation, not just routing. |
| A11 | App | **Add /metrics endpoint.** Expose queue depth, agent latency, error rate, token cost, dead-letter count. | Enables operational monitoring of the orchestrator itself. |
| A12 | Strategy Engine | **Add regime exclusion P&L computation.** When detecting regime mismatch, compute exact P&L with those trades excluded. Simple filter on curated data, no simulation needed. | Makes regime gate suggestions quantified: "excluding ranging trades would have saved $X/week." |

### Implementation Order

```
Phase 1 (Immediate — 1 week)
├── B1: k_stock post-exit backfill
├── B2: k_stock signal factors enrichment
├── B3: All bots strategy_params_at_entry
├── A3: Wire SlippageAnalyzer → CostModel
└── A8: QualityGate graceful degradation

Phase 2 (Short-term — 2 weeks)
├── B4: k_stock filter margin_pct
├── B5: All bots MFE/MAE tracking (most valuable single change)
├── A1: Signal factor attribution pipeline
├── A2: Exit efficiency computation
└── A12: Strategy engine regime exclusion computation

Phase 3 (Medium-term — 3-4 weeks)
├── B6: k_stock drawdown state
├── B7: k_stock portfolio state
├── B8: swing_trader drawdown tier
├── A4: Filter sensitivity curve analyzer
├── A9: Brain error frequency tracking
└── A11: /metrics endpoint

Phase 4 (Longer-term — 4-6 weeks)
├── B9: All bots exit_efficiency metric (requires B5)
├── A5: Counterfactual trade replay simulator
├── A6: Exit strategy comparison simulator
└── A7: Automated outcome measurement
```

### Expected Impact

If all recommendations were implemented:

| Metric | Current | After Phase 2 | After Phase 4 |
|--------|---------|---------------|---------------|
| Suggestion specificity | Directional ("consider relaxing") | Semi-quantified ("excluding regime X saves $Y") | Fully quantified ("change threshold to X: +$Y/mo, +Z% drawdown") |
| Signal improvement capability | None (detect decay only) | Factor-level attribution | Factor-specific threshold optimization |
| Exit optimization | None (detector exists, no data) | Exit efficiency tracking | Comparative exit strategy testing |
| Cost model accuracy | Fixed bps approximation | Empirical per-symbol/hour | Dynamic per-regime/condition |
| Structural change proposals | Qualitative only | Some quantified (regime gates, filter sensitivity) | Full counterfactual simulation |
| Feedback loop completeness | Manual outcome measurement | Same | Automated before/after tracking |

---

*This evaluation is a point-in-time assessment. The system is actively evolving — the gaps identified here should be re-evaluated after each implementation phase.*
