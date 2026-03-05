# Comprehensive Ecosystem Evaluation: Trading Bot Repositories + Trading Assistant

**Date:** 2026-03-05

## Context

This evaluation assesses whether the three trading bot repositories (`k_stock_trader`, `momentum_trader`, `swing_trader`) and the `trading_assistant` orchestrator collectively achieve their stated goal: **monitoring, automatically dealing with bugs/errors, analysing, and continuously improving trading performance over time**. Special focus on portfolio-level intelligence — risk allocation, position sizing, strategy proportion optimization, cross-strategy synergies, and structural improvement capability.

---

# Part 1: k_stock_trader

## Current State

4 strategies sharing a centralized OMS: **KMP** (momentum breakout), **KPR** (VWAP pullback), **NULRIMOK** (swing flow), **PCIM** (AI premarket intelligence). Capital allocation 50% each (overlapping), max 15 concurrent positions, regime-based exposure caps.

### Strengths

1. **Rich instrumentation layer.** TradeEvent captures signal_id, signal_strength, signal_factors (per-factor contributions with thresholds), filter_decisions (threshold vs observed with margin), regime_context (multi-timeframe), sizing_inputs (base_risk, quality/time/regime multipliers), portfolio_state_at_entry, MFE/MAE tracking with exit_efficiency, spread_at_entry_bps, entry/exit latency and slippage. This is close to gold-standard.

2. **Deterministic process quality scoring** (21 root causes). Scoring rubric is transparent and produces a 0-100 score + classification (good/neutral/bad) + result_tag. Root causes use a controlled taxonomy shared with the assistant.

3. **Sidecar + relay pipeline** is production-grade: watermark-based exactly-once delivery, HMAC-SHA256 signing, exponential backoff retry, deterministic event_id deduplication.

4. **Per-strategy config in YAML** — all indicator periods, TP/SL levels, filters, time windows are parametric. Simulation policies defined for missed opportunity backfill.

5. **Drawdown tier system** with automatic position sizing reduction (normal/caution/stress/critical).

### Critical Gaps

#### Gap K1: Inconsistent missed opportunity integration (HIGH)
`missed_opportunity.py` exists but integration across all 4 strategies is uneven. Some strategies log gate blocks to text only, not structured JSONL. Without consistent `MissedOpportunityEvent` with hypothetical backfill outcomes across all strategies, the assistant cannot compute **filter cost** (the most actionable daily metric — "this filter blocked 31 winners and saved 8 losers").

**Recommendation:** Audit each strategy's entry flow. Every gate block (OMS risk denial, spread gate, regime gate, time gate) must emit a `MissedOpportunityEvent` with `hypothetical_entry_price`, `blocked_by`, `filter_decisions`, `simulation_policy`, and schedule backfill for 1h/4h/24h outcomes. This is the single highest-ROI instrumentation fix.

#### Gap K2: No `strategy_params_at_entry` snapshot (HIGH)
When parameters change mid-session (conservative mode toggle, runtime adjustments), trades cannot be attributed to the correct parameter set. This breaks WFO parameter attribution entirely — the assistant's `param_optimizer.py` grid search becomes unreliable because it can't know which parameter set produced which trades.

**Recommendation:** Capture a frozen dict of strategy-specific parameters at trade entry time: `{"atr_period": 14, "tp_mult": 2.0, "sl_mult": 1.0, "signal_threshold": 0.3, ...}`. Include the hash of this dict as `param_set_id` for efficient grouping.

#### Gap K3: No cross-strategy interaction logging (HIGH for portfolio optimization)
When OMS rejects a trade due to portfolio-level limits (max 15 positions, sector cap 30%, gross exposure 90%), this is logged as a missed opportunity but **not attributed to which existing position caused the rejection**. The assistant cannot answer: "KPR was blocked 12 times this week because NULRIMOK held sector-concentrated positions."

**Recommendation:** Add `blocking_positions` field to MissedOpportunityEvent when blocked_by is a portfolio-level gate: `[{"strategy": "NULRIMOK", "symbol": "005930", "exposure_pct": 12.5}]`. This enables the assistant to propose capital reallocation between strategies.

#### Gap K4: No intra-day strategy-level P&L tracking (MEDIUM)
Daily snapshot aggregates all strategies together. No per-strategy breakdown within the day. The assistant can decompose by trade, but for real-time monitoring and proactive scanning, per-strategy intraday P&L curves would enable: "KMP is -3R at 11:00 — entering strategy-level daily stop territory."

**Recommendation:** Extend `DailySnapshot` with `per_strategy_summary`: `{"KMP": {"trades": 3, "pnl": -250, "heat_r": 1.2}, "KPR": {...}}`.

#### Gap K5: No execution cascade audit trail (MEDIUM)
Signal generation → OMS decision → order submission → fill confirmation — individual timestamps not captured. Only `entry_latency_ms` (total). Cannot distinguish "slow signal computation" from "slow OMS" from "slow KIS API."

**Recommendation:** Add `execution_timeline` dict: `{"signal_generated_at": ts, "oms_received_at": ts, "order_submitted_at": ts, "fill_confirmed_at": ts}`.

#### Gap K6: Market snapshot coverage is inconsistent (MEDIUM)
`MarketSnapshotService` exists but not all strategies attach full market context (volume profile, VWAP, tick dynamics) at entry time. KPR naturally has this (VWAP-based), KMP partially, NULRIMOK and PCIM less so.

**Recommendation:** Standardize: every trade entry must include `market_snapshot_at_entry` with at minimum `{bid, ask, spread_bps, atr_14, volume_1m, volume_5m, vwap_distance_pct}`.

#### Gap K7: No A/B experiment outcome tracking (LOW)
Experiment ID fields exist in config but no automated significance testing or outcome aggregation. When parameter changes are deployed, there's no structured way to compare before/after.

**Recommendation:** Add `experiment_id` and `experiment_variant` to every TradeEvent. The assistant's `auto_outcome_measurer.py` should compute A/B comparison metrics when experiment_id changes.

### Strategy Proportion & Allocation Observations

- **Capital allocation is static 50% per strategy with overlap.** There's no data-driven basis for this — KMP and KPR could both try to use 100% of capital simultaneously. The OMS enforces position limits reactively, not proactively.
- **No capital efficiency tracking.** Which strategy generates the most PnL per unit of capital allocated? Without per-strategy capital utilization metrics, the assistant cannot suggest "shift 10% allocation from PCIM to KMP."
- **PCIM (AI-based) has fundamentally different signal characteristics** (event-driven, premarket-only, YouTube-sourced) than the other 3. Its hit rate and holding period suggest it should have different allocation rules, but this isn't captured in the data.

---

# Part 2: momentum_trader

## Current State

3 strategies on NQ futures via IBKR: **Helix v4.0** (1H pullback continuation), **NQDTC v2.1** (30m box breakout), **Vdubus NQ v4.0** (15m VWAP pullback swing). Portfolio daily stop 1.5R, per-strategy heat caps, drawdown tiers.

### Strengths

1. **Most complete instrumentation of the three bots.** Every field the assistant expects is populated: signal_factors with factor_name/value/threshold/contribution, filter_decisions with margin_pct, regime classification, MFE/MAE with exit_efficiency, sizing_inputs with all multipliers, portfolio_state_at_entry, spread/volume/ATR snapshots.

2. **Missed opportunity tracking is fully integrated** across all 3 strategies. Backfill at 1h/4h with simulation_confidence scoring. `MissedOpportunityEvent` includes filter_decisions (why blocked), hypothetical_entry_price, would_have_hit_tp/sl, bars_to_tp/sl.

3. **Process scoring rules in external YAML** — strategy-specific preferred/adverse regimes, configurable thresholds. This makes WFO-driven scoring rule changes possible without code changes.

4. **Experiment tracking built in** — `experiment_id` and `experiment_variant` in config, attached to events.

5. **Drawdown state tracking is granular** — `drawdown_pct`, `drawdown_tier` (full/half/quarter/halt), `drawdown_size_mult` all captured per trade.

### Critical Gaps

#### Gap M1: No order book depth at entry (HIGH for futures)
For NQ futures where liquidity can thin dramatically, only `spread_at_entry_bps` is captured. No Level 2 data (order book depth, cumulative size at levels). Market impact on a 3-lot NQ order is negligible, but for larger position sizes or less liquid contracts, this becomes critical for slippage attribution.

**Recommendation:** Capture `order_book_snapshot_at_entry`: `{"bid_size_l1": 45, "ask_size_l1": 38, "depth_5_levels_bid": 312, "depth_5_levels_ask": 287}`. This enables the assistant's `slippage_analyzer.py` to decompose slippage into market impact vs. adverse selection vs. latency cost.

#### Gap M2: No per-candle signal evolution logging (HIGH for signal decay analysis)
Signals are computed every bar but only the triggering bar's values are logged. The assistant's `detect_signal_decay()` looks at signal-outcome correlation over time, but can't see **how signals build up before triggering**. False starts (signal nearly fired but didn't) are invisible.

**Recommendation:** For each strategy, log a `signal_evolution` array on MissedOpportunityEvent and TradeEvent: the last N bars' signal component values leading up to the entry decision. Even N=5 (5 bars of pre-entry signal state) would enable: "signals that build gradually over 3+ bars have 72% win rate vs 41% for sudden spikes."

#### Gap M3: No session transition tracking (MEDIUM)
`session_type` at entry is captured (RTH_PRIME1, ETH_OVERNIGHT, etc.) but no tracking of **how positions behave across session boundaries**. For an NQ position entered in ETH that transitions to RTH, the RTH open gap and volume surge create a regime change mid-trade.

**Recommendation:** Add `session_transitions` list to exit events: `[{"from": "ETH", "to": "RTH", "gap_pct": 0.3, "unrealized_pnl_at_transition": -0.5}]`. This lets the assistant propose time-based exit rules at session boundaries.

#### Gap M4: No cross-strategy correlation data emitted (MEDIUM)
All 3 strategies trade NQ. When Helix and NQDTC both enter long NQ simultaneously, effective exposure doubles. The OMS tracks `concurrent_positions_at_entry` but doesn't capture **which other strategies hold correlated positions** or the effective combined exposure.

**Recommendation:** Add `correlated_open_positions` to portfolio_state_at_entry: `[{"strategy": "Helix", "direction": "LONG", "risk_r": 1.2, "unrealized_pnl": 450}]`. This is critical because all 3 strategies trade the same instrument — the assistant needs this to compute effective single-name concentration risk.

#### Gap M5: Parameters are in Python dataclasses, not external config (MEDIUM)
Unlike k_stock_trader's YAML configs, momentum_trader's strategy parameters are hardcoded in `strategy/config.py`, `strategy_2/config.py`, `strategy_3/config.py` as frozen dataclasses. Changing parameters requires code changes, not config file updates.

**Recommendation:** This isn't a blocking issue for the assistant (WFO proposals generate PRs anyway), but it means the assistant can't distinguish "parameter change" from "code change" in PR reviews. Add `strategy_params_at_entry` snapshot (same as K2) so the assistant can track parameter evolution over time.

#### Gap M7: No per-strategy P&L in daily snapshot (HIGH for portfolio optimization)
Daily snapshot aggregates all 3 strategies (Helix, NQDTC, Vdubus) into a single P&L figure. The assistant cannot decompose which strategy contributed what to daily returns without re-parsing every individual trade. This is the **prerequisite** for cross-strategy synergy analysis (A2) and intra-bot proportion optimization (A4). Without it, the assistant cannot answer: "Helix generated 80% of this week's P&L with 33% of the heat — should it get more allocation?"

**Recommendation:** Extend `DailySnapshot` with `per_strategy_summary`: `{"helix": {"trades": 2, "pnl": 850, "win_rate": 1.0, "heat_r": 1.2, "max_drawdown_r": 0.3}, "nqdtc": {"trades": 4, "pnl": -200, "win_rate": 0.25, "heat_r": 2.1, "max_drawdown_r": 1.4}, "vdubus": {"trades": 1, "pnl": 120, "win_rate": 1.0, "heat_r": 0.5, "max_drawdown_r": 0.1}}`. Include per-strategy win count, loss count, gross PnL, net PnL, avg slippage, and process score distribution. This is a moderate change in `daily_snapshot.py` — the data is already available per-trade (each trade has `strategy_type`), it just needs aggregation.

#### Gap M6: News/macro event impact not quantified (LOW)
News blocking windows are defined in config, but when a trade is affected by a macro event (FOMC, NFP, etc.), there's no structured capture of which event it was or the estimated impact.

**Recommendation:** Add `macro_event_context` to trades near scheduled events: `{"event": "FOMC", "minutes_before": 45, "expected_impact": "high"}`. Low priority because the assistant can infer this from timestamps, but structured data enables automated regime analysis.

### Strategy Proportion & Allocation Observations

- **All 3 strategies trade the same instrument (NQ).** This is fundamentally a concentration risk — the diversification benefit is between *strategy types* (momentum, breakout, mean-reversion), not between *instruments*. The assistant's `compute_portfolio_risk.py` detects this via `crowding_alerts`, but has no framework for answering: "Given that all 3 strategies trade NQ, what's the optimal heat allocation between them?"
- **Per-strategy heat caps are static** (3.5R each, portfolio 1.5R daily stop). There's no adaptive allocation based on which strategy is performing better in the current regime. In a trending regime, Helix should arguably get more capital than NQDTC (box breakout), but this isn't captured.
- **Strategy priority system exists** (0=Helix > 1=NQDTC > 2=Vdubus) but it's for OMS conflict resolution, not capital allocation. It answers "which trade wins when both want the same capital" but not "how much capital should each strategy have."

---

# Part 3: swing_trader

## Current State

5 strategies via IBKR: **ATRSS** (ATR swing, priority 0), **S5_PB** (Keltner momentum on IBIT), **S5_DUAL** (dual Keltner on GLD+IBIT), **SWING_BREAKOUT_V3** (breakout), **AKC_HELIX** (mean-reversion, lowest priority). Cross-strategy coordination via `StrategyCoordinator`, EMA crossover overlay engine on QQQ/GLD.

### Strengths

1. **Most sophisticated cross-strategy coordination.** `StrategyCoordinator` implements concrete rules: "tighten Helix stop when ATRSS enters" (protective adjustment when higher-priority strategy takes exposure). This is the only bot that has explicit cross-strategy interaction logic.

2. **Multi-asset diversification.** Trades MNQ, MCL, MGC, MBT, NQ, CL, GC, BRR, QQQ, GLD, IBIT — genuine instrument diversification, not single-name concentration like momentum_trader.

3. **Overlay engine provides macro regime gating** — EMA(13,48) crossover on QQQ and GLD acts as a portfolio-level regime filter. This is a structural advantage the assistant should evaluate.

4. **Priority-weighted position sizing** — per-strategy `unit_risk_pct` scales from account equity (1.2%, 0.8%, 0.8%, 0.5%, 0.5%), reflecting conviction levels.

5. **Post-exit tracking exists** (`post_exit/post_exit_YYYY-MM-DD.jsonl`) with 1h/4h price backfill — enables exit efficiency analysis.

### Critical Gaps

#### Gap S1: No MFE/MAE tracking (CRITICAL)
This is the single most important missing metric. Without Maximum Favorable Excursion and Maximum Adverse Excursion per trade, the assistant cannot:
- Compute exit efficiency (`actual_pnl / mfe`)
- Identify if stops are too tight (MAE shows how often price touches stop before reversing)
- Optimize take-profit levels (MFE shows where price peaks before reversing)
- Run the `exit_strategy_simulator.py` meaningfully (needs MFE/MAE as input)

The other two bots have this. swing_trader is the only one without it, and it runs the most diverse portfolio.

**Recommendation:** Implement MFE/MAE tracking in the trade management loop. On every bar while a position is open: `if price > mfe_price: mfe_price = price`. On exit, compute `exit_efficiency = actual_pnl_pct / mfe_pct`. This is a 50-line change with outsized analytical value.

#### Gap S2: No `strategy_params_at_entry` snapshot (HIGH)
Same as K2 and M5. With 5 strategies and 10+ instruments, parameter attribution is critical. When the assistant runs WFO and proposes "change ATRSS MNQ atr_daily_period from 20 to 25", it needs to verify which trades used period=20 vs. period=25.

**Recommendation:** Snapshot frozen dataclass fields as dict at entry time.

#### Gap S3: Drawdown tier not logged with precision (HIGH)
`drawdown_pct_at_entry` exists but `drawdown_tier_at_entry` classification and `position_size_multiplier` applied are not consistently populated. The assistant can't validate: "were positions correctly sized during the caution tier?"

**Recommendation:** Ensure all 3 fields (`drawdown_pct`, `drawdown_tier`, `drawdown_size_mult`) are populated on every TradeEvent.

#### Gap S4: StrategyCoordinator interactions not logged (HIGH for synergy analysis)
The coordinator makes cross-strategy decisions (tighten Helix stop when ATRSS enters, size boosts) but these decisions are not captured in the event stream. The assistant has no visibility into: "Helix was stopped out 8 times this month because ATRSS triggered the stop-tightening rule — is that rule helping or hurting?"

**Recommendation:** Emit `coordinator_action` events: `{"action": "tighten_stop", "target_strategy": "Helix", "trigger_strategy": "ATRSS", "old_stop": 4520, "new_stop": 4535, "reason": "higher_priority_entry"}`. Wire into the sidecar for relay to the assistant.

#### Gap S5: Overlay engine decisions not logged (HIGH for structural analysis)
The EMA crossover overlay on QQQ/GLD gates entry for some strategies, but the overlay state is not captured in trade events. The assistant can't analyze: "the overlay blocked 45 signals this month — what would have happened?"

**Recommendation:** Add `overlay_state` to TradeEvent and MissedOpportunityEvent: `{"qqq_ema_bullish": true, "gld_ema_bullish": false, "overlay_gate_passed": true}`. For missed opportunities blocked by the overlay, this enables filter sensitivity analysis on the overlay itself.

#### Gap S8: No per-strategy P&L in daily snapshot (HIGH for portfolio optimization)
Daily snapshot aggregates all 5 strategies (ATRSS, S5_PB, S5_DUAL, Breakout, Helix) into a single P&L figure. This is the same gap as K4 and M7 but particularly impactful here because swing_trader has the most strategies (5) and the most complex cross-strategy interactions (StrategyCoordinator, overlay engine). Without per-strategy breakdowns, the assistant cannot evaluate: "ATRSS generates 60% of returns but Helix provides diversification benefit that reduces portfolio drawdown by 15% — is the 0.5% allocation justified despite 34% win rate?"

**Recommendation:** Extend `DailySnapshot` with `per_strategy_summary`: `{"atrss": {"trades": 3, "pnl": 1200, "win_rate": 0.67, "heat_r": 1.5, "symbols_traded": ["MNQ", "MCL"], "max_drawdown_r": 0.8}, "s5_pb": {...}, "s5_dual": {...}, "breakout": {...}, "helix": {...}}`. Include per-strategy: trade count, win/loss count, gross/net PnL, avg slippage, process score distribution, and symbols traded. Additionally, include `overlay_state_summary`: `{"signals_gated": 12, "signals_passed": 45, "pct_time_bullish_qqq": 0.65, "pct_time_bullish_gld": 0.48}` to enable overlay engine evaluation.

#### Gap S6: No overnight gap tracking (MEDIUM)
For multi-day swing holds, the gap between previous close and next open is a significant risk factor. Not tracked.

**Recommendation:** Add `overnight_gap_pct` to trades that span sessions. Enables the assistant to propose: "for MNQ positions held overnight, gap risk averages 0.3% — consider sizing down or using overnight stop levels."

#### Gap S7: Signal factor attribution incomplete (MEDIUM)
`signal_factors` array exists in the schema but not all strategies populate it with the same richness. ATRSS has detailed factors, Helix and Breakout strategies may have sparser factor data.

**Recommendation:** Standardize signal_factors across all 5 strategies. Each factor should have `factor_name`, `factor_value`, `threshold`, `contribution`. The assistant's `factor_attribution.py` schema expects this structure.

### Strategy Proportion & Allocation Observations

- **Priority-weighted sizing is a good start** (ATRSS 1.2% > S5_PB/S5_DUAL 0.8% > Breakout/Helix 0.5%), but it's static. The assistant should be able to propose: "ATRSS Sharpe this quarter is 0.8 while S5_PB is 2.1 — consider shifting allocation."
- **AKC_HELIX has 34% win rate** and lowest priority. The assistant should be evaluating: "Is Helix contributing positive expected value to the portfolio, or is it a drag? Does it provide diversification benefit (decorrelated returns) that justifies its negative Sharpe?"
- **StrategyCoordinator creates dependencies** between strategies — tightening Helix stop when ATRSS enters means Helix P&L is partially a function of ATRSS activity. This interaction effect is completely invisible to the assistant without Gap S4 being fixed.
- **Overlay engine is a portfolio-level structural feature** — it's a meta-strategy that gates individual strategies. This is exactly the kind of structural component the assistant should be able to evaluate and optimize, but can't without Gap S5.

---

# Part 4: trading_assistant

## Current State

1019 tests, 7-phase architecture (orchestrator → brain → worker → handlers → prompt assembly → Claude CLI → notification). Daily/weekly/WFO analysis, bug triage, proactive scanning, memory governance.

### What It Does Well

1. **Daily analysis is genuinely portfolio-aware.** Loads all bots' curated data + portfolio risk card, asks Claude to analyze cross-bot patterns, crowding, correlation. The prompt structure (portfolio-first, then per-bot, then cross-patterns) is well-designed.

2. **Weekly analysis runs meaningful simulations.** Strategy engine detects issues → filter sensitivity, counterfactual, and exit strategy simulators quantify the impact → results injected into Claude's prompt. This is a strong feedback loop.

3. **Memory governance prevents stale recommendations.** Rejected suggestions tracked, temporal decay on findings (90-day window, 50-entry cap), outcome measurements feed back into the next cycle. The Ralph Loop V2 pattern (failure log) prevents repeating mistakes.

4. **10 strategy engine detectors** cover parameter, filter, regime, alpha decay, signal decay, exit timing, correlation, time-of-day, drawdown, and position sizing issues.

5. **Quality gate with graceful degradation** — missing data doesn't block analysis, just flags it. This is operationally critical for a system that depends on 3 unreliable VPS data streams.

6. **Context builder loads rich historical context** — corrections, failure patterns, outcome measurements, consolidated patterns, session history — all temporally decayed to prioritize recency.

### Critical Gaps

#### Gap A1: No portfolio allocation optimization (CRITICAL)

**The problem:** Soul.md says "maximise expected returns whilst minimising max drawdown" and prioritizes Calmar ratio, but the system has **no mechanism to compute or propose optimal capital allocation** across bots or across strategies within a bot.

**What exists:**
- `compute_portfolio_risk.py` computes exposure, concentration (Herfindahl HHI), crowding alerts
- Weekly prompt asks Claude to "assess cross-bot patterns"
- Strategy engine has `detect_correlation_breakdown()` for pair-wise correlation

**What's missing:**
- No schema for allocation recommendations (`AllocationRecommendation` with bot_id, current_pct, suggested_pct, expected_calmar_change, rationale)
- No optimization algorithm — even simple Sharpe-maximization given historical returns + covariance matrix
- No rebalancing logic or rebalancing frequency policy
- No capital efficiency metric (PnL / capital_allocated per bot)
- No ROIC tracking over time (is Bot A's return on invested capital improving or degrading?)

**Impact:** The user manually decides capital allocation without quantified guidance. When one bot outperforms, there's no structured recommendation to rebalance. When drawdowns concentrate in one bot, the system alerts but doesn't propose "reduce allocation to Bot B from 40% to 25%."

**Recommendation:** Build `skills/portfolio_allocator.py` that:
1. Takes per-bot weekly summaries (returns, volatility, max drawdown) + return correlation matrix
2. Computes mean-variance efficient frontier (or simplified: risk-parity + Calmar-weighted tilt)
3. Outputs `AllocationRecommendation` per bot with current vs. suggested allocation, expected portfolio Calmar change
4. Respects constraints from `allocation_rules.md` (min/max per bot, rebalancing frequency, max single-rebalance magnitude)
5. Wire into weekly handler alongside strategy engine

#### Gap A2: No cross-strategy synergy analysis (CRITICAL)

**The problem:** Each bot runs multiple strategies internally (k_stock_trader: 4, momentum_trader: 3, swing_trader: 5), but the assistant analyzes **per-bot**, not **per-strategy-within-bot**. It cannot answer:
- "KMP and KPR in k_stock_trader have 0.6 return correlation — are they redundant?"
- "Helix in momentum_trader and AKC_HELIX in swing_trader are the same strategy on different instruments — do they diversify or concentrate?"
- "S5_PB and S5_DUAL both trade IBIT — is this intentional overlap?"
- "ATRSS on MNQ and Helix on NQ are correlated at 0.85 — effective exposure is 2x what it looks like"

**What exists:**
- `detect_correlation_breakdown()` operates at the bot level, not strategy level
- Portfolio risk card shows crowding by symbol, not by strategy
- No schema for strategy-pair analysis

**What's missing:**
- Per-strategy return time series (currently aggregated at bot level in daily snapshot)
- Strategy-pair correlation matrix
- Diversification benefit quantification ("adding Helix improves portfolio Sharpe by 0.12 despite negative individual Sharpe because correlation with ATRSS is -0.3")
- Synergy scoring (complementary vs. redundant vs. cannibalistic)
- Regime-conditional correlation ("ATRSS and S5_PB are uncorrelated in trending markets but correlated at 0.7 in ranging markets")

**Recommendation:** Build `skills/synergy_analyzer.py`:
1. Require per-strategy P&L time series from bots (Gap K4 / equivalent in other bots must be addressed first)
2. Compute rolling correlation matrix (30d, 60d, 90d windows)
3. Compute marginal contribution to portfolio Sharpe per strategy
4. Identify redundant pairs (correlation > 0.7, similar signal type) and complementary pairs (correlation < 0.2, different regime preferences)
5. Output `SynergyReport` with strategy_pair, correlation, diversification_benefit, regime_conditional_correlation, recommendation

#### Gap A3: No structural improvement proposal framework (HIGH)

**The problem:** The strategy engine detects **parameter-level issues** (tight stops, costly filters, regime mismatch) and the weekly prompt asks Claude to propose **structural changes**, but there's no structured framework for evaluating structural quality. Claude is asked "assess whether any bot's signal logic needs structural changes" but given no quantified structural metrics.

**What "structural" means:**
- Signal architecture: Is the signal type (momentum, mean-reversion, breakout) appropriate for the instrument and regime distribution?
- Exit architecture: Does the exit strategy match the signal type? (Momentum signals need trailing stops, mean-reversion signals need time-based exits)
- Regime gating: Should a regime filter be added/removed? What's the opportunity cost?
- Position sizing model: Is the current model (risk parity, Kelly, fixed fractional) optimal for this strategy's win rate + payoff ratio?
- Strategy lifecycle: Is this strategy in its growth, maturity, or decay phase? Should it be scaled up, maintained, or wound down?

**What's missing:**
- `StructuralProposal` schema (bot_id, strategy_id, category, current_state, proposed_change, evidence, expected_impact, effort_level, reversibility)
- Quantified structural metrics: signal-outcome correlation trend, win rate by regime over time, filter ROI
- Strategy lifecycle classification (alpha decay rate, regime sensitivity drift)
- Architecture mismatch detection ("this momentum strategy uses a fixed TP — momentum strategies should use trailing exits")

**Recommendation:** Build `skills/structural_analyzer.py`:
1. For each strategy, compute: signal hit rate trend (is edge decaying?), regime sensitivity (does it only work in one regime?), exit efficiency distribution (are exits systematically early/late?), filter ROI (does each filter save more than it costs?)
2. Classify strategy lifecycle: growing (improving Sharpe), mature (stable), decaying (declining Sharpe over 90d)
3. Detect architecture mismatches using rules: momentum + fixed_tp = mismatch, mean_reversion + trailing_stop = mismatch
4. Output `StructuralReport` with per-strategy lifecycle status, detected mismatches, proposed structural changes with effort/impact scoring

#### Gap A4: No intra-bot strategy proportion optimization (HIGH)

**The problem:** The user specifically wants optimizing "the rules and proportion between the 4-5 active strategies and EMA crossover strategy in swing_trader, or optimising the rules and proportion between the 3 strategies in momentum_trader." The assistant has **zero capability** for this.

**What exists:**
- Each bot's internal strategy allocation is hardcoded (swing_trader: ATRSS 1.2%, S5_PB 0.8%, etc.; momentum_trader: 3.5R per strategy)
- The assistant sees aggregate bot-level P&L but not per-strategy P&L within a bot
- No schema for within-bot allocation recommendations

**What's needed:**
1. **Bots must emit per-strategy P&L in daily snapshot** (Gap K4, equivalent for others). This is the prerequisite — without per-strategy data, the assistant is blind.
2. **New schema:** `IntraBotAllocation` with `{bot_id, strategy_id, current_unit_risk_pct, current_heat_cap_r, suggested_unit_risk_pct, suggested_heat_cap_r, rationale, evidence_period_days, expected_portfolio_calmar_change}`
3. **New skill:** `skills/strategy_proportion_optimizer.py` — takes per-strategy returns + correlation, computes optimal within-bot risk allocation subject to OMS constraints
4. **Wire into weekly handler:** Run after strategy engine, before prompt assembly. Include intra-bot allocation recommendations in the weekly prompt.

**Special cases:**
- **swing_trader overlay engine:** The EMA crossover overlay should be treated as a separate "strategy" whose allocation question is: "How much does the overlay improve portfolio metrics? What if we relaxed it for ATRSS but kept it for Helix?" This requires Gap S5 to be fixed first.
- **momentum_trader same-instrument concentration:** All 3 strategies trade NQ. Allocation here is fundamentally about: "Given 3 different edges on the same price series, how much capital should each get, and are their signals independent enough to justify running all 3?"

#### Gap A5: No regime-conditional analysis pipeline (MEDIUM)

**The problem:** The strategy engine detects regime mismatch and the counterfactual simulator can exclude regimes, but there's no **regime-conditional optimization pipeline**. The system can answer "performance was bad in ranging markets" but not "in ranging markets, strategy A should reduce size by 40% and strategy B should increase size by 20% because their regime sensitivities are complementary."

**What's missing:**
- Regime-conditional strategy correlation matrix
- Regime-conditional optimal allocation (different weights per regime)
- Regime transition detection (e.g., "we transitioned from trending to ranging at 14:00 — should sizing adjust intraday?")
- Regime probability estimation (what % of time do we spend in each regime? Is this changing?)

**Recommendation:** Extend `strategy_engine.py` with `compute_regime_conditional_metrics()`:
1. For each regime × strategy combination: win rate, expectancy, Sharpe, max drawdown
2. Compute optimal allocation per regime
3. Track regime distribution over time (is trending becoming less common?)
4. Output regime-conditional suggestions: "In ranging NQ markets, reduce Helix allocation to 0.3% (from 0.5%) and increase NQDTC to 1.0% (from 0.8%)"

#### Gap A6: No strategy interaction effect modeling (MEDIUM)

**The problem:** swing_trader's StrategyCoordinator creates causal dependencies between strategies (tightening stops, size boosts), but the assistant has no visibility into these interactions and cannot model their effects.

**What's missing:**
- No coordinator event consumption (Gap S4 must be fixed first)
- No interaction effect quantification: "The stop-tightening rule caused Helix to exit 8 trades early, costing $1,200. Without it, Helix drawdown would have been $800 worse. Net benefit: $400."
- No interaction optimization: "Should the tightening multiplier be 1.5x instead of 2x?"

**Recommendation:** After Gap S4 is fixed:
1. Parse coordinator events in `build_daily_metrics.py`, add `coordinator_impact` section to curated data
2. Build `skills/interaction_analyzer.py` that replays trades with/without coordinator rules
3. Include interaction analysis in weekly prompt

#### Gap A7: Claude prompt instructions lack portfolio allocation framework (MEDIUM)

**The problem:** The daily and weekly prompt `_INSTRUCTIONS` ask Claude to analyze "cross-bot patterns" and "correlation" but provide no structured framework for **what to do about it**. Claude is told to flag issues but not given a methodology for proposing allocation changes.

**Recommendation:** Add to `_WEEKLY_INSTRUCTIONS`:
```
13. PORTFOLIO ALLOCATION ASSESSMENT
    For each bot and each strategy within each bot:
    a. Compute capital efficiency: PnL / (unit_risk × max_positions × days_active)
    b. Compute marginal Sharpe contribution: portfolio Sharpe with vs. without this strategy
    c. If capital efficiency differs >2x between strategies, propose reallocation with:
       - Current allocation and suggested allocation (as % of equity and R-units)
       - Expected Calmar ratio change
       - Minimum observation period before re-evaluating
    d. For same-instrument strategies (e.g., all NQ), assess signal independence:
       - Entry overlap rate (% of entries within 2 bars of each other)
       - Return correlation at trade level (not just daily)
       - If correlation > 0.6, recommend consolidation or differentiation
```

#### Gap A8: No historical allocation tracking (LOW)

**The problem:** Even if the assistant starts proposing allocation changes, there's no historical record of what allocations were active at what times. Without this, it can't evaluate: "We shifted from 50/30/20 to 40/40/20 allocation in February — did portfolio Calmar improve?"

**Recommendation:** Add `findings/allocation_history.jsonl` that records: `{"date", "bot_id", "strategy_id", "allocation_pct", "unit_risk_pct", "heat_cap_r", "source": "manual|suggested"}`. Feed into context builder for temporal analysis.

---

# Part 5: Cross-Cutting Findings

## The Data Gap Dependency Chain

Many assistant-side improvements are **blocked by bot-side data gaps**. The critical dependency chain:

```
Bot-side prerequisites:
  K4 + M7 + S8 (per-strategy P&L in daily snapshot for all 3 bots)
    → Enables: A2 (synergy analysis), A4 (proportion optimization)

  K1 (consistent missed opportunities)
    → Enables: accurate filter sensitivity across all k_stock_trader strategies

  S1 (MFE/MAE tracking)
    → Enables: exit strategy simulation for swing_trader

  K2 + M5 + S2 (params at entry)
    → Enables: reliable WFO parameter attribution

  M4 + S4 + S5 (cross-strategy correlation data, coordinator + overlay logging)
    → Enables: A6 (interaction effects), structural analysis of overlay engine, NQ concentration risk modeling
```

## Priority-Ordered Implementation Roadmap

### Tier 1: Highest ROI (do first)
| # | Where | What | Why |
|---|-------|------|-----|
| 1 | swing_trader | Add MFE/MAE tracking (S1) | Unlocks exit optimization for most diversified bot |
| 2 | k_stock_trader | Consistent missed opportunity integration (K1) | Unlocks filter cost analysis across all 4 strategies |
| 3 | All bots | Add `strategy_params_at_entry` (K2/M5/S2) | Unlocks reliable WFO attribution |
| 4 | k_stock_trader | Add per-strategy P&L to daily snapshot (K4) | **Prerequisite for A2 + A4** — extend DailySnapshot with `per_strategy_summary` for KMP/KPR/NULRIMOK/PCIM |
| 4b | momentum_trader | Add per-strategy P&L to daily snapshot (M7) | **Prerequisite for A2 + A4** — extend DailySnapshot with `per_strategy_summary` for Helix/NQDTC/Vdubus |
| 4c | swing_trader | Add per-strategy P&L to daily snapshot (S8) | **Prerequisite for A2 + A4** — extend DailySnapshot with `per_strategy_summary` for ATRSS/S5_PB/S5_DUAL/Breakout/Helix + overlay summary |
| 5 | trading_assistant | Build `portfolio_allocator.py` (A1) | Direct response to user's primary need |

### Tier 2: High Value (do after Tier 1)
| # | Where | What | Why |
|---|-------|------|-----|
| 6 | trading_assistant | Build `synergy_analyzer.py` (A2) | Cross-strategy redundancy/complementarity |
| 7 | trading_assistant | Build `strategy_proportion_optimizer.py` (A4) | Intra-bot allocation optimization |
| 8 | swing_trader | Log StrategyCoordinator decisions (S4) | Enables interaction effect analysis |
| 9 | swing_trader | Log overlay engine state (S5) | Enables overlay optimization |
| 10 | k_stock_trader | Cross-strategy interaction logging (K3) | OMS rejection attribution |

### Tier 3: Medium Value (do after Tier 2)
| # | Where | What | Why |
|---|-------|------|-----|
| 11 | trading_assistant | Build `structural_analyzer.py` (A3) | Strategy lifecycle + architecture fitness |
| 12 | trading_assistant | Regime-conditional optimization (A5) | Dynamic allocation by market state |
| 13 | momentum_trader | Signal evolution logging (M2) | Signal build-up pattern analysis |
| 14 | trading_assistant | Interaction effect analyzer (A6) | Coordinator rule optimization |
| 15 | trading_assistant | Portfolio allocation prompt framework (A7) | Structured Claude guidance |

### Tier 4: Polish (do last)
| # | Where | What | Why |
|---|-------|------|-----|
| 16 | All bots | Execution cascade timestamps (K5/M1) | Latency component attribution |
| 17 | momentum_trader | Session transition tracking (M3) | Cross-session position behavior |
| 18 | swing_trader | Overnight gap tracking (S6) | Gap risk quantification |
| 19 | trading_assistant | Historical allocation tracking (A8) | Allocation change evaluation |
| 20 | All bots | Experiment A/B tracking (K7) | Automated significance testing |
