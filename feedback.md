## K_STOCK_TRADER (Korean Equity Multi-Strategy Bot)

### What It Does Well

- **Comprehensive event capture.** `TradeEvent` in `instrumentation/src/trade_logger.py` captures 30+ fields per trade: entry/exit prices, signal metadata, regime, filters, slippage, latency, strategy params frozen at entry, and full market snapshots at both entry and exit. This is among the richest trade instrumentation I've seen.
- **Missed opportunity tracking with outcome backfill.** `MissedOpportunityLogger` in `instrumentation/src/missed_opportunity.py` records every blocked signal, then asynchronously backfills hypothetical outcomes at 1h/4h/24h using per-strategy simulation policies. This is the single most valuable data source for filter optimization.
- **Process quality scoring.** `ProcessScorer` in `instrumentation/src/process_scorer.py` assigns 0-100 scores with a controlled 21-element root cause taxonomy, per-strategy YAML rules, and separate classification of process vs. outcome. This correctly separates luck from skill at the individual trade level.
- **InstrumentationKit facade.** `instrumentation/facade.py` provides a clean 6-method API (`on_entry_fill`, `on_exit_fill`, `on_signal_blocked`, `periodic_tick`, `build_daily_snapshot`, `shutdown`) that never crashes the strategy. This is exactly how instrumentation should integrate.
- **Sidecar with HMAC signing.** `instrumentation/src/sidecar.py` handles local-first logging, watermark-based dedup, batched forwarding with exponential backoff, and sort_keys canonicalization for HMAC verification. Robust and well-designed.
- **Per-strategy simulation policies in YAML.** Different strategies (KMP, KPR, PCIM, NULRIMOK) have different TP/SL logic and cost assumptions for hypothetical outcome calculation. Simulation assumptions are transparent and auditable.

### Critical Gaps

1. **No signal confluence logging.** The `entry_signal` field is a bare string (e.g., "kmp_breakout"). The orchestrator has no idea what combination of indicators/conditions caused the signal to fire. Without this, the system can optimize when to trade (regime gates, filter thresholds) but cannot assess whether the signal logic itself is good. **Recommendation:** Log the top 3-5 confluence factors that triggered the signal (e.g., `["RSI_oversold", "volume_spike_2.3x", "MA_cross_confirmed", "sector_momentum_positive"]`) with their numeric values. This is the highest-value single improvement across all bots.

2. **No position sizing decision logging.** The system records `position_size` and `position_size_quote` but not the sizing model's inputs: target risk (R), account equity at time, volatility-adjusted size, any scaling factors applied. Without this, the assistant cannot assess whether position sizing is optimal. **Recommendation:** Add `sizing_model: str`, `target_risk_pct: float`, `account_equity: float`, `volatility_at_sizing: float` to TradeEvent.

3. **Active filters logged without threshold context.** `active_filters: ["volume_gate", "spread_gate"]` and `passed_filters: ["regime_gate"]` are recorded, but not the threshold values. The assistant knows a filter was active but can't tell the user "volume_gate threshold is 1.5x — the actual volume was 1.3x, so you missed by 13%". **Recommendation:** Change `active_filters` from `list[str]` to `list[dict]` with `{name, threshold, actual_value, passed: bool}`.

4. **Regime classifier is single-timeframe.** `RegimeClassifier` uses 50-period MA + ADX(14) + ATR percentile on a single timeframe. This misses multi-timeframe regime context (e.g., trending on H1 but ranging on D1). For Korean equities with strong sector rotation, sector-level regime would also be valuable. **Recommendation:** Add a `regime_context` dict to events: `{primary_regime, higher_tf_regime, sector_regime}`.

5. **No error event type forwarded via sidecar.** Instrumentation errors (`instrumentation_errors_*.jsonl`) are logged locally but never forwarded. The sidecar maps the `errors/` directory to the `error` event type, but these are instrumentation failures, not trading bot errors. Actual bot exceptions (OMS failures, API errors, connectivity issues) need a separate, explicitly emitted error event. **Recommendation:** Add an explicit `emit_error(severity, error_type, message, stack_trace)` method to `InstrumentationKit`.

6. **No heartbeat event.** The daily snapshot provides end-of-day health, but there's no real-time heartbeat. If the bot goes silent, the orchestrator won't know until the evening when the daily snapshot is expected. **Recommendation:** Add `emit_heartbeat()` to the facade, called every 30 seconds, forwarded by sidecar as `heartbeat` event type.

7. **KIS REST bid/ask limitation.** MarketSnapshot always records `bid=0, ask=0, spread_bps=0` because KIS REST doesn't provide real-time bid/ask. This means slippage analysis lacks market microstructure context. **Recommendation:** If KIS WebSocket is available, use it for snapshots. If not, document this limitation explicitly in the event so the assistant doesn't draw conclusions from zero spread.

8. **No tracking of how price moved after exit.** The system records exit price but not what happened next. Was the exit premature (price continued favorably)? Was it optimal (price reversed)? The process scorer checks `price_moved_pct` but this field isn't populated by the trade logger. **Recommendation:** Backfill post-exit price movement at 1h/4h intervals (similar to missed opportunity backfill) and populate `price_moved_pct` and `stop_distance_pct` on the exit event.

---

### Highest Impact (Bot-Side Data Capture)

1. **Add signal confluence logging to all bots.** Change `entry_signal: str` to `entry_signal: str` + `signal_factors: list[dict]` with `{factor_name, factor_value, threshold, contribution}`. This is the single change that would most expand the system's improvement capability.

2. **Add filter threshold context.** Change `active_filters: list[str]` to `filter_decisions: list[dict]` with `{filter_name, threshold, actual_value, passed, margin_pct}`. This transforms filter optimization from "should we keep this filter?" to "what threshold maximizes edge?"

3. **Add position sizing inputs.** Log `target_risk_pct`, `account_equity`, `volatility_basis`, `sizing_model` on every trade.

4. **Create InstrumentationKit facades for momentum_trader and swing_trader.** All 3 bots should have the same clean integration API.

5. **Add post-exit price tracking.** Backfill 1h/4h post-exit price movement on completed trades.