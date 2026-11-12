KALCB execution recovery research preset

This opt-in preset was evaluated with the frozen round-3 guarded candidate pool, KRW 100,000,000 initial equity, 2x buying power, and the standard capacity-aware broker. It is not a live promotion or a replacement for the round-1 finalist.

mutations.json is an overlay for scripts/kalcb_bounded_replay.py --mutations; use the reference result and frozen snapshots recorded in provenance.json. effective_config.json is the complete resulting KALCBConfig mapping. Applying only the overlay to a different base recipe or pool does not reproduce the study.

The 13:00 liquidity check uses the lower observed volume order statistic from twelve completed five-minute bars, assumes 0.5% participation, and reserves half the forecast exit capacity. Price-lock handling uses a separately enabled prior-close proxy because these historical snapshots lack sourced dated upper-limit prices. The proxy is a research heuristic, not verified order-book depth. Entry expiry is implemented but disabled in this selected preset after its measured return/frequency trade-off.

Full results and reconstruction evidence: E:/CodexResearch/k_stock_trader/kalcb_execution_recovery_20261005
