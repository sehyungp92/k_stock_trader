# Live/Backtest Parity Alignment

Status: pass
Scope: portfolio_arbitration_layer

This pass means the portfolio arbitration layer no longer depends on post-trade KALCB/OLR outcomes for live admission or sizing. It is not a substitute for a full source-engine replay or paper/live OMS parity audit.

## Checks

- PASS: score_component_count_lte_7
- PASS: source_fingerprint_present
- PASS: feature_manifest_hash_present
- PASS: candidate_snapshot_hash_present
- PASS: holdout_excluded_from_optimization
- PASS: no_same_bar_fills
- PASS: no_forced_replay_closes
- PASS: no_rejected_orders
- PASS: no_end_open_positions
- PASS: legacy_eod_outcome_boost_not_active
- PASS: portfolio_rules_use_live_known_fields

## Field Policy

Live-known fields: `strategy,symbol,sector,session_index,entry_minute,rank_bucket,route,score_band,expected_r,quality_score,gross_unit,sector_confirmed,symbol_confirmed,conflict,early_exit_reason`

Offline-only diagnostics fields: `net_r,mae_r,mfe_r,pnl_cash,weighted_r,blocked_avg_r,accepted_avg_r`