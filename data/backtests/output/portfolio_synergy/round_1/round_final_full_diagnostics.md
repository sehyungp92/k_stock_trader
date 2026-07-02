# KALCB + OLR Portfolio Synergy Full End-of-Round Diagnostics

Generated: 2026-05-27T10:52:57.391921+00:00
Round: 1
Promotion status: research_only
Parity status: pass

## Headline Metrics

- Score: 113.7233
- Return: 461.24%
- Trades/21 sessions: 45.77
- Accepted/blocked: 473/63
- Positive-alpha block rate: 6.90%
- Accepted avg R vs blocked avg R: +1.245 vs +0.732
- Max drawdown: 4.13%
- Profit factor: 4.60

## Selected Phases

- Phase 1: score 94.7031 -> 97.3455; kept p1_light_agreement_boost, p1_kalcb_olr_55_45_risk
- Phase 2: score 97.3455 -> 101.8273; kept p2_gross_cap_2p10
- Phase 3: score 101.8273 -> 101.9607; kept p3_kalcb_rank6_10_confirmed_half_size
- Phase 4: score 101.9607 -> 103.4951; kept p4_confirmed_kalcb_size_112, p4_agreement_boost_118, p4_olr_boost_after_kalcb_strong_live_path_115
- Phase 5: score 0.0000 -> 108.7883; kept p5_kalcb_secondary_confirmed, p5_olr_slot6_guarded
- Phase 6: score 108.7883 -> 113.7233; kept p6_reference_risk_62bp_probe, p6_kalcb_confirmed_size_125
- Phase 7: score 113.7233 -> 113.7233; kept (none)

## Block Diagnostics

```json
{
  "daily_gross_cap": 63
}
```

## Live/Backtest Parity

This pass means the portfolio arbitration layer no longer depends on post-trade KALCB/OLR outcomes for live admission or sizing. It is not a substitute for a full source-engine replay or paper/live OMS parity audit.

```json
[
  {
    "details": "",
    "name": "score_component_count_lte_7",
    "passed": true
  },
  {
    "details": "",
    "name": "source_fingerprint_present",
    "passed": true
  },
  {
    "details": "",
    "name": "feature_manifest_hash_present",
    "passed": true
  },
  {
    "details": "",
    "name": "candidate_snapshot_hash_present",
    "passed": true
  },
  {
    "details": "",
    "name": "holdout_excluded_from_optimization",
    "passed": true
  },
  {
    "details": "",
    "name": "no_same_bar_fills",
    "passed": true
  },
  {
    "details": "",
    "name": "no_forced_replay_closes",
    "passed": true
  },
  {
    "details": "",
    "name": "no_rejected_orders",
    "passed": true
  },
  {
    "details": "",
    "name": "no_end_open_positions",
    "passed": true
  },
  {
    "details": "",
    "name": "legacy_eod_outcome_boost_not_active",
    "passed": true
  },
  {
    "details": "",
    "name": "portfolio_rules_use_live_known_fields",
    "passed": true
  }
]
```