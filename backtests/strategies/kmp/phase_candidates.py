from __future__ import annotations

from backtests.auto.shared.types import Experiment

BASE_MUTATIONS = {
    "fixed_qty": 10,
    "target_pct": 0.03,
    "accept_timeout_min": 10.0,
    "spread_bps": 8.0,
}

PHASE_FOCUS = {
    1: ("signal extraction, OR range, surge, trend anchor", ["decision_count", "expected_total_r"]),
    2: ("discrimination gates, RVOL, spread, VI, regime breadth, chop", ["profit_factor", "max_drawdown_pct"]),
    3: ("acceptance and entry mechanics", ["expectancy", "mfe_capture"]),
    4: ("sizing, liquidity cap, NAV cap, sector cap", ["net_profit", "max_drawdown_pct"]),
    5: ("trade management and exits", ["mfe_capture", "expected_total_r"]),
    6: ("robustness, costs, walk-forward, stress", ["profit_factor", "max_drawdown_pct"]),
}


def get_phase_candidates(phase: int) -> list[Experiment]:
    raw = {
        1: [
            ("surge_threshold_relaxed", {"min_surge_base": 1.2}),
            ("surge_threshold_strict", {"min_surge_base": 2.0}),
        ],
        2: [
            ("spread_gate_tighter", {"spread_bps": 5.0}),
            ("spread_gate_wider", {"spread_bps": 12.0}),
        ],
        3: [
            ("acceptance_fast", {"accept_timeout_min": 5.0}),
            ("acceptance_patient", {"accept_timeout_min": 15.0}),
        ],
        4: [
            ("size_half", {"fixed_qty": 5}),
            ("size_double", {"fixed_qty": 20}),
        ],
        5: [
            ("target_fast", {"target_pct": 0.02}),
            ("target_patient", {"target_pct": 0.045}),
        ],
        6: [
            ("robust_small_size", {"fixed_qty": 7, "target_pct": 0.025}),
            ("robust_cost_buffer", {"spread_bps": 6.0, "accept_timeout_min": 8.0}),
        ],
    }.get(phase, [])
    return [Experiment(name, mutations) for name, mutations in raw]

