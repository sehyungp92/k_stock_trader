from __future__ import annotations

from backtests.auto.shared.types import Experiment

BASE_MUTATIONS = {"fixed_qty": 10, "required_closes": 1}

PHASE_FOCUS = {
    1: ("VWAP depth, HOD drop, panic/drift setup extraction", ["decision_count", "expected_total_r"]),
    2: ("confidence and flow discrimination", ["profit_factor", "win_rate"]),
    3: ("reclaim acceptance and entry timing", ["expectancy", "mfe_capture"]),
    4: ("sizing, time-of-day, stale-flow penalties, sector caps", ["net_profit", "max_drawdown_pct"]),
    5: ("partial/full/trailing/time exits", ["mfe_capture", "profit_factor"]),
    6: ("robustness, cost stress, walk-forward", ["max_drawdown_pct", "profit_factor"]),
}


def get_phase_candidates(phase: int) -> list[Experiment]:
    raw = {
        1: [("setup_more_sensitive", {"required_closes": 1}), ("setup_confirmed", {"required_closes": 2})],
        2: [("confidence_yellow_allowed", {"confidence_mode": "yellow_ok"}), ("confidence_green_only", {"confidence_mode": "green_only"})],
        3: [("one_close_accept", {"required_closes": 1}), ("two_close_accept", {"required_closes": 2})],
        4: [("size_half", {"fixed_qty": 5}), ("size_double", {"fixed_qty": 20})],
        5: [("trail_fast", {"trail_variant": "fast"}), ("trail_patient", {"trail_variant": "patient"})],
        6: [("robust_small_size", {"fixed_qty": 7}), ("robust_confirmed", {"required_closes": 2, "fixed_qty": 8})],
    }.get(phase, [])
    return [Experiment(name, mutations) for name, mutations in raw]

