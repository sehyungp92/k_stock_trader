from __future__ import annotations

from backtests.auto.shared.types import Experiment

BASE_MUTATIONS = {"fixed_qty": 10, "confirm_bars": 3, "target_pct": 0.04, "band_lower_mult": 0.985, "band_upper_mult": 1.010}

PHASE_FOCUS = {
    1: ("DSE signal extraction, regime, flow, RS, trend, AVWAP", ["decision_count", "expected_total_r"]),
    2: ("ranking, tradable cut, active set, overflow rotation", ["profit_factor", "expected_total_r"]),
    3: ("IEPE entry band, volume dry-up, confirmation", ["expectancy", "mfe_capture"]),
    4: ("sizing, risk budget, exposure headroom, sector cap", ["net_profit", "max_drawdown_pct"]),
    5: ("setup-aware exits, flow reversal, orphan exits, time stops", ["mfe_capture", "profit_factor"]),
    6: ("multiday robustness, costs, walk-forward", ["max_drawdown_pct", "profit_factor"]),
}


def get_phase_candidates(phase: int) -> list[Experiment]:
    raw = {
        1: [("avwap_band_wide", {"band_lower_mult": 0.980, "band_upper_mult": 1.015}), ("avwap_band_tight", {"band_lower_mult": 0.990, "band_upper_mult": 1.006})],
        2: [("active_set_conservative", {"active_set_k": 3}), ("active_set_expanded", {"active_set_k": 6})],
        3: [("confirm_fast", {"confirm_bars": 2}), ("confirm_patient", {"confirm_bars": 4})],
        4: [("size_half", {"fixed_qty": 5}), ("size_double", {"fixed_qty": 20})],
        5: [("target_fast", {"target_pct": 0.025}), ("target_patient", {"target_pct": 0.055})],
        6: [("robust_small_size", {"fixed_qty": 7, "confirm_bars": 3}), ("robust_tight_band", {"band_lower_mult": 0.990, "band_upper_mult": 1.008})],
    }.get(phase, [])
    return [Experiment(name, mutations) for name, mutations in raw]

