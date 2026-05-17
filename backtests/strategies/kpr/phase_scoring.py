from __future__ import annotations

from backtests.auto.shared.types import GateCriterion
from backtests.strategies.common.scoring import hard_reject_reason, score_metrics

ULTIMATE_TARGETS = {
    "expectancy": 1_000.0,
    "profit_factor": 1.35,
    "expected_total_r": 0.5,
    "mfe_capture": 0.30,
    "max_drawdown_pct": 0.06,
}

PHASE_HARD_REJECTS = {phase: {"min_trades": 1, "max_dd_pct": 0.14, "max_same_bar_fills": 0} for phase in range(1, 7)}

PHASE_SCORING_WEIGHTS = {
    phase: {"expectancy": 0.30, "profit_factor": 0.25, "expected_total_r": 0.20, "mfe_capture": 0.15, "max_drawdown_pct": -0.10}
    for phase in range(1, 7)
}


def score_kpr_phase(phase: int, metrics: dict[str, float], weights: dict[str, float] | None = None) -> float:
    return score_metrics(metrics, weights or PHASE_SCORING_WEIGHTS.get(phase))


def kpr_reject_reason(phase: int, metrics: dict[str, float], hard_rejects: dict[str, float] | None = None) -> str:
    return hard_reject_reason(metrics, hard_rejects or PHASE_HARD_REJECTS.get(phase, {}), phase=phase)


def gate_criteria(phase: int, metrics: dict[str, float], hard_rejects: dict[str, float] | None = None) -> list[GateCriterion]:
    hard = hard_rejects or PHASE_HARD_REJECTS.get(phase, {})
    return [
        GateCriterion("hard_total_trades", float(hard.get("min_trades", 0.0)), float(metrics.get("total_trades", 0.0)), float(metrics.get("total_trades", 0.0)) >= float(hard.get("min_trades", 0.0))),
        GateCriterion("hard_same_bar_fills", float(hard.get("max_same_bar_fills", 0.0)), float(metrics.get("same_bar_fill_count", 0.0)), float(metrics.get("same_bar_fill_count", 0.0)) <= float(hard.get("max_same_bar_fills", 0.0))),
        GateCriterion("max_drawdown_pct", float(hard.get("max_dd_pct", 1.0)), float(metrics.get("max_drawdown_pct", 0.0)), float(metrics.get("max_drawdown_pct", 0.0)) <= float(hard.get("max_dd_pct", 1.0))),
        GateCriterion("profit_factor", 1.0, float(metrics.get("profit_factor", 0.0)), float(metrics.get("profit_factor", 0.0)) >= 1.0),
    ]

