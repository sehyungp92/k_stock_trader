from __future__ import annotations

import math

from backtests.auto.shared.types import GateCriterion


ULTIMATE_TARGETS = {
    "expected_total_r": 8.0,
    "official_mtm_net_return_pct": 0.0060,
    "profit_factor": 1.45,
    "entry_count": 60.0,
    "avg_r": 0.10,
    "mfe_capture": 0.35,
    "max_drawdown_pct": 0.0080,
}

IMMUTABLE_SCORE_COMPONENTS = {
    "official_mtm_net_return_pct": 0.24,
    "expected_total_r": 0.22,
    "profit_factor": 0.16,
    "avg_r": 0.14,
    "entry_count": 0.12,
    "mfe_capture": 0.06,
    "max_drawdown_pct": -0.06,
}

PHASE_HARD_REJECTS = {
    phase: {
        "min_trades": 12,
        "max_dd_pct": 0.0120,
        "min_pf": 0.95,
        "max_same_bar_fills": 0,
    }
    for phase in range(1, 7)
}
PHASE_HARD_REJECTS[5] = {**PHASE_HARD_REJECTS[5], "min_trades": 18}


def score_kalcb_phase(phase: int, metrics: dict[str, float], weights: dict[str, float] | None = None) -> float:
    del phase
    components = weights or IMMUTABLE_SCORE_COMPONENTS
    if len(components) > 7:
        raise ValueError("KALCB immutable score supports at most 7 components")
    score = 0.0
    for name, weight in components.items():
        score += float(weight) * _scaled_component(name, metrics)
    return float(100.0 * score)


def kalcb_reject_reason(phase: int, metrics: dict[str, float], hard_rejects: dict[str, float] | None = None) -> str:
    hard = hard_rejects or PHASE_HARD_REJECTS.get(phase, {})
    total = _entry_frequency(metrics)
    if total < float(hard.get("min_trades", 0.0)):
        return f"phase{phase}_too_few_entries ({total:.0f} < {float(hard.get('min_trades', 0.0)):.0f})"
    if float(metrics.get("same_bar_fill_count", 0.0)) > float(hard.get("max_same_bar_fills", 0.0)):
        return f"phase{phase}_same_bar_fill"
    dd = float(metrics.get("max_drawdown_pct", 0.0))
    if "max_dd_pct" in hard and dd > float(hard["max_dd_pct"]):
        return f"phase{phase}_max_dd ({dd:.2%} > {float(hard['max_dd_pct']):.2%})"
    pf = float(metrics.get("profit_factor", 0.0))
    if "min_pf" in hard and pf < float(hard["min_pf"]):
        return f"phase{phase}_low_pf ({pf:.2f} < {float(hard['min_pf']):.2f})"
    return ""


def gate_criteria(phase: int, metrics: dict[str, float], hard_rejects: dict[str, float] | None = None) -> list[GateCriterion]:
    hard = hard_rejects or PHASE_HARD_REJECTS.get(phase, {})
    min_trades = float(hard.get("min_trades", 12.0))
    entries = _entry_frequency(metrics)
    return [
        GateCriterion("hard_entry_count", min_trades, entries, entries >= min_trades),
        GateCriterion("hard_same_bar_fills", float(hard.get("max_same_bar_fills", 0.0)), float(metrics.get("same_bar_fill_count", 0.0)), float(metrics.get("same_bar_fill_count", 0.0)) <= float(hard.get("max_same_bar_fills", 0.0))),
        GateCriterion("max_drawdown_pct", float(hard.get("max_dd_pct", 0.0100)), float(metrics.get("max_drawdown_pct", 0.0)), float(metrics.get("max_drawdown_pct", 0.0)) <= float(hard.get("max_dd_pct", 0.0100))),
        GateCriterion("profit_factor", 1.0, float(metrics.get("profit_factor", 0.0)), float(metrics.get("profit_factor", 0.0)) >= 1.0),
        GateCriterion("expected_total_r", 0.0, float(metrics.get("expected_total_r", 0.0)), float(metrics.get("expected_total_r", 0.0)) >= 0.0),
        GateCriterion("live_parity", 0.0, float(metrics.get("same_bar_fill_count", 0.0)), float(metrics.get("same_bar_fill_count", 0.0)) == 0.0),
    ]


def _scaled_component(name: str, metrics: dict[str, float]) -> float:
    value = _official_return_pct(metrics) if name in {"net_return_pct", "official_mtm_net_return_pct"} else float(metrics.get(name, 0.0))
    if name in {"net_return_pct", "official_mtm_net_return_pct"}:
        return math.tanh(value / 0.0100)
    if name == "expected_total_r":
        return math.tanh(value / 10.0)
    if name == "entry_count":
        return math.tanh(max(value - 20.0, 0.0) / 55.0)
    if name == "profit_factor":
        return _clip((min(value, 3.5) - 1.05) / 1.20)
    if name == "avg_r":
        return _clip((value - 0.02) / 0.20)
    if name == "mfe_capture":
        return _clip((value - 0.30) / 0.30)
    if name == "max_drawdown_pct":
        return _clip(value / 0.0120, 0.0, 2.0)
    return value


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _entry_frequency(metrics: dict[str, float]) -> float:
    entries = float(metrics.get("entry_count", 0.0) or 0.0)
    return entries if entries > 0 else float(metrics.get("total_trades", 0.0) or 0.0)


def _official_return_pct(metrics: dict[str, float]) -> float:
    if metrics.get("official_mtm_net_return_pct") is not None:
        return float(metrics.get("official_mtm_net_return_pct", 0.0) or 0.0)
    if metrics.get("net_return_pct_basis"):
        return float(metrics.get("net_return_pct", 0.0) or 0.0)
    return 0.0
