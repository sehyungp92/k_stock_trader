from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

SCORE_WEIGHTS: dict[str, float] = {
    "capital_normalized_mtm_return": 0.24,
    "risk_normalized_total_r": 0.16,
    "trade_frequency": 0.17,
    "block_selectivity": 0.15,
    "drawdown_control": 0.11,
    "profit_quality": 0.10,
    "robust_balance": 0.07,
}

SCORE_COMPONENTS: tuple[str, ...] = tuple(SCORE_WEIGHTS)

BASELINE_TARGETS: dict[str, float] = {
    "isolated_baseline_return_pct": 4.1728475583307265,
    "isolated_baseline_total_r": 690.0,
    "target_trades_per_21_sessions": 40.0,
    "drawdown_comfort_pct": 0.06,
    "drawdown_hard_pct": 0.14,
}


@dataclass(frozen=True)
class PortfolioScore:
    components: dict[str, float]
    total: float
    rejected: bool = False
    reject_reason: str = ""


def score_portfolio_metrics(
    metrics: dict[str, Any],
    *,
    scoring_weights: dict[str, float] | None = None,
    hard_rejects: dict[str, float] | None = None,
) -> PortfolioScore:
    weights = _normalise_weights(scoring_weights or SCORE_WEIGHTS)
    if len(weights) > 7:
        raise ValueError(f"Portfolio synergy score has {len(weights)} components; max is 7.")

    components = _components(metrics)
    rejection = _reject_reason(metrics, hard_rejects or {})
    total = 100.0 * sum(weights.get(name, 0.0) * components.get(name, 0.0) for name in SCORE_COMPONENTS)
    if rejection:
        return PortfolioScore(components=components, total=0.0, rejected=True, reject_reason=rejection)
    return PortfolioScore(components=components, total=total)


def _components(metrics: dict[str, Any]) -> dict[str, float]:
    isolated_return = max(0.90 * _metric(metrics, "isolated_baseline_return_pct", BASELINE_TARGETS["isolated_baseline_return_pct"]), 1.0)
    isolated_total_r = max(0.88 * _metric(metrics, "isolated_baseline_total_r", BASELINE_TARGETS["isolated_baseline_total_r"]), 1.0)
    trades_target = _metric(metrics, "target_trades_per_21_sessions", BASELINE_TARGETS["target_trades_per_21_sessions"])

    net_return = _metric(metrics, "official_mtm_net_return_pct", _metric(metrics, "net_return_pct"))
    total_r = _metric(metrics, "total_r")
    trades_per_21 = _metric(metrics, "trades_per_21_sessions")
    accepted_avg_r = _metric(metrics, "accepted_avg_r")
    blocked_avg_r = _metric(metrics, "blocked_avg_r", accepted_avg_r - 0.10)
    positive_alpha_block_rate = _metric(metrics, "positive_alpha_block_rate")
    max_dd = _metric(metrics, "max_drawdown_pct")
    profit_factor = _metric(metrics, "profit_factor")
    positive_slices = _metric(metrics, "positive_slices")
    min_strategy_capture = _metric(metrics, "min_strategy_trade_capture")
    max_strategy_r_share = _metric(metrics, "max_strategy_r_share")

    separation = accepted_avg_r - blocked_avg_r
    block_selectivity = 0.55 * _clip(separation / 0.35, 0.0, 1.20) + 0.45 * _clip(1.0 - positive_alpha_block_rate / 0.18, 0.0, 1.10)
    drawdown_control = 1.15 if max_dd <= 0.06 else 1.15 * _clip((0.14 - max_dd) / 0.08)
    balance = (
        _clip(positive_slices / 4.0)
        + _clip(min_strategy_capture / 0.70)
        + _clip(1.0 - max(0.0, max_strategy_r_share - 0.70) / 0.30)
    ) / 3.0

    return {
        "capital_normalized_mtm_return": _clip(net_return / isolated_return, 0.0, 1.30),
        "risk_normalized_total_r": _clip(total_r / isolated_total_r, 0.0, 1.25),
        "trade_frequency": _clip(trades_per_21 / max(trades_target, 1.0), 0.0, 1.25),
        "block_selectivity": _clip(block_selectivity, 0.0, 1.20),
        "drawdown_control": _clip(drawdown_control, 0.0, 1.15),
        "profit_quality": _clip((profit_factor - 1.25) / (2.75 - 1.25), 0.0, 1.15),
        "robust_balance": _clip(balance, 0.0, 1.10),
    }


def _reject_reason(metrics: dict[str, Any], hard_rejects: dict[str, float]) -> str:
    for metric_name in ("same_bar_fill_count", "forced_replay_close_count", "rejected_order_count"):
        limit = float(hard_rejects.get(f"max_{metric_name}", 0.0) or 0.0)
        actual = _metric(metrics, metric_name)
        if actual > limit:
            return f"{metric_name} {actual:.0f} > {limit:.0f}"

    max_dd = hard_rejects.get("max_drawdown_pct")
    if max_dd is not None and _metric(metrics, "max_drawdown_pct") > float(max_dd):
        return f"max drawdown {_metric(metrics, 'max_drawdown_pct'):.2%} > {float(max_dd):.2%}"

    min_trades_per_21 = hard_rejects.get("min_trades_per_21_sessions")
    if min_trades_per_21 is not None and _metric(metrics, "trades_per_21_sessions") < float(min_trades_per_21):
        return f"trades/21 {_metric(metrics, 'trades_per_21_sessions'):.2f} < {float(min_trades_per_21):.2f}"

    min_pf = hard_rejects.get("min_profit_factor")
    if min_pf is not None and _metric(metrics, "profit_factor") < float(min_pf):
        return f"profit factor {_metric(metrics, 'profit_factor'):.2f} < {float(min_pf):.2f}"

    max_block_rate = hard_rejects.get("max_block_rate")
    if max_block_rate is not None and _metric(metrics, "block_rate") > float(max_block_rate):
        return f"block rate {_metric(metrics, 'block_rate'):.2%} > {float(max_block_rate):.2%}"

    max_positive_alpha_block_rate = hard_rejects.get("max_positive_alpha_block_rate")
    if max_positive_alpha_block_rate is not None and _metric(metrics, "positive_alpha_block_rate") > float(max_positive_alpha_block_rate):
        return (
            "positive-alpha block rate "
            f"{_metric(metrics, 'positive_alpha_block_rate'):.2%} > {float(max_positive_alpha_block_rate):.2%}"
        )

    min_strategy_capture = hard_rejects.get("min_strategy_trade_capture")
    if min_strategy_capture is not None and _metric(metrics, "min_strategy_trade_capture") < float(min_strategy_capture):
        return (
            f"min strategy capture {_metric(metrics, 'min_strategy_trade_capture'):.2%} "
            f"< {float(min_strategy_capture):.2%}"
        )

    if _metric(metrics, "entries_blocked_by_portfolio") > 0 and bool(hard_rejects.get("require_accepted_avg_r_gt_blocked_avg_r", 1.0)):
        if _metric(metrics, "accepted_avg_r") <= _metric(metrics, "blocked_avg_r"):
            return f"accepted avg R {_metric(metrics, 'accepted_avg_r'):.3f} <= blocked avg R {_metric(metrics, 'blocked_avg_r'):.3f}"

    return ""


def _normalise_weights(weights: dict[str, float]) -> dict[str, float]:
    unexpected = sorted(set(weights) - set(SCORE_COMPONENTS))
    if unexpected:
        raise ValueError(f"Unknown score component(s): {', '.join(unexpected)}")
    total = sum(float(value) for value in weights.values())
    if total <= 0:
        return {name: 1.0 / len(SCORE_COMPONENTS) for name in SCORE_COMPONENTS}
    return {name: float(weights.get(name, 0.0)) / total for name in SCORE_COMPONENTS}


def _metric(metrics: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = float(metrics.get(key, default) if metrics.get(key, default) is not None else default)
    except (TypeError, ValueError):
        return float(default)
    return value if isfinite(value) else float(default)


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    if not isfinite(value):
        return low
    return min(max(value, low), high)

