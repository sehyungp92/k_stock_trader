from __future__ import annotations

from typing import Any

from backtests.auto.shared.types import GateCriterion


ULTIMATE_TARGETS: dict[str, float] = {
    # Round 2 Phase 5/6 candidates clear 100% MTM, so the return target must
    # stay above the active frontier or the score stops seeing real alpha.
    "official_mtm_net_return_pct": 1.50,
    # Entry-level R is net of partial-leg accounting inflation. At the target
    # trade frequency this implies roughly +0.18R per entry.
    "expected_total_r": 65.0,
    "total_trades": 320.0,
    "profit_factor": 3.00,
    "max_drawdown_pct": 0.10,
    "olr_alpha_capture": 0.40,
    "olr_discrimination_quality": 0.70,
}

# Keep the immutable score compact: return, realized R, frequency, quality,
# drawdown, alpha capture, and signal discrimination.
IMMUTABLE_SCORE_COMPONENTS: dict[str, float] = {
    "official_mtm_net_return_pct": 0.24,
    "expected_total_r": 0.15,
    "total_trades": 0.13,
    "profit_factor": 0.08,
    "max_drawdown_pct": 0.12,
    "olr_alpha_capture": 0.16,
    "olr_discrimination_quality": 0.12,
}


PHASE_HARD_REJECTS: dict[int, dict[str, float]] = {
    1: {
        "min_total_trades": 130.0,
        "min_profit_factor": 1.03,
        "max_max_drawdown_pct": 0.170,
        "max_same_bar_fill_count": 0.0,
        "max_rejected_order_count": 0.0,
        "max_forced_replay_close_count": 0.0,
        "max_end_open_position_count": 0.0,
    },
    2: {
        "min_total_trades": 135.0,
        "min_profit_factor": 1.05,
        "max_max_drawdown_pct": 0.165,
        "max_same_bar_fill_count": 0.0,
        "max_rejected_order_count": 0.0,
        "max_forced_replay_close_count": 0.0,
        "max_end_open_position_count": 0.0,
    },
    3: {
        "min_total_trades": 220.0,
        "min_profit_factor": 1.06,
        "max_max_drawdown_pct": 0.160,
        "max_same_bar_fill_count": 0.0,
        "max_rejected_order_count": 0.0,
        "max_forced_replay_close_count": 0.0,
        "max_end_open_position_count": 0.0,
    },
    4: {
        "min_total_trades": 125.0,
        "min_profit_factor": 1.07,
        "max_max_drawdown_pct": 0.155,
        "max_same_bar_fill_count": 0.0,
        "max_rejected_order_count": 0.0,
        "max_forced_replay_close_count": 0.0,
        "max_end_open_position_count": 0.0,
    },
    5: {
        "min_total_trades": 125.0,
        "min_profit_factor": 1.08,
        "max_max_drawdown_pct": 0.155,
        "max_same_bar_fill_count": 0.0,
        "max_rejected_order_count": 0.0,
        "max_forced_replay_close_count": 0.0,
        "max_end_open_position_count": 0.0,
    },
    6: {
        "min_total_trades": 230.0,
        "min_profit_factor": 1.10,
        "max_max_drawdown_pct": 0.150,
        "max_same_bar_fill_count": 0.0,
        "max_rejected_order_count": 0.0,
        "max_forced_replay_close_count": 0.0,
        "max_end_open_position_count": 0.0,
    },
}


def score_olr_phase(
    phase: int,
    metrics: dict[str, Any],
    scoring_weights: dict[str, float] | None = None,
) -> float:
    del phase
    weights = _normalized_weights(scoring_weights or IMMUTABLE_SCORE_COMPONENTS)
    components = {
        "official_mtm_net_return_pct": _return_score(_f(metrics, "official_mtm_net_return_pct", _f(metrics, "net_return_pct")), target=ULTIMATE_TARGETS["official_mtm_net_return_pct"]),
        "expected_total_r": _linear_score(_expected_total_r(metrics), target=ULTIMATE_TARGETS["expected_total_r"]),
        "total_trades": _linear_score(_entry_frequency(metrics), target=ULTIMATE_TARGETS["total_trades"]),
        "profit_factor": _profit_factor_score(_profit_factor(metrics)),
        "max_drawdown_pct": _drawdown_score(_f(metrics, "max_drawdown_pct", _f(metrics, "official_mtm_max_drawdown_pct"))),
        "olr_alpha_capture": _linear_score(_alpha_capture(metrics), target=ULTIMATE_TARGETS["olr_alpha_capture"]),
        "olr_discrimination_quality": _linear_score(_f(metrics, "olr_discrimination_quality"), target=ULTIMATE_TARGETS["olr_discrimination_quality"]),
    }
    return 100.0 * sum(weights.get(key, 0.0) * components.get(key, 0.0) for key in weights)


def olr_reject_reason(
    phase: int,
    metrics: dict[str, Any],
    hard_rejects: dict[str, float] | None = None,
) -> str:
    hard_rejects = hard_rejects or PHASE_HARD_REJECTS.get(phase, {})
    for criterion in gate_criteria(phase, metrics, hard_rejects):
        if criterion.name.startswith("hard_") and not criterion.passed:
            return f"{criterion.name} ({criterion.actual:.6g} vs {criterion.target:.6g})"
    return ""


def gate_criteria(
    phase: int,
    metrics: dict[str, Any],
    hard_rejects: dict[str, float] | None = None,
) -> list[GateCriterion]:
    hard_rejects = hard_rejects or PHASE_HARD_REJECTS.get(phase, {})
    criteria = [_criterion_from_reject(key, target, metrics) for key, target in hard_rejects.items()]
    criteria.extend(
        [
            GateCriterion(
                "official_mtm_net_return_pct",
                _phase_floor(phase, "official_mtm_net_return_pct"),
                _f(metrics, "official_mtm_net_return_pct", _f(metrics, "net_return_pct")),
                _f(metrics, "official_mtm_net_return_pct", _f(metrics, "net_return_pct")) >= _phase_floor(phase, "official_mtm_net_return_pct"),
            ),
            GateCriterion(
                "expected_total_r",
                _phase_floor(phase, "expected_total_r"),
                _expected_total_r(metrics),
                _expected_total_r(metrics) >= _phase_floor(phase, "expected_total_r"),
            ),
            GateCriterion(
                "olr_alpha_capture",
                _phase_floor(phase, "olr_alpha_capture"),
                _alpha_capture(metrics),
                _alpha_capture(metrics) >= _phase_floor(phase, "olr_alpha_capture"),
            ),
            GateCriterion(
                "olr_discrimination_quality",
                _phase_floor(phase, "olr_discrimination_quality"),
                _f(metrics, "olr_discrimination_quality"),
                _f(metrics, "olr_discrimination_quality") >= _phase_floor(phase, "olr_discrimination_quality"),
            ),
            GateCriterion(
                "selected_negative_label_share",
                _phase_negative_label_ceiling(phase),
                _f(metrics, "selected_negative_label_share", _f(metrics, "olr_selected_negative_label_share")),
                _f(metrics, "selected_negative_label_share", _f(metrics, "olr_selected_negative_label_share")) <= _phase_negative_label_ceiling(phase),
            ),
        ]
    )
    return criteria


def _criterion_from_reject(key: str, target: float, metrics: dict[str, Any]) -> GateCriterion:
    if key.startswith("min_"):
        metric = key[4:]
        actual = _metric_value(metrics, metric)
        return GateCriterion(f"hard_{metric}", float(target), actual, actual >= float(target))
    if key.startswith("max_"):
        metric = key[4:]
        actual = _metric_value(metrics, metric)
        return GateCriterion(f"hard_{metric}", float(target), actual, actual <= float(target))
    actual = _metric_value(metrics, key)
    return GateCriterion(f"hard_{key}", float(target), actual, actual >= float(target))


def _phase_floor(phase: int, key: str) -> float:
    ratios = {
        1: 0.46,
        2: 0.48,
        3: 0.48,
        4: 0.50,
        5: 0.52,
        6: 0.55,
    }
    if key == "olr_alpha_capture":
        return {1: 0.03, 2: 0.04, 3: 0.05, 4: 0.06, 5: 0.06, 6: 0.07}.get(phase, 0.05)
    if key == "olr_discrimination_quality":
        return {1: 0.12, 2: 0.14, 3: 0.14, 4: 0.15, 5: 0.15, 6: 0.16}.get(phase, 0.14)
    return ULTIMATE_TARGETS[key] * ratios.get(phase, 0.50)


def _phase_negative_label_ceiling(phase: int) -> float:
    return {1: 0.72, 2: 0.70, 3: 0.70, 4: 0.69, 5: 0.69, 6: 0.68}.get(phase, 0.70)


def _metric_value(metrics: dict[str, Any], metric: str) -> float:
    if metric == "total_trades":
        return _entry_frequency(metrics)
    if metric == "profit_factor":
        return _profit_factor(metrics)
    if metric == "max_drawdown_pct":
        return _f(metrics, "max_drawdown_pct", _f(metrics, "official_mtm_max_drawdown_pct"))
    if metric == "olr_alpha_capture":
        return _alpha_capture(metrics)
    if metric == "expected_total_r":
        return _expected_total_r(metrics)
    if metric == "selected_negative_label_share":
        return _f(metrics, "selected_negative_label_share", _f(metrics, "olr_selected_negative_label_share"))
    return _f(metrics, metric)


def _normalized_weights(weights: dict[str, float]) -> dict[str, float]:
    allowed = {key: float(value) for key, value in dict(weights or {}).items() if key in IMMUTABLE_SCORE_COMPONENTS}
    total = sum(max(value, 0.0) for value in allowed.values())
    if total <= 0.0:
        return dict(IMMUTABLE_SCORE_COMPONENTS)
    return {key: max(value, 0.0) / total for key, value in allowed.items()}


def _expected_total_r(metrics: dict[str, Any]) -> float:
    if _has_metric(metrics, "entry_level_expected_total_r"):
        return _f(metrics, "entry_level_expected_total_r")
    if _has_metric(metrics, "expected_total_r"):
        return _f(metrics, "expected_total_r")
    return _f(metrics, "entry_level_avg_r", _f(metrics, "avg_r")) * _entry_frequency(metrics)


def _entry_frequency(metrics: dict[str, Any]) -> float:
    if _has_metric(metrics, "entry_fill_count"):
        return _f(metrics, "entry_fill_count")
    if _has_metric(metrics, "entry_level_trade_count"):
        return _f(metrics, "entry_level_trade_count")
    return _f(metrics, "total_trades")


def _profit_factor(metrics: dict[str, Any]) -> float:
    if _has_metric(metrics, "entry_level_profit_factor"):
        return _f(metrics, "entry_level_profit_factor")
    return _f(metrics, "profit_factor")


def _alpha_capture(metrics: dict[str, Any]) -> float:
    if _has_metric(metrics, "olr_alpha_capture"):
        return _f(metrics, "olr_alpha_capture")
    if _has_metric(metrics, "mfe_capture"):
        return _f(metrics, "mfe_capture")
    return _f(metrics, "close_to_close_alpha_capture_pct")


def _linear_score(value: float, *, target: float) -> float:
    return _clip01(float(value) / max(float(target), 1e-9))


def _return_score(value: float, *, target: float) -> float:
    floor = -0.12
    return _clip01((float(value) - floor) / (max(float(target), 1e-9) - floor))


def _profit_factor_score(value: float) -> float:
    return _clip01((float(value) - 0.95) / (ULTIMATE_TARGETS["profit_factor"] - 0.95))


def _drawdown_score(value: float) -> float:
    drawdown = abs(float(value))
    if drawdown <= 0.07:
        return 1.0
    if drawdown >= 0.18:
        return 0.0
    return _clip01(1.0 - (drawdown - 0.07) / 0.11)


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _f(metrics: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = metrics.get(key, default)
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _has_metric(metrics: dict[str, Any], key: str) -> bool:
    value = metrics.get(key)
    return key in metrics and value is not None and value != ""
