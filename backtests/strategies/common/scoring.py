from __future__ import annotations


def score_metrics(metrics: dict[str, float], weights: dict[str, float] | None = None) -> float:
    weights = weights or {
        "expectancy": 0.30,
        "profit_factor": 0.22,
        "expected_total_r": 0.16,
        "mfe_capture": 0.12,
        "win_rate": 0.10,
        "max_drawdown_pct": -0.10,
    }
    score = 0.0
    for key, weight in weights.items():
        value = float(metrics.get(key, 0.0))
        if key == "profit_factor":
            value = min(value, 5.0) / 5.0
        elif key in {"win_rate", "mfe_capture", "max_drawdown_pct"}:
            value = min(max(value, 0.0), 1.0)
        elif key == "expected_total_r":
            value = value / 10.0
        elif key == "expectancy":
            value = value / 10_000.0
        score += weight * value
    return float(score)


def hard_reject_reason(metrics: dict[str, float], hard_rejects: dict[str, float] | None, *, phase: int) -> str:
    hard = hard_rejects or {}
    total = float(metrics.get("total_trades", 0.0))
    if total < float(hard.get("min_trades", 0.0)):
        return f"phase{phase}_too_few_trades ({total:.0f} < {float(hard.get('min_trades', 0.0)):.0f})"
    dd = float(metrics.get("max_drawdown_pct", 0.0))
    if "max_dd_pct" in hard and dd > float(hard["max_dd_pct"]):
        return f"phase{phase}_max_dd ({dd:.2%} > {float(hard['max_dd_pct']):.2%})"
    pf = float(metrics.get("profit_factor", 0.0))
    if "min_pf" in hard and pf < float(hard["min_pf"]):
        return f"phase{phase}_low_pf ({pf:.2f} < {float(hard['min_pf']):.2f})"
    if "min_expectancy" in hard and float(metrics.get("expectancy", 0.0)) < float(hard["min_expectancy"]):
        return f"phase{phase}_low_expectancy"
    if "max_same_bar_fills" in hard and float(metrics.get("same_bar_fill_count", 0.0)) > float(hard["max_same_bar_fills"]):
        return f"phase{phase}_same_bar_fill"
    return ""

