from __future__ import annotations

from .runner import StrategyBacktestResult


def build_kmp_diagnostics(result: StrategyBacktestResult) -> dict[str, float | str]:
    metrics = result.metrics
    return {
        "strategy": "kmp",
        "capability_level": result.capability_level,
        "signal_extraction_quality": metrics.get("decision_count", 0.0),
        "net_expectancy": metrics.get("expectancy", 0.0),
        "mfe_capture": metrics.get("mfe_capture", 0.0),
        "same_bar_fill_count": metrics.get("same_bar_fill_count", 0.0),
        "source_fingerprint": result.source_fingerprint,
    }


def format_kmp_diagnostics(result: StrategyBacktestResult) -> str:
    diag = build_kmp_diagnostics(result)
    return "\n".join(
        [
            "KMP Backtest Diagnostics",
            f"Capability: {diag['capability_level']}",
            f"Trades: {result.metrics.get('total_trades', 0.0):.0f}",
            f"Net profit: {result.metrics.get('net_profit', 0.0):.2f}",
            f"Profit factor: {result.metrics.get('profit_factor', 0.0):.2f}",
            f"Max drawdown: {result.metrics.get('max_drawdown_pct', 0.0):.2%}",
            f"MFE capture: {result.metrics.get('mfe_capture', 0.0):.2f}",
            f"Source: {diag['source_fingerprint']}",
        ]
    )

