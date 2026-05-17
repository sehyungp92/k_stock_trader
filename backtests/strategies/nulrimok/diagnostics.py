from __future__ import annotations

from .runner import StrategyBacktestResult


def build_nulrimok_diagnostics(result: StrategyBacktestResult) -> dict:
    return {
        "strategy": "nulrimok",
        "capability_level": result.capability_level,
        "selection_top_rank_forward_r": result.selection_attribution.get("top_rank_avg_forward_r", 0.0),
        "realized_net_expectancy": result.metrics.get("expectancy", 0.0),
        "overflow_opportunity_cost": result.selection_attribution.get("overflow_opportunity_cost", 0.0),
        "source_fingerprint": result.source_fingerprint,
    }


def format_nulrimok_diagnostics(result: StrategyBacktestResult) -> str:
    return "\n".join(
        [
            "Nulrimok Backtest Diagnostics",
            f"Capability: {result.capability_level}",
            f"Trades: {result.metrics.get('total_trades', 0.0):.0f}",
            f"Net profit: {result.metrics.get('net_profit', 0.0):.2f}",
            f"Profit factor: {result.metrics.get('profit_factor', 0.0):.2f}",
            f"Selection top-rank forward R: {result.selection_attribution.get('top_rank_avg_forward_r', 0.0):.2f}",
            f"Source: {result.source_fingerprint}",
        ]
    )

