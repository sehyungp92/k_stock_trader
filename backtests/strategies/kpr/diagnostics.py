from __future__ import annotations

from .runner import StrategyBacktestResult


def build_kpr_diagnostics(result: StrategyBacktestResult) -> dict:
    return {
        "strategy": "kpr",
        "capability_level": result.capability_level,
        "vwap_depth_alpha": result.metrics.get("expected_total_r", 0.0),
        "confidence_quality": result.metrics.get("profit_factor", 0.0),
        "partial_exit_value": result.metrics.get("mfe_capture", 0.0),
        "source_fingerprint": result.source_fingerprint,
    }


def format_kpr_diagnostics(result: StrategyBacktestResult) -> str:
    return "\n".join(
        [
            "KPR Backtest Diagnostics",
            f"Capability: {result.capability_level}",
            f"Trades: {result.metrics.get('total_trades', 0.0):.0f}",
            f"Net profit: {result.metrics.get('net_profit', 0.0):.2f}",
            f"Profit factor: {result.metrics.get('profit_factor', 0.0):.2f}",
            f"VWAP depth alpha proxy: {result.metrics.get('expected_total_r', 0.0):.2f}",
            f"Source: {result.source_fingerprint}",
        ]
    )

