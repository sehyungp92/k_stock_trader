from __future__ import annotations

from datetime import timezone
from typing import Any

from strategy_common.events import DecisionEvent, TradeOutcome


def trade_outcome_to_instrumentation_dict(trade: TradeOutcome) -> dict[str, Any]:
    return {
        "trade_id": f"{trade.strategy_id}:{trade.symbol}:{trade.entry_fill_time.isoformat()}",
        "bot_id": "k_stock_trader",
        "strategy_id": trade.strategy_id,
        "pair": trade.symbol,
        "side": "LONG",
        "entry_time": trade.entry_fill_time.isoformat(),
        "exit_time": trade.exit_fill_time.isoformat() if trade.exit_fill_time else None,
        "entry_price": trade.entry_price,
        "exit_price": trade.exit_price,
        "position_size": trade.qty,
        "position_size_quote": trade.qty * trade.entry_price,
        "pnl": trade.net_pnl,
        "fees_paid": trade.commission,
        "exit_reason": trade.exit_reason,
        "stage": "exit" if trade.realized else "entry",
        "event_metadata": {
            "event_type": "trade",
            "timestamp": (trade.exit_fill_time or trade.entry_fill_time).astimezone(timezone.utc).isoformat(),
            "data_source_id": "backtest",
            "source_artifact_hash": trade.source_artifact_hash,
        },
    }


def decision_to_filter_dicts(decision: DecisionEvent) -> list[dict[str, Any]]:
    filters = decision.metadata.get("filter_decisions", [])
    rows: list[dict[str, Any]] = []
    if not isinstance(filters, list):
        return rows
    for item in filters:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "bot_id": "k_stock_trader",
                "pair": decision.symbol,
                "timestamp": decision.timestamp.isoformat(),
                "filter_name": item.get("filter", item.get("filter_name", "")),
                "passed": bool(item.get("passed", False)),
                "threshold": float(item.get("threshold", 0.0) or 0.0),
                "actual_value": float(item.get("actual", item.get("actual_value", 0.0)) or 0.0),
                "signal_name": decision.decision_code,
                "strategy_type": decision.strategy_id.lower(),
            }
        )
    return rows

