"""Family daily snapshot helpers for KRX strategy rollups."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Mapping


def build_family_daily_snapshot(
    *,
    trade_date: date,
    family_id: str = "krx_equity",
    strategy_summaries: Mapping[str, Mapping[str, Any]] | None = None,
    portfolio_summary: Mapping[str, Any] | None = None,
    replay_parity_status: str = "",
) -> dict[str, Any]:
    summaries = {str(key).upper().strip(): dict(value) for key, value in dict(strategy_summaries or {}).items()}
    total_trades = sum(int(row.get("total_trades") or row.get("trades") or 0) for row in summaries.values())
    wins = sum(int(row.get("wins") or 0) for row in summaries.values())
    losses = sum(int(row.get("losses") or 0) for row in summaries.values())
    return {
        "record_type": "family_daily_snapshot",
        "family_id": family_id,
        "trade_date": trade_date.isoformat(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "strategy_summaries": summaries,
        "portfolio_summary": dict(portfolio_summary or {}),
        "replay_parity_status": replay_parity_status,
    }
