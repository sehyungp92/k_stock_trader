"""Read-only decision previews using the ordinary KALCB core.

Readiness is a submitted intent, never a future fill or realised outcome. This
module deliberately has no broker, label, or market-data lookup dependency.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta
from typing import Iterable

from strategy_common.actions import SubmitEntry
from strategy_common.market import MarketBar
from strategy_kalcb.config import KALCBConfig
from strategy_kalcb.models import KALCBDailySnapshot
from .core_models import KALCBPortfolioView
from .logic import step_kalcb_core
from .state import KALCBState


def preview_entry(
    bars: Iterable[MarketBar], snapshot: KALCBDailySnapshot,
    config: KALCBConfig, portfolio: KALCBPortfolioView, *,
    available_at: datetime, state: KALCBState | None = None,
) -> dict:
    """Preview new completed bars on a private state copy.

    ``state`` must precede the supplied bars; callers must not replay an already
    ingested bar. Bars after the cutoff are ignored, including their values.
    Account reservations belong in the supplied portfolio/state. A standalone
    empty-account preview is not a shared-account feasibility certificate.
    """
    if available_at.tzinfo is None:
        raise ValueError("A timezone-aware information cutoff is required")
    if snapshot.generated_at and snapshot.generated_at > available_at:
        raise ValueError("Candidate snapshot is not available at the cutoff")
    prefix = sorted((b for b in bars if b.timestamp + timedelta(minutes=5) <= available_at),
                    key=lambda b: (b.timestamp, b.symbol))
    if len({(b.symbol, b.timestamp) for b in prefix}) != len(prefix):
        raise ValueError("Duplicate preview bars")
    if any(b.timestamp.date() != snapshot.trade_date for b in prefix):
        raise ValueError("Preview must stay within the snapshot session")
    private = deepcopy(state) if state is not None else KALCBState()
    account = deepcopy(portfolio)
    intents, decisions = [], []
    for bar in prefix:
        old = private.symbols.get(bar.symbol)
        if old and old.bars_today and old.bars_today[-1].timestamp >= bar.timestamp:
            raise ValueError("Preview bars overlap existing state")
        result = step_kalcb_core(private, bar, config, snapshot, account)
        private = result.state
        decisions.extend(d.to_json_dict() for d in result.decisions)
        for action in result.actions:
            if isinstance(action, SubmitEntry):
                intents.append({"symbol": action.symbol, "quantity": action.qty,
                    "order_type": action.order_type, "limit_price": action.limit_price,
                    "stop_price": action.stop_price, "reason": action.reason,
                    "available_at": (bar.timestamp + timedelta(minutes=5)).isoformat(),
                    "metadata": dict(action.metadata)})
    return {"available_at": available_at.isoformat(), "ready": bool(intents),
            "intents": intents, "decisions": decisions,
            "observed_bars": len(prefix), "scope": "supplied_account_state_only"}
