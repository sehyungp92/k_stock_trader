from __future__ import annotations

from dataclasses import dataclass, field
from typing import Awaitable, Callable

from oms_client import IntentStatus
from oms_client.client import AccountState, OMSClient
from strategy_common.actions import StrategyAction
from strategy_common.events import DecisionEvent
from strategy_common.market import MarketBar, require_completed_bar

from .config import KALCBConfig, KALCB_CORE_VERSION, STRATEGY_ID
from .core.core_models import KALCBFillEvent, KALCBOrderUpdateEvent, KALCBPortfolioView
from .core.logic import on_kalcb_fill, on_kalcb_order_update, on_kalcb_timer, remember_submitted_order, step_kalcb_core
from .core.state import KALCBState
from .execution import action_to_intent, normalize_action_prices
from .models import KALCBDailySnapshot


SubmitFn = Callable[[StrategyAction], str | None]
AsyncSubmitFn = Callable[[StrategyAction], Awaitable[str | None]]


@dataclass(slots=True)
class KALCBEngine:
    """Thin live/paper adapter over the shared KALCB core."""

    config: KALCBConfig = field(default_factory=KALCBConfig)
    state: KALCBState = field(default_factory=KALCBState)
    candidate_snapshot: KALCBDailySnapshot | None = None
    decisions: list[DecisionEvent] = field(default_factory=list)

    def on_bar(self, bar: MarketBar, portfolio: KALCBPortfolioView, submit: SubmitFn) -> list[DecisionEvent]:
        require_completed_bar(bar)
        result = step_kalcb_core(self.state, bar, self.config, self.candidate_snapshot, portfolio)
        self._submit_actions(result.actions, submit)
        self.decisions.extend(result.decisions)
        return list(result.decisions)

    def on_fill(self, fill: KALCBFillEvent, submit: SubmitFn) -> list[DecisionEvent]:
        result = on_kalcb_fill(self.state, fill, self.config)
        self._submit_actions(result.actions, submit)
        self.decisions.extend(result.decisions)
        return list(result.decisions)

    def on_timer(self, timestamp, submit: SubmitFn) -> list[DecisionEvent]:
        result = on_kalcb_timer(self.state, timestamp, self.config)
        self._submit_actions(result.actions, submit)
        self.decisions.extend(result.decisions)
        return list(result.decisions)

    def on_order_update(self, update: KALCBOrderUpdateEvent, submit: SubmitFn | None = None) -> list[DecisionEvent]:
        result = on_kalcb_order_update(self.state, update)
        if submit is not None:
            self._submit_actions(result.actions, submit)
        self.decisions.extend(result.decisions)
        return list(result.decisions)

    def reconcile_submitted_order(self, order_id: str | None, action: StrategyAction) -> None:
        remember_submitted_order(self.state, order_id, action)

    def _submit_actions(self, actions: list[StrategyAction], submit: SubmitFn) -> None:
        defer_memory = _defers_order_memory(submit)
        for action in actions:
            normalized = normalize_action_prices(action)
            order_id = submit(normalized)
            if not defer_memory:
                remember_submitted_order(self.state, order_id, normalized)

    def health_status(self) -> dict:
        return {
            "strategy_id": STRATEGY_ID,
            "status": "ready",
            "strategy_core_version": KALCB_CORE_VERSION,
            "live_parity_fill_timing": self.config.live_parity_fill_timing,
            "auction_mode": self.config.auction_mode,
            "snapshot_hash": self.state.snapshot_hash,
            "open_positions": sum(1 for item in self.state.symbols.values() if item.position is not None),
        }


@dataclass(slots=True)
class KALCBOMSLiveAdapter:
    """OMS HTTP adapter for paper/live modes.

    KIS REST polling is deliberately kept outside the signal hot path. Account
    refreshes can return stale/None under EGW00201 pressure; the core sees a
    no-op portfolio view until the next completed 5m bar.
    """

    oms: OMSClient
    config: KALCBConfig = field(default_factory=KALCBConfig)
    dry_run: bool = False
    last_account_state: AccountState | None = None

    async def refresh_portfolio(self) -> KALCBPortfolioView:
        account = await self.oms.get_account_state()
        positions = await self.oms.get_all_positions()
        if account is not None:
            self.last_account_state = account
        cash = float((account.buyable_cash if account is not None else None) or (self.last_account_state.buyable_cash if self.last_account_state else 0.0))
        equity = float((account.equity if account is not None else None) or (self.last_account_state.equity if self.last_account_state else 0.0))
        session_return_pct = float((account.daily_pnl_pct if account is not None else None) or (self.last_account_state.daily_pnl_pct if self.last_account_state else 0.0))
        position_qty: dict[str, int] = {}
        open_notional = 0.0
        if positions:
            for symbol, position in positions.items():
                qty = position.get_allocation(STRATEGY_ID)
                if qty <= 0:
                    continue
                position_qty[symbol] = qty
                open_notional += qty * float(position.avg_price or 0.0)
        return KALCBPortfolioView(
            cash=cash,
            positions=position_qty,
            open_positions=len(position_qty),
            sector_counts={},
            open_notional=open_notional,
            equity=equity,
            session_return_pct=session_return_pct,
        )

    async def submit(self, action: StrategyAction) -> str | None:
        normalized = normalize_action_prices(action)
        if self.dry_run:
            if getattr(self.oms, "record_only", False):
                result = await self.oms.submit_intent(action_to_intent(normalized))
                return result.order_id or result.intent_id
            return f"dry-run:{normalized.strategy_id}:{normalized.symbol}:{normalized.reason}"
        result = await self.oms.submit_intent(action_to_intent(normalized))
        if result.status in {IntentStatus.REJECTED, IntentStatus.CANCELLED, IntentStatus.DEFERRED}:
            return None
        return result.order_id or result.intent_id


def _defers_order_memory(submit: SubmitFn) -> bool:
    return bool(getattr(submit, "defer_order_memory", False) or getattr(getattr(submit, "__self__", None), "defer_order_memory", False))
