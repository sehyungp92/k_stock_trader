from __future__ import annotations

from dataclasses import dataclass, field
from typing import Awaitable, Callable

from oms_client import IntentStatus
from oms_client.client import AccountState, OMSClient
from strategy_common.actions import StrategyAction
from strategy_common.events import DecisionEvent
from strategy_common.market import MarketBar, require_completed_bar

from .config import OLRConfig, OLR_CORE_VERSION, STRATEGY_ID
from .core.core_models import OLRExpiredOrderEvent, OLRFillEvent, OLROrderUpdateEvent, OLRPortfolioView
from .core.logic import on_olr_fill, on_olr_order_expired, on_olr_order_update, on_olr_timer, remember_submitted_order, step_olr_core
from .core.state import OLRState
from .execution import action_to_intent, normalize_action_prices
from .models import OLRDailySnapshot


SubmitFn = Callable[[StrategyAction], str | None]
AsyncSubmitFn = Callable[[StrategyAction], Awaitable[str | None]]


@dataclass(slots=True)
class OLREngine:
    """Thin live/paper adapter over the shared OLR core."""

    config: OLRConfig = field(default_factory=OLRConfig)
    state: OLRState = field(default_factory=OLRState)
    candidate_snapshot: OLRDailySnapshot | None = None
    decisions: list[DecisionEvent] = field(default_factory=list)

    def on_bar(self, bar: MarketBar, portfolio: OLRPortfolioView, submit: SubmitFn) -> list[DecisionEvent]:
        require_completed_bar(bar)
        result = step_olr_core(self.state, bar, self.config, self.candidate_snapshot, portfolio)
        self._submit_actions(result.actions, submit)
        self.decisions.extend(result.decisions)
        return list(result.decisions)

    def on_fill(self, fill: OLRFillEvent, submit: SubmitFn) -> list[DecisionEvent]:
        result = on_olr_fill(self.state, fill, self.config)
        self._submit_actions(result.actions, submit)
        self.decisions.extend(result.decisions)
        return list(result.decisions)

    def on_timer(self, timestamp, submit: SubmitFn) -> list[DecisionEvent]:
        result = on_olr_timer(self.state, timestamp, self.config)
        self._submit_actions(result.actions, submit)
        self.decisions.extend(result.decisions)
        return list(result.decisions)

    def on_order_expired(self, expired: OLRExpiredOrderEvent, submit: SubmitFn) -> list[DecisionEvent]:
        result = on_olr_order_expired(self.state, expired, self.config)
        self._submit_actions(result.actions, submit)
        self.decisions.extend(result.decisions)
        return list(result.decisions)

    def on_order_update(self, update: OLROrderUpdateEvent, submit: SubmitFn | None = None) -> list[DecisionEvent]:
        result = on_olr_order_update(self.state, update, self.config)
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
            "strategy_core_version": OLR_CORE_VERSION,
            "live_parity_fill_timing": self.config.live_parity_fill_timing,
            "entry_mode": self.config.entry_mode,
            "exit_mode": self.config.exit_mode,
            "snapshot_hash": self.state.snapshot_hash,
            "open_positions": sum(1 for item in self.state.symbols.values() if item.position is not None),
        }


@dataclass(slots=True)
class OLROMSLiveAdapter:
    """OMS HTTP adapter for OLR paper/live modes."""

    oms: OMSClient
    config: OLRConfig = field(default_factory=OLRConfig)
    dry_run: bool = False
    last_account_state: AccountState | None = None

    async def refresh_portfolio(self) -> OLRPortfolioView:
        account = await self.oms.get_account_state()
        positions = await self.oms.get_all_positions()
        if account is not None:
            self.last_account_state = account
        equity = float((account.equity if account is not None else None) or (self.last_account_state.equity if self.last_account_state else 0.0))
        cash = float((account.buyable_cash if account is not None else None) or (self.last_account_state.buyable_cash if self.last_account_state else 0.0))
        position_qty: dict[str, int] = {}
        open_notional = 0.0
        if positions:
            for symbol, position in positions.items():
                qty = position.get_allocation(STRATEGY_ID)
                if qty <= 0:
                    continue
                position_qty[str(symbol).zfill(6)] = qty
                open_notional += qty * float(position.avg_price or 0.0)
        gross = open_notional / equity if equity > 0 else 0.0
        return OLRPortfolioView(
            cash=cash,
            equity=equity,
            positions=position_qty,
            open_positions=len(position_qty),
            open_notional=open_notional,
            gross_exposure_pct=gross,
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
