from __future__ import annotations

from typing import Iterable

from oms_client import Intent
from strategy_common.actions import StrategyAction

from .risk import round_price_for_krx


def normalize_action_prices(action: StrategyAction) -> StrategyAction:
    """Normalize neutral KALCB actions to conservative KRX tick prices."""

    from dataclasses import replace
    from strategy_common.actions import ReplaceProtectiveStop, SubmitEntry, SubmitExit, SubmitPartialExit, SubmitProtectiveStop

    if isinstance(action, SubmitEntry):
        limit = round_price_for_krx(action.limit_price, "buy_limit") if action.limit_price else None
        stop = round_price_for_krx(action.stop_price, "protective_stop") if action.stop_price else None
        return replace(action, limit_price=limit, stop_price=stop)
    if isinstance(action, (SubmitExit, SubmitPartialExit)):
        limit = round_price_for_krx(action.limit_price, "sell_limit") if action.limit_price else None
        return replace(action, limit_price=limit)
    if isinstance(action, SubmitProtectiveStop):
        return replace(action, stop_price=round_price_for_krx(action.stop_price, "protective_stop"))
    if isinstance(action, ReplaceProtectiveStop):
        return replace(action, stop_price=round_price_for_krx(action.stop_price, "protective_stop"))
    return action


def normalize_actions(actions: Iterable[StrategyAction]) -> list[StrategyAction]:
    return [normalize_action_prices(action) for action in actions]


def action_to_intent(action: StrategyAction) -> Intent:
    from strategy_common.oms_adapter import action_to_intent as shared_action_to_intent

    return shared_action_to_intent(normalize_action_prices(action))
