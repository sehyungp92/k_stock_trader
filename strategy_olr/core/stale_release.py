"""Optional, bounded release of weak residual inventory after its native horizon.

This is a testable risk heuristic, not a learned estimate of continuation alpha.
It never recognizes cash until ordinary execution confirms a sale.
"""
from dataclasses import asdict
from datetime import datetime, timedelta

from kis_core.trading_calendar import get_trading_calendar
from strategy_common.actions import CancelOrders, SubmitPartialExit
from strategy_common.execution_session import bar_execution_issue, bar_has_periodic_auction, bar_reopens_after_label
from strategy_olr.config import OLR_CORE_VERSION, STRATEGY_ID
from strategy_olr.execution import _running_vwap
from .core_models import OLRCoreResult


def maybe_release_stale_inventory(state, bar, config):
    from .logic import _decision, _exit_plan, _position_risk, _submit_managed_market_exit
    if not config.capacity_stale_release:
        return OLRCoreResult(state=state)
    symbol_state = state.symbol_state(bar.symbol)
    position = symbol_state.position
    if position is None or position.qty_open <= 0 or symbol_state.pending_exit_order_id:
        return OLRCoreResult(state=state)
    calendar = get_trading_calendar()
    day = bar.timestamp.date()
    earliest = calendar.next_trading_day(calendar.next_trading_day(position.entry_time.date()))
    if day < earliest or position.metadata.get('stale_release_date') == day.isoformat():
        return OLRCoreResult(state=state)
    opening, _ = calendar.session_hours(day)
    start = datetime.combine(day, opening, bar.timestamp.tzinfo)
    available = bar.timestamp + timedelta(minutes=5)
    if available < start + timedelta(minutes=15):
        return OLRCoreResult(state=state)
    prefix = [b for b in symbol_state.session_bars if b.timestamp.date() == day and b.timestamp <= bar.timestamp]
    if (len(prefix) < 3 or prefix[0].timestamp != start or prefix[-1].timestamp != bar.timestamp
            or any(b.timestamp-a.timestamp != timedelta(minutes=5) for a,b in zip(prefix,prefix[1:]))
            or any(bar_execution_issue(b) or bar_has_periodic_auction(b) or bar_reopens_after_label(b) for b in prefix)
            or sum(b.volume for b in prefix) <= 0):
        return OLRCoreResult(state=state)
    vwap = _running_vwap(prefix)
    if bar.close >= vwap:
        return OLRCoreResult(state=state)
    quantity = max(1, position.qty_open // 2)
    plan = _exit_plan(config)
    position.metadata['stale_release_date'] = day.isoformat()
    if quantity == position.qty_open:
        return _submit_managed_market_exit(state, symbol_state, bar, position, plan,
                                          'stale_inventory_release', _position_risk(position))
    metadata = {'strategy_core_version': OLR_CORE_VERSION, 'source_artifact_hash': position.source_artifact_hash,
        'candidate_rank': position.candidate_rank, 'candidate_score': position.candidate_score,
        'trade_exit_plan': asdict(plan), 'order_role': 'CAPACITY_SLICE', 'risk_per_share': _position_risk(position),
        'stale_release': {'entry_time': position.entry_time.isoformat(), 'earliest_session': earliest.isoformat(),
                          'available_at': available.isoformat(), 'completed_vwap': vwap, 'completed_close': bar.close,
                          'quantity_before': position.qty_open, 'fraction': .5}}
    action = SubmitPartialExit(STRATEGY_ID, position.symbol, quantity, 'MARKET', None,
                              'stale_inventory_release', metadata)
    return OLRCoreResult(state=state,
        actions=[CancelOrders(STRATEGY_ID, position.symbol, 'stale_inventory_release'), action],
        decisions=[_decision(bar.timestamp, position.symbol, 'PARTIAL_EXIT_SUBMITTED', 'stale_inventory_release',
                             {**metadata, 'qty': quantity})])
