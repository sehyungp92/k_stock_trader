"""Prepared research-only entry remainder revalidation, disabled by default.

This module is not imported or enabled by production/live/replay adapters.
It cancels only where the adapter's symbol-wide cancellation is equivalent to
cancelling one original entry. It deliberately does not use the liquidation
helper: normal racing fills remain part of the original managed lifetime.
Parent exits and terminal liquidation continue to own their usual behavior.
"""
from datetime import datetime, timedelta
from math import isfinite

from strategy_common.actions import CancelOrders

KEY = 'pending_entry_revalidation_v1'
REASON = 'research_pending_entry_revalidation_keep_filled_v1'


def revalidate(state, symbol, clock, pending_orders):
    """Called after ordinary parent bar/timer decisions with a complete order view.

    `pending_orders` must be the actual account's complete current open-order
    list of neutral dictionaries; None means unavailable and fails closed.
    The fixed 09:35 contrast uses exactly the existing V47/V49 price predicate.
    No original order selection/quantity/price is changed before this clock.
    """
    if not state.meta.get(KEY + '_enabled', False):
        return [], 'disabled'
    ss = state.symbols.get(symbol)
    if ss is None or not ss.pending_entry_order_id:
        return [], 'no_pending_entry'
    oid = ss.pending_entry_order_id
    meta = ss.pending_entry_metadata
    if meta.get('entry_route') != 'first30_open_anchor':
        return [], 'other_route'
    available = meta.get('decision_available_at')
    # Core intent metadata records the signal bar; broker metadata adds the
    # availability clock. The original explicit next-5m-OPEN contract makes
    # that completed-bar boundary available without consulting future data.
    if not available and meta.get('signal_bar') and meta.get('fill_timing') == 'next_5m_open':
        available = (datetime.fromisoformat(str(meta['signal_bar'])) + timedelta(minutes=5)).isoformat()
    if not available:
        return [], 'missing_original_clock'
    available = datetime.fromisoformat(str(available))
    if (available.hour, available.minute) != (9, 30):
        return [], 'unsupported_opening_clock'
    if clock != available + timedelta(minutes=5):
        return [], 'not_fixed_clock'
    records = state.meta.setdefault(KEY, {})
    if oid in records:
        return [], 'already_evaluated'
    if pending_orders is None:
        return [], 'unavailable_complete_order_view'
    own = [o for o in pending_orders if o.get('symbol') == symbol]
    # Both existing adapters implement CancelOrders at symbol scope. Fail
    # closed if a protective/partial/other-strategy order would also be hit.
    if len(own) != 1 or own[0].get('order_id') != oid:
        return [], 'symbol_cancellation_not_single_entry'
    order = own[0]
    observed_clock = order.get('metadata', {}).get('decision_available_at')
    if observed_clock and datetime.fromisoformat(str(observed_clock)) != available:
        return [], 'original_clock_mismatch'
    if order.get('strategy_id') != 'KALCB' or order.get('side') != 'BUY' or order.get('order_type') != 'MARKET':
        return [], 'not_original_market_entry'
    remaining = order.get('qty')
    if not isinstance(remaining, int) or isinstance(remaining, bool) or remaining <= 0:
        return [], 'invalid_pending_quantity'
    pos = ss.position
    if pos and (pos.entry_order_id != oid or pos.exit_in_flight or pos.partial_taken
                or pos.qty_open != pos.qty_entry or pos.metadata.get('liquidation_reason')):
        return [], 'parent_exit_sale_or_other_lifetime'
    if meta.get('cancel_late_fill_reason'):
        return [], 'parent_cancellation_already_owned'
    planned = meta.get('risk_reserved_entry_qty')
    filled = pos.qty_entry if pos else 0
    if not isinstance(planned, int) or filled + remaining != planned:
        return [], 'pending_quantity_not_reconciled'
    bars = ss.bars_today
    if len(bars) != 7 or any(not b.is_completed or b.volume <= 0 for b in bars):
        return [], 'invalid_completed_prefix'
    start = available - timedelta(minutes=30)
    if any(b.timestamp != start + timedelta(minutes=5*i) for i,b in enumerate(bars)):
        return [], 'nonconsecutive_prefix'
    if any(not all(isfinite(float(x)) and x > 0 for x in (b.open,b.high,b.low,b.close,b.volume))
           or b.low > min(b.open,b.close) or b.high < max(b.open,b.close) for b in bars):
        return [], 'invalid_source_bar'
    reference = meta.get('entry_price_ref')
    if reference != bars[5].close:
        return [], 'original_reference_mismatch'
    vwap = sum((b.high+b.low+b.close)*b.volume for b in bars)/(3*sum(b.volume for b in bars))
    weak = bars[-1].close < reference or bars[-1].close < vwap
    records[oid] = dict(at=clock.isoformat(),symbol=symbol,original_qty=planned,
        held_qty=filled,remaining_qty=remaining,completed_close=bars[-1].close,
        original_reference=reference,completed_vwap=vwap,cancel_requested=weak,
        pending_status_observed=True,successful_cancellation=False)
    if not weak:
        return [], 'confirmation_preserved'
    # Keep pending identity until the actual acknowledgement. Ordinary racing
    # fills are reconciled by the original core; do not pretend they vanished
    # or set cancel_late_fill_reason, which would liquidate the WHOLE position.
    state.order_roles.setdefault(oid, {})['research_cancel_preserves_filled'] = True
    return [CancelOrders('KALCB', symbol, REASON,
        {'entry_order_id': oid, 'research_cancel_preserves_filled': True,
         'decision_available_at': clock.isoformat()})], 'cancel_requested'
