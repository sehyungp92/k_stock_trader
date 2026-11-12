"""Completed opening-thesis failure; pure predicate for a registered ablation."""
from datetime import timedelta


def evidence(symbol_state, position, bar, minimum_hold):
    if minimum_hold not in (6, 8):
        raise ValueError('Unregistered opening-failure hold')
    if not bar.is_completed:
        raise ValueError('Opening failure requires a completed bar')
    if (position.metadata.get('entry_route') != 'first30_open_anchor'
            or position.hold_bars < minimum_hold or position.qty_open <= 0
            or position.exit_in_flight or not bar.close < position.entry_price):
        return None
    # Gross losing is deliberately stricter than losing after positive costs:
    # never exit a gross winner just because an estimated cost hurdle is large.
    prefix = [b for b in symbol_state.bars_today if b.timestamp <= bar.timestamp]
    if len(prefix) < 6 or prefix[-1].timestamp != bar.timestamp:
        return None
    if any(not b.is_completed for b in prefix):
        raise ValueError('Uncompleted observation in opening failure')
    if any(b.timestamp + timedelta(minutes=5) != c.timestamp for b, c in zip(prefix, prefix[1:])):
        return None
    last = prefix[-2:]
    if last[0].timestamp < position.entry_time:
        return None
    # First six completed bars are the unchanged parent's opening structure.
    high = max(b.high for b in prefix[:6]); low = min(b.low for b in prefix[:6])
    if not high > low > 0:
        return None
    midpoint = (high + low) / 2
    from strategy_kalcb.signals import compute_session_vwap
    confirmations = []
    for i in (len(prefix)-2, len(prefix)-1):
        item = prefix[i]; vwap = compute_session_vwap(prefix[:i+1])
        if not vwap > 0 or not item.close < min(vwap, midpoint):
            return None
        confirmations.append({'bar': item.timestamp.isoformat(), 'close': item.close,
                              'own_vwap': vwap, 'opening_midpoint': midpoint})
    return {'version': 'opening-failure-v1', 'minimum_hold': minimum_hold,
            'hold_bars': position.hold_bars, 'entry_price_at_decision': position.entry_price,
            'qty_open': position.qty_open, 'gross_loss_fraction': bar.close / position.entry_price - 1,
            'available_at': (bar.timestamp + timedelta(minutes=5)).isoformat(),
            'confirmations': confirmations}
