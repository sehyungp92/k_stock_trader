"""Existing-path diagnostic for the frozen minute35 confirmation; no policy."""
from datetime import datetime, timedelta
from math import isfinite
from statistics import median


def clock(value):
    return datetime.fromisoformat(str(value))


def confirmation(exposure, bars):
    """Only seven bars that are complete at the proposed confirmation clock."""
    action = exposure['action']; meta = action['metadata']
    available = clock(exposure['decision_available_at'])
    at = available + timedelta(minutes=5)
    result = {'completion_at': at.isoformat(), 'planned_qty': action['qty'],
              'initial_qty': action['qty']//2 if action['qty'] >= 2 else action['qty']}
    if (available.hour, available.minute) != (9, 30):
        return {**result, 'category': 'unsupported_opening_clock'}
    prefix = [b for b in bars if clock(b[0])+timedelta(minutes=5) <= at]
    start = at.replace(hour=9, minute=0, second=0, microsecond=0)
    if len(prefix) != 7 or any(clock(b[0]) != start+timedelta(minutes=5*i)
                              for i, b in enumerate(prefix)):
        return {**result, 'category': 'invalid_prefix'}
    if any(not all(isfinite(float(v)) for v in b[1:]) or min(b[1:]) <= 0
           or b[3] > min(b[1], b[4]) or b[2] < max(b[1], b[4]) for b in prefix):
        return {**result, 'category': 'invalid_prefix'}
    ref = float(meta['entry_price_ref'])
    if ref != float(prefix[5][4]):
        raise ValueError('Original opening reference differs from completed raw close')
    vwap = sum((b[2]+b[3]+b[4])/3*b[5] for b in prefix)/sum(b[5] for b in prefix)
    close = float(prefix[-1][4]); passed = close >= ref and close >= vwap
    return {**result, 'category': 'single_share_unchanged' if action['qty'] < 2 else
            'price_confirmed' if passed else 'price_not_confirmed',
            'price_confirmed': passed, 'original_reference': ref,
            'completed_close35': close, 'completed_vwap35': vwap,
            'completed_quote_change_bps': (close/ref-1)*10000,
            'raw_prefix': prefix}


def summarize(rows):
    positions = [r for r in rows if r['position_id'] is not None]
    ids = [r['position_id'] for r in positions]
    if len(ids) != len(set(ids)):
        raise ValueError('Lifetime PnL would be duplicated')
    premiums = [r['completed_quote_change_bps'] for r in rows
                if 'completed_quote_change_bps' in r]
    bps = [r['parent_net_pnl_krw']/r['parent_entry_cost_krw']*10000 for r in positions]
    return {'submitted_orders': len(rows), 'issuer_days': len({(r['symbol'], r['day']) for r in rows}),
            'sessions': len({r['day'] for r in rows}), 'filled_unique_lifetimes': len(positions),
            'no_fill_orders': len(rows)-len(positions),
            'parent_net_krw': sum(r['parent_net_pnl_krw'] for r in positions),
            'parent_winner_count': sum(r['parent_net_pnl_krw'] > 0 for r in positions),
            'parent_positive_net_krw': sum(max(0, r['parent_net_pnl_krw']) for r in positions),
            'parent_negative_net_krw': sum(min(0, r['parent_net_pnl_krw']) for r in positions),
            'parent_entry_cost_krw': sum(r['parent_entry_cost_krw'] for r in positions),
            'parent_fees_krw': sum(r['parent_fees_krw'] for r in positions),
            'parent_mean_net_entry_notional_bps': sum(bps)/len(bps) if bps else None,
            'median_completed_quote_change_bps': median(premiums) if premiums else None,
            'parent_full_quantity_filled_before35': sum(r['parent_full_quantity_filled_before35'] for r in rows),
            'parent_sell_intent_at_or_before35': sum(r['parent_sell_intent_at_or_before35'] for r in rows),
            'parent_sale_before35': sum(r['parent_sale_before35'] for r in rows),
            'initial_account_has_other_inventory_or_pending': sum(r['initial_nonflat_or_pending'] for r in rows),
            'scope': 'Observed unchanged-parent outcomes only. No staged returns, avoided loss, funded availability or independent support.'}
