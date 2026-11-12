"""Observed entry-remainder exposure, not cancellation outcomes or a policy."""
from datetime import datetime


def stamp(x):
    return datetime.fromisoformat(str(x))


def measure(row, fills):
    cutoff = stamp(row['completion_at'])
    buys = [f for f in fills if f['order_id'] == row['order_id'] and f['side'] == 'BUY']
    if len({f['metadata']['execution_id'] for f in buys}) != len(buys):
        raise ValueError('Duplicate physical entry fill')
    before = [f for f in buys if stamp(f['timestamp']) < cutoff]
    later = [f for f in buys if stamp(f['timestamp']) >= cutoff]
    q0 = sum(f['qty'] for f in before)
    q1 = sum(f['qty'] for f in later)
    if q0 != row['parent_qty_bought_before35'] or q0 + q1 > row['planned_qty']:
        raise ValueError('Original exact quantity reconciliation failed')
    # Diagnostic cohort uses completed evidence and previous fills only.
    # Remaining requested shares do not prove the order is still pending.
    supported = row['category'] in {'price_confirmed', 'price_not_confirmed', 'single_share_unchanged'}
    weak = supported and not row['price_confirmed']
    blocked = row['parent_sell_intent_at_or_before35'] or row['parent_sale_before35']
    potential = bool(weak and row['planned_qty'] > q0 and not blocked)
    ref = row.get('original_reference', 0.)
    return {
        'order_id': row['order_id'], 'symbol': row['symbol'], 'day': row['day'],
        'decision_at': row['completion_at'], 'position_id': row['position_id'],
        'prefix_supported': supported, 'weak_price_at35': weak,
        'parent_exit_or_sale_before_decision': blocked,
        'planned_qty': row['planned_qty'], 'before_qty': q0, 'later_qty': q1,
        'requested_unfilled_at35': row['planned_qty'] - q0,
        'potential_remainder_observation': potential,
        'subsequent_fill_observed': q1 > 0,
        'original_entry_cost_krw': row['parent_entry_cost_krw'],
        'original_whole_lifetime_net_krw': row['parent_net_pnl_krw'],
        'later_purchase_notional_krw': sum(f['qty'] * f['price'] for f in later),
        'later_purchase_fee_krw': sum(f['metadata']['fee'] for f in later),
        'later_price_minus_original_reference_krw': sum(f['qty'] * (f['price'] - ref) for f in later) if supported else None,
        'later_execution_ids': [f['metadata']['execution_id'] for f in later],
        'future_fill_fraction_of_total_qty': q1 / (q0 + q1) if q0 + q1 else None,
        'known_future_fills_are_not_policy_inputs': True,
        'pending_status_at35_qualified': False,
        'candidate_pnl': None,
    }


def summary(rows):
    ids = [r['position_id'] for r in rows if r['position_id']]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate lifetime attribution')
    return {
        'orders': len(rows), 'issuer_days': len({(r['symbol'], r['day']) for r in rows}),
        'sessions': len({r['day'] for r in rows}),
        'filled_lifetimes': len(ids), 'no_fill_orders': sum(not r['position_id'] for r in rows),
        'later_filled_orders': sum(r['later_qty'] > 0 for r in rows),
        'before_qty': sum(r['before_qty'] for r in rows), 'later_qty': sum(r['later_qty'] for r in rows),
        'later_purchase_notional_krw': sum(r['later_purchase_notional_krw'] for r in rows),
        'original_entry_cost_krw': sum(r['original_entry_cost_krw'] for r in rows),
        'later_purchase_fee_krw': sum(r['later_purchase_fee_krw'] for r in rows),
        'later_price_minus_original_reference_krw': sum(r['later_price_minus_original_reference_krw'] or 0. for r in rows),
        'original_whole_lifetime_net_krw': sum(r['original_whole_lifetime_net_krw'] for r in rows),
        'original_winners': sum(r['original_whole_lifetime_net_krw'] > 0 for r in rows),
        'original_positive_whole_lifetime_net_krw': sum(max(0., r['original_whole_lifetime_net_krw']) for r in rows),
        'original_negative_whole_lifetime_net_krw': sum(min(0., r['original_whole_lifetime_net_krw']) for r in rows),
        'scope': 'Shares/notional are descriptive, not independent support or saved PnL. Whole lifetime outcomes cannot be assigned to remainders.',
    }
