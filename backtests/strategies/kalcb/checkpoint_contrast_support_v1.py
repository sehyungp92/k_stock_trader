"""Read-only chronology and native-prefix comparability; no model or policy."""
from collections import defaultdict
from datetime import datetime


def stamp(value):
    return datetime.fromisoformat(str(value))


def validate_source_context(value, context):
    if not context['context_supported'] or value['checkpoint_at'] != context['available_at']:
        raise ValueError('Unsupported or wrong source clock')
    if value['own_features'] != context['own_features'] or value['context_features'] != context['context_features']:
        raise ValueError('Changed independently reconstructed raw features')
    held = value['hold']['checkpoint']['held_quantity']
    if not isinstance(held, int) or held < 0 or not isinstance(value['planned_quantity'], int):
        raise ValueError('Noninteger native quantity')
    exact = held * context['completed_close']
    if abs(exact - value['held_notional_krw']) > .000001:
        raise ValueError('Wrong actual integer held-notional denominator')


def mature_support(rows, sessions, windows, selected=None):
    """Count unique source sessions, never quantity/scenario replications."""
    result = []
    for first, last in windows:
        allowed = set(sessions[:max(0, sessions.index(first) - 1)])
        cutoff = max(allowed, default=None)
        ready = [r for r in rows if r['joint_payoff_valid'] and r['day'] in allowed
                 and r['available_at'] and r['available_at'][:10] <= cutoff
                 and (selected is None or r['action_id'] in selected)]
        names = {(r['day'], r['symbol']) for r in ready}
        days = {d for d, _ in names}
        result.append({'valid_from': first, 'valid_through': last,
                       'embargo_cutoff': cutoff, 'mature_sessions': len(days),
                       'mature_issuer_days': len(names), 'dependent_action_rows': len(ready),
                       'minimum_60_sessions_met': len(days) >= 60})
    return result


def joint_row(baseline, combined):
    for key in ('opportunity_id', 'day', 'symbol', 'quantity_scale', 'planned_quantity'):
        if baseline[key] != combined[key]:
            raise ValueError('Mismatched paired scenarios: ' + key)
    if baseline['scenario'] != 'baseline' or combined['scenario'] != 'combined':
        raise ValueError('Wrong scenario pairing')
    valid = all(r['paired_label_valid'] for r in (baseline, combined))
    safe = all(r['liquidate']['independent_execution_pass']
               and r['liquidate']['independent_account_pass']
               and not r['liquidate']['deadline_violation']
               and r['liquidate']['checkpoint']['action_submitted']
               and r['liquidate']['actual_changed_exit_quantity'] == r['hold']['checkpoint']['held_quantity']
               and r['hold']['checkpoint']['held_quantity'] > 0
               for r in (baseline, combined))
    return {'action_id': baseline['opportunity_id'], 'day': baseline['day'],
            'symbol': baseline['symbol'], 'quantity_scale': baseline['quantity_scale'],
            'joint_payoff_valid': valid, 'liquidation_physically_supported': safe,
            'available_at': max(r['label_available_at'] for r in (baseline, combined)) if valid else None,
            'hold_deadline_observed': any(r['hold']['deadline_violation'] for r in (baseline, combined))}


def native_prefix(fills, submissions, terminals, order_ids, symbol, clock):
    """Rebuild only the entry orders of one actual lifetime at completed45.

    Parent decisions at raw09:40 have completed45 availability. Fills at09:45
    belong to the next bar and cannot affect this state. Amendments for partial
    remainders are reconstructed from fills rather than their stale submit time.
    """
    end = stamp(clock)
    order_ids = set(order_ids)
    source = [submissions[o] for o in order_ids]
    if any(o['symbol'] != symbol or o['side'] != 'BUY' for o in source):
        raise ValueError('Wrong lifetime entry order')
    source = [o for o in source if stamp(o['metadata']['decision_available_at']) <= end]
    included = {o['order_id'] for o in source}
    planned = sum(o['qty'] for o in source)
    filled = defaultdict(int)
    held = 0
    prices = []
    gross_cash = 0.0
    for f in fills:
        if stamp(f['timestamp']) >= end:
            continue
        if f['symbol'] != symbol:
            raise ValueError('Mixed lifetime symbols')
        if f['side'] == 'BUY':
            if f['order_id'] not in included:
                raise ValueError('Unbound lifetime entry fill')
            filled[f['order_id']] += f['qty']
            held += f['qty']
            gross_cash -= f['qty'] * f['price']
            prices.append((f['qty'], f['price']))
        elif f['side'] == 'SELL':
            held -= f['qty']
            gross_cash += f['qty'] * f['price']
        else:
            raise ValueError('Unknown fill side')
        if held < 0:
            raise ValueError('Negative prefix inventory')
    pending = 0
    for order in source:
        remaining = order['qty'] - filled[order['order_id']]
        if remaining < 0:
            raise ValueError('Entry overfill')
        closed = any(stamp(t) < end for t in terminals.get(order['order_id'], ()))
        if not closed:
            pending += remaining
    return {'held_quantity': held, 'planned_quantity': planned,
            'pending_buy_quantity': pending, 'gross_cash_flow': gross_cash,
            'mean_entry_fill_price': sum(n*p for n, p in prices)/sum(n for n, _ in prices) if prices else None}


def state_comparison(native, reference):
    fields = ('held_quantity', 'planned_quantity', 'pending_buy_quantity')
    return {k + '_exact': native[k] == reference[k] for k in fields}
