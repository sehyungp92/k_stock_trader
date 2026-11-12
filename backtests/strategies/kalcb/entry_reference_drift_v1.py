"""Read-only opening execution decomposition, never a counterfactual PnL model."""
from datetime import datetime
import math


def stamp(x):
    value = datetime.fromisoformat(str(x))
    if value.tzinfo is None:
        raise ValueError('Naive execution clock')
    return value


def measure(row, fills, slippage_bps):
    buys = sorted((f for f in fills if f['order_id'] == row['order_id'] and f['side'] == 'BUY'),
                  key=lambda f: (stamp(f['timestamp']), f['metadata']['execution_id']))
    ids = [f['metadata']['execution_id'] for f in buys]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate physical execution')
    result = {k: row[k] for k in ['order_id', 'symbol', 'day', 'position_id', 'planned_qty',
              'decision_available_at', 'parent_net_pnl_krw', 'parent_entry_cost_krw']}
    result.update(first_fill_at=None, first_base_gap_bps=None, first_fill_gap_bps=None,
                  first_qty=0, later_qty=0, reference_notional_krw=0., actual_notional_krw=0.,
                  initial_market_drift_krw=0., later_market_drift_krw=0.,
                  initial_slippage_krw=0., later_slippage_krw=0., entry_fees_krw=0.,
                  execution_ids=ids, candidate_pnl=None)
    if not buys:
        if row['position_id'] or row['parent_entry_cost_krw'] or row['parent_net_pnl_krw']:
            raise ValueError('Unfilled order has lifetime attribution')
        result['cohort'] = 'unfilled'
        return result
    if not row['position_id']:
        raise ValueError('Filled order lacks lifetime')
    first = stamp(buys[0]['timestamp'])
    reference = float(buys[0]['metadata']['entry_price_ref'])
    if not math.isfinite(reference) or reference <= 0:
        raise ValueError('Invalid entry reference')
    if row.get('original_reference') and abs(row['original_reference'] - reference) > 1e-6:
        raise ValueError('Reference differs from reviewed decision')
    result['first_fill_at'] = first.isoformat()
    result['reference_price'] = reference
    initial_bases = []
    for f in buys:
        meta = f['metadata']; q = f['qty']; base = meta['base_price']; price = f['price']
        if not isinstance(q, int) or isinstance(q, bool) or q <= 0:
            raise ValueError('Noninteger physical shares')
        if f['symbol'] != row['symbol'] or stamp(f['timestamp']) < stamp(row['decision_available_at']):
            raise ValueError('Wrong symbol or predecision fill')
        if meta['entry_price_ref'] != reference or meta['entry_route'] != 'first30_open_anchor':
            raise ValueError('Mixed reference or route')
        if not all(math.isfinite(x) and x > 0 for x in [base, price]):
            raise ValueError('Invalid execution prices')
        bar = meta['execution_bar']
        if meta['execution_phase'] != 'open' or abs(base - bar['open']) > 1e-6:
            raise ValueError('Opening-market execution contract changed')
        if bar['volume'] <= 0 or stamp(bar['timestamp']) != stamp(f['timestamp']):
            raise ValueError('Invalid execution bar')
        if abs(price - base * (1 + slippage_bps / 10000)) > max(1e-6, price * 1e-10):
            raise ValueError('Modeled slippage mismatch')
        initial = stamp(f['timestamp']) == first
        part = 'initial' if initial else 'later'
        result['first_qty' if initial else 'later_qty'] += q
        result[part + '_market_drift_krw'] += q * (base - reference)
        result[part + '_slippage_krw'] += q * (price - base)
        result['reference_notional_krw'] += q * reference
        result['actual_notional_krw'] += q * price
        result['entry_fees_krw'] += meta['fee']
        if initial:
            initial_bases.append((q, base, price))
    if result['first_qty'] + result['later_qty'] > row['planned_qty']:
        raise ValueError('Overfilled original order')
    if abs(result['actual_notional_krw'] - row['parent_entry_cost_krw']) > .01:
        raise ValueError('Lifetime entry notional does not reconcile')
    q0 = result['first_qty']
    result['first_base_gap_bps'] = (sum(q * b for q, b, _ in initial_bases) / q0 / reference - 1) * 10000
    result['first_fill_gap_bps'] = (sum(q * p for q, _, p in initial_bases) / q0 / reference - 1) * 10000
    gap = result['initial_market_drift_krw']
    result['cohort'] = 'adverse_initial_gap' if gap > 1e-6 else 'favorable_initial_gap' if gap < -1e-6 else 'flat_initial_gap'
    parts = sum(result[k] for k in ['initial_market_drift_krw', 'later_market_drift_krw', 'initial_slippage_krw', 'later_slippage_krw'])
    if abs(parts - (result['actual_notional_krw'] - result['reference_notional_krw'])) > .01:
        raise ValueError('Execution decomposition does not add up')
    return result


SUM_FIELDS = ['first_qty', 'later_qty', 'reference_notional_krw', 'actual_notional_krw',
              'initial_market_drift_krw', 'later_market_drift_krw', 'initial_slippage_krw',
              'later_slippage_krw', 'entry_fees_krw', 'parent_net_pnl_krw', 'parent_entry_cost_krw']


def summary(rows):
    positions = [r['position_id'] for r in rows if r['position_id']]
    if len(positions) != len(set(positions)):
        raise ValueError('Duplicate lifetime PnL')
    out = {k: sum(r[k] for r in rows) for k in SUM_FIELDS}
    out.update(orders=len(rows), filled_lifetimes=len(positions),
               sessions=len({r['day'] for r in rows}), issuer_days=len({(r['day'], r['symbol']) for r in rows}),
               winners=sum(r['parent_net_pnl_krw'] > 0 for r in rows),
               positive_parent_net_krw=sum(max(0, r['parent_net_pnl_krw']) for r in rows),
               negative_parent_net_krw=sum(min(0, r['parent_net_pnl_krw']) for r in rows))
    denom = out['reference_notional_krw']
    out['initial_market_drift_per_all_filled_notional_bps'] = out['initial_market_drift_krw'] / denom * 10000 if denom else None
    return out
