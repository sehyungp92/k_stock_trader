"""Read-only physical exit lifecycle accounting; no counterfactual policy or model."""
from collections import defaultdict
from datetime import datetime, timedelta


def stamp(value):
    return datetime.fromisoformat(str(value)).isoformat()


def close(left, right, message):
    if abs(left - right) > max(0.05, abs(right) * 1e-9):
        raise ValueError(f'{message}: {left} != {right}')


def lifetimes(fills, positions):
    """Consume actual execution order, retaining partials within one lifetime."""
    expected = {p['position_id']: p for p in positions}
    if len(expected) != len(positions):
        raise ValueError('Duplicate expected lifetime')
    active, quantity, mapped = {}, defaultdict(int), defaultdict(list)
    seen = set()
    previous = ''
    for fill in fills:
        when = stamp(fill['timestamp'])
        if when < previous:
            raise ValueError('Nonchronological fills')
        previous = when
        execution = fill['metadata']['execution_id']
        if execution in seen:
            raise ValueError('Repeated physical execution')
        seen.add(execution)
        symbol = fill['symbol']
        if fill['side'] == 'BUY' and not quantity[symbol]:
            key = f'{symbol}/{when}'
            if key not in expected or key in mapped:
                raise ValueError('Unknown/repeated lifetime')
            active[symbol] = key
        if symbol not in active:
            raise ValueError('Exit without inventory')
        key = active[symbol]
        quantity[symbol] += fill['qty'] * (1 if fill['side'] == 'BUY' else -1)
        if quantity[symbol] < 0:
            raise ValueError('Oversold inventory')
        mapped[key].append(fill)
        if not quantity[symbol]:
            del active[symbol]
    if active or set(mapped) != set(expected):
        raise ValueError('Unclosed or missing lifetime')
    for key, rows in mapped.items():
        gross = sum(f['qty'] * f['price'] * (-1 if f['side'] == 'BUY' else 1) for f in rows)
        fees = sum(f['metadata']['fee'] for f in rows)
        close(gross, expected[key]['gross_pnl_krw'], 'Lifetime gross')
        close(fees, expected[key]['fees_krw'], 'Lifetime fees')
        close(gross-fees, expected[key]['net_pnl_krw'], 'Lifetime net')
        if sum(f['qty'] for f in rows if f['side'] == 'BUY') != expected[key]['qty']:
            raise ValueError('Lifetime quantity')
    return mapped


def completed_mark(symbol, available_at, marks):
    # Raw source timestamps denote bar OPEN. The bar must be complete at the decision.
    bar_at = (datetime.fromisoformat(stamp(available_at))-timedelta(minutes=5)).isoformat()
    key = (bar_at, symbol)
    if key not in marks or marks[key] <= 0:
        raise ValueError(f'Missing completed raw close {key}')
    return bar_at, marks[key]


def decision_snapshot(position, rows, available_at, marks):
    available_at = stamp(available_at)
    bar_at, price = completed_mark(position['symbol'], available_at, marks)
    gross, qty, fees = 0.0, 0, 0.0
    for f in rows:
        # Same-clock next-bar OPEN fills occur after the completed-bar decision.
        if stamp(f['timestamp']) >= available_at:
            continue
        sign = 1 if f['side'] == 'BUY' else -1
        qty += sign * f['qty']
        gross -= sign * f['qty'] * f['price']
        fees += f['metadata']['fee']
    if qty <= 0:
        raise ValueError('Exit intent not preceded by held inventory')
    return {'available_at': available_at, 'raw_bar_at': bar_at, 'raw_close': price,
            'held_qty': qty, 'gross_marked_pnl_krw': gross + qty * price,
            'fees_paid_krw': fees}


def reconcile_curve(fills, curve, marks, initial=100000000.0):
    cash, qty, latest, cursor = initial, defaultdict(int), {}, 0
    for when, expected in curve:
        when = stamp(when)
        while cursor < len(fills) and stamp(fills[cursor]['timestamp']) <= when:
            f = fills[cursor]; sign = 1 if f['side'] == 'BUY' else -1
            qty[f['symbol']] += sign * f['qty']
            cash -= sign * f['qty'] * f['price'] + f['metadata']['fee']
            if qty[f['symbol']] < 0:
                raise ValueError('Negative inventory at mark')
            cursor += 1
        for symbol, size in qty.items():
            if size:
                if (when, symbol) in marks:
                    latest[symbol] = marks[(when, symbol)]
                if symbol not in latest:
                    raise ValueError('Unmarked holding')
        actual = cash + sum(size * latest.get(s, 0) for s, size in qty.items())
        close(actual, expected, 'Every raw-close equity mark')
    if cursor != len(fills) or any(qty.values()):
        raise ValueError('Unconsumed execution/terminal holding')
    return len(curve)


def account(positions, fills, submits, curve, marks):
    mapped = lifetimes(fills, positions)
    mark_count = reconcile_curve(fills, curve, marks)
    sell_orders = {}
    for order in submits:
        if order['side'] == 'SELL':
            oid = order['order_id']
            if oid in sell_orders and sell_orders[oid] != order:
                raise ValueError('Conflicting initial sell submission')
            sell_orders[oid] = order
    assigned = defaultdict(list)
    executed_order_lifetime = {}
    for key, rows in mapped.items():
        for f in rows:
            if f['side'] == 'SELL':
                oid = f['order_id']
                if oid in executed_order_lifetime and executed_order_lifetime[oid] != key:
                    raise ValueError('Sell order spans lifetimes')
                executed_order_lifetime[oid] = key
    for order in sell_orders.values():
        available = stamp(order['metadata']['decision_available_at'])
        matches = [p for p in positions if p['symbol'] == order['symbol']
                   and stamp(p['entry_fill_time']) <= available <= stamp(p['exit_fill_time'])]
        if order['order_id'] in executed_order_lifetime:
            matches = [p for p in matches if p['position_id'] == executed_order_lifetime[order['order_id']]]
        if len(matches) != 1:
            raise ValueError(f'Unbound/ambiguous sell intent {order["order_id"]}')
        assigned[matches[0]['position_id']].append(order)
    output = []
    for p in positions:
        key = p['position_id']; rows = mapped[key]
        full = sorted([o for o in assigned[key] if o['order_type'] == 'MARKET'
                       and o['metadata'].get('order_role') == 'EXIT'],
                      key=lambda o: (stamp(o['metadata']['decision_available_at']), o['order_id']))
        sells = [f for f in rows if f['side'] == 'SELL']
        record = {k: p[k] for k in ['position_id', 'symbol', 'route', 'entry_fill_time',
                  'exit_fill_time', 'qty', 'entry_cost_krw', 'entry_equity_krw',
                  'gross_pnl_krw', 'fees_krw', 'net_pnl_krw']}
        record.update({'sell_execution_count': len(sells), 'unique_sell_order_count': len({f['order_id'] for f in sells}),
                       'initial_sell_intents': len(assigned[key]), 'full_market_exit_intents': len(full)})
        record['executed_slippage_krw'] = sum((f['price']-f['metadata']['base_price']) * f['qty'] *
                        (1 if f['side'] == 'BUY' else -1) for f in rows)
        if full:
            first = full[0]; when = stamp(first['metadata']['decision_available_at'])
            point = decision_snapshot(p, rows, when, marks)
            before = point['gross_marked_pnl_krw']
            after = p['gross_pnl_krw'] - before
            close(before + after - p['fees_krw'], p['net_pnl_krw'], 'Exact loss-stage identity')
            record.update({'first_full_exit': {'order_id': first['order_id'], 'reason': first['reason'],
                           'submitted_bar_timestamp': first['submitted_at'], 'request_qty': first['qty'], **point},
                           'before_first_full_exit_gross_krw': before,
                           'after_first_full_exit_gross_krw': after,
                           'entry_fills_at_or_after_exit_decision': sum(stamp(f['timestamp']) >= when and f['side']=='BUY' for f in rows),
                           'minutes_decision_to_flat': (datetime.fromisoformat(stamp(p['exit_fill_time']))-datetime.fromisoformat(when)).total_seconds()/60,
                           'minutes_entry_to_decision': (datetime.fromisoformat(when)-datetime.fromisoformat(stamp(p['entry_fill_time']))).total_seconds()/60,
                           'sell_executions_before_full_exit': sum(stamp(f['timestamp']) < when for f in sells)})
        else:
            record['full_exit_missing_reason'] = 'Closed without a submitted MARKET/EXIT role; retained separately, no invented decision.'
        output.append(record)
    return {'all_raw_close_equity_marks_reconciled': mark_count, 'all_lifetimes_reconciled': len(output),
            'unique_sell_submissions_bound': len(sell_orders), 'positions': output,
            'scope': 'Observed accounting decomposition, not executable exit benefit, independent labels or policy qualification.'}
