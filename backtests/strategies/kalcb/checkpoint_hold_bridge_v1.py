"""Read-only isolated HOLD / actual native lifetime comparison."""
from datetime import datetime
import math


def stamp(value):
    return datetime.fromisoformat(value)


def fills_canonical(fills, symbol, before=None):
    rows = []
    for f in fills:
        t = stamp(f['timestamp'])
        if before is not None and t >= stamp(before):
            continue
        if f['symbol'] != symbol or f['side'] not in ('BUY', 'SELL'):
            raise ValueError('Wrong lifetime symbol or side')
        if type(f['qty']) is not int or f['qty'] <= 0 or not math.isfinite(f['price']) or f['price'] <= 0:
            raise ValueError('Invalid fill quantity/price')
        rows.append((t.isoformat(), f['side'], f['qty'], f['price'], f['reason']))
    return rows


def account_from_fills(fills, costs):
    gross = fees = 0.; inventory = 0
    for f in fills:
        q = f['qty']; notional = q*f['price']; buy = f['side'] == 'BUY'
        inventory += q if buy else -q
        if inventory < 0:
            raise ValueError('Negative lifetime inventory')
        gross += -notional if buy else notional
        fees += notional*(costs['commission_bps'] + (0 if buy else costs['tax_bps_on_sell']))/10000
    return {'gross_cash_flow': gross, 'fees': fees, 'terminal_inventory': inventory,
            'net_cash_flow': gross-fees}


def compare(native, isolated, symbol, clock, costs, native_position):
    prefix = fills_canonical(native, symbol, clock) == fills_canonical(isolated, symbol, clock)
    full = fills_canonical(native, symbol) == fills_canonical(isolated, symbol)
    n = account_from_fills(native, costs); i = account_from_fills(isolated, costs)
    if n['terminal_inventory'] != native_position['live_qty']:
        raise ValueError('Native inventory differs from authenticated lifetime')
    if abs(n['fees']-native_position['fees_krw']) > 1e-5:
        raise ValueError('Native cost model differs from authenticated fees')
    if not n['terminal_inventory'] and abs(n['net_cash_flow']-native_position['net_pnl_krw']) > 1e-5:
        raise ValueError('Native cash flow differs from authenticated PnL')
    same_cost = abs(n['fees']-i['fees']) < 1e-5
    same_cash = abs(n['net_cash_flow']-i['net_cash_flow']) < 1e-5
    return {'exact_fill_prefix': prefix, 'exact_all_fills': full,
            'exact_fees': same_cost, 'exact_net_cash': same_cash,
            'hold_lifetime_bridge': prefix and full and same_cost and same_cash and not n['terminal_inventory'] and not i['terminal_inventory'],
            'native_fill_count': len(native), 'reference_fill_count': len(isolated),
            'native_terminal_inventory': n['terminal_inventory'], 'reference_terminal_inventory': i['terminal_inventory'],
            'native_net_cash_flow': n['net_cash_flow'], 'reference_net_cash_flow': i['net_cash_flow']}
