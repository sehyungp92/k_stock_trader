"""Pure sleeve funding contract; native cores retain their order lifecycles.

Pending native quantities are requests, not additional cash. This ledger assigns
only feasible reservations in the existing execution order. Actual fills must
be checked again at their execution price. No exit order supplies buying power.
"""
from dataclasses import dataclass
from datetime import datetime
import hashlib
from itertools import groupby
import math


@dataclass(frozen=True)
class NativeBudgetConfig:
    kalcb_sizing_fraction: float = 1.0
    olr_fraction: float = 1.0
    kalcb_budget_fraction: float = 2.0
    max_kalcb_stop_risk_fraction: float = 0.0
    allocation_mode: str = 'symbol'
    allocation_seed: int = 0
    capital_mode: str = 'shared'
    kalcb_capital_fraction: float = .5

    def __post_init__(self):
        for name, low, high in (
            ('kalcb_sizing_fraction', .25, 2.), ('olr_fraction', .05, 1.),
            ('kalcb_budget_fraction', .25, 2.), ('max_kalcb_stop_risk_fraction', 0., .1),
            ('kalcb_capital_fraction', .05, .95)):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError('Invalid native budget: ' + name)
        if self.allocation_mode not in ('symbol', 'native_rank', 'proportional', 'random'):
            raise ValueError('Invalid native allocation mode')
        if type(self.allocation_seed) is not int or self.allocation_seed < 0:
            raise ValueError('Invalid allocation seed')
        if self.capital_mode not in ('shared', 'partition', 'daily_rebalanced'):
            raise ValueError('Invalid capital mode')


def native_rank(order):
    value = (order.get('metadata') or {}).get('candidate_rank')
    return int(value) if value is not None and float(value) > 0 else 1_000_000


def cash_limited_transfer(wealth, cash, kalcb_fraction):
    """Move only available donor cash toward the target marked-wealth split."""
    desired = sum(wealth.values())*kalcb_fraction-wealth['KALCB']
    actual = min(desired, max(0.,cash['OLR'])) if desired > 0 else max(desired,-max(0.,cash['KALCB']))
    return desired,actual


def order_key(order, priority, config=None):
    """Match NativePolicyMixin with its learned/rank overrides disabled."""
    stamp = order['submitted_at']
    if isinstance(stamp, str):
        stamp = datetime.fromisoformat(stamp)
    prefix = (stamp, priority.index(order['strategy_id']))
    mode = config.allocation_mode if config else 'symbol'
    if mode in ('native_rank', 'proportional'):
        prefix += (native_rank(order),)
    elif mode == 'random':
        # Stable across processes and independent of outcomes and submission sequence.
        token = f"{config.allocation_seed}|{stamp.isoformat()}|{order['strategy_id']}|{order['symbol']}"
        prefix += (hashlib.sha256(token.encode()).hexdigest(),)
    return (*prefix, order['symbol'], order['order_id'])


def proportional_grants(items, available):
    """Integer max-min completion of native requests within one causal cohort.

    Items contain remaining quantity, already filled quantity, and resource use
    per share. Credit past partial fills so successive execution callbacks cannot
    repeatedly favor the first order. Freeze only orders using a saturated
    resource, then distribute residual resources to other feasible orders.
    """
    grants = [0. for _ in items]
    active = {i for i, x in enumerate(items) if x['qty'] > 0 and x['unit'] > 0}
    remaining = {k: max(0., v) for k, v in available.items()}
    while active:
        def increments(level):
            return {i: max(0., min(items[i]['qty'], level * (items[i]['qty'] + items[i]['filled'])
                - items[i]['filled']) - grants[i]) for i in active}
        def use(inc, key):
            return sum(q * items[i]['resources'].get(key, 0.) for i, q in inc.items())
        low, high = 0., 1.
        for _ in range(52):
            mid = (low + high) / 2
            inc = increments(mid)
            if all(use(inc, k) <= v + 1e-9 for k, v in remaining.items()): low = mid
            else: high = mid
        inc = increments(low)
        for k in remaining: remaining[k] = max(0., remaining[k] - use(inc, k))
        for i, q in inc.items(): grants[i] += q
        saturated = {k for k, v in remaining.items() if v <= 1e-6}
        frozen = {i for i in active if grants[i] >= items[i]['qty'] - 1e-8
                  or any(items[i]['resources'].get(k, 0.) > 0 for k in saturated)}
        if not frozen:
            raise ArithmeticError('Proportional allocation did not converge')
        active -= frozen
    result = [min(x['qty'], max(0, math.floor(q + 1e-8))) for x, q in zip(items, grants)]
    residual = {k: max(0., v - sum(q * x['resources'].get(k, 0.) for x, q in zip(items, result)))
                for k, v in available.items()}
    # Largest fractional remainder, with the already sorted native rank as tie-break.
    for i in sorted(range(len(items)), key=lambda i: (-(grants[i] - result[i]), i)):
        x = items[i]
        if x['unit'] > 0 and result[i] < x['qty'] and all(v <= residual[k] + 1e-9 for k, v in x['resources'].items()):
            result[i] += 1
            for k, v in x['resources'].items(): residual[k] -= v
    return result


def reservation_ledger(*, cash, holdings, orders, marks, costs, leverage, caps,
                       sectors, priority, config, execution=None, sizing_equity_cap=0., sleeve_equities=None):
    """Allocate remaining native requests against one current account.

Holdings are dictionaries with strategy_id, symbol, qty, and risk_per_share.
execution optionally supplies (order_id, execution_unit_cost). No supplied price
may come from a future bar. Missing price evidence funds zero shares.
"""
    by_strategy = {'KALCB': 0., 'OLR': 0.}
    by_symbol, by_sector = {}, {}
    gross = risk = 0.
    for h in holdings:
        sid, sym, qty = h['strategy_id'], h['symbol'], h['qty']
        amount = qty * marks[sym]
        gross += amount
        by_strategy[sid] += amount
        by_symbol[sym] = by_symbol.get(sym, 0.) + amount
        sector = sectors.get(sym, 'UNKNOWN')
        by_sector[sector] = by_sector.get(sector, 0.) + amount
        if sid == 'KALCB':
            risk += qty * max(0., h.get('risk_per_share', 0.))
    equity = max(0., cash + gross)
    sizing = min(equity, sizing_equity_cap) if sizing_equity_cap > 0 else equity
    sizing_by_strategy = {sid:sizing for sid in by_strategy}
    if config.capital_mode != 'shared':
        if sleeve_equities is None or set(sleeve_equities) != set(by_strategy):
            raise ValueError('Partition requires reconciled sleeve equity')
        if not math.isclose(sum(sleeve_equities.values()), cash+gross, rel_tol=1e-12, abs_tol=1e-6):
            raise ValueError('Partition wealth does not reconcile')
        for sid in by_strategy:
            fraction = config.kalcb_capital_fraction if sid == 'KALCB' else 1-config.kalcb_capital_fraction
            sizing_by_strategy[sid] = max(0., min(sleeve_equities[sid], sizing_equity_cap*fraction)
                if sizing_equity_cap > 0 else sleeve_equities[sid])
    budgets = {'KALCB': min(leverage['KALCB'], config.kalcb_budget_fraction) * sizing_by_strategy['KALCB'],
               'OLR': min(leverage['OLR'], config.olr_fraction) * sizing_by_strategy['OLR']}
    funding = max(0., min(equity * max(leverage.values()), caps['max_gross_notional']) - gross)
    risk_limit = equity * config.max_kalcb_stop_risk_fraction
    allocations = {}
    ordered = sorted((o for o in orders if o['side'] == 'BUY'), key=lambda o: order_key(o, priority, config))
    def unit_cost(order):
        sid, sym = order['strategy_id'], order['symbol']
        meta = order.get('metadata') or {}
        estimate = max(float(meta.get('estimated_entry_price') or 0.),
                       float(meta.get('entry_price_ref') or 0.), float(marks.get(sym, 0.)))
        if execution and execution[0] == order['order_id']: return execution[1]
        return estimate * (1 + (costs[sid]['commission_bps'] + costs[sid]['slippage_bps']) / 10000)
    proportional = {}
    if config.allocation_mode == 'proportional':
        resources = {'physical_funding': funding,
            **{'strategy:'+s: max(0., v-by_strategy[s]) for s, v in budgets.items()},
            **{'symbol:'+o['symbol']: max(0., caps['max_symbol_notional']-by_symbol.get(o['symbol'], 0.)) for o in ordered},
            **{'sector:'+sectors.get(o['symbol'], 'UNKNOWN'): max(0., caps['max_sector_notional']-by_sector.get(sectors.get(o['symbol'], 'UNKNOWN'), 0.)) for o in ordered}}
        if risk_limit > 0: resources['kalcb_stop_risk'] = max(0., risk_limit-risk)
        for _, group in groupby(ordered, key=lambda o: order_key(o, priority, config)[:2]):
            group = list(group); items = []
            for o in group:
                sid, sym = o['strategy_id'], o['symbol']; unit = unit_cost(o)
                rps = max(0., float((o.get('metadata') or {}).get('risk_per_share') or 0.))
                use = {'physical_funding': unit, 'strategy:'+sid: unit, 'symbol:'+sym: unit,
                       'sector:'+sectors.get(sym, 'UNKNOWN'): unit}
                qty = int(o['qty'] or 0)
                if sid == 'KALCB' and risk_limit > 0:
                    use['kalcb_stop_risk'] = rps
                    if rps <= 0: qty = 0
                items.append({'qty': qty, 'filled': int(o.get('filled_qty', 0)), 'unit': unit, 'resources': use})
            for o, x, qty in zip(group, items, proportional_grants(items, resources)):
                proportional[o['order_id']] = qty
                for k, v in x['resources'].items(): resources[k] = max(0., resources[k]-qty*v)
    for order in ordered:
        sid, sym, oid = order['strategy_id'], order['symbol'], order['order_id']
        meta = order.get('metadata') or {}
        sector = sectors.get(sym, 'UNKNOWN')
        unit = unit_cost(order)
        constraints = {'physical_funding': funding,
            'strategy_budget': budgets[sid] - by_strategy[sid],
            'symbol_budget': caps['max_symbol_notional'] - by_symbol.get(sym, 0.),
            'sector_budget': caps['max_sector_notional'] - by_sector.get(sector, 0.)}
        available = max(0., min(constraints.values()))
        qty = min(int(order['qty'] or 0), math.floor((available + 1e-9) / unit)) if unit > 0 else 0
        rps = max(0., float(meta.get('risk_per_share') or 0.)) if sid == 'KALCB' else 0.
        bindings = [k for k, v in constraints.items() if v <= available + 1e-7] if qty < int(order['qty'] or 0) else []
        if unit <= 0:
            bindings.append('missing_price')
        if risk_limit > 0 and sid == 'KALCB':
            risk_qty = max(0, math.floor((risk_limit - risk + 1e-9) / rps)) if rps > 0 else 0
            if risk_qty < qty:
                bindings.append('kalcb_stop_risk')
            qty = min(qty, risk_qty)
        if oid in proportional:
            if proportional[oid] < qty: bindings.append('proportional_cohort')
            qty = min(qty, proportional[oid])
        amount = qty * unit
        allocations[oid] = {'qty': qty, 'unit_cost': unit, 'reserved_notional': amount,
            'strategy_id': sid, 'requested_qty': int(order['qty'] or 0),
            'binding_constraints': bindings,
            'planned_risk': qty * rps}
        funding -= amount
        by_strategy[sid] += amount
        by_symbol[sym] = by_symbol.get(sym, 0.) + amount
        by_sector[sector] = by_sector.get(sector, 0.) + amount
        risk += qty * rps
    return {'equity': equity, 'held_gross': gross, 'budgets': budgets,
            'allocations': allocations, 'remaining_funding': max(0., funding),
            'committed_planned_kalcb_risk': risk}
