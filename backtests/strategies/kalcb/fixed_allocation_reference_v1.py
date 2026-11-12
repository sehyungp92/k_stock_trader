"""Research benchmark: fixed nominal allocation reference, actual funded account.

The strategy's cash-based sizing/heat budget uses reference minus invested cost.
Broker cash, equity, drawdown/session state, capacity and fills stay real. Profits
are not added to the allocation reference and losses are not replenished. Actual
funding can reject an otherwise eligible order. This is a fixed allocation-budget
comparison, NOT an externally replenished constant-balance account or live rule.
"""
from dataclasses import replace
import math
import json
from pathlib import Path


def allocation_cash(reference, invested):
    reference, invested = float(reference), float(invested)
    if not math.isfinite(reference) or reference <= 0 or not math.isfinite(invested) or invested < 0:
        raise ValueError('Invalid fixed allocation reference or invested cost')
    return max(0., reference-invested)


def validate_scope(config):
    # These mechanisms introduce other capital bases. A separate registered
    # benchmark is necessary rather than silently claiming full fixed sizing.
    if (config.risk_dynamic_notional_enabled or config.risk_max_single_equity_pct or
            config.risk_max_gross_equity_pct or config.risk_daily_vol_target or
            config.frontier_shadow_enabled):
        raise ValueError('Unsupported capital-dependent or shadow allocation mechanism')


class FixedAllocationReference:
    def __init__(self, reference, *, enabled):
        self.reference = float(reference)
        allocation_cash(self.reference, 0.)
        self.enabled = bool(enabled)
        self.installed = False
        self.rows = []
        self.seen = set()

    def install(self):
        if self.installed: raise ValueError('Allocation reference already installed')
        from backtests.strategies.kalcb.runner import KALCBReplayAdapter
        original = KALCBReplayAdapter._portfolio_view
        self.original = original
        observer = self

        def view(adapter, broker):
            actual = original(adapter, broker)
            if observer.enabled: validate_scope(adapter.config)
            if observer.enabled and actual.equity <= 0:
                raise ValueError('Nonpositive funded equity cannot be replaced by an allocation reference')
            budget = allocation_cash(observer.reference, actual.open_notional) if observer.enabled else actual.cash
            # Keep actual equity. The core updates high-water/session returns
            # from it; replacing equity would inadvertently erase risk signals.
            result = replace(actual, cash=budget)
            stamp = str(broker.timestamps[-1]) if broker.timestamps else None
            identity = (stamp, actual.cash, actual.equity, actual.open_notional, budget)
            if identity not in observer.seen:
                observer.seen.add(identity)
                # This is the last broker mark, not necessarily the current
                # decision clock. Use native decision/order/fill journals for
                # causal funding attribution; do not join on this field.
                observer.rows.append({'last_mark_timestamp':stamp,'actual_cash':actual.cash,'actual_equity':actual.equity,
                    'invested_cost':actual.open_notional,'allocation_reference':observer.reference,
                    'sizing_cash_reference':budget,'enabled':observer.enabled})
            return result

        KALCBReplayAdapter._portfolio_view = view
        self.installed = True

    def restore(self):
        if not self.installed: raise ValueError('Allocation reference not installed')
        from backtests.strategies.kalcb.runner import KALCBReplayAdapter
        KALCBReplayAdapter._portfolio_view = self.original
        self.installed = False

    def evidence(self):
        return {'version':'fixed-allocation-reference-v1','enabled':self.enabled,'reference':self.reference,
            'rows':self.rows,'ledger_cash_mutations':0,'external_capital_flows':0,
            'scope':'Actual funded native account with fixed nominal allocation cash reference minus invested cost. Real equity/drawdown and broker funding retained. No cash reset or replenishment; this is not a constant-balance account. Identity mode requires exact native bridge; fixed mode requires source/account/funding audit before comparison.'}


def verify_identity_bridge(actual,reference):
    """Check economic invariance of installing the disabled benchmark code."""
    actual,reference=Path(actual),Path(reference)
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    for f in ('equity.json','daily.json'):
        if read(actual/f)!=read(reference/f):raise ValueError('Fixed-allocation identity account drift '+f)
    def physical(f):
        return {**{k:f[k] for k in ('order_id','strategy_id','symbol','side','qty','price','timestamp','reason')},
                'execution':{k:f['metadata'].get(k) for k in ('execution_phase','base_price','fee',
                             'decision_available_at','submitted_at','execution_bar')}}
    if list(map(physical,read(actual/'fills.json')))!=list(map(physical,read(reference/'fills.json'))):
        raise ValueError('Fixed-allocation identity physical fill drift')
    return {'passed':True,'reference':str(reference),'equity_marks':len(read(actual/'equity.json')),
            'scope':'All native daily balances, timestamped MTM marks and physical fills exactly reproduced.'}
