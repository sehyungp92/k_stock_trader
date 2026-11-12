"""Current-NAV path risk with explicit service and common dated scenarios.

Inputs are causal externally qualified scenarios, never a peer liquidity loss
substituted for a name's price return. Missing name/path support fails closed.
This module computes evidence; it does not invent expected alpha or authorize
new risk. Every candidate quantity is simulated again through its own service.
"""
from dataclasses import dataclass
from datetime import datetime
from math import isfinite


@dataclass(frozen=True)
class Book:
    cash: float
    high_water: float
    marks: dict
    quantities: dict


def project(book,scenario,at,*,sell_targets=None,commission_bps=2,tax_bps=20,slippage_bps=5):
    """Project a committed sell quantity over a supplied whole-book path.

Each step supplies actual candidate-name prices and share capacity on the SAME
scenario date. Partial liquidation returns cash only when that step serves it.
"""
    known=datetime.fromisoformat(scenario['known_at'])
    if known.tzinfo is None or at.tzinfo is None or known>=at:raise ValueError('Scenario is not strictly matured')
    if not scenario.get('source_hash') or not scenario.get('date'):raise ValueError('Unbound scenario')
    q=dict(book.quantities);cash=float(book.cash);marks=dict(book.marks);peak=float(book.high_water)
    if any(type(v) is not int or v<0 for v in q.values()):raise ValueError('Invalid book quantity')
    if any(s not in marks for s,v in q.items() if v):raise ValueError('Unmarked book')
    nav=cash+sum(v*marks[s] for s,v in q.items());peak=max(peak,nav)
    if not isfinite(nav) or nav<=0 or cash<0:raise ValueError('Invalid account')
    pending={s:q[s]-t for s,t in (sell_targets or {}).items()}
    if any(type(t) is not int or t<0 or t>q.get(s,-1) for s,t in (sell_targets or {}).items()):raise ValueError('Invalid target inventory')
    dd=1-nav/peak;points=[];costs=0;last=-1
    for step in scenario['path']:
        if step['offset']<=last:raise ValueError('Reversed scenario path')
        last=step['offset']
        if any(s not in step['prices'] or s not in step['capacity'] for s,v in q.items() if v):
            raise ValueError('Missing whole-book scenario support')
        for s,p in step['prices'].items():
            if not isfinite(p) or p<=0:raise ValueError('Invalid scenario price')
            marks[s]=p
        # Price movement is observed before service; include that marked loss.
        nav=cash+sum(v*marks[s] for s,v in q.items());peak=max(peak,nav);dd=max(dd,1-nav/peak)
        fills={}
        for s,left in pending.items():
            cap=step['capacity'].get(s,0)
            if type(cap) is not int or cap<0:raise ValueError('Invalid own-name service')
            n=min(left,cap,q[s]);price=marks[s]*(1-slippage_bps/10000)
            fee=n*price*(commission_bps+tax_bps)/10000
            cash+=n*price-fee;costs+=n*marks[s]-n*price+fee;q[s]-=n;pending[s]-=n
            if n:fills[s]=n
        nav=cash+sum(v*marks[s] for s,v in q.items());peak=max(peak,nav);dd=max(dd,1-nav/peak)
        points.append(dict(offset=last,nav=nav,high_water=peak,cash=cash,quantities=dict(q),fills=fills))
    if not points:raise ValueError('Empty scenario path')
    return dict(scenario_date=scenario['date'],source_hash=scenario['source_hash'],max_drawdown_fraction=dd,
                final_equity=nav,high_water=peak,remaining_orders=pending,remaining_inventory=q,costs=costs,path=points)


def compare_actions(book,scenarios,at,actions,**costs):
    """Requote every declared dose; do not reuse risk from a larger quantity."""
    if len({s['date'] for s in scenarios})!=len(scenarios):raise ValueError('Duplicate common shock date')
    if not scenarios:raise ValueError('No causal scenario support')
    return {name:[project(book,s,at,sell_targets=targets,**costs) for s in scenarios]
            for name,targets in actions.items()}
