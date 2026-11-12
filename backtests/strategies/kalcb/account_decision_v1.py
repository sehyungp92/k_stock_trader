"""Complete-state forks and path-valued account decisions, independent of search.

Prefixes are content-addressed serialized receipts, never shared mutable broker
objects. Every intervention starts before the same native timestamp batch.
"""
from dataclasses import dataclass
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path

from .pending_entry_account_pilot_v1 import state_receipt, canonical
from .account_history_v1 import clone_complete_account


@dataclass(frozen=True)
class AccountBoundary:
    state_sha256: str
    available_at: str
    equity: float
    running_peak: float
    marks: int
    fills: int
    pending_orders: int


def capture(adapter, broker, available_at, store=None):
    receipt = state_receipt(adapter, broker)
    if store is not None:
        path = Path(store) / (receipt['sha256'] + '.json.gz')
        encoded = canonical(receipt['state']).encode()
        if path.exists():
            with gzip.open(path, 'rb') as stream:
                if hashlib.sha256(stream.read()).hexdigest() != receipt['sha256']:
                    raise ValueError('Corrupt immutable account prefix')
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(path, 'xb', compresslevel=1) as stream:
                stream.write(encoded)
    return AccountBoundary(receipt['sha256'], available_at.isoformat(),
        broker._portfolio_equity(), max(broker.equity_curve),
        len(broker.timestamps), len(broker.fills), len(broker.orders))


def fork(adapter, broker, boundary):
    if state_receipt(adapter, broker)['sha256'] != boundary.state_sha256:
        raise ValueError('Account changed since captured boundary')
    result = clone_complete_account(adapter, broker)
    if state_receipt(*result)['sha256'] != boundary.state_sha256:
        raise ValueError('Incomplete account fork')
    return result


def assert_identity(left, right):
    if state_receipt(*left)['sha256'] != state_receipt(*right)['sha256']:
        raise ValueError('Disabled account continuation differs from parent')


def suffix_value(boundary, broker):
    """Path risk includes the inherited peak; observed marks are never fills."""
    equity = broker.equity_curve[boundary.marks + 1:]
    stamps = broker.timestamps[boundary.marks:]
    if not equity or len(equity) != len(stamps):
        raise ValueError('Missing complete account continuation marks')
    peak = boundary.running_peak
    worst = 0.0
    for value in equity:
        peak = max(peak, value)
        worst = max(worst, 1.0 - value / peak)
    fills = broker.fills[boundary.fills:]
    # Original entry order identifies a lifetime, not each partial execution.
    buys = {}
    for fill in fills:
        if fill.side == 'BUY':
            key = (fill.symbol, fill.order_id)
            buys[key] = buys.get(key, 0.0) + fill.qty * fill.price
    return dict(terminal_equity=equity[-1],
        net_change_krw=equity[-1] - boundary.equity,
        max_drawdown_from_inherited_peak=worst,
        worst_change_from_decision_krw=min(equity) - boundary.equity,
        funded_entry_orders=len(buys),
        material_funded_entries=sum(v >= .01 * boundary.equity for v in buys.values()),
        buy_notional_krw=sum(buys.values()),
        terminal_inventory={str(k): p.qty for k, p in broker.positions.items()},
        terminal_pending=len(broker.orders),
        equity_path=[(str(t), e) for t, e in zip(stamps, equity)])


def contrast(boundary, parent, candidate):
    left, right = suffix_value(boundary, parent), suffix_value(boundary, candidate)
    if [t for t, _ in left['equity_path']] != [t for t, _ in right['equity_path']]:
        raise ValueError('Account contrast has different mark clocks')
    return dict(parent=left, candidate=right,
        marginal_terminal_wealth_krw=right['terminal_equity'] - left['terminal_equity'],
        marginal_max_drawdown=right['max_drawdown_from_inherited_peak'] - left['max_drawdown_from_inherited_peak'],
        interpretation='Mutually exclusive whole-account suffix under declared parent continuation. '
        'Includes same-session funding descendants; cannot be summed into sequential strategy returns.')


def verify_suffix(boundary, broker, initial_cash, initial_positions, initial_marks):
    """Independent suffix cash/inventory/MTM from native fills and observed bars.

    The unchanged peer's source-qualified stream supplies prices. Missing quotes
    retain the last observed mark exactly as the native broker does; never fills.
    """
    cash=float(initial_cash)
    quantities=defaultdict(int,initial_positions)
    marks=dict(initial_marks)
    fills=defaultdict(list)
    for fill in broker.fills[boundary.fills:]:
        fills[fill.timestamp].append(fill)
    batches=broker.expected_batches[boundary.marks:]
    stamps=broker.timestamps[boundary.marks:]
    equity=broker.equity_curve[boundary.marks+1:]
    if len(batches)!=len(stamps) or len(stamps)!=len(equity):
        raise ValueError('Suffix source/mark count differs')
    for stamp,target,batch in zip(stamps,equity,batches):
        for fill in fills.pop(stamp,[]):
            if not isinstance(fill.qty,int) or fill.qty<=0:
                raise ValueError('Invalid suffix integer fill')
            notional=fill.qty*fill.price
            cost=notional*(broker.costs.commission_bps+(broker.costs.tax_bps_on_sell if fill.side=='SELL' else 0))/10000
            if abs(cost-float(fill.metadata['fee']))>.011:
                raise ValueError('Suffix fill fee differs from independent cost')
            if fill.side=='BUY':
                cash-=notional+cost;quantities[fill.symbol]+=fill.qty
            elif fill.side=='SELL':
                cash+=notional-cost;quantities[fill.symbol]-=fill.qty
            else:
                raise ValueError('Unknown suffix fill side')
            if quantities[fill.symbol]<0:
                raise ValueError('Suffix sold unowned inventory')
        for bar in batch:
            if bar['timestamp']!=stamp.isoformat():
                raise ValueError('Suffix source clock differs from account mark')
            marks[bar['symbol']]=bar['close']
        value=cash+sum(q*marks[s] for s,q in quantities.items() if q)
        if abs(value-target)>.011:
            raise ValueError('Independent suffix MTM differs')
    actual={s:p.qty for (_,s),p in broker.positions.items() if p.qty}
    if fills or abs(cash-broker.cash)>.011 or {s:q for s,q in quantities.items() if q}!=actual:
        raise ValueError('Independent suffix terminal cash/inventory differs')
    return {'passed':True,'marks':len(stamps),'fills':len(broker.fills)-boundary.fills,'terminal_cash':cash}
