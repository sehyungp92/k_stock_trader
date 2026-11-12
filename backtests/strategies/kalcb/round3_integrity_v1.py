"""C17 read-only lifecycle evidence and strict physical-path comparison.

No selection, price, sizing, or fill implementation lives here. The observer
delegates every batch to the installed native kernel. Optional hydration is a
preflight probe, never a selectable trading mechanism.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
import gzip
import hashlib
import json
import math
from enum import Enum
from pathlib import Path
import pickle


def neutral(value):
    return json.loads(json.dumps(value, default=lambda x:x.value if isinstance(x,Enum) else str(x), allow_nan=False))


def emit(stream, value):
    stream.write(json.dumps(neutral(value), separators=(',', ':')) + '\n')


def rows(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for row in stream:
            yield json.loads(row)


def exact_paths(actual, reference):
    """Do not discard metadata, event ordering, fills, or quantity differences."""
    actual, reference = Path(actual), Path(reference)
    checks = {}
    for name in ('fills.json','trades.json','equity.json','daily.json','horizon_actions.json',
                 'decisions.jsonl.gz','orders.jsonl.gz'):
        if name.endswith('.gz'):
            left, right = list(rows(actual/name)), list(rows(reference/name))
        else:
            left, right = (json.loads((p/name).read_text(encoding='utf-8')) for p in (actual,reference))
        if left != right:
            mismatch = next((i for i,(a,b) in enumerate(zip(left,right),1) if a != b), min(len(left),len(right))+1)
            raise ValueError(f'Physical parity failed: {name} row {mismatch}; counts {len(left)}/{len(right)}')
        checks[name] = len(left)
    return dict(passed=True, full_ordered_json_equality=True, artifacts=checks)


def hydrate(adapter, broker, directory, label):
    """Persist complete adapter/broker state, rehash source prefix, and hydrate.

    Pickle is read only from the just-written local checkpoint, never external
    content. The broker hash object is reconstructed from its retained ordered
    source prefix. Core state uses the production JSON serializer.
    """
    from strategy_kalcb.core.serializers import snapshot_state, restore_state
    from backtests.auto.shared.cache_keys import stable_signature
    from .pending_entry_account_pilot_v1 import neutral as state_value
    payload = snapshot_state(adapter.state)
    state = dict(vars(adapter)); state.pop('state')
    ledger = dict(vars(broker)); source_hash = ledger.pop('source_events').hexdigest()
    packet = dict(adapter=state, core=payload, broker=ledger, source_hash=source_hash)
    path = Path(directory)/(label+'.checkpoint.pkl')
    with path.open('xb') as stream:
        pickle.dump(packet, stream, protocol=5)
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open('rb') as stream:
        loaded = pickle.load(stream)
    if hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
        raise ValueError('Checkpoint changed')
    core = restore_state(deepcopy(loaded['core']))
    if snapshot_state(core) != payload:
        raise ValueError('Core JSON hydration drift')
    prefix_hash = hashlib.sha256()
    # process_batch preserves the incoming batch order when hashing source bars.
    for batch in loaded['broker'].get('expected_batches', ()):
        for bar in batch:
            prefix_hash.update(json.dumps(bar, sort_keys=True, default=str).encode())
    if prefix_hash.hexdigest() != source_hash:
        raise ValueError('Cannot reconstruct complete source hash at restart')
    before = stable_signature(state_value(dict(adapter=state, core=payload, broker=ledger)))
    adapter.__dict__.clear(); adapter.__dict__.update(loaded['adapter']); adapter.state=core
    broker.__dict__.clear(); broker.__dict__.update(loaded['broker']); broker.source_events=prefix_hash
    after = stable_signature(state_value(dict(adapter={k:v for k,v in vars(adapter).items() if k!='state'},
        core=snapshot_state(adapter.state),broker={k:v for k,v in vars(broker).items() if k!='source_events'})))
    if before != after:
        raise ValueError('Incomplete adapter/broker hydration')
    return dict(passed=True, label=label, state_hash=before, checkpoint_sha256=checksum,
        source_prefix_hash=source_hash, pending=len(broker.orders), positions=len(broker.positions),
        scope='disk checkpoint hydration in the same process; suffix parity required separately')


class LifecycleObserver:
    def __init__(self, output, *, enabled=False, hydration=False):
        self.output=Path(output); self.enabled=enabled; self.hydration=hydration
        self.cursor=0; self.batches=0; self.hydrated=set(); self.receipts=[]
        self.kernel=None; self.journal=None; self.accounts=None

    def install(self):
        if not self.enabled and not self.hydration:
            return
        from backtests.engine import replay
        self.kernel=replay.drive_timestamp_batch
        replay.drive_timestamp_batch=self.drive
        if self.enabled:
            self.journal=gzip.open(self.output/'broker_journal.jsonl.gz','xt',encoding='utf-8',compresslevel=1)
            self.accounts=gzip.open(self.output/'account_states.jsonl.gz','xt',encoding='utf-8',compresslevel=1)

    def restore(self):
        if self.kernel is not None:
            from backtests.engine import replay
            replay.drive_timestamp_batch=self.kernel
        for stream in (self.journal,self.accounts):
            if stream is not None: stream.close()

    def drive(self, broker, bars, adapters, *, accepts=None):
        if len(adapters)!=1 or accepts is not None:
            raise ValueError('Observer requires one whole native account')
        bars=tuple(bars)
        result=self.kernel(broker,bars,adapters)
        self.batches+=1
        if self.enabled:
            for index in range(self.cursor,len(broker.journal)):
                emit(self.journal,dict(row=index+1,record=broker.journal[index]))
            self.cursor=len(broker.journal)
            emit(self.accounts,dict(batch=self.batches,timestamp=bars[0].timestamp,
                journal_rows=self.cursor,cash=broker.cash,equity=broker._portfolio_equity(),
                inventory={s:p.qty for (_,s),p in broker.positions.items()},
                orders=[asdict(o) for o in broker.orders],marks=broker.last_prices,
                source_event_count=broker.source_event_count))
        if self.hydration:
            label='pending' if broker.orders and 'pending' not in self.hydrated else (
                'inventory' if broker.positions and 'inventory' not in self.hydrated else None)
            if label:
                self.receipts.append(hydrate(adapters[0],broker,self.output,label));self.hydrated.add(label)
        return result


def reconstruct_journal(directory, initial):
    """Independent arithmetic, no broker/core methods or reported NAV inputs."""
    directory=Path(directory); cash=float(initial); qty=Counter(); prices={}; orders={}
    fees=0.; slip=0.; peak=float(initial); worst=0.; observed=[]; fills=0; count=0
    stream=iter(rows(directory/'broker_journal.jsonl.gz')); max_error=0.; min_cash=cash
    for state in rows(directory/'account_states.jsonl.gz'):
        while count < state['journal_rows']:
            item=next(stream);count+=1
            if item['row'] != count: raise ValueError('Journal sequence gap')
            r=item['record']; kind=r['event']; order=r.get('order',{}); oid=order.get('order_id')
            if kind in ('submit','amend'):
                orders[oid]=deepcopy(order)
            elif kind in ('cancel','expire','reject'):
                orders.pop(oid,None)
            elif kind in ('opening_marks','closing_marks'):
                prices.update(r['prices'])
                if kind=='closing_marks':
                    nav=cash+sum(q*prices[s] for s,q in qty.items() if q)
                    peak=max(peak,nav);worst=max(worst,100*(peak-nav)/peak);observed.append([r['timestamp'],nav])
            elif kind=='fill':
                f=r['fill']; sign=1 if f['side']=='BUY' else -1; q=f['qty']; price=f['price']; c=r['costs']
                if type(q) is not int or q<=0 or price<=0:raise ValueError('Invalid physical fill')
                fee=q*price*(c['commission_bps']+(c['tax_bps_on_sell'] if sign<0 else 0))/10000
                if abs(fee-f['metadata']['fee'])>.02 or abs(cash-r['cash_before'])>.02:
                    raise ValueError('Journal fee/cash-before mismatch')
                cash-=sign*q*price+fee;qty[f['symbol']]+=sign*q;fees+=fee;fills+=1
                min_cash=min(min_cash,cash);slip+=sign*q*(price-f['metadata']['base_price'])
                if qty[f['symbol']]<0 or abs(cash-r['cash_after'])>.02:
                    raise ValueError('Negative inventory/cash-after mismatch')
                available=datetime.fromisoformat(order['metadata']['decision_available_at'])
                if datetime.fromisoformat(f['timestamp']) < available:
                    raise ValueError('Fill precedes information availability')
                orders.pop(oid,None) # A partial remainder is an explicit later amendment.
        actual=cash+sum(q*prices[s] for s,q in qty.items() if q)
        max_error=max(max_error,abs(cash-state['cash']),abs(actual-state['equity']))
        if {s:q for s,q in qty.items() if q}!=state['inventory'] or max_error>.02:
            raise ValueError('Intermediate account mismatch')
        expected={o['order_id']:o for o in state['orders']}
        if orders!=expected:raise ValueError('Pending reservation/order mismatch')
    if next(stream,None) is not None:raise ValueError('Unconsumed journal')
    native=json.loads((directory/'equity.json').read_text())
    if len(native)!=len(observed) or any(datetime.fromisoformat(a[0])!=datetime.fromisoformat(b[0]) or abs(a[1]-b[1])>.02 for a,b in zip(native,observed)):
        raise ValueError('Full native MTM curve mismatch')
    return dict(passed=True,events=count,marks=len(observed),fills=fills,terminal_cash=cash,
        terminal_inventory={s:q for s,q in qty.items() if q},pending=len(orders),
        max_account_error_krw=max_error,fees_krw=fees,embedded_slippage_krw=slip,
        max_drawdown_pct=worst,minimum_cash=min_cash,
        collateral='Native levered buying power; no separately modelled collateral settlement ledger',
        interest_borrow='No native financing/borrow charge; not evidence of zero real-world cost')
