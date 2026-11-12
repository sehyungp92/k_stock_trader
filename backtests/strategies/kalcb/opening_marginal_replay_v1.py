"""Native single-opening-route omissions from identical funded account states."""
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict
from datetime import datetime, timedelta
import gzip
import json
from pathlib import Path
import time

from .account_history_v1 import clone_complete_account
from .account_decision_v1 import capture, fork, assert_identity, contrast, verify_suffix
from .retention_research_io_v1 import bound, write

_omission = ContextVar('opening_marginal_omission_v1', default=None)


@contextmanager
def omit_opening(symbol, stamp, hits):
    token = _omission.set((symbol, stamp, hits) if symbol else None)
    try: yield
    finally: _omission.reset(token)


def apply_omission(original, symbol, qty, metadata):
    action = _omission.get()
    if action and symbol == action[0] and metadata.get('entry_route') == 'first30_open_anchor' and qty > 0:
        record = dict(symbol=symbol, timestamp=str(action[1]), requested_qty=qty, chosen_qty=0,
                      mechanism='single_symbol_session_first30_veto')
        metadata['opening_marginal_omission_v1'] = record
        action[2].append(record)
        return 0
    return original(symbol, qty, metadata)


def activity(broker, boundary, trade_start):
    entries = {}; profits = defaultdict(float)
    for fill in broker.fills[boundary.fills:]:
        if fill.side != 'BUY': continue
        stamp = datetime.fromisoformat(fill.metadata['submitted_at']).isoformat()
        key = fill.symbol+'/'+stamp
        row = entries.setdefault(key, dict(notional=0., equity=float(fill.metadata['portfolio_equity'])))
        row['notional'] += fill.qty*fill.price
    for trade in broker.trades[trade_start:]:
        key = trade.symbol+'/'+trade.entry_decision_time.isoformat()
        profits[key] += trade.net_pnl
    if set(profits)-set(entries): raise ValueError('Suffix trade lacks funded entry')
    rows = {k:dict(**v, net_pnl_krw=profits[k], material=v['notional']>=.01*v['equity']) for k,v in entries.items()}
    return dict(opportunities=rows, funded=len(rows), material=sum(v['material'] for v in rows.values()),
                profitable_material=sum(v['material'] and v['net_pnl_krw']>0 for v in rows.values()))


class OpeningMarginalReplay:
    def __init__(self, output, expected_units=None):
        self.output=Path(output); self.expected_units=set(expected_units) if expected_units is not None else None
        self.seen=set(); self.records=[]; self.pairs=[]; self.active=None; self.started=time.monotonic()

    def install(self):
        from backtests.engine import replay
        from strategy_kalcb.core import account_allocation_v1 as allocation
        self.kernel=replay.drive_timestamp_batch; self.original_allocation=allocation.allocate_quantity
        replay.drive_timestamp_batch=self.drive
        allocation.allocate_quantity=lambda s,q,m:apply_omission(self.original_allocation,s,q,m)

    def restore(self):
        from backtests.engine import replay
        from strategy_kalcb.core import account_allocation_v1 as allocation
        replay.drive_timestamp_batch=self.kernel; allocation.allocate_quantity=self.original_allocation

    def drive(self, broker, bars, adapters, *, accepts=None):
        bars=tuple(bars); stamp=bars[0].timestamp
        if len(adapters)!=1 or accepts is not None: raise ValueError('One native adapter required')
        adapter=adapters[0]; day=str(stamp.date())
        if self.active and self.active['day']!=day: self.finish_day()
        if (stamp.hour,stamp.minute)==(9,25):
            aa,bb=clone_complete_account(adapter,broker); old_ids={o.order_id for o in bb.orders}
            self.kernel(bb,bars,(aa,))
            orders=[o for o in bb.orders if o.order_id not in old_ids and o.side=='BUY'
                    and o.strategy_id=='KALCB' and o.metadata.get('entry_route')=='first30_open_anchor']
            if orders:
                if self.active: raise ValueError('Multiple opening boundaries in one session')
                if broker.positions or broker.orders: raise ValueError('Unexpected occupied opening account; revise population contract')
                symbols=[o.symbol for o in orders]
                if len(set(symbols))!=len(symbols): raise ValueError('Multiple first30 intents per symbol')
                keys={day+'/'+s for s in symbols}
                if self.expected_units is not None and not keys<=self.expected_units: raise ValueError('Unregistered native opening population')
                if self.seen & keys: raise ValueError('Repeated native opening opportunity')
                self.seen.update(keys)
                boundary=capture(adapter,broker,stamp+timedelta(minutes=5),self.output/'prefixes')
                self.active=dict(day=day,boundary=boundary,initial_cash=broker.cash,initial_positions={},
                    initial_marks=dict(broker.last_prices),trade_start=len(broker.trades),
                    arms={name:fork(adapter,broker,boundary) for name in ['hold',*symbols]},
                    hits={s:[] for s in symbols})
                self.records.append(dict(day=day,boundary=asdict(boundary),intents=[asdict(o) for o in orders]))
        result=self.kernel(broker,bars,adapters)
        if self.active:
            for name,(aa,bb) in self.active['arms'].items():
                with omit_opening(None if name=='hold' else name,stamp,self.active['hits'].get(name,[])):
                    self.kernel(bb,bars,(aa,))
            self.active['parent']=(adapter,broker)
        return result

    def finish_day(self):
        if self.active is None: return
        from backtests.engine.integrity import audit_broker
        p=self.active; boundary=p['boundary']; hold=p['arms']['hold'][1]
        assert_identity(p['parent'],p['arms']['hold'])
        parent_activity=activity(hold,boundary,p['trade_start'])
        for name,(aa,bb) in p['arms'].items():
            audit=audit_broker(bb)
            if not audit['execution_validity_pass']: raise ValueError('Native omission execution ledger failed')
            check=verify_suffix(boundary,bb,p['initial_cash'],p['initial_positions'],p['initial_marks'])
            if bb.positions or bb.orders: raise ValueError('Unresolved omission suffix account')
            value=contrast(boundary,hold,bb); acted=p['hits'].get(name,[])
            if name!='hold':
                if not acted: raise ValueError('Omission failed to activate')
                if any(f.side=='BUY' and f.symbol==name and f.metadata.get('entry_route')=='first30_open_anchor'
                       for f in bb.fills[boundary.fills:]): raise ValueError('Vetoed opening route still filled')
            donor=sum(t.net_pnl for t in hold.trades[p['trade_start']:] if t.symbol==name
                      and t.route_metadata.get('entry_route')=='first30_open_anchor') if name!='hold' else 0.
            candidate_activity=activity(bb,boundary,p['trade_start'])
            path=self.output/'forks'/p['day']/(name+'.json.gz');path.parent.mkdir(parents=True,exist_ok=True)
            payload=dict(day=p['day'],omitted_symbol=None if name=='hold' else name,boundary=asdict(boundary),
                initial_cash=p['initial_cash'],initial_positions=p['initial_positions'],initial_marks=p['initial_marks'],
                hits=acted,exact_disabled_identity=True,independent_suffix=check,engine_checks=audit,
                fills=[asdict(f) for f in bb.fills[boundary.fills:]],
                trades=[t.to_json_dict() for t in bb.trades[p['trade_start']:]],source_batches=bb.expected_batches[boundary.marks:],
                contrast=value,donor_first30_control_pnl_krw=donor,
                other_opportunity_effect_krw=value['marginal_terminal_wealth_krw']+donor,
                parent_activity=parent_activity,candidate_activity=candidate_activity)
            with gzip.open(path,'xt',encoding='utf-8',compresslevel=1) as stream:
                json.dump(payload,stream,default=str,separators=(',',':'),allow_nan=False)
            self.pairs.append(dict(day=p['day'],omitted_symbol=payload['omitted_symbol'],ledger=bound(path),
                delta_wealth_krw=value['marginal_terminal_wealth_krw'],donor_first30_control_pnl_krw=donor,
                other_opportunity_effect_krw=payload['other_opportunity_effect_krw'],
                delta_dd=value['marginal_max_drawdown'],delta_profitable_material=candidate_activity['profitable_material']-parent_activity['profitable_material']))
        self.active=None
        progress=dict(completed_opening_sessions=len(self.records),completed_omission_targets=len(self.seen),
                      seconds=time.monotonic()-self.started,last_day=p['day'],all_completed_suffix_checks_passed=True)
        write(self.output/'target_progress.json',progress)
        if len(self.records)==10: write(self.output/'pilot_checkpoint.json',progress)
        print(json.dumps(progress),flush=True)

    def finish(self):
        self.finish_day()
        if self.expected_units is not None and self.seen!=self.expected_units: raise ValueError('Incomplete registered opening population')
        write(self.output/'opening_decisions.json',self.records);write(self.output/'marginal_targets.json',self.pairs)
