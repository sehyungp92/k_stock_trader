"""Repeated native holding decisions and sequential, exact-account suffixes.

Future bars are accessible only to the counterfactual simulator. Observation
construction and policy selection receive the completed native prefix alone.
No branch value is compounded or added to a strategy return.
"""
from dataclasses import asdict
from datetime import datetime,timedelta
import gzip
import json
from itertools import groupby
from pathlib import Path
import time

from . import account_fork_v2 as states
from .account_decision_v1 import suffix_value,verify_suffix
from .holding_action_v1 import observations,apply
from .retention_research_io_v1 import write,bound


def save_gzip(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(path,'xt',encoding='utf-8',compresslevel=1) as f:
        json.dump(value,f,default=str,separators=(',',':'),allow_nan=False)
    return bound(path)


def ledger(boundary,broker,cash,positions,marks):
    audit=verify_suffix(boundary,broker,cash,positions,marks)
    fills=broker.fills[boundary.fills:]
    buys={}
    for f in fills:
        available=datetime.fromisoformat(f.metadata['decision_available_at'])
        if f.timestamp<available:raise ValueError('Anticipatory branch execution')
        if f.side=='BUY':
            key=(f.symbol,f.order_id)
            item=buys.setdefault(key,{'notional':0.,'equity':float(f.metadata['portfolio_equity'])})
            item['notional']+=f.qty*f.price
    value=suffix_value(boundary,broker)
    value['max_drawdown_from_inherited_peak']=max(1-boundary.equity/boundary.running_peak,
        value['max_drawdown_from_inherited_peak'])
    value['material_funded_entry_orders']=sum(x['notional']>=.01*x['equity'] for x in buys.values())
    value['fees_krw']=sum(float(f.metadata['fee']) for f in fills)
    return {'value':value,'independent_suffix_audit':audit,
        'full_source_qualification':'authenticated_by_sequential_control_not_branch_alone',
        'fills':[asdict(f) for f in fills], 'terminal_cash':broker.cash,
        'terminal_inventory':{str(k):p.qty for k,p in broker.positions.items()},
        'terminal_pending':[asdict(o) for o in broker.orders]}


class HoldingAccountReplay:
    def __init__(self,output,*,mode='labels',limit=None,selector=None):
        if mode not in ('labels','policy','disabled'):raise ValueError('Unknown holding replay mode')
        self.output=Path(output);self.mode=mode;self.limit=limit;self.selector=selector
        self.rows=0;self.forks=0;self.day=None;self.batches=();self.expected_hold=None
        self.checked_days=[];self.actions=[];self.parent=None;self.started=time.monotonic()
        self.done_lifetimes=set()

    def install(self):
        from backtests.engine import replay
        from scripts import kalcb_executed_value_replay as native
        self.kernel=replay.drive_timestamp_batch;replay.drive_timestamp_batch=self.drive
        self.old_prefix=native.observation_prefix
        def source(*args,**kwargs):
            bars=self.old_prefix(*args,**kwargs)
            self.set_source(bars)
            return bars
        native.observation_prefix=source

    def restore(self):
        from backtests.engine import replay
        from scripts import kalcb_executed_value_replay as native
        replay.drive_timestamp_batch=self.kernel;native.observation_prefix=self.old_prefix

    def set_source(self,bars):
        self.finish_day()
        self.batches=tuple(tuple(g) for _,g in groupby(bars,key=lambda b:b.timestamp))
        if not self.batches:raise ValueError('Missing full native source day')
        self.day=str(self.batches[0][0].timestamp.date());self.done_lifetimes=set()
        if self.mode=='labels':
            self.source_binding=save_gzip(self.output/'sources'/(self.day+'.json.gz'),
                [[b.to_json_dict() for b in batch] for batch in self.batches])

    def finish_day(self):
        if self.expected_hold is not None:
            if states.receipt(*self.parent)['sha256']!=self.expected_hold:
                raise ValueError('No-action EOD full-state identity failed')
            self.checked_days.append({'day':self.day,'exact_hold_eod_state':self.expected_hold})
            write(self.output/'hold_identity.json',self.checked_days)
        self.expected_hold=None

    def drive(self,broker,bars,adapters,*,accepts=None):
        bars=tuple(bars)
        if len(adapters)!=1 or accepts is not None:raise ValueError('One complete native account required')
        result=self.kernel(broker,bars,adapters)
        adapter=adapters[0];self.parent=(adapter,broker)
        if self.mode=='disabled':return result
        stamp=bars[0].timestamp
        if str(stamp.date())!=self.day:raise ValueError('Exact filtered full-day source was not captured')
        rows=observations(adapter,broker,bars)
        for row in rows:
            with (self.output/'observations.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(row,separators=(',',':'))+'\n')
        eligible=[r for r in rows if r['reason']=='eligible']
        if self.mode=='policy':
            # One intervention per lifetime; observations continue so omissions
            # and changed state support remain measurable in the sequential path.
            offers=[]
            for row in eligible:
                if row['entry_order_id'] in self.done_lifetimes:continue
                decision=self.selector(row)
                self.actions.append({'observation':row,'decision':decision})
                if decision['arm']!='hold':offers.append((decision['predicted_value_bps'],row,decision))
            # One simultaneous alternative, as in the training contract. Future
            # completed observations can reconsider other funded positions.
            if offers:
                _,row,decision=max(offers,key=lambda r:(r[0],r[1]['symbol']))
                action=apply(adapter,broker,row,decision['arm'])
                self.done_lifetimes.add(row['entry_order_id'])
                self.actions.append({'executed':action,'observation':row,'decision':decision})
            return result
        if not eligible or (self.limit is not None and self.rows>=self.limit):return result
        future=[b for b in self.batches if b[0].timestamp>stamp]
        if not future:return result
        boundary,prefix=states.capture(adapter,broker,stamp+timedelta(minutes=5))
        stem=stamp.strftime('%Y%m%dT%H%M')
        prefix_binding=save_gzip(self.output/'prefixes'/(stem+'.json.gz'),prefix)
        cash=broker.cash;positions={s:p.qty for (_,s),p in broker.positions.items()};marks=dict(broker.last_prices)
        aa,bb=states.fork(adapter,broker,boundary)
        for batch in future:self.kernel(bb,batch,(aa,))
        hold=ledger(boundary,bb,cash,positions,marks)
        final_hash=states.receipt(aa,bb)['sha256']
        if self.expected_hold not in (None,final_hash):raise ValueError('Conflicting parent EOD continuations')
        self.expected_hold=final_hash
        hold_binding=save_gzip(self.output/'forks'/stem/'hold.json.gz',hold)
        del aa,bb
        for row in eligible:
            if self.limit is not None and self.rows>=self.limit:break
            contrasts={}
            for arm in ('half','exit'):
                aa,bb=states.fork(adapter,broker,boundary)
                action=apply(aa,bb,row,arm)
                for batch in future:self.kernel(bb,batch,(aa,))
                detail=ledger(boundary,bb,cash,positions,marks)
                actual=[f for f in detail['fills'] if f['reason']=='research_remaining_account_'+arm+'_v1']
                if not actual:raise ValueError('Declared executable action produced no fill')
                if detail['terminal_inventory'] or detail['terminal_pending']:raise ValueError('Incomplete action liquidation at EOD')
                value=detail['value'];base=hold['value']
                target={'value_bps':(value['terminal_equity']-base['terminal_equity'])/boundary.equity*10000,
                    'dd_change_bps':(value['max_drawdown_from_inherited_peak']-base['max_drawdown_from_inherited_peak'])*10000,
                    'material_entry_change':value['material_funded_entry_orders']-base['material_funded_entry_orders'],
                    'fee_change_krw':value['fees_krw']-base['fees_krw']}
                contrasts[arm]={'target':target,'action':action,'actual_action_qty':sum(f['qty'] for f in actual),
                    'first_fill':actual[0]['timestamp'],'last_fill':actual[-1]['timestamp'],
                    'ledger':save_gzip(self.output/'forks'/stem/(row['symbol']+'_'+arm+'.json.gz'),detail)}
                self.forks+=1
                del aa,bb
            record={'observation':row,'boundary':asdict(boundary),'prefix':prefix_binding,'source':self.source_binding,
                'source_suffix_offset':len(self.batches)-len(future),'initial_cash':cash,'initial_positions':positions,
                'initial_marks':marks,'hold':hold_binding,'hold_eod_state':final_hash,'contrasts':contrasts}
            with (self.output/'labels.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record,default=str,separators=(',',':'))+'\n')
            self.rows+=1
            write(self.output/'holding_progress.json',{'day':self.day,'available_at':row['available_at'],'rows':self.rows,
                'action_forks':self.forks,'elapsed_seconds':time.monotonic()-self.started,'complete':False})
        return result

    def finish(self):
        self.finish_day()
        write(self.output/'holding_actions.json',self.actions)
        write(self.output/'holding_complete.json',{'complete':True,'rows':self.rows,'action_forks':self.forks,
            'exact_hold_days':len(self.checked_days),'elapsed_seconds':time.monotonic()-self.started})
