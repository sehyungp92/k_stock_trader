"""Bounded native mechanics pilot. Isolated episodes are not portfolio returns."""
from contextlib import ExitStack
from datetime import datetime, timedelta
from dataclasses import asdict
import gzip
import json
from pathlib import Path
from unittest.mock import patch

from backtests.strategies.kalcb import executed_value as ev
from backtests.strategies.kalcb.pending_funding_labels_v1 import exact_snapshot
from backtests.strategies.kalcb.checkpoint_contrast_labels_v1 import SourceIndexes, independent_account
from backtests.strategies.kalcb.late_entry_v1 import continuous_connection
from backtests.strategies.kalcb.retention_research_io_v1 import read, write, bound, verify
from strategy_kalcb.core.staged_commitment_v1 import KEY


def episode(spec,snapshot,bars,cfg,conn,sessions,indexes,arm,bridge=False):
    from scripts.kalcb_executed_value_study import run_episode
    from backtests.engine import sim_broker
    from backtests.engine.integrity import audit_broker
    from backtests.strategies.kalcb import runner
    from strategy_kalcb.core import logic
    available=datetime.fromisoformat(spec['available_at'])
    prefix=[b for b in bars if b.timestamp+timedelta(minutes=5)<=available]
    snap=exact_snapshot(snapshot,spec['symbol'],spec['parent_qty'],spec['parent_qty'],cfg,prefix,spec['initial_equity'],available)
    row={'opportunity_id':spec['id'],'trade_date':spec['day'],'symbol':spec['symbol'],
         'feature_available_at':available.isoformat(),
         'candidate':next(c.to_json_dict() for c in snap.candidates if c.symbol==spec['symbol'])}
    brokers=[];adapters=[];old_broker=sim_broker.SimBroker;old_init=runner.KALCBReplayAdapter.__init__
    def new_broker(*a,**kw):
        b=old_broker(*a,**kw);brokers.append(b);return b
    def new_init(self,*a,**kw):
        old_init(self,*a,**kw);adapters.append(self)
        if arm=='staged':self.state.meta['staged_commitment_enabled']=True
    with ExitStack() as stack:
        stack.enter_context(patch.object(ev,'REFERENCE_EQUITY',spec['initial_equity']))
        stack.enter_context(patch.object(ev,'snapshot_for',lambda *a,**kw:snap))
        stack.enter_context(patch.object(sim_broker,'SimBroker',new_broker))
        stack.enter_context(patch.object(runner.KALCBReplayAdapter,'__init__',new_init))
        if bridge:
            for name,func in logic._staged_parent_api.items():stack.enter_context(patch.object(runner,name,func))
        value,detail=run_episode(row,bars,cfg,spec['scenario'],'staged-commitment-mechanics-v1',conn,sessions)
        if len(brokers)!=1 or len(adapters)!=1:raise ValueError('Unexpected native account count')
        broker=brokers[0];adapter=adapters[0];audit=audit_broker(broker);checked=0
        for batch in broker.expected_batches:
            for b in batch:
                day=datetime.fromisoformat(b['timestamp']).date().isoformat()
                expected=indexes[day].get((b['symbol'],b['timestamp'],b['timeframe']))
                if expected!=tuple(b[k] for k in ('open','high','low','close','volume','is_completed')):
                    raise ValueError('Non-source native bar')
                checked+=1
        account=independent_account(broker,indexes) if not value['unresolved'] else {'passed':False,'unresolved':True}
    submits=[r['order'] for r in broker.journal if r['event']=='submit' and r['order']['side']=='BUY']
    target=spec['parent_qty']//2 if arm=='staged' and spec['parent_qty']>=2 else spec['parent_qty']
    if not submits or submits[0]['qty']!=target:raise ValueError('Initial native quantity differs')
    commitments=adapter.state.meta.get(KEY,{})
    for c in commitments.values():
        if c['bought']>c['planned_qty']:raise ValueError('Exceeded original total quantity')
        if c['completion_id']:
            physical=[f for f in broker.fills if f.order_id==c['completion_id']]
            if any(f.timestamp<datetime.fromisoformat(c['completion_at']) for f in physical):
                raise ValueError('Completion filled before fresh observation')
    value.update(arm=arm,independent_account_pass=account['passed'],
        independent_execution_pass=audit['execution_validity_pass'],raw_bars_verified=checked,
        full_portfolio_audit_status=audit['audit_status'],full_portfolio_qualified=False,
        planned_parent_quantity=spec['parent_qty'],initial_quantity=target,
        completion_filled_quantity=sum(c['completion_filled'] for c in commitments.values()),
        unique_position_commitments=1,addition_is_not_new_profitable_trade=True,
        isolated_not_funded_portfolio_pnl=True)
    detail.update(arm=arm,spec=spec,staged_commitments=commitments,journal=broker.journal,
        equity_curve=broker.equity_curve,timestamps=broker.timestamps,terminal_cash=broker.cash,
        independent_account=account,independent_engine_audit=audit,
        costs={k:getattr(broker.costs,k) for k in ('commission_bps','tax_bps_on_sell','slippage_bps')})
    return value,detail


def run(root):
    from scripts.kalcb_executed_value_study import config_for,load_day
    from strategy_kalcb.models import KALCBDailySnapshot
    p=read(root/'protocol.json');specs=read(p['episodes']['path'])
    out=root/'labels';out.mkdir(exist_ok=False);conn=continuous_connection(p['databases'])
    sessions=[x[0] for x in conn.execute('SELECT DISTINCT day FROM bars ORDER BY day')]
    if len(sessions)!=286 or sum(d<'2026-04-01' for d in sessions)!=217:raise ValueError('History changed')
    indexes=SourceIndexes(p['databases']);outputs=[];summary=[]
    for i,spec in enumerate(specs,1):
        cfg=config_for(read(spec['profile']['plan']['path']),spec['scenario'])
        if cfg.risk_max_gross_equity_pct or cfg.exit_hard_stop_enabled:raise ValueError('Unsupported pilot configuration')
        snap=KALCBDailySnapshot.from_json_dict(next(x for x in read(spec['snapshot']['path']) if x['trade_date']==spec['day']))
        bars=load_day(conn,spec['day'],'staged-commitment-existing-raw',{spec['symbol']})
        arms={};details={}
        for arm in ('control','staged'):
            value,detail=episode(spec,snap,bars,cfg,conn,sessions,indexes,arm)
            arms[arm]=value;details[arm]=detail
        if spec['bridge']:
            bv,bd=episode(spec,snap,bars,cfg,conn,sessions,indexes,'control',bridge=True)
            keys=('fills','orders','journal','equity_curve','timestamps','terminal_cash')
            if any(json.dumps(bd[k],default=str,sort_keys=True)!=json.dumps(details['control'][k],default=str,sort_keys=True) for k in keys):
                raise ValueError('Disabled derivative changed parent native episode')
            if bv['net_value_krw']!=arms['control']['net_value_krw']:raise ValueError('Disabled value changed')
            arms['original_bridge']=bv;details['original_bridge']=bd
        for arm,value in arms.items():
            folder=out/spec['id']/arm;folder.mkdir(parents=True)
            write(folder/'label.json',value)
            with gzip.open(folder/'events.json.gz','wt',encoding='utf-8',compresslevel=1) as f:json.dump(details[arm],f,default=str,allow_nan=False)
            outputs.append({'label':bound(folder/'label.json'),'events':bound(folder/'events.json.gz')})
        summary.append({'id':spec['id'],'day':spec['day'],'symbol':spec['symbol'],'scenario':spec['scenario'],
            'control_net_krw':arms['control']['net_value_krw'],'staged_net_krw':arms['staged']['net_value_krw'],
            'completion_quantity':arms['staged']['completion_filled_quantity'],
            'all_checks_pass':all(v['label_valid'] and v['independent_execution_pass'] and v['independent_account_pass'] and not v['deadline_violation'] for v in arms.values())})
        write(root/'progress.json',{'completed_pairs':i,'expected_pairs':len(specs),'latest':summary[-1]})
        print(json.dumps(summary[-1]),flush=True)
    conn.close();verify(p['bindings']);write(out/'completion.json',{'complete':True,'study_complete':False,
       'episodes':len(outputs),'pairs':len(summary),'outputs':outputs,'summary':summary,
       'data_only':True,'mechanics_pilot_not_economic_efficacy_screen':True})
