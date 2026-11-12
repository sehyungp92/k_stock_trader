"""Fresh isolated full-cost outcomes for frozen pending-reservation quantities.

Use the existing native episode primitive. No portfolio return rescaling,
selector refitting, future-outcome selection, or additional funded capital.
"""
from collections import Counter,defaultdict
from dataclasses import replace
from datetime import datetime,timedelta
import gzip
import json
import os
from pathlib import Path
import sqlite3
from unittest.mock import patch

from backtests.strategies.kalcb import executed_value as ev
from backtests.strategies.kalcb.late_entry_v1 import continuous_connection
from backtests.strategies.kalcb.retention_research_io_v1 import read,write,bound,verify,digest


def raw_index(protocol,day):
    db=protocol['databases'][int(day>='2026-04-01')]
    with sqlite3.connect('file:'+Path(db).as_posix()+'?mode=ro',uri=True) as conn:
        return {(s,t,'5m'):(o,h,l,c,v,True) for s,t,o,h,l,c,v in conn.execute(
            'SELECT symbol,ts,o,h,l,c,v FROM bars WHERE day=?',(day,))}


def exact_snapshot(snapshot,symbol,target_qty,parent_qty,cfg,prefix,capital,available):
    from strategy_kalcb.core.preview import preview_entry
    from strategy_kalcb.core.core_models import KALCBPortfolioView
    account=KALCBPortfolioView(capital,equity=capital)
    def quantity(snap):
        result=preview_entry(prefix,snap,cfg,account,available_at=available)
        intents=result['intents']
        if len(intents)!=1 or intents[0]['symbol']!=symbol:raise ValueError('Isolated original action is not unique')
        return intents[0]['quantity']
    if quantity(snapshot)!=parent_qty:raise ValueError('Isolated parent quantity mismatch')
    if target_qty==parent_qty:return snapshot
    if not 0<target_qty<parent_qty or cfg.entry_quantity_multiplier!=1:
        raise ValueError('Unsupported quantity request')
    def change(scale):
        return replace(snapshot,candidates=tuple(replace(c,metadata={**c.metadata,'action_quantity_scale':scale})
            if c.symbol==symbol else c for c in snapshot.candidates))
    full=quantity(change(1.0));scale=(target_qty+.5)/full
    parent_scale=next(c.metadata['action_quantity_scale'] for c in snapshot.candidates if c.symbol==symbol)
    if not 0<scale<parent_scale:raise ValueError('Quantity reconstruction is not reduction-only')
    changed=change(scale)
    if quantity(changed)!=target_qty:raise ValueError('Integer quantity reconstruction mismatch')
    return changed


def episode(spec,snapshot,bars,cfg,conn,sessions,protocol,indexer=raw_index):
    from scripts.kalcb_executed_value_study import run_episode
    from backtests.engine import sim_broker
    from backtests.engine.integrity import audit_broker
    available=datetime.fromisoformat(spec['available_at']);symbol=spec['symbol']
    prefix=[b for b in bars if b.timestamp+timedelta(minutes=5)<=available]
    modified=exact_snapshot(snapshot,symbol,spec['target_qty'],spec['parent_qty'],cfg,prefix,
        spec['initial_equity'],available)
    reference=spec['reference_price']*spec['target_qty'];capital=spec['initial_equity']
    scale=next(c.metadata['action_quantity_scale'] for c in modified.candidates if c.symbol==symbol)
    row={'opportunity_id':spec['episode_id'],'trade_date':spec['day'],'symbol':symbol,
        'feature_available_at':available.isoformat(),'candidate':next(c.to_json_dict() for c in modified.candidates if c.symbol==symbol)}
    brokers=[];original=sim_broker.SimBroker
    def observe(*a,**kw):
        broker=original(*a,**kw);brokers.append(broker);return broker
    # The isolated account's INITIAL capital matches the observed action basis.
    # This never resets or replenishes an ongoing portfolio account.
    with patch.object(ev,'REFERENCE_EQUITY',capital),patch.object(ev,'snapshot_for',lambda *a,**kw:modified),patch.object(sim_broker,'SimBroker',observe):
        value,detail=run_episode(row,bars,cfg,spec['scenario'],'pending-funding-quantity-label-v1',conn,sessions)
        if len(brokers)!=1:raise ValueError('Unexpected native broker count')
        broker=brokers[0];audit=audit_broker(broker)
    indexes={};checked=0
    for batch in broker.expected_batches:
        for b in batch:
            day=datetime.fromisoformat(b['timestamp']).date().isoformat()
            if day not in indexes:indexes[day]=indexer(protocol,day)
            expected=indexes[day].get((b['symbol'],b['timestamp'],b['timeframe']))
            if expected!=tuple(b[k] for k in ('open','high','low','close','volume','is_completed')):
                raise ValueError('Native label consumed non-source bar')
            checked+=1
    submits=[r for r in broker.journal if r['event']=='submit' and r['order']['side']=='BUY']
    if not submits or submits[0]['order']['qty']!=spec['target_qty']:
        raise ValueError('Native submitted quantity differs from frozen request')
    if datetime.fromisoformat(submits[0]['order']['metadata']['decision_available_at'])!=available:
        raise ValueError('Native action clock mismatch')
    if not audit['execution_validity_pass']:
        value.update(label_valid=False,status='execution_invalid',label_available_at=None,
            net_value_krw=None,net_value_equity_bps=None)
    value.update(episode_id=spec['episode_id'],aliases=spec['aliases'],action_kind=spec['action_kind'],
        initial_equity_krw=capital,planned_quantity=spec['target_qty'],native_action_quantity_scale=scale,
        planned_notional_krw=reference,net_value_planned_notional_bps=(value['net_value_krw']/reference*10000 if value['label_valid'] else None),
        independent_execution_pass=audit['execution_validity_pass'],raw_execution_bars_verified=checked,
        full_portfolio_audit_status=audit['audit_status'],full_native_portfolio_qualification=False,
        isolated_not_funded_candidate_pnl=True)
    detail.update(initial_equity_krw=capital,scenario=spec['scenario'],costs=vars(broker.costs) if hasattr(broker.costs,'__dict__') else
        {k:getattr(broker.costs,k) for k in ('commission_bps','tax_bps_on_sell','slippage_bps')},
        independent_engine_audit=audit,raw_execution_bars_verified=checked,journal=broker.journal,
        equity_curve=broker.equity_curve,timestamps=broker.timestamps,terminal_cash=broker.cash,
        spec=spec,scope='Isolated matching-quantity native label; no portfolio benefit or exchangeability claim.')
    return value,detail


def reuse_smoke(root,spec,target):
    source=root/'smoke'/spec['episode_id'];receipt=read(root/'smoke/completion.json')
    bindings=[b for row in receipt['outputs'] for b in row.values() if Path(b['path']).parent==source]
    if len(bindings)!=2:raise ValueError('Unauthenticated smoke reuse')
    verify(bindings)
    with gzip.open(source/'events.json.gz','rt',encoding='utf-8') as stream:details=json.load(stream)
    if details['spec']!=spec:raise ValueError('Smoke action contract changed')
    value=read(source/'label.json')
    for name in ('label.json','events.json.gz'):os.link(source/name,target/name)
    write(target/'authenticated_smoke_reuse.json',{'sources':bindings,'same_episode':spec['episode_id']})
    return value,details


def run(root,smoke=False):
    from scripts.kalcb_executed_value_study import config_for,load_day,emit
    from strategy_kalcb.models import KALCBDailySnapshot
    p=read(root/'protocol.json');verify(p['bindings']);specs=read(p['episodes']['path'])
    if smoke:
        specs=[next(r for r in specs if r['scenario']==s and r['action_kind']=='capped') for s in ('baseline','combined')]
    out=root/('smoke' if smoke else 'labels');out.mkdir(parents=True,exist_ok=False)
    conn=continuous_connection(p['databases']);sessions=[r[0] for r in conn.execute('SELECT DISTINCT day FROM bars ORDER BY day')]
    if len(sessions)!=286 or sum(d<'2026-04-01' for d in sessions)!=217:raise ValueError('Changed continuous market history')
    counts=Counter();outputs=[]
    for i,spec in enumerate(specs,1):
        profile=spec['profile'];cfg=config_for(read(profile['plan']['path']),spec['scenario'])
        if cfg.risk_max_gross_equity_pct:raise ValueError('Quantity labels must not invent pending accounts')
        raw=next(r for r in read(spec['snapshot']['path']) if r['trade_date']==spec['day'])
        snap=KALCBDailySnapshot.from_json_dict(raw)
        bars=load_day(conn,spec['day'],'authenticated-pending-funding-labels',{spec['symbol']})
        target=out/spec['episode_id'];target.mkdir()
        source=root/'smoke'/spec['episode_id']
        reused=not smoke and source.exists()
        if reused:
            value,details=reuse_smoke(root,spec,target)
        else:
            value,details=episode(spec,snap,bars,cfg,conn,sessions,p)
            write(target/'label.json',value)
            with gzip.open(target/'events.json.gz','wt',encoding='utf-8',compresslevel=1) as stream:
                json.dump(details,stream,default=str,allow_nan=False)
        outputs.append({'label':bound(target/'label.json'),'events':bound(target/'events.json.gz')})
        counts[value['status']]+=1;counts['physical_failures']+=len(value['physical_fill_failures'])
        counts['deadline_violations']+=bool(value['deadline_violation']);counts['independent_execution_failures']+=not value['independent_execution_pass']
        write(root/'progress.json',{'completed':i,'expected':len(specs),'counts':dict(counts),'latest':spec['episode_id']})
        print(json.dumps({'completed':i,'expected':len(specs),'status':value['status'],'episode':spec['episode_id'],'authenticated_smoke_reuse':reused}),flush=True)
    conn.close();verify(p['bindings'])
    write(out/'completion.json',{'complete':True,'study_complete':False,'episodes':len(outputs),'counts':dict(counts),
        'outputs':outputs,'portfolio_results':0,'aliases_dependent':True,'no_fits_or_policies':True,
        'scope':'Fresh original/capped quantity full-cost native isolated labels. Failure/no-fill/loss retained; whole-portfolio source qualification remains unverified.'})
