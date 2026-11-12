"""Fresh minute90 action preparation and independently audited isolated labels.

No model, selection policy, optimizer, or portfolio replay is owned here. The
existing native episode primitive and shared core/broker own all transitions.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import date, datetime, timedelta
from itertools import groupby
from pathlib import Path
from unittest.mock import patch
import gzip
import json
import math
import sqlite3

from backtests.strategies.kalcb import executed_value as ev, continuation_value as cv
from backtests.strategies.kalcb.retention_research_io_v1 import read,write,bound,verify,digest
from backtests.strategies.kalcb.late_entry_v1 import continuous_connection
from backtests.strategies.kalcb.reclaim_source_audit_v1 import verify_rows


def geometry(bars, cutoff):
    from kis_core.trading_calendar import get_trading_calendar
    opening=datetime.combine(cutoff.date(),get_trading_calendar().session_hours(cutoff.date())[0],cutoff.tzinfo)
    if cutoff!=opening+timedelta(minutes=90):raise ValueError('Undeclared reclaim clock')
    prefix=[b for b in bars if opening<=b.timestamp and b.timestamp+timedelta(minutes=5)<=cutoff]
    if len(prefix)!=18 or [b.timestamp for b in prefix]!=[opening+timedelta(minutes=5*i) for i in range(18)]:
        raise ValueError('Incomplete reclaim prefix')
    if any(b.volume<=0 or b.low<=0 or b.low>min(b.open,b.close) or b.high<max(b.open,b.close) for b in prefix):
        raise ValueError('Invalid reclaim prefix')
    cash=volume=0.;vwaps=[]
    for b in prefix:
        cash+=(b.high+b.low+b.close)/3*b.volume;volume+=b.volume;vwaps.append(cash/volume)
    flags=(any(prefix[i].close>vwaps[i] for i in range(6,12)),
           any(prefix[i].close<vwaps[i] for i in range(12,16)),
           all(prefix[i].close>vwaps[i] and prefix[i].close>=prefix[i].open for i in (16,17)))
    return prefix,all(flags),{'reclaim_confirmation_margin':min(prefix[i].close/vwaps[i]-1 for i in (16,17)),
        'reclaim_pullback_depth':min(prefix[i].close/vwaps[i]-1 for i in range(12,16)),
        'reclaim_impulse_margin':max(prefix[i].close/vwaps[i]-1 for i in range(6,12))}


def original_population(protocol):
    population=defaultdict(list);seen=set()
    for index,folder in enumerate(protocol['opening_prepared']):
        for row in ev.read_jsonl(Path(folder)/'opportunities.jsonl'):
            day=row['trade_date']
            if (index==0 and day>'2026-03-31') or (index==1 and day<'2026-04-01'):continue
            key=day,row['symbol']
            if key in seen:raise ValueError('Duplicate source population')
            seen.add(key);population[day].append(row)
    return population


def independent_day_index(protocol,day):
    # Direct read of the original immutable table, independent of load_day/union.
    db=protocol['databases'][int(day>='2026-04-01')]
    with sqlite3.connect('file:'+Path(db).as_posix()+'?mode=ro',uri=True) as connection:
        return {(s,t,'5m'):(o,h,l,c,v,True) for s,t,o,h,l,c,v in connection.execute(
            'SELECT symbol,ts,o,h,l,c,v FROM bars WHERE day=? ORDER BY symbol,ts',(day,))}


def prepare(root):
    from scripts.kalcb_executed_value_study import config_for,load_day,emit
    from scripts.kalcb_action_value_study import action_features
    from scripts.kalcb_checkpoint_prepare import checkpoint_features
    from scripts.kalcb_action_quantity_labels import scaled_row
    from strategy_kalcb.core.preview import preview_entry
    from strategy_kalcb.core.core_models import KALCBPortfolioView
    p=read(root/'protocol.json');verify(p['data_bindings'])
    target=root/'prepared';target.mkdir(exist_ok=False)
    conn=continuous_connection(p['databases']);source=ev.signature(p['data_bindings'])
    sessions=[r[0] for r in conn.execute('SELECT DISTINCT day FROM bars ORDER BY day')]
    if len(sessions)!=286 or sum(d<'2026-04-01' for d in sessions)!=217:raise ValueError('Continuous calendar differs')
    cfg=config_for(read(p['producer_runtime']['plan']['path']))
    if not cfg.entry_action_quantity_scale_enabled:raise ValueError('Parent quarter quantity support missing')
    account=KALCBPortfolioView(cash=ev.REFERENCE_EQUITY,equity=ev.REFERENCE_EQUITY)
    census={ (r['trade_date'],r['symbol']):r for r in ev.read_jsonl(p['census_opportunities']['path']) }
    population=original_population(p);counts=Counter();proofs=[];seen=set()
    with (target/'opportunities.jsonl').open('x',encoding='utf-8') as out,gzip.open(target/'previews.jsonl.gz','xt',encoding='utf-8',compresslevel=1) as previews:
        for number,day in enumerate(sorted(population),1):
            rows=population[day];cutoff=datetime.fromisoformat(rows[0]['feature_available_at'])+timedelta(minutes=60)
            bars=defaultdict(list)
            for b in load_day(conn,day,source,set(p['watch_symbols'])):
                if b.timestamp+timedelta(minutes=5)<=cutoff:bars[b.symbol].append(b)
            prior=defaultdict(list);index=sessions.index(day)
            if index:
                for b in load_day(conn,sessions[index-1],source,set(p['watch_symbols'])):
                    prior[b.symbol].append(dict(timestamp=b.timestamp.isoformat(),**{k:getattr(b,k) for k in ('open','high','low','close','volume')}))
            valid={};sectors=defaultdict(list)
            for original in rows:
                key=day,original['symbol'];seen.add(key);counts['considered']+=1
                try:prefix,confirmed,extra=geometry(bars[original['symbol']],cutoff)
                except ValueError:
                    if census[key]['status']=='complete':raise ValueError('Census complete-prefix mismatch')
                    counts['excluded_prefix']+=1;continue
                if census[key]['status']!='complete' or confirmed!=census[key]['qualified']:raise ValueError('Registered census geometry changed')
                valid[original['symbol']]=prefix,confirmed,extra
                sectors[original['sector']].append(prefix[-1].close/prefix[0].open-1)
            produced=[]
            for original in rows:
                if original['symbol'] not in valid:continue
                row=deepcopy(original);row.pop('quality_target',None)
                prefix,confirmed,extra=valid[row['symbol']]
                row['opportunity_id']+='/minute90';row['feature_available_at']=cutoff.isoformat()
                row['candidate']['metadata'].update(entry_checkpoint_at=cutoff.isoformat(),feature_available_at=cutoff.isoformat())
                preview=preview_entry(prefix,ev.snapshot_for(day,[(row,0.)],source=source),cfg,account,available_at=cutoff)
                row['original_guard_at_rank1']=row['guard_at_rank1'];row['guard_at_rank1']=confirmed
                row['action']={'ready':preview['ready'],'quantity':sum(a['quantity'] for a in preview['intents']),
                    'routes':[a['metadata'].get('entry_route') for a in preview['intents']],
                    'reasons':[d['reason'] for d in preview['decisions']]}
                row['opening_bars']=[dict(symbol=b.symbol,timestamp=b.timestamp.isoformat(),**{k:getattr(b,k) for k in ('open','high','low','close','volume')}) for b in prefix]
                sector=sectors[row['sector']]
                row['features'].update(cv.prior_path(prior.get(row['symbol'],[]),day),**action_features(preview,prefix),
                    **checkpoint_features(prefix,cutoff),**extra,checkpoint_sector_return=sum(sector)/len(sector),
                    checkpoint_sector_breadth=sum(v>0 for v in sector)/len(sector))
                quarter=scaled_row(row,.25)
                small=preview_entry(prefix,ev.snapshot_for(day,[(quarter,0.)],source=source),cfg,account,available_at=cutoff)
                if sum(a['quantity'] for a in small['intents'])!=quarter['action']['quantity']:raise ValueError('Quarter integer quantity mismatch')
                row['quarter_preview']={'ready':small['ready'],'quantity':quarter['action']['quantity'],
                    'reasons':[d['reason'] for d in small['decisions']]}
                row['parent_snapshot_eligibility']=census[(day,row['symbol'])]['parent_snapshot_eligibility']
                produced.append(row);emit(out,row);emit(previews,{'opportunity_id':row['opportunity_id'],'full':preview,'quarter':small})
                counts['complete']+=1;counts['geometry']+=confirmed;counts['geometry_quarter_ready']+=bool(confirmed and small['ready'])
            proof=verify_rows(rows,produced,independent_day_index(p,day),{date.fromisoformat(day)})
            proofs.append({'day':day,**proof})
            print(json.dumps({'stage':'prepare90','session':number,'day':day,**counts}),flush=True)
    conn.close()
    if seen!=set(census):raise ValueError('Census population mismatch')
    write(target/'source_population_audit.json',{'passed':True,'sessions':proofs,'census':p['census_opportunities'],
        'scope':'Independent raw minute90 values/population/sector/geometry. Existing minute45 native source obligations remain mandatory for future portfolio cases.'})
    write(target/'prepared.json',{'complete':True,'sessions':sessions,'counts':dict(counts),'protocol':bound(root/'protocol.json'),
        'opportunities_sha256':digest(target/'opportunities.jsonl'),'previews':bound(target/'previews.jsonl.gz'),
        'source_audit':bound(target/'source_population_audit.json'),'first_oos_prior_sessions':217,
        'scope':'Rank1 empty-account native preview is an isolated feasibility check, not actual parent-account availability.'})


def audited_episode(row,bars,cfg,scenario,source,conn,sessions,protocol):
    from scripts.kalcb_executed_value_study import run_episode
    from backtests.engine import sim_broker
    from backtests.engine.integrity import audit_broker
    captured=[];original=sim_broker.SimBroker
    def observer(*args,**kwargs):
        broker=original(*args,**kwargs);captured.append(broker);return broker
    # Instrument the existing primitive without replacing its core or replay loop.
    with patch.object(sim_broker,'SimBroker',observer):
        value,detail=run_episode(row,bars,cfg,scenario,source,conn,sessions)
    if len(captured)!=1:raise ValueError('Unexpected episode broker count')
    broker=captured[0];indexes={};checked=0
    for batch in broker.expected_batches:
        for b in batch:
            day=datetime.fromisoformat(b['timestamp']).date().isoformat()
            if day not in indexes:indexes[day]=independent_day_index(protocol,day)
            expected=indexes[day].get((b['symbol'],b['timestamp'],b['timeframe']))
            if expected!=tuple(b[k] for k in ('open','high','low','close','volume','is_completed')):
                raise ValueError('Episode consumed a non-source bar')
            checked+=1
    audit=audit_broker(broker)
    # Preserve the engine's truthful unverified whole-portfolio qualification;
    # this independent episode receipt is never relabelled a full native audit.
    if not audit['execution_validity_pass']:
        value.update(label_valid=False,status='execution_invalid',label_available_at=None,
                     net_value_krw=None,net_value_equity_bps=None)
    reference=row['features']['action_notional_equity']*ev.REFERENCE_EQUITY
    qty=row['action']['quantity']
    if reference<0 or (qty==0)!=(reference==0):raise ValueError('Invalid planned-notional denominator')
    value.update(base_opportunity_id=row['base_opportunity_id'],quantity_scale=.25,planned_quantity=qty,
        planned_notional_krw=reference,net_value_planned_notional_bps=(value['net_value_krw']/reference*10000
            if value['label_valid'] and reference>0 else None),
        independent_episode_execution_pass=audit['execution_validity_pass'],raw_execution_bars_verified=checked,
        full_native_portfolio_qualification=False)
    detail.update(independent_engine_audit=audit,raw_execution_bars_verified=checked,
        journal=broker.journal,equity_curve=broker.equity_curve,timestamps=broker.timestamps,
        terminal_cash=broker.cash,scope='Fresh isolated action target, not funded multi-issuer portfolio PnL or parent-account readiness')
    return value,detail


def labels(root,scenario,smoke=False):
    from scripts.kalcb_executed_value_study import config_for,load_day,emit
    from scripts.kalcb_action_quantity_labels import scaled_row
    p=read(root/'protocol.json');verify(p['data_bindings'])
    prepared=root/'prepared';meta=read(prepared/'prepared.json')
    if digest(prepared/'opportunities.jsonl')!=meta['opportunities_sha256']:raise ValueError('Changed prepared actions')
    verify([meta['source_audit']]);target=root/('smoke' if smoke else 'labels')/scenario;target.mkdir(parents=True,exist_ok=False)
    plan=read(p['producer_runtime']['plan']['path']);cfg=config_for(plan,scenario)
    if not cfg.entry_action_quantity_scale_enabled:raise ValueError('Missing quarter support')
    conn=continuous_connection(p['databases']);source=ev.signature(p['data_bindings']);counts=Counter();ids=set()
    for day,group in groupby(ev.read_jsonl(prepared/'opportunities.jsonl'),lambda r:r['trade_date']):
        # Replay every geometry action, including guard-rejected/no-order attempts.
        # No outcome, parent membership or forecast is used to choose labels.
        rows=[r for r in group if r['guard_at_rank1']]
        bars=defaultdict(list)
        for b in load_day(conn,day,source,{r['symbol'] for r in rows}):bars[b.symbol].append(b)
        with (target/(day+'.jsonl')).open('x',encoding='utf-8') as out,gzip.open(target/(day+'.events.jsonl.gz'),'xt',encoding='utf-8',compresslevel=1) as events:
            for r in rows:
                action=scaled_row(r,.25)
                value,detail=audited_episode(action,bars[r['symbol']],cfg,scenario,source,conn,meta['sessions'],p)
                if action['opportunity_id'] in ids:raise ValueError('Duplicate isolated action')
                ids.add(action['opportunity_id']);emit(out,value);emit(events,detail);counts[value['status']]+=1
                if not value['independent_episode_execution_pass']:counts['independent_execution_failures']+=1
                if smoke and value['status']=='filled' and value['independent_episode_execution_pass']:break
        print(json.dumps({'stage':'smoke90' if smoke else 'labels90','scenario':scenario,'day':day,'actions':len(ids),**counts}),flush=True)
        if smoke and counts['filled']:break
    conn.close()
    if smoke and (not counts['filled'] or counts['independent_execution_failures']):raise ValueError('No independently reconciled filled smoke')
    if not smoke and len(ids)!=meta['counts']['geometry']:raise ValueError('Incomplete fresh action-label population')
    write(target/'completion.json',{'complete':True,'smoke':smoke,'scenario':scenario,'actions':len(ids),'counts':dict(counts),
        'prepared':bound(prepared/'prepared.json'),'config':bound(plan['config']),
        'files':{f.name:digest(f) for f in target.iterdir() if f.is_file()},
        'scope':'Independent isolated actions, same shared core/broker; full portfolio qualification and fitting remain pending.'})
