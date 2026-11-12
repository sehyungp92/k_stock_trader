"""Causal first-hour action research. No optimizer or execution loop of its own.

Existing shared-core preview and isolated native episode primitives define the
action and target. Earlier candidate snapshots remain immutable incumbents.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import replace
from datetime import date, datetime, timedelta
from itertools import groupby, chain
import gzip
import json
from pathlib import Path
import sqlite3

from backtests.strategies.kalcb import executed_value as ev, continuation_value as cv, action_value as av, paired_value as pv
from backtests.strategies.kalcb.retention_research_io_v1 import read, write, bound, verify, digest, link


def continuous_connection(databases):
    """Read-only union; March31 is retained from IS exactly once, never reset."""
    conn=sqlite3.connect(':memory:',uri=True)
    for alias,path in zip(('history','forward'),databases):
        conn.execute('ATTACH DATABASE ? AS '+alias,('file:'+Path(path).as_posix()+'?mode=ro',))
        conn.execute('PRAGMA '+alias+'.cache_size=-8192')
    conn.execute("CREATE TEMP VIEW bars AS SELECT day,symbol,ts,o,h,l,c,v FROM history.bars WHERE day<='2026-03-31' UNION ALL SELECT day,symbol,ts,o,h,l,c,v FROM forward.bars WHERE day>='2026-04-01'")
    conn.execute('PRAGMA query_only=ON')
    return conn


def completed_confirmation(bars,cutoff):
    """Two completed closes above their contemporaneous VWAP; no later data."""
    from kis_core.trading_calendar import get_trading_calendar
    opening=datetime.combine(cutoff.date(),get_trading_calendar().session_hours(cutoff.date())[0],cutoff.tzinfo)
    if cutoff!=opening+timedelta(minutes=60):raise ValueError('Undeclared first-hour clock')
    prefix=[b for b in bars if opening<=b.timestamp and b.timestamp+timedelta(minutes=5)<=cutoff]
    if len(prefix)!=12 or [b.timestamp for b in prefix]!=[opening+timedelta(minutes=5*i) for i in range(12)]:
        raise ValueError('Incomplete first-hour prefix')
    vwaps=[];cash=volume=0.
    for b in prefix:
        cash+=(b.high+b.low+b.close)/3*b.volume;volume+=b.volume;vwaps.append(cash/max(volume,1))
    confirmed=all(prefix[i].close>=vwaps[i] and prefix[i].close>=prefix[i].open for i in (10,11))
    pullback=any(prefix[i].low<=vwaps[i] for i in range(6,10))
    return prefix,confirmed,{'later_vwap_pullback':float(pullback),'later_confirmation_margin':min(prefix[i].close/max(vwaps[i],1e-9)-1 for i in (10,11))}


def flat_blocked(state,portfolio):
    return bool(portfolio.open_positions or any(portfolio.positions.values()) or
                any(s.pending_entry_order_id or s.position is not None for s in state.symbols.values()))


def cumulative_timed_policy(directory,manifest,snapshots):
    """An incumbent 45-minute action may persist unchanged at minute60."""
    from strategy_kalcb.models import KALCBDailySnapshot
    filename=manifest.get('timed_snapshot_file')
    if not filename:return {}
    admission=manifest.get('timed_admission_file')
    if filename not in manifest['files'] or admission not in manifest['files']:raise ValueError('Unbound timed policy')
    decisions=ev.read_jsonl(Path(directory)/admission)
    expected={(r['trade_date'],r['symbol'],r['available_at']):r for r in decisions}
    if len(expected)!=len(decisions):raise ValueError('Duplicate timed admission')
    result={};used=set()
    for raw in ev.read_jsonl(Path(directory)/filename):
        snap=KALCBDailySnapshot.from_json_dict(raw)
        if snap.artifact_hash!=raw['artifact_hash'] or snap.trade_date not in snapshots or snap.generated_at is None:
            raise ValueError('Invalid timed snapshot')
        previous=result.setdefault(snap.trade_date,[])
        base=(previous[-1] if previous else snapshots[snap.trade_date]).by_symbol()
        if previous and previous[-1].generated_at>=snap.generated_at:raise ValueError('Unordered update')
        symbols=[c.symbol for c in snap.candidates]
        if len(symbols)!=len(set(symbols)) or not set(base).issubset(symbols) or len(symbols)>manifest['contract']['timed_budget']:
            raise ValueError('Incumbent removal or invalid budget')
        for c in snap.candidates:
            if c.symbol in base:
                if c.to_json_dict()!=base[c.symbol].to_json_dict():raise ValueError('Changed incumbent')
                continue
            key=(str(snap.trade_date),c.symbol,str(c.metadata.get('entry_checkpoint_at')))
            if c.symbol not in manifest['observation_symbols'] or key not in expected or key in used:raise ValueError('Unregistered timed action')
            if datetime.fromisoformat(key[2])!=snap.generated_at or c.selection_score!=expected[key]['score']:
                raise ValueError('Timed score/clock mismatch')
            if datetime.fromisoformat(c.metadata['feature_available_at'])>snap.generated_at:raise ValueError('Future feature')
            used.add(key)
        previous.append(snap)
    if used!=set(expected):raise ValueError('Unused timed action')
    return result


def prepare(root,limit=None):
    from scripts.kalcb_executed_value_study import config_for,load_day,emit
    from scripts.kalcb_action_value_study import action_features
    from scripts.kalcb_checkpoint_prepare import checkpoint_features
    from strategy_kalcb.core.preview import preview_entry
    from strategy_kalcb.core.core_models import KALCBPortfolioView
    protocol=read(root/'protocol.json');verify(protocol['data_bindings'])
    target=root/('prepared' if limit is None else 'preflight/prepared');target.mkdir(parents=True,exist_ok=False)
    conn=continuous_connection(protocol['databases']);source=ev.signature(protocol['data_bindings'])
    sessions=[r[0] for r in conn.execute('SELECT DISTINCT day FROM bars ORDER BY day')]
    if len([d for d in sessions if d<'2026-04-01'])!=217:raise ValueError('Continuous history differs')
    cfg=config_for(read(protocol['profiles']['100']['is']['plan']['path']))
    portfolio=KALCBPortfolioView(cash=ev.REFERENCE_EQUITY,equity=ev.REFERENCE_EQUITY)
    counts=Counter();previous=None;last_day=None;count=0
    with (target/'opportunities.jsonl').open('x',encoding='utf-8') as out,gzip.open(target/'previews.jsonl.gz','xt',encoding='utf-8',compresslevel=1) as events:
        for prepared in protocol['opening_prepared']:
            with (Path(prepared)/'opportunities.jsonl').open(encoding='utf-8') as stream:
                for day,group in groupby(map(json.loads,stream),lambda r:r['trade_date']):
                    if last_day is not None and day<=last_day:continue
                    if limit is not None and count>=limit:break
                    rows=list(group);last_day=day;count+=1
                    cutoff=datetime.fromisoformat(rows[0]['feature_available_at'])+timedelta(minutes=30)
                    bars=defaultdict(list)
                    for b in load_day(conn,day,source,set(protocol['watch_symbols'])):
                        if b.timestamp+timedelta(minutes=5)<=cutoff:bars[b.symbol].append(b)
                    prior=defaultdict(list)
                    if previous:
                        for b in load_day(conn,previous,source,set(protocol['watch_symbols'])):
                            prior[b.symbol].append(dict(timestamp=b.timestamp.isoformat(),**{k:getattr(b,k) for k in ('open','high','low','close','volume')}))
                    previous=day
                    valid={};sectors=defaultdict(list)
                    for row in rows:
                        counts['considered']+=1
                        try:p,confirmation,extra=completed_confirmation(bars[row['symbol']],cutoff)
                        except ValueError:counts['incomplete_prefix']+=1;continue
                        valid[row['symbol']]=(p,confirmation,extra)
                        sectors[row['sector']].append(p[-1].close/p[0].open-1)
                    for row in rows:
                        if row['symbol'] not in valid:continue
                        p,confirmed,extra=valid[row['symbol']]
                        row.pop('quality_target',None)
                        row['opportunity_id']+='/minute60'
                        row['feature_available_at']=cutoff.isoformat()
                        row['candidate']['metadata'].update(entry_checkpoint_at=cutoff.isoformat(),feature_available_at=cutoff.isoformat())
                        snap=ev.snapshot_for(day,[(row,0.)],source=source)
                        preview=preview_entry(p,snap,cfg,portfolio,available_at=cutoff)
                        row['original_guard_at_rank1']=row['guard_at_rank1'];row['guard_at_rank1']=confirmed
                        row['action']={'ready':preview['ready'],'quantity':sum(a['quantity'] for a in preview['intents']),
                            'routes':[a['metadata'].get('entry_route') for a in preview['intents']],
                            'reasons':[d['reason'] for d in preview['decisions']]}
                        row['opening_bars']=[dict(symbol=b.symbol,timestamp=b.timestamp.isoformat(),**{k:getattr(b,k) for k in ('open','high','low','close','volume')}) for b in p]
                        sector=sectors[row['sector']]
                        row['features'].update(cv.prior_path(prior.get(row['symbol'],[]),day),**action_features(preview,p),
                            **checkpoint_features(p,cutoff),**extra,checkpoint_sector_return=sum(sector)/len(sector),
                            checkpoint_sector_breadth=sum(v>0 for v in sector)/len(sector))
                        emit(out,row);emit(events,{'opportunity_id':row['opportunity_id'],**preview})
                        counts['native_ready']+=bool(preview['ready']);counts['confirmed_ready']+=bool(preview['ready'] and confirmed)
                    print(json.dumps({'stage':'prepare60','session':count,'day':day,**counts}),flush=True)
    conn.close()
    write(target/'prepared.json',{'sessions':sessions,'opportunities_sha256':digest(target/'opportunities.jsonl'),
        'previews':bound(target/'previews.jsonl.gz'),'counts':counts,'prepared_sessions':count,'complete':limit is None,
        'protocol':bound(root/'protocol.json'),'first_oos_prior_sessions':217,'no_outcome_membership':True})
    return target


def labels(root,scenario,smoke=False):
    from scripts.kalcb_executed_value_study import config_for,load_day,run_episode,emit
    from scripts.kalcb_action_quantity_labels import scaled_row
    protocol=read(root/'protocol.json');verify(protocol['data_bindings'])
    prepared=root/('preflight/prepared' if smoke else 'prepared');meta=read(prepared/'prepared.json')
    if digest(prepared/'opportunities.jsonl')!=meta['opportunities_sha256']:raise ValueError('Changed population')
    target=root/('preflight/labels' if smoke else 'labels')/scenario;target.mkdir(parents=True,exist_ok=False)
    plan=read(protocol['profiles']['100']['is']['plan']['path']);cfg=config_for(plan,scenario)
    conn=continuous_connection(protocol['databases']);source=ev.signature(protocol['data_bindings']);counts=Counter();number=0
    with (prepared/'opportunities.jsonl').open(encoding='utf-8') as stream:
        for day,group in groupby(map(json.loads,stream),lambda r:r['trade_date']):
            rows=[r for r in group if r['action']['ready'] and r['guard_at_rank1']]
            if smoke:rows=rows[:1]
            bars=defaultdict(list)
            for b in load_day(conn,day,source,{r['symbol'] for r in rows}):bars[b.symbol].append(b)
            with (target/(day+'.jsonl')).open('x',encoding='utf-8') as values,gzip.open(target/(day+'.events.jsonl.gz'),'xt',encoding='utf-8',compresslevel=1) as events:
                for row in rows:
                    for q in (.25,.5):
                        action=scaled_row(row,q)
                        value,detail=run_episode(action,bars[row['symbol']],cfg,scenario,source,conn,meta['sessions'])
                        value.update(base_opportunity_id=row['opportunity_id'],quantity_scale=q)
                        emit(values,value);emit(events,detail);counts[value['status']]+=1
            number+=1;print(json.dumps({'stage':'fresh_labels','scenario':scenario,'day':day,'session':number,**counts}),flush=True)
            if smoke and rows:break
    conn.close()
    write(target/'completion.json',{'complete':True,'smoke':smoke,'scenario':scenario,'counts':counts,
        'source_generation_hash':__import__('backtests.auto.shared.cache_keys',fromlist=['source_generation_hash']).source_generation_hash(),
        'prepared':bound(prepared/'prepared.json'),'config':bound(plan['config']),
        'files':{p.name:digest(p) for p in target.iterdir() if p.is_file()}})


def fit(root):
    """One regularized expanding learner, original global 20-session clock."""
    import joblib
    from threadpoolctl import threadpool_limits
    from scripts.kalcb_action_quantity_labels import scaled_row
    from scripts.kalcb_management_refit import failure_outcomes
    meta=read(root/'prepared/prepared.json')
    if not meta['complete'] or digest(root/'prepared/opportunities.jsonl')!=meta['opportunities_sha256']:raise ValueError('Unfinished preparation')
    rows=[scaled_row(r,q) for r in ev.read_jsonl(root/'prepared/opportunities.jsonl') if r['action']['ready'] and r['guard_at_rank1'] for q in (.25,.5)]
    grid={};proofs=[];target=root/'models';target.mkdir(exist_ok=False)
    for scenario in ('baseline','combined'):
        label=root/'labels'/scenario;done=read(label/'completion.json')
        if not done['complete'] or done['smoke']:raise ValueError('Incomplete fresh labels')
        verify([{'path':str(label/f),'sha256':h} for f,h in done['files'].items()])
        values=[r for p in sorted(label.glob('*.jsonl')) for r in ev.read_jsonl(p)]
        grid[scenario]={r['opportunity_id']:r for r in values}
        if len(grid[scenario])!=len(values) or set(grid[scenario])!={r['opportunity_id'] for r in rows}:raise ValueError('Unpaired fresh label grid')
    outcomes=failure_outcomes(grid);skipped=[]
    with (target/'forecasts.jsonl').open('x',encoding='utf-8') as stream,threadpool_limits(limits=1):
        for fold,(first,through) in enumerate(ev.outer_windows(meta['sessions'],min_train=60,block=20)):
            try:
                risk_model,risk_predictions,proof=av.fit_feasibility(rows,outcomes,meta['sessions'],first,through)
                fits={s:cv.fit_forecast(rows,grid[s],meta['sessions'],first,through,columns=cv.COLUMNS+pv.CHECKPOINT_COLUMNS) for s in grid}
            except ValueError as e:
                if 'Insufficient mature' not in str(e):raise
                skipped.append({'first':first,'through':through,'reason':str(e)});continue
            risk=dict((r['opportunity_id'],v) for r,v in risk_predictions);models=[('risk',risk_model,proof)]+[(s,fits[s][0],fits[s][2]) for s in fits]
            for kind,model,proof in models:
                p=target/f'late60_{fold}_{kind}.joblib';joblib.dump(model,p);proof.update(scenario=kind,model_path=p.name,model_sha256=digest(p));proofs.append(proof)
            scores={s:{r['opportunity_id']:v for r,v in fits[s][1]} for s in fits}
            for row,value in fits['baseline'][1]:
                key=row['opportunity_id'];stream.write(json.dumps({'opportunity_id':key,'valid_from':first,'baseline':value,'combined':scores['combined'][key],'failure_probability':risk[key]},allow_nan=False)+'\n')
            print(json.dumps({'stage':'fit60','first':first,'through':through,'train_rows':proof['train_rows']}),flush=True)
    write(target/'chronology.json',{'folds':proofs,'skipped_folds':skipped,'scope':'continuous past-only first-hour targets; one complete-session embargo; immature outcomes excluded; reused development'})
    write(target/'completion.json',{'complete':True,'rows':len(rows),'files':{p.name:digest(p) for p in target.iterdir() if p.is_file()}})


def append_snapshot(base,row,value,flat_only):
    snap=ev.snapshot_for(row['trade_date'],[(row,value)],source='registered_first_hour_v1')
    candidate=replace(snap.candidates[0],metadata={**snap.candidates[0].metadata,'candidate_rank':len(base.candidates)+1,
        'frontier_rank':len(base.candidates)+1,'later_entry_flat_only':flat_only,'later_entry_research':'first_hour_v1'})
    candidates=base.candidates+(candidate,)
    # Preserve incumbent global context too; update only inventory of candidates.
    metadata={**base.metadata,'active_symbols':[c.symbol for c in candidates]}
    for k in ('active_symbol_count','candidate_pool_count','selection_count','frontier_symbol_count'):metadata[k]=len(candidates)
    return replace(base,generated_at=snap.generated_at,candidates=candidates,metadata=metadata)


def build_policies(root):
    from scripts.kalcb_action_quantity_labels import scaled_row
    from scripts.kalcb_action_quantity_ranker import capacity_ok
    from scripts.kalcb_executed_value_replay import verify_policy
    protocol=read(root/'protocol.json');meta=read(root/'prepared/prepared.json')
    models=read(root/'models/completion.json');verify([{'path':str(root/'models'/f),'sha256':h} for f,h in models['files'].items()])
    scores={r['opportunity_id']:r for r in ev.read_jsonl(root/'models/forecasts.jsonl')};daily=defaultdict(list)
    for raw in ev.read_jsonl(root/'prepared/opportunities.jsonl'):
        if raw['guard_at_rank1'] and raw['action']['ready']:
            for q in (.25,.5):
                r=scaled_row(raw,q);s=scores.get(r['opportunity_id'])
                if s and s['failure_probability']<.001 and min(s['baseline'],s['combined'])>48*r['features']['action_notional_equity'] and capacity_ok(r):daily[r['trade_date']].append((r,s))
    for name,recipe in protocol['recipes'].items():
        for period in ('is','oos'):
            old=protocol['profiles'][str(recipe['prior'])][period];source=Path(old['policy']['path']).parent
            m,opening=verify_policy(source,'interaction_cost_margin');updates=cumulative_timed_policy(source,m,opening)
            target=root/'policies'/name/period;target.mkdir(parents=True,exist_ok=False)
            for f in m['files']:link(source/f,target/f)
            timed=[];admissions=ev.read_jsonl(source/m['timed_admission_file']);added=[]
            for day,base in sorted(opening.items()):
                parents=updates.get(day,[]);timed.extend(s.to_json_dict() for s in parents)
                latest=parents[-1] if parents else base
                choices=[(r,s) for r,s in daily[str(day)] if r['symbol'] not in latest.by_symbol() and r['features']['action_quantity_scale']==recipe['quantity']]
                if recipe['quantity']==0 or not choices:continue
                row,score=max(choices,key=lambda x:(min(x[1]['baseline'],x[1]['combined']),x[1]['baseline'],x[0]['symbol']))
                value=min(score['baseline'],score['combined']);snap=append_snapshot(latest,row,value,recipe['flat_only']);timed.append(snap.to_json_dict())
                added.append({'trade_date':str(day),'symbol':row['symbol'],'available_at':row['feature_available_at'],'score':value,
                    'action_id':row['opportunity_id'],'quantity_scale':recipe['quantity'],'valid_from':score['valid_from'],'flat_only':recipe['flat_only']})
            # New names avoid mutating hard-linked parent policy files.
            for f,values in [('later60.snapshots.jsonl',timed),('later60.admission.jsonl',admissions+added),('later60.new.jsonl',added)]:
                with (target/f).open('x',encoding='utf-8') as stream:
                    for r in values:stream.write(json.dumps(r,default=str,allow_nan=False)+'\n')
            chronology=read(source/'chronology.json');chronology['folds']+=read(root/'models/chronology.json')['folds']
            # Chronology is a hardlink: replace with a new inode, never overwrite.
            (target/'chronology.json').unlink();write(target/'chronology.json',chronology)
            for f in models['files']:link(root/'models'/f,target/(f if f.endswith('.joblib') else 'later_model_'+f))
            write(target/'later_lineage.json',{'registration':bound(root/'protocol.json'),'preparation':bound(root/'prepared/prepared.json'),
                'models':bound(root/'models/completion.json'),'label_manifests':[bound(root/'labels'/s/'completion.json') for s in ('baseline','combined')],
                'parent_policy':old['policy'],'parent_snapshots_preserved':True,'recipe':recipe})
            m.update(version='kalcb-cumulative-first-hour-v1',observation_minutes=60,timed_snapshot_file='later60.snapshots.jsonl',
                timed_admission_file='later60.admission.jsonl',contract={**m['contract'],'timed_budget':13,'new_budget':1},
                files={p.name:digest(p) for p in target.iterdir() if p.is_file()})
            write(target/'policies_manifest.json',m);write(target/'completion.json',{'complete':True})
            actual,ss=verify_policy(target,'interaction_cost_margin');cumulative_timed_policy(target,actual,ss)
            print(json.dumps({'stage':'policy60','recipe':name,'period':period,'added_decisions':len(added)}),flush=True)
    write(root/'policies/completion.json',{'complete':True,'independent_validation':False})
