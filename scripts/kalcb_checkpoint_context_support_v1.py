"""Finite outcome-free source/support preflight; no native/model dispatch."""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone,timedelta
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
import time

REPO=Path(__file__).resolve().parents[1]
BASE=Path('E:/CodexResearch/k_stock_trader/kalcb_late_entry_20261011')
HELPER=REPO/'backtests/strategies/kalcb/checkpoint_context_support_v1.py'
TEST=REPO/'tests/backtests/strategies/test_kalcb_checkpoint_context_support_v1.py'
CALENDAR=BASE.parent/'kalcb_aligned_walk_forward_20261010/v3/fit/calendar.json'
spec=importlib.util.spec_from_file_location('registered_checkpoint_context_support',HELPER)
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)

def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def bound(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False)
def verify(rows):
    for r in rows:
        if sha(r['path'])!=r['sha256']:raise ValueError('Changed bound source '+r['path'])
def origins():
    values=[]
    for name,mod in tuple(sys.modules.items()):
        f=getattr(mod,'__file__',None)
        if f and Path(f).resolve().is_relative_to(REPO):
            if Path(f).resolve() not in {HELPER.resolve(),Path(__file__).resolve()}:
                raise ValueError('Unexpected application import '+str(f))
            values.append({'module':name,**bound(f)})
    return values

def register(root,tests):
    test=read(tests)
    if not test['passed']:raise ValueError('Tests failed')
    verify(test['programs'])
    old=read(BASE/'v18_reclaim_labels/protocol.json')
    cases=read(BASE/'v30_checkpoint_information/protocol.json')['cases']
    bindings=list(old['data_bindings'])+[bound(CALENDAR),bound(tests),
        bound(BASE/'v18_reclaim_labels/protocol.json'),bound(BASE/'v18_reclaim_labels/prepared/prepared.json'),
        bound(BASE/'v30_checkpoint_information/protocol.json'),bound(BASE/'v30_checkpoint_information/assessment.json'),
        bound(BASE/'v26_opening_decision_audit_v2/decisions.jsonl.gz'),
        bound(BASE/'v27_opening_conditional_value_v2/protocol.json')]
    for case in cases:
        bindings += [bound(BASE/'v30_checkpoint_information/accounts'/Path(case['key']).with_suffix('.json')),bound(case['assessment'])]
    bindings=list({b['path']:b for b in bindings}.values());verify(bindings)
    root.mkdir(exist_ok=False)
    write(root/'protocol.json',{'version':'checkpoint-context-source-support-v1','registered_at':now(),
        'programs':[bound(__file__),bound(HELPER),bound(TEST)],'bindings':bindings,'tests':bound(tests),
        'opening_prepared':old['opening_prepared'],'databases':old['databases'],'cases':cases,
        'own_features':m.OWN_FEATURES,'context_features':m.CONTEXT_FEATURES,
        'clock':'Original per-row opening+45min; only nine completed positive-volume valid5m bars. Original first six bars match raw source exactly.',
        'population':'Original immutable causal opening opportunity rows, original effective sector, never outcomes or future constituents. Peers are this opportunity population, not a broad market index.',
        'formulas':{'breadth':'Leave target out. Mean indicator(close45>VWAP45) minus mean indicator(close30>VWAP30), on the SAME complete-at45 peers.',
            'relative_return':'(target close45/close30-1 minus mean peer close45/close30-1)/target prior daily_atr_pct.',
            'unsupported':'No same-sector peer or missing prefix => unavailable context; never zero-filled. Both breadth endpoints become available only at45.'},
        'source_support':'Chronological source count uses original20session fit windows and one-session embargo. Source count cannot certify held-state/mature-label support.',
        'comparisons':'Source availability for all original rows and all existing native checkpoint holdings; existing V27 static features are not proposed as new information.',
        'dd_materiality':'Join existing independently reconciled MTM episode contributions to checkpoint/source availability. Descriptive exposure overlap, NOT saved PnL or causal benefit.',
        'next':'Exact labeled episode population, matching parent/native state and runtime must be established before a separately registered one-clock hold-versus-liquidate producer. No automatic fit/replay.',
        'runtime_estimate':'Under10min for registered read-only source census; overall design preflight budget30-60min.',
        'model_fits':0,'native_cases':0,'action_labels':0,'study_complete':False})
    print(json.dumps({'registered':str(root),'bindings':len(bindings)}),flush=True)

def run(root):
    if (root/'supervisor_start.json').exists():raise ValueError('Already started; do not restart')
    import psutil
    process=psutil.Process();start=time.perf_counter()
    write(root/'supervisor_start.json',{'pid':process.pid,'creation_epoch':process.create_time(),'command':process.cmdline(),'at':now()})
    p=read(root/'protocol.json');verify(p['programs']+p['bindings']);before=origins()
    calendar=read(CALENDAR);sessions=calendar['sessions'];windows=calendar['windows']
    if len(sessions)!=286 or sessions.index('2026-04-01')!=217:raise ValueError('Changed continuous history')
    population=defaultdict(list);seen=set()
    for i,folder in enumerate(p['opening_prepared']):
        with (Path(folder)/'opportunities.jsonl').open(encoding='utf-8') as f:
            for line in f:
                row=json.loads(line);day=row['trade_date']
                if (i==0 and day>'2026-03-31') or (i==1 and day<'2026-04-01'):continue
                key=(day,row['symbol'])
                if key in seen:raise ValueError('Duplicate original issuer-day')
                seen.add(key)
                row['candidate']={k:row['candidate'][k] for k in ('sector','trade_date')}
                row['features']={'daily_atr_pct':row['features']['daily_atr_pct']}
                population[day].append({k:row[k] for k in ('symbol','sector','trade_date','candidate','features','opening_bars','feature_available_at','action')})
    if len(seen)!=7676 or set(population)!=set(sessions):raise ValueError('Unexpected original source population')
    connections=[sqlite3.connect('file:'+Path(db).as_posix()+'?mode=ro',uri=True) for db in p['databases']]
    allrows=[];excluded=[];proofs=[]
    with gzip.open(root/'contexts.jsonl.gz','xt',encoding='utf-8',compresslevel=1) as output:
        for index,day in enumerate(sessions):
            raw=defaultdict(list)
            for symbol,ts,o,h,l,c,v in connections[int(day>='2026-04-01')].execute('SELECT symbol,ts,o,h,l,c,v FROM bars WHERE day=? ORDER BY symbol,ts',(day,)):
                raw[symbol].append({'symbol':symbol,'timestamp':datetime.fromisoformat(ts).isoformat(),'open':o,'high':h,'low':l,'close':c,'volume':v})
            rows,missing=m.context_rows(population[day],raw)
            bysym={r['symbol']:r for r in population[day]}
            for row in rows:
                source=bysym[row['symbol']]
                row.update(original_ready=source['action']['ready'],original_reference_quantity=source['action']['quantity'],
                           period='pre_is' if day<'2025-08-07' else 'is' if day<'2026-04-01' else 'oos')
                output.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
                allrows.append({k:v for k,v in row.items() if k!='consumed_bars'})
            excluded.extend(missing)
            proof={'day':day,'original':len(population[day]),'raw_complete':len(rows),'context_supported':sum(r['context_supported'] for r in rows),'excluded':len(missing)}
            proofs.append(proof)
            if index%25==0:print(json.dumps(proof),flush=True)
    for conn in connections:conn.close()
    lookup={(r['day'],r['symbol']):r for r in allrows};mapped={};dd={};assessment_cache={}
    for case in p['cases']:
        observed=read(BASE/'v30_checkpoint_information/accounts'/Path(case['key']).with_suffix('.json'))['observations']
        rows=[]
        for r in observed:
            context=lookup.get((r['day'],r['symbol']))
            rows.append({'position_id':r['position_id'],'day':r['day'],'symbol':r['symbol'],
                'held_quantity':r['held_qty'],'context_supported':bool(context and context['context_supported']),
                'prefix_present':context is not None,'parent_exit_already_decided':r['parent_exit_already_decided'],
                'available_at':context['available_at'] if context else r['features']['available_at'],
                'future_partial_flag_descriptive_only':r['entry_fills_at_or_after_checkpoint']>0})
        active=[r for r in rows if not r['parent_exit_already_decided']]
        mapped[case['key']]={'rows':rows,'chronology':m.chronological_support(active,sessions,windows),
            'total':len(rows),'no_prior_exit':len(active),'context_supported':sum(r['context_supported'] for r in active)}
        if case['assessment'] not in assessment_cache:assessment_cache[case['assessment']]=read(case['assessment'])
        account=assessment_cache[case['assessment']]['accounts'][case['assessment_key']]
        byid={r['position_id']:r for r in active};episodes=[]
        for episode in account['mtm']['episodes']:
            trough=datetime.fromisoformat(episode['trough_at'])+timedelta(minutes=5)
            contributions=[]
            for x in episode['position_contributions']:
                observed=byid.get(x['position_id'])
                contributions.append({**x,'checkpoint_observed':bool(observed),
                    'context_supported':bool(observed and observed['context_supported']),
                    'checkpoint_before_trough_close':bool(observed and datetime.fromisoformat(observed['available_at'])<=trough)})
            episodes.append({**{k:v for k,v in episode.items() if k!='position_contributions'},'position_contributions':contributions})
        dd[case['key']]={'all_native_marks_previously_reconciled':account['mtm']['all_native_marks_reconciled'],
            'maximum_episode':max(episodes,key=lambda e:e['drawdown_pct']),
            'scope':'Existing exact peak/trough position contribution; overlap is not recoverable loss or causal action effect.'}
    totals={period:{'source_rows':sum(r['period']==period for r in allrows),
                   'context_rows':sum(r['period']==period and r['context_supported'] for r in allrows),
                   'context_sessions':len({r['day'] for r in allrows if r['period']==period and r['context_supported']})}
            for period in ('pre_is','is','oos')}
    write(root/'population_source_proof.json',{'sessions':proofs,'excluded':excluded,'complete_raw_prefixes':len(allrows),
        'strict_original_first6_raw_match':True,'peer_symbols_explicit_in_context_rows':True})
    write(root/'native_support.json',mapped);write(root/'drawdown_overlap.json',dd)
    write(root/'chronology.json',m.chronological_support(allrows,sessions,windows))
    write(root/'summary.json',{'totals':totals,'source_context_supported':sum(r['context_supported'] for r in allrows),
        'unsupported_reasons':dict(Counter(r['unsupported_reason'] for r in allrows if not r['context_supported'])),
        'all_original_population_sessions':len(population),'first_oos_prior_sessions':217,
        'registered_source_window':[sessions[0],sessions[-1]],'independent_unused_window_claimed':False,
        'native_account_counts':{key:{k:v for k,v in value.items() if k not in ('rows','chronology')} for key,value in mapped.items()},
        'missing_action_evidence':'Held-state maturity, execution of early liquidation, conditional exchangeability across candidate vs selected native account and exact parent bridges are not established by this source census.',
        'native_replays':0,'fits':0,'action_labels':0,'elapsed_seconds':time.perf_counter()-start})
    verify(p['programs']+p['bindings']);after=origins()
    write(root/'completion.json',{'complete':True,'at':now(),'programs':p['programs'],'bindings':p['bindings'],
        'artifacts':[bound(root/f) for f in ('contexts.jsonl.gz','population_source_proof.json','native_support.json','drawdown_overlap.json','chronology.json','summary.json')],
        'origins_before':before,'origins_after':after,'study_complete':False,'native_cases':0,'fits':0,'action_labels':0})
    print(json.dumps({'complete':True,'totals':totals}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('register','run'));parser.add_argument('--root',type=Path,required=True);parser.add_argument('--tests',type=Path)
    args=parser.parse_args()
    if args.mode=='register':register(args.root,args.tests)
    else:
        try:run(args.root)
        except Exception as error:
            write(args.root/'failure.json',{'at':now(),'error':repr(error),'study_complete':False});raise
