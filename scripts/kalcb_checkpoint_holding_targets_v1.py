"""Register and derive current-holding targets from already audited native arms."""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import gzip
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
BASE=Path('E:/CodexResearch/k_stock_trader/kalcb_late_entry_20261011')
SOURCE=BASE/'v33_checkpoint_contrast_labels'
SCOPE=BASE/'v35_checkpoint_hold_bridge/scope_review_v1'
HELPER=REPO/'backtests/strategies/kalcb/checkpoint_holding_target_v1.py'
LIFE=REPO/'backtests/strategies/kalcb/checkpoint_episode_scope_v1.py'
TEST=REPO/'tests/backtests/strategies/test_kalcb_checkpoint_holding_target_v1.py'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def bound(p):
    with Path(p).open('rb') as f:s=hashlib.file_digest(f,'sha256').hexdigest()
    return {'path':str(Path(p).resolve()),'sha256':s}
def verify(items):
    for x in items:
        if bound(x['path'])['sha256']!=x['sha256']:raise ValueError('Changed source '+x['path'])
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False)
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def origins():
    out=[]
    for name,m in tuple(sys.modules.items()):
        f=getattr(m,'__file__',None)
        if f and Path(f).resolve().is_relative_to(REPO):
            if Path(f).resolve() not in (Path(__file__).resolve(),HELPER.resolve(),LIFE.resolve()):raise ValueError('Unexpected application import '+name)
            out.append({'module':name,**bound(f)})
    return out
def closed():
    import psutil
    x=read(SCOPE/'launch_receipt.json')
    try:
        p=psutil.Process(x['pid'])
        if p.cmdline()==x['command'] and abs(p.create_time()-x['process_created'])<.001:raise ValueError('Scope audit still running')
    except psutil.NoSuchProcess:pass
    if not read(SCOPE/'completion.json')['complete']:raise ValueError('Scope audit incomplete')
def register(root,tests):
    closed();t=read(tests)
    if t['exit_code']!=0:raise ValueError('Failed tests')
    verify(t['programs']+[t['log'],t['historical_preflight']]);p=read(SCOPE/'protocol.json');done=read(SCOPE/'completion.json')
    bindings=p['bindings']+[done['scope'],bound(SCOPE/'completion.json'),bound(SCOPE/'protocol.json'),bound(tests),t['historical_preflight']]
    bindings=list({x['path']:x for x in bindings}.values());verify(bindings)
    root.mkdir(exist_ok=False)
    write(root/'protocol.json',{'at':datetime.now(timezone.utc).isoformat(),'programs':[bound(__file__),bound(HELPER),bound(LIFE),bound(TEST)],'bindings':bindings,'tests':bound(tests),
        'population':'All38508 existing V33 paired observations unchanged, including no positions, failures and unresolved outcomes; zero native reruns or new quantity choices.',
        'target':'Net cash difference of the actual position lifetime held immediately before completed45, LIQUIDATE minus HOLD, divided by exact actual held shares times completed45 close. Require original independently valid paired episode and both current lifetimes closed; preserve original conservative whole-episode label availability. Original episode and other-lifetime differences retained separately.',
        'causal_state_features':{'log_held_notional':'log(actual held integer shares * completed45 close)','log_held_to_last_bar_volume':'log1p(actual held integer shares / latest completed5m source volume)','held_to_planned_quantity':'actual held / original planned integer quantity','pending_to_held_quantity':'remaining submitted BUY shares / actual held shares','marked_entry_return_atr':'(completed45 close / actual filled entry price -1) / prior daily ATR percent'},
        'market_features':'Exact independently reconstructed V31 own six and leave-one-out temporal context four; no stale geometry substitutions or future peers.',
        'scope_gate':'For any later proposed transfer, pending-entry states remain unsupported until matching funding evidence exists. No-pending is a decision-time state, not a label-outcome filter. Current research data retains every state. Quantity/state support and original physical gate remain separate.',
        'next_model':'Not registered here. Before any fit freeze at most own-state versus incremental temporal-context contrast, all features/formulas/support/domain/calibration, original20session clock/60mature sessions/one-session embargo. Preserve no-extrapolation and unsupported abstention; no generic penalty/feature grid.',
        'limits':'Current-holding action value is a separately versioned research estimand, not a funded portfolio return. Other positions and later entries remain for whole-portfolio qualification; no post-hoc PnL scaling or summing alternatives. No new policy or live change.',
        'models':0,'policies':0,'native_cases':0,'study_complete':False})
def run(root):
    import psutil
    closed();p=read(root/'protocol.json');verify(p['programs']+p['bindings']);proc=psutil.Process()
    write(root/'supervisor_start.json',{'pid':proc.pid,'process_created':proc.create_time(),'command':proc.cmdline(),'at':datetime.now(timezone.utc).isoformat()})
    h=load('registered_holding_target',HELPER);life=load('registered_lifetime_split',LIFE);before=origins()
    with gzip.open(BASE/'v31_checkpoint_context_support/contexts.jsonl.gz','rt',encoding='utf-8') as f:contexts={(r['day'],r['symbol']):r for r in map(json.loads,f)}
    artifacts=read(SOURCE/'labels/completion.json')['artifacts'];byname={Path(x['path']).name:x for x in artifacts};seen=set();counts=Counter();stats=defaultdict(Counter);joint=defaultdict(dict);consumed=[]
    with gzip.open(root/'targets.jsonl.gz','xt',encoding='utf-8') as out:
        for binding in sorted((x for x in artifacts if '.events.' not in x['path']),key=lambda x:x['path']):
            path=Path(binding['path']);events=path.with_name(path.name.replace('.jsonl.gz','.events.jsonl.gz'));eb=byname[events.name];verify([binding,eb]);consumed.extend([binding,eb])
            with gzip.open(path,'rt',encoding='utf-8') as vf,gzip.open(events,'rt',encoding='utf-8') as ef:
                for line,event in itertools.zip_longest(vf,ef):
                    if line is None or event is None:raise ValueError('Label/event length disagreement')
                    v=json.loads(line);e=json.loads(event);key=(v['opportunity_id'],v['scenario'])
                    if key in seen or e['pair']['opportunity_id']!=key[0] or e['pair']['scenario']!=key[1]:raise ValueError('Wrong/duplicate pair')
                    seen.add(key);row=h.derive(v,e,contexts[v['day'],v['symbol']],life.lifetimes)
                    out.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n');counts['rows']+=1
                    period='pre_is' if row['day']<'2025-08-07' else 'is' if row['day']<'2026-04-01' else 'oos'
                    bucket=stats[period+'/'+row['scenario']];bucket['rows']+=1;bucket['valid']+=row['current_holding_label_valid'];bucket['no_pending']+=row['no_pending_entry_state']
                    if row['current_holding_label_valid']:
                        bucket['nonzero_other_lifetimes']+=abs(row['other_lifetimes_advantage_bps'])>1e-7
                        bucket['initial_episode_sign_disagreement']+=(row['current_holding_advantage_bps']>0)!=(row['original_episode_advantage_bps']>0)
                    joint[row['opportunity_id']][row['scenario']]={k:row[k] for k in ('day','symbol','current_holding_label_valid','label_available_at','no_pending_entry_state')}
            print(json.dumps({'file':path.name,'rows':counts['rows']}),flush=True)
    if counts['rows']!=38508 or any(set(x)!=set(('baseline','combined')) for x in joint.values()):raise ValueError('Wrong finite population')
    # Coverage only, not a fit. Scenarios and quantities do not multiply sessions.
    calendar=read(BASE.parent/'kalcb_aligned_walk_forward_20261010/v3/fit/calendar.json');chronology=[]
    for first,last in calendar['windows']:
        allowed=set(calendar['sessions'][:max(0,calendar['sessions'].index(first)-1)]);cutoff=max(allowed)
        ready=[x for x in joint.values() if all(r['day'] in allowed and r['current_holding_label_valid'] and r['no_pending_entry_state'] and r['label_available_at'][:10]<=cutoff for r in x.values())]
        issuer={(x['baseline']['day'],x['baseline']['symbol']) for x in ready}
        chronology.append({'valid_from':first,'valid_through':last,'embargo_cutoff':cutoff,'mature_sessions':len({x[0] for x in issuer}),'issuer_days':len(issuer),'dependent_quantities':len(ready)})
    write(root/'summary.json',{'counts':dict(counts),'period_scenario':{k:dict(v) for k,v in stats.items()},'no_pending_original_population_chronology':chronology,
          'limits':p['limits'],'selected_parent_maturity':'V34 known selected chronology retained; full population is not selected-parent evidence.'})
    verify(p['programs']+p['bindings']+consumed)
    write(root/'completion.json',{'complete':True,'at':datetime.now(timezone.utc).isoformat(),'programs':p['programs'],'bindings':p['bindings'],'consumed':consumed,
          'origins_before':before,'origins_after':origins(),'artifacts':[bound(root/'targets.jsonl.gz'),bound(root/'summary.json')],
          'models':0,'policies':0,'native_cases':0,'study_complete':False})
if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('mode',choices=('register','run'));a.add_argument('--root',type=Path,required=True);a.add_argument('--tests',type=Path)
    args=a.parse_args()
    if args.mode=='register':register(args.root,args.tests)
    else:
        try:run(args.root)
        except Exception as e:write(args.root/'failure.json',{'at':datetime.now(timezone.utc).isoformat(),'error':repr(e),'study_complete':False});raise
