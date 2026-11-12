"""Registered read-only accounting and mechanism assessment after native exit."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter, defaultdict
import argparse
import importlib.util
import json
import os
import sys
import time

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from backtests.strategies.kalcb.retention_research_io_v1 import read,write,bound,verify
from backtests.auto.shared.cache_keys import stable_signature

GENERIC=Path('E:/CodexResearch/k_stock_trader/kalcb_late_entry_20261011/v10_signal/assessment_v1/assess.py')
GLOBAL={'baseline':73.1303145472,'combined':47.8121465627}


def exact_process(receipt):
    import psutil
    try:
        p=psutil.Process(receipt['pid'])
        return abs(p.create_time()-receipt['process_created'])<.01 and p.cmdline()==receipt['command']
    except (psutil.NoSuchProcess,psutil.AccessDenied):
        return False


def check_gate(root,receipt):
    """Completion includes unique cases, eight bridges and supervisor departure."""
    status=read(root/'status.json') if (root/'status.json').exists() else {}
    if status.get('stage')=='attention_required':
        raise RuntimeError('Native study requires a versioned repair; preserve all evidence')
    if exact_process(receipt):return False
    if not (root/'stage_completion.json').exists():
        raise RuntimeError('Exact native supervisor exited before completion')
    completion=read(root/'stage_completion.json')
    rows=read(root/'progress.json')
    recipes=read(root/'protocol.json')['recipes']
    expected={(n,p,s) for n in recipes for p in ('is','oos') for s in ('baseline','combined')}
    actual=[(r['name'],r['period'],r['scenario']) for r in rows]
    if not completion.get('complete') or completion.get('native_cases')!=24 or len(actual)!=24 or set(actual)!=expected:
        raise ValueError('Missing, duplicate or unregistered native cases')
    if sum(r['is_identity'] for r in rows)!=8 or not read(root/'bridge_gate.json').get('passed'):
        raise ValueError('Eight exact parent bridges required')
    return True


def activity(folder):
    trades=read(folder/'trades.json'); fills=read(folder/'fills.json')
    days=len(read(folder/'daily.json'))
    return {'closed_trades':len(trades),'closed_trades_per_session':len(trades)/days,
        'filled_entry_issuer_days':len({(r['symbol'],r['timestamp'][:10]) for r in fills if r['side']=='BUY'}),
        'filled_entry_orders':len({r['order_id'] for r in fills if r['side']=='BUY'}),
        'profitable_trades':sum(t['net_pnl']>0 for t in trades),
        'win_rate':sum(t['net_pnl']>0 for t in trades)/len(trades) if trades else None,
        'average_positive_net_pnl':sum(max(t['net_pnl'],0) for t in trades)/max(1,sum(t['net_pnl']>0 for t in trades)),
        'average_negative_net_pnl':sum(min(t['net_pnl'],0) for t in trades)/max(1,sum(t['net_pnl']<0 for t in trades))}


def entry_outcomes(folder):
    result=defaultdict(lambda:{'net_pnl':0.,'qty':0})
    for t in read(folder/'trades.json'):
        key=(t['symbol'],t['entry_fill_time'])
        result[key]['net_pnl']+=t['net_pnl'];result[key]['qty']+=t['qty']
    return result


def tail_comparison(candidate,parent):
    c,p=entry_outcomes(candidate),entry_outcomes(parent)
    top=sorted(p,key=lambda k:p[k]['net_pnl'],reverse=True)[:10]
    return {'parent_top10_entries':[{'symbol':k[0],'entry_fill_time':k[1],
        'parent':p[k],'candidate':c.get(k)} for k in top],
        'parent_top10_net_pnl':sum(p[k]['net_pnl'] for k in top),
        'candidate_same_entry_net_pnl':sum(c.get(k,{}).get('net_pnl',0) for k in top),
        'new_entry_keys':len(set(c)-set(p)),'absent_parent_entry_keys':len(set(p)-set(c)),
        'scope':'Entry-key attribution; quantities, partial exits and funding descendants may differ. No isolated causal PnL claim.'}


def support(folder):
    e=read(folder/'mechanism_evidence.json'); fills=read(folder/'fills.json')
    actual={f['metadata']['execution_id']:f for f in fills}
    seen=set()
    for x in e['summary']['executions']:
        if x['execution_id'] in seen:raise ValueError('Duplicate intervention execution')
        seen.add(x['execution_id']);f=actual[x['execution_id']]
        if f['side']!='SELL' or f['order_id']!=x['order_id'] or f['reason']!='opening_thesis_failure' or f['qty']!=x['qty']:
            raise ValueError('Intervention is not an actual matching native exit')
        if x['parent_exit_reason'] is not None:
            raise ValueError('Parent exit priority was violated')
    entries=e['entry_probes']
    classifications=Counter()
    for x in entries:
        if not x.get('native_prefix_matched'):raise ValueError('Unbound alternative prefix')
        if x.get('already_candidate'):classifications['already_candidate']+=1
        elif x.get('native_order_submitted'):classifications['native_submission_in_copy']+=1
        else:
            reasons={d['reason'] for d in x.get('decisions',[])}
            classifications.update('blocked:'+r for r in reasons or {'no_decision'})
    return {'summary':e['summary'],'entry_preview_classification':dict(classifications),
        'observed_entry_previews':entries,
        'low_support':e['summary']['direct_filled_positions']<20,
        'interpretation':'Fewer than20 positions flagged as low support descriptively, not a performance acceptance threshold. Previews submit only to account copies, not funded fills or economic benefit; retries/descendants dependent.'}


def assess(out):
    reg=read(out/'protocol.json');verify(reg['programs']+reg['bindings'])
    root=Path(reg['native_root'])
    if not check_gate(root,reg['native_supervisor']):raise ValueError('Native study is still active')
    if (out/'assessment.json').exists():raise ValueError('Preserve completed assessment')
    spec=importlib.util.spec_from_file_location('registered_account_attribution',GENERIC)
    generic=importlib.util.module_from_spec(spec);spec.loader.exec_module(generic)
    from backtests.strategies.kalcb.late_artifact_validation_v2 import validate_native
    from backtests.strategies.kalcb.population_research_io_v1 import bridge_accounts
    protocol=read(root/'protocol.json');accounts={};bindings=[];bridges=[];aliases=defaultdict(list)
    rows=read(root/'progress.json')
    for row in rows:
        name,period,scenario=(row[k] for k in ('name','period','scenario'))
        folder=root/'cases'/name/period/scenario
        item=read(folder.parent/'input_profile.json')
        metrics=validate_native(folder,item,scenario,save=False)
        if row['is_identity']:
            reference=protocol['profiles'][str(protocol['recipes'][name]['prior'])][period]['references'][scenario]
            bridge=bridge_accounts(folder,Path(reference))
            if not bridge['passed']:raise ValueError('Parent bridge changed')
            bridges.append({'case':[name,period,scenario],'bridge':bridge})
        attr=generic.account_attribution(folder)
        if len(attr['daily_dates'])!={'is':156,'oos':69}[period]:raise ValueError('Shortened period')
        key='/'.join((name,period,scenario))
        accounts[key]={'metrics':metrics,'attribution':attr,'activity':activity(folder),'support':support(folder)}
        paths={f:read(folder/f) for f in ('daily.json','equity.json')}
        paths['physical_fills']=[{k:f[k] for k in ('symbol','side','qty','price','timestamp')} for f in read(folder/'fills.json')]
        aliases[stable_signature(paths)].append(key)
        for f in ('result.json','completion.json','request.json','execution_audit.json','daily.json','equity.json','fills.json','trades.json','mechanism_evidence.json'):
            bindings.append(bound(folder/f))
        print(json.dumps({'revalidated':key}),flush=True)
    comparisons=[]
    for name,recipe in protocol['recipes'].items():
        if not recipe['hold']:continue
        parent='control' if recipe['prior']==100 else 'control_prior125'
        for period in ('is','oos'):
            for scenario in ('baseline','combined'):
                ck='/'.join((name,period,scenario));pk='/'.join((parent,period,scenario))
                c,p=accounts[ck],accounts[pk]
                if c['attribution']['daily_dates']!=p['attribution']['daily_dates']:raise ValueError('Unpaired calendars')
                cm,pm=c['metrics'],p['metrics']
                deltas={k:cm[k]-pm[k] for k in ('return_pct','drawdown_pct','trades','active_days')}
                sacrifice={}
                if period=='is':
                    sacrifice={'local':pm['return_pct']-cm['return_pct'],'global':GLOBAL[scenario]-cm['return_pct']}
                comparisons.append({'candidate':ck,'parent':pk,'deltas':deltas,'IS_sacrifice_pp':sacrifice,
                    'IS_bands':{str(b):{k:v<=b for k,v in sacrifice.items()} for b in (2,5,10)},
                    'activity_deltas':{k:c['activity'][k]-p['activity'][k] for k in ('closed_trades','filled_entry_issuer_days','profitable_trades')},
                    'right_tail':tail_comparison(root/'cases'/name/period/scenario,root/'cases'/parent/period/scenario),
                    'paired_uncertainty':[generic.paired_blocks(c['attribution']['daily_returns'],p['attribution']['daily_returns'],b) for b in (5,10)]})
    verify(reg['programs']+reg['bindings']+bindings)
    write(out/'assessment.json',{'registered':bound(out/'protocol.json'),'accounts':accounts,'comparisons':comparisons,
        'bridges':bridges,'realized_path_aliases':dict(aliases),'bindings':bindings,
        'study_complete':False,'canonical_replaced':False,
        'next':'Economic judgment on exit mechanism, incremental target/population/reserve work based on funded observations, supported interactions, stress/stability/funded comparisons and existing-engine final reports.',
        'limitations':['April-July reused development, not independent validation.','Management outcomes are funded paths, not refit action labels.','Opportunity previews and parent-path interventions are not isolated causal PnL.']})
    write(out/'completion.json',{'complete':True,'study_complete':False,'source_account_cases':24,'exact_bridges':8,'assessment':bound(out/'assessment.json')})


def register(out,root,tests):
    test=read(tests)
    if test['exit_code']!=0:raise ValueError('Tests must pass')
    verify(test['programs']+[test['log']])
    protocol=read(root/'protocol.json');verify(protocol['programs']+protocol['bindings'])
    out.mkdir(parents=True,exist_ok=False)
    write(out/'protocol.json',{'registered_at_utc':datetime.now(timezone.utc).isoformat(),
        'native_root':str(root),'native_supervisor':read(root/'supervisor_start.json'),
        'programs':[bound(__file__),bound(GENERIC),*protocol['programs']],
        'bindings':[bound(root/'protocol.json'),bound(root/'study.json'),bound(root/'supervisor_start.json'),bound(tests)],
        'tests':test,'gates':'All24 unique audited cases,8exactbridges,stage_completion AND exact native supervisor exit.',
        'scope':'Read-only mechanism/account assessment; shared native PhaseRunner remains lifecycle owner; no duplicate replays, no final-report completion claim.'})


def supervise(out):
    import psutil,msvcrt
    reg=read(out/'protocol.json');verify(reg['programs']+reg['bindings'])
    lock=(out/'run.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    if (out/'supervisor_start.json').exists():raise ValueError('Previous supervisor must be preserved')
    state={'pid':os.getpid(),'process_created':psutil.Process().create_time(),'command':psutil.Process().cmdline()}
    write(out/'supervisor_start.json',state)
    try:
        while not check_gate(Path(reg['native_root']),reg['native_supervisor']):
            write(out/'status.json',{**state,'stage':'waiting_native_cases_and_exact_exit'});time.sleep(30)
        write(out/'status.json',{**state,'stage':'read_only_assessment'})
        assess(out)
        write(out/'status.json',{**state,'stage':'assessment_complete_economic_review_pending','study_complete':False})
    except Exception as error:
        write(out/'status.json',{**state,'stage':'attention_required','error':repr(error)});raise
    finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',choices=('register','supervise'))
    ap.add_argument('--out',required=True,type=Path);ap.add_argument('--root',type=Path);ap.add_argument('--tests',type=Path)
    args=ap.parse_args()
    if args.stage=='register':register(args.out.resolve(),args.root.resolve(),args.tests)
    else:supervise(args.out.resolve())
