"""Freeze a finite intent-allocation experiment and select within explicit risk bounds.

Reuses the previous market/native-candidate generation. Only an explicit file
allowlist is replaced, isolating this work from unrelated workspace changes.
Research profiles only; this program cannot activate deployment.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from backtests.strategies.kalcb.executed_value import digest, signature, write_json

CHANGED_FILES=(
    'backtests/engine/contracts.py','backtests/engine/integrity.py','backtests/engine/replay.py',
    'strategy_olr/core/core_models.py','strategy_olr/core/logic.py',
    'deployment/olr_kalcb/intent_allocator.py',
    'backtests/strategies/portfolio_synergy/intent_replay.py',
    'scripts/portfolio_synergy_study.py','scripts/portfolio_synergy_allocation_study.py')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def recipes(selected):
    selected=copy.deepcopy(selected)
    plain={**copy.deepcopy(selected),'quality':False,'decision_rank':False}
    aggregate={**copy.deepcopy(plain),'policy':{**plain['policy'],'same_symbol_policy':'aggregate'}}
    def allocated(base,**kw):return {**copy.deepcopy(base),'allocation':kw}
    capacity={'olr_budget_fraction':.75,'entry_capacity_bars':3.,'exit_capacity_quantile':.25}
    return {'legacy_selected':selected,
        'reservation_block':allocated(plain),
        'reservation_aggregate':allocated(aggregate),
        'exit_capacity':allocated(aggregate,exit_capacity_quantile=.25),
        'overnight_75':allocated(aggregate,olr_budget_fraction=.75),
        'overnight_65':allocated(aggregate,olr_budget_fraction=.65),
        'overnight_50':allocated(aggregate,olr_budget_fraction=.5),
        'capacity_75':allocated(aggregate,**capacity),
        'deferred_capacity_75':allocated(aggregate,**capacity,defer_minutes=5),
        'value_capacity_75':allocated(aggregate,**capacity,defer_minutes=5,rank_mode='supported_value',occupation_bps_per_session=1.),
        'half_split':allocated(aggregate,kalcb_sizing_fraction=.5,olr_sizing_fraction=.5,
            kalcb_budget_fraction=1.,olr_budget_fraction=.5),
        'kalcb_size_125':{**plain,'reference_sizing':1.25},
        'kalcb_size_150':{**plain,'reference_sizing':1.5}}


def prepare(parent_path,output):
    parent=read(parent_path);manifest=read(parent['manifest']);expected=manifest.pop('manifest_hash')
    if signature(manifest)!=expected:raise ValueError('Changed parent manifest')
    output.mkdir(parents=True,exist_ok=False);root=output/'inputs/files';root.mkdir(parents=True)
    hashes={};changed={}
    for name,sha in manifest['files'].items():
        target=root/name;target.parent.mkdir(parents=True,exist_ok=True)
        if name in CHANGED_FILES:
            shutil.copy2(REPO/name,target);hashes[name]=digest(target)
            changed[name]={'parent':sha,'new':hashes[name]}
        else:
            os.link(Path(parent['code'])/name,target);hashes[name]=sha
    for name in CHANGED_FILES:
        if name in hashes:continue
        target=root/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REPO/name,target)
        hashes[name]=digest(target);changed[name]={'parent':None,'new':hashes[name]}
    spec=read(Path(parent['code'])/parent['spec_path'])
    spec['candidates']=recipes(spec['candidates']['active_block_duplicates'])
    spec['initial_mutations']={'recipe':'legacy_selected'}
    spec['registration_revision']='intent-allocation-v2; shared forecast capacity; finite risk-budget bracket'
    spec['allocation_selection']={
        'scope':'reused_development_only',
        'baseline_max_dd_pct':11.,'combined_max_dd_pct':12.5,
        'frequency_floor':'at least native KALCB baseline and stressed trade frequency',
        'return_floor':'beat every feasible registered standalone sizing reference in each scenario',
        'objective':'highest combined-stress return among risk-feasible non-dominated portfolios',
        'near_return_band_pct_points':1.,
        'tie_break':'within 1 percentage point of top stressed return, prefer higher minimum trade count, then lower maximum DD',
        'qualification':'repeat finalist baseline/combined with full independent source and execution qualification',
        'independent_validation':False,'live_activation':False}
    name='research/portfolio_current/intent_allocation_v1.json';write_json(root/name,spec);hashes[name]=digest(root/name)
    program='from kis_core.trading_calendar import get_trading_calendar;get_trading_calendar();from backtests.auto.shared.cache_keys import source_generation_hash;print(source_generation_hash())'
    source=subprocess.check_output([sys.executable,'-B','-c',program],cwd=root,text=True).strip()
    evidence={**manifest['evidence'],'portfolio_current_study':name,'intent_allocation_parent_manifest_hash':expected}
    payload={**manifest,'files':hashes,'source_generation_hash':source,'evidence':evidence}
    write_json(output/'inputs/manifest.json',{**payload,'manifest_hash':signature(payload)})
    write_json(output/'study.json',{**parent,**spec,'manifest':str((output/'inputs/manifest.json').resolve()),
        'code':str(root.resolve()),'spec_path':name,'source_generation_hash':source})
    jobs=[]
    for recipe in spec['candidates']:
        for scenario in ('baseline','combined'):
            sid='KALCB' if recipe.startswith('kalcb_size_') else None
            jobs.append({'name':recipe+'_'+scenario,'recipe':recipe,'scenario':scenario,'standalone':sid})
    for sid in ('KALCB','OLR'):
        for scenario in ('baseline','combined'):
            jobs.append({'name':sid.lower()+'_native_'+scenario,'recipe':'legacy_selected','scenario':scenario,'standalone':sid})
    write_json(output/'registration.json',{'parent_study':str(parent_path),'parent_manifest_hash':expected,
        'source_generation_hash':source,'changed_files':changed,'jobs':jobs,'selection':spec['allocation_selection'],
        'market_and_native_candidate_inputs':'unchanged frozen parent files','initial_equity':spec['initial_equity'],
        'pilot_disclosure':'Three runtime_v1 baseline replays preceded this final registration. Reservation aggregation improved the old profile; the capacity-75 pilot exceeded the fixed 11% DD bound. Added a finite 65%/50% overnight budget bracket without changing the acceptance bounds. All results remain reused development.'})
    print(json.dumps({'prepared':str(output),'jobs':len(jobs),'source':source}),flush=True)


def run_case(study_path,job,full_audit=False):
    study=read(study_path);base=study_path.parent
    output=base/('qualified' if full_audit else 'cases')/job['name']
    if (output/'completion.json').exists():return verify_case(output,study)
    if output.exists():raise FileExistsError('Preserving incomplete replay: '+str(output))
    command=[sys.executable,'-B','-u',str(Path(study['code'])/'scripts/portfolio_synergy_study.py'),
        'replay','--study',str(study_path),'--recipe',job['recipe'],'--scenario',job['scenario'],'--output',str(output)]
    if job.get('standalone'):command+=['--standalone',job['standalone']]
    if full_audit:command+=['--full-audit']
    log=base/'logs'/((('qualified_' if full_audit else '')+job['name'])+'.log');log.parent.mkdir(exist_ok=True)
    print(json.dumps({'started':job['name'],'full_audit':full_audit}),flush=True)
    with log.open('x',encoding='utf-8') as stream:
        result=subprocess.run(command,cwd=study['code'],stdout=stream,stderr=subprocess.STDOUT,
            env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'})
    write_json(base/'receipts'/((('qualified_' if full_audit else '')+job['name'])+'.json'),{'command':command,'returncode':result.returncode})
    if result.returncode:raise RuntimeError('Replay failed: '+str(log))
    result=verify_case(output,study)
    print(json.dumps({'completed':job['name'],'return':result['return_pct'],'dd':result['drawdown_pct'],
        'trades':result['trades'],'execution_valid':result['execution_validity_pass'],'allocation_audit':result.get('allocation_audit')}),flush=True)
    return result


def verify_case(path,study):
    completion=read(path/'completion.json');result=read(path/'result.json')
    if (completion['result_sha256']!=digest(path/'result.json') or completion['audit_sha256']!=digest(path/'execution_audit.json')
            or completion['source_generation_hash']!=study['source_generation_hash']):raise ValueError('Changed replay artifact')
    if (not result['execution_validity_pass'] or result['terminal_inventory'] or result['terminal_pending']
            or result['intraday_residual_sessions']):raise ValueError('Replay execution/inventory gate failed: '+str(path))
    if result.get('allocation_audit') and (not result['allocation_audit']['pass'] or result['allocation_audit']['terminal_deferred_intents']):
        raise ValueError('Allocation audit failed: '+str(path))
    return result


def run(study_path,workers=3,only=None):
    registered=read(study_path.parent/'registration.json')['jobs']
    jobs=[j for j in registered if not only or j['name'] in only]
    errors={};results={}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(run_case,study_path,j):j for j in jobs}
        for future in as_completed(futures):
            job=futures[future]
            try:results[job['name']]=future.result()
            except Exception as e:
                errors[job['name']]=str(e);print(json.dumps({'failed':job['name'],'error':str(e)}),flush=True)
    write_json(study_path.parent/('partial_results.json' if only else 'screening_results.json'),{'results':results,'errors':errors})
    if errors:raise RuntimeError('Preserved failed cases; see results')


def select(study_path,reference_study=None):
    registration=read(study_path.parent/'registration.json');selection=registration['selection']
    study=read(study_path)
    results={j['name']:verify_case(study_path.parent/'cases'/j['name'],study) for j in registration['jobs']}
    def pair(name):return results[name+'_baseline'],results[name+'_combined']
    def feasible(b,s):return b['drawdown_pct']<=selection['baseline_max_dd_pct'] and s['drawdown_pct']<=selection['combined_max_dd_pct']
    references={name:pair(name) for name in ['kalcb_native','olr_native',
        *[name for name in study['candidates'] if name.startswith('kalcb_size_')]]}
    reference_challenge=None
    if reference_study:
        extra=read(reference_study);extra_registration=read(reference_study.parent/'registration.json')
        if (extra['source_generation_hash']!=study['source_generation_hash']
                or extra_registration['parent_study_sha256']!=digest(study_path)
                or extra['initial_equity']!=study['initial_equity']
                or extra['configuration']!=study['configuration']):
            raise ValueError('Standalone challenge is not bound to the same primary generation')
        for job in extra_registration['jobs']:
            if job['standalone']!='KALCB' or job['scenario'] not in ('baseline','combined'):
                raise ValueError('Only registered standalone challenges are supported')
            results[job['name']]=verify_case(reference_study.parent/'cases'/job['name'],extra)
        for name in sorted({j['recipe'] for j in extra_registration['jobs']}):references[name]=pair(name)
        reference_challenge={'study':str(reference_study.resolve()),'study_sha256':digest(reference_study),
            'registration_sha256':digest(reference_study.parent/'registration.json'),
            'case_evidence':{j['name']:{f:digest(reference_study.parent/'cases'/j['name']/f)
                for f in ('completion.json','result.json','execution_audit.json')} for j in extra_registration['jobs']}}
    reference_frontier={n:p for n,p in references.items() if feasible(*p)}
    floors=[max(p[i]['return_pct'] for p in reference_frontier.values()) for i in range(2)]
    native=pair('kalcb_native');candidates={};rejected={}
    for name in study['candidates']:
        if name.startswith('kalcb_size_'):continue
        b,s=pair(name);reasons=[]
        if not feasible(b,s):reasons.append('drawdown_envelope')
        if any(p['return_pct']<=floors[i] for i,p in enumerate((b,s))):reasons.append('standalone_return_frontier')
        if any(p['trades']<native[i]['trades'] for i,p in enumerate((b,s))):reasons.append('frequency_floor')
        if reasons:rejected[name]=reasons
        else:candidates[name]=(b,s)
    frontier={}
    def vector(p):return [p[0]['return_pct'],p[1]['return_pct'],min(p[0]['trades'],p[1]['trades']),-p[0]['drawdown_pct'],-p[1]['drawdown_pct']]
    for name,p in candidates.items():
        v=vector(p)
        if not any(all(a>=b for a,b in zip(vector(other),v)) and any(a>b for a,b in zip(vector(other),v))
                   for other_name,other in candidates.items() if other_name!=name):frontier[name]=p
    winner=None
    if frontier:
        top=max(p[1]['return_pct'] for p in frontier.values())
        shortlist={n:p for n,p in frontier.items() if p[1]['return_pct']>=top-selection['near_return_band_pct_points']}
        winner=max(shortlist,key=lambda n:(min(p['trades'] for p in shortlist[n]),-max(p['drawdown_pct'] for p in shortlist[n]),shortlist[n][1]['return_pct']))
    receipt={'winner':winner,'recipe':study['candidates'].get(winner),'frontier':list(frontier),'rejected':rejected,
        'reference_frontier':list(reference_frontier),'return_floors':floors,'selection':selection,'source_generation_hash':study['source_generation_hash'],
        'reference_challenge':reference_challenge,
        'status':'frozen_research_finalist_pending_qualification' if winner else 'no_candidate_passed_do_not_promote',
        'results':{n:[{k:r[k] for k in ('return_pct','drawdown_pct','trades','active_days')} for r in pair(n)]
                   for n in [*references,*[n for n in study['candidates'] if not n.startswith('kalcb_size_')]]}}
    path=study_path.parent/'finalist.json'
    if path.exists() and read(path)!=receipt:raise ValueError('Finalist already frozen')
    write_json(path,receipt);print(json.dumps(receipt,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','run','select','qualify'))
    p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path);p.add_argument('--workers',type=int,default=3)
    p.add_argument('--reference-study',type=Path)
    p.add_argument('--only',nargs='*');args=p.parse_args()
    if args.command=='prepare':prepare(args.study,args.output)
    elif args.command=='run':run(args.study,args.workers,args.only)
    elif args.command=='select':select(args.study,args.reference_study)
    else:
        finalist=read(args.study.parent/'finalist.json')
        if not finalist['winner']:raise ValueError('No selected finalist')
        jobs=[{'name':finalist['winner']+'_'+s,'recipe':finalist['winner'],'scenario':s} for s in ('baseline','combined')]
        for job in jobs:
            result=run_case(args.study,job,True)
            if result['audit_status']!='passed':raise ValueError('Full qualification failed')


if __name__=='__main__':main()
