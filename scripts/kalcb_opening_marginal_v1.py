"""Register and run two frozen IS traversals for exp0031 marginal targets."""
import argparse
from collections import Counter,defaultdict
from copy import deepcopy
from datetime import datetime,timezone
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from backtests.strategies.kalcb.retention_research_io_v1 import read,write,bound,verify,digest,link,source_hash,env
from backtests.auto.shared.cache_keys import stable_signature
from backtests.strategies.kalcb.late_artifact_validation_v2 import validate_native
from backtests.strategies.kalcb.capital_sensitivity_diagnostics_v1 import funding_review

BASE=Path('E:/CodexResearch/k_stock_trader/kalcb_opening_marginal_20261016')
DESIGN=REPO/'docs/optimisation/kalcb/experiments/kalcb-exp-0031.md'
WORKER='scripts/kalcb_opening_marginal_worker_v1.py'
OVERLAYS=(WORKER,'backtests/strategies/kalcb/opening_marginal_replay_v1.py',
          'backtests/strategies/kalcb/fixed_allocation_reference_v1.py')
PROGRAMS=OVERLAYS+('scripts/kalcb_opening_marginal_v1.py','tests/test_kalcb_opening_marginal_v1.py',
    'backtests/strategies/kalcb/late_artifact_validation_v2.py','backtests/strategies/kalcb/qualified_artifacts.py',
    'backtests/strategies/kalcb/retention_research_io_v1.py','backtests/strategies/kalcb/capital_sensitivity_diagnostics_v1.py')


def status(root,phase,**kwargs):
    import psutil
    row=dict(at_utc=datetime.now(timezone.utc).isoformat(),phase=phase,pid=os.getpid(),
        created_at=psutil.Process().create_time(),command=sys.argv,**kwargs)
    write(root/'status.tmp',row);os.replace(root/'status.tmp',root/'status.json');print(__import__('json').dumps(row),flush=True)


def no_other_producer():
    import psutil
    for process in psutil.process_iter(['pid','name','cmdline']):
        if process.pid==os.getpid() or 'python' not in (process.info['name'] or '').lower():continue
        args=process.info['cmdline'] or []
        if any('kalcb' in a.lower() and a.lower().endswith('.py') and any(x in a.lower() for x in
               ('worker','opening_marginal_v1','capital_sensitivity_v','frontier_completion','semantic_feature_ablation')) for a in args):
            raise ValueError('Another KALCB producer is active: '+str(process.pid))


def register(root,tests):
    no_other_producer();preflight=read(BASE/'preflight_v1.json');receipt=read(tests)
    if receipt['exit_code']!=0:raise ValueError('Failed preflight tests')
    verify(receipt['bindings']+preflight['bindings']+[preflight['program'],preflight['design']])
    parent=preflight['parent_profile'];verify([parent[k] for k in ('manifest','plan','policy')])
    root.mkdir(parents=True,exist_ok=False);(root/'temp').mkdir()
    shutil.copyfile(DESIGN,root/'registered_design.md')
    inputs=preflight['bindings']+[bound(BASE/'preflight_v1.json'),bound(tests),bound(root/'registered_design.md')]
    inputs+=receipt['bindings']+[parent[k] for k in ('manifest','plan','policy')]
    protocol=dict(at_utc=datetime.now(timezone.utc).isoformat(),experiment='kalcb-exp-0031',criteria='C7',
        parent=parent,inputs=inputs,programs=[bound(REPO/p) for p in PROGRAMS],
        cases={s:dict(expected_units=list(preflight['cases'][s]['units']),bridge=preflight['cases'][s]['folder']) for s in ('baseline','combined')},
        native_control_traversals=2,planned_omission_targets=546,fit_count=0,candidate_policies=0,
        scope='Mutually exclusive native same-session marginal targets, not a sequential strategy or untouched validation.')
    write(root/'protocol.json',protocol);status(root,'source_authentication',files_checked=0)
    manifest_path=Path(parent['manifest']['path']);payload=read(manifest_path);signature=payload.pop('manifest_hash')
    if stable_signature(payload)!=signature:raise ValueError('Changed parent manifest payload')
    code=root/'inputs/files';files={};seen=set()
    for i,(rel,sha) in enumerate(payload['files'].items()):
        source=manifest_path.parent/'files'/rel;stat=source.stat();key=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,sha)
        if key not in seen:
            if digest(source)!=sha:raise ValueError('Changed frozen source '+rel)
            seen.add(key)
        if rel not in OVERLAYS:link(source,code/rel);files[rel]=sha
        if i%500==0:status(root,'source_authentication',files_checked=i+1)
    for rel in OVERLAYS:
        (code/rel).parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(REPO/rel,code/rel);files[rel]=digest(code/rel)
    generation=source_hash(code);payload['files']=files;payload['source_generation_hash']=generation
    payload['evidence']['opening_marginal_diagnostic']=dict(protocol=bound(root/'protocol.json'),
        population=bound(BASE/'preflight_v1.json'),parent_manifest=parent['manifest'],disabled_parent=True)
    write(root/'inputs/manifest.json',{**payload,'manifest_hash':stable_signature(payload)})
    profile=dict(name='control',period='is',folder=str(root),source_generation_hash=generation,
        manifest=bound(root/'inputs/manifest.json'),plan=parent['plan'],
        policy=bound(code/'research/kalcb_executed_value/policies/policies_manifest.json'))
    if profile['policy']['sha256']!=parent['policy']['sha256']:raise ValueError('Policy changed')
    environment=env(code);environment.update(TEMP=str(root/'temp'),TMP=str(root/'temp'))
    result=subprocess.run([sys.executable,'-B',str(code/WORKER),'--help'],cwd=code,env=environment,capture_output=True,text=True,check=True)
    write(root/'source_preflight.json',dict(full_hash_passed=True,inherited_files=len(payload['files'])-len(OVERLAYS),
        executing_generation=generation,worker_help=result.stdout,profile=profile,
        original_core_and_execution_unchanged=True,only_new_omission_observer_and_worker=True))
    verify(protocol['programs']+inputs)
    write(root/'registration.json',dict(at_utc=datetime.now(timezone.utc).isoformat(),protocol=bound(root/'protocol.json'),
        source_preflight=bound(root/'source_preflight.json'),profile=profile,programs=protocol['programs'],inputs=inputs))
    status(root,'registered',omission_targets=546,native_traversals=2)


def run(root):
    import psutil,msvcrt
    no_other_producer();reg=read(root/'registration.json');protocol=read(root/'protocol.json')
    verify(reg['programs']+reg['inputs']+[reg['protocol'],reg['source_preflight']])
    if (root/'cases').exists():raise ValueError('Preserve existing generation')
    lock=(root/'run.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    code=root/'inputs/files';environment=env(code);environment.update(TEMP=str(root/'temp'),TMP=str(root/'temp'))
    try:
        for scenario,case in protocol['cases'].items():
            verify(reg['programs']);item=reg['profile'];target=root/'cases'/scenario;target.parent.mkdir(exist_ok=True)
            command=[sys.executable,'-B','-u',str(code/WORKER),'--code',str(code),'--plan',item['plan']['path'],
                '--policies',str(Path(item['policy']['path']).parent),'--audit-manifest',item['manifest']['path'],
                '--output',str(target),'--population',str(BASE/'preflight_v1.json'),'--bridge',case['bridge'],'--scenario',scenario]
            with (root/(scenario+'.log')).open('x',encoding='utf-8') as log:
                child=subprocess.Popen(command,cwd=code,env=environment,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                status(root,'native_targets',scenario=scenario,child=dict(pid=child.pid,
                    created_at=psutil.Process(child.pid).create_time(),command=command));child.wait()
            if child.returncode:raise RuntimeError('Native targets failed: '+scenario)
            validate_native(target,item,scenario);write(target/'funding_review.json',funding_review(target))
            status(root,'case_validated',scenario=scenario,omission_targets=read(target/'completion.json')['omission_targets'])
        assess(root,reg);status(root,'assessment_complete')
    finally:
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()


def assess(root,reg):
    summaries={};inputs=[]
    for scenario in ('baseline','combined'):
        folder=root/'cases'/scenario;validate_native(folder,reg['profile'],scenario,save=False)
        rows=read(folder/'marginal_targets.json');verify([r['ledger'] for r in rows]);actual=[r for r in rows if r['omitted_symbol']]
        if len(actual)!=273:raise ValueError('Target population differs from registration')
        monthly=defaultdict(list)
        for row in actual:monthly[row['day'][:7]].append(row)
        def summary(items):
            sign=lambda x:1 if x>.01 else -1 if x<-.01 else 0
            return dict(targets=len(items),sessions=len({r['day'] for r in items}),
                positive_omission_wealth=sum(r['delta_wealth_krw']>.01 for r in items),
                externality_abs_ge_100k=sum(abs(r['other_opportunity_effect_krw'])>=100000 for r in items),
                donor_vs_retention_sign_reversals=sum(sign(r['donor_first30_control_pnl_krw'])!=0 and
                    sign(-r['delta_wealth_krw'])!=0 and sign(r['donor_first30_control_pnl_krw'])!=sign(-r['delta_wealth_krw']) for r in items),
                absolute_externality_krw=sum(abs(r['other_opportunity_effect_krw']) for r in items),
                absolute_total_contrast_krw=sum(abs(r['delta_wealth_krw']) for r in items),
                mean_externality_krw=sum(r['other_opportunity_effect_krw'] for r in items)/len(items))
        summaries[scenario]=dict(overall=summary(actual),monthly={m:summary(v) for m,v in monthly.items()},
            largest_externalities=sorted(actual,key=lambda r:abs(r['other_opportunity_effect_krw']),reverse=True)[:15])
        inputs.extend([bound(folder/f) for f in ('result.json','completion.json','identity_bridge.json','marginal_targets.json',
            'independent_account_verification.json','funding_review.json')]+[r['ledger'] for r in rows])
    verify(reg['programs']+reg['inputs'])
    write(root/'assessment.json',dict(at_utc=datetime.now(timezone.utc).isoformat(),registration=bound(root/'registration.json'),
        summaries=summaries,bindings=inputs,scope='Dependent mutually exclusive diagnostic targets; sums are magnitude diagnostics, never strategy returns.',
        economic_decision_required=True,economic_objective_achieved=False))
    write(root/'completion.json',dict(complete=True,assessment=bound(root/'assessment.json'),native_controls=2,
        omission_targets=546,qualified_candidate_policies=0,economic_objective_achieved=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--phase',choices=('register','run','all'),required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--tests',type=Path);args=parser.parse_args()
    try:
        if args.phase in ('register','all'):register(args.output,args.tests)
        if args.phase in ('run','all'):run(args.output)
    except Exception as error:
        if args.output.exists():
            write(args.output/('failure_'+args.phase+'.json'),dict(at_utc=datetime.now(timezone.utc).isoformat(),error=repr(error),traceback=traceback.format_exc()))
            status(args.output,'failed',error=repr(error))
        raise
