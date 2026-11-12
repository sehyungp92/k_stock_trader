"""Complete eight missing native cells of two existing funded frontier policies.

No new alpha model or parameter selection. Existing immutable workers perform
the replay; this runner binds their inputs and applies the standard assessment.
"""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from backtests.strategies.kalcb.retention_research_io_v1 import read, write, bound, verify, digest, env, validate_native
from backtests.strategies.kalcb.fixed_allocation_funding_v1 import funding_review
from backtests.auto.shared.cache_keys import stable_signature
from scripts.kalcb_population_assessment_v1 import account, paired_attribution

BASE = Path('E:/CodexResearch/k_stock_trader')
V7 = BASE/'kalcb_selective_recovery_20261010/v7_funded_qualification'
V16 = BASE/'kalcb_late_entry_20261011/v16_exit_qualification'
SCENARIOS = ('baseline','participation','cost','combined')
METRICS_CODE = REPO/'docs/optimisation/kalcb/evidence/reconstruct_frontier.py'


def status(root, phase, **kw):
    import psutil
    data = dict(at_utc=datetime.now(timezone.utc).isoformat(), phase=phase,
                pid=os.getpid(), created_at=psutil.Process().create_time(), command=sys.argv, **kw)
    write(root/'status.tmp',data); os.replace(root/'status.tmp',root/'status.json')
    print(json.dumps(data),flush=True)


def profiles():
    result={}
    for period in ('is','oos'):
        a=read(V7/'cases/expanding_removal_prior100'/period/'input_profile.json')
        request=read(V16/'fixed/cases/prior100_failure8'/period/'baseline/request.json')
        folder=V16/'runtime/fixed/prior100_failure8'/period
        b=dict(name='prior100_failure8',period=period,folder=str(folder),
               source_generation_hash=request['source_generation_hash'],manifest=bound(folder/'inputs/manifest.json'),
               plan=bound(folder/'plan.json'),policy=bound(Path(request['arguments']['policies'])/'policies_manifest.json'))
        result['fixed_prior100/'+period]=a;result['fixed_prior100_hold8/'+period]=b
    return result


def existing_cases():
    result={}
    for period in ('is','oos'):
        for scenario in SCENARIOS:
            result['compound_prior100/'+period+'/'+scenario]=str(V16/'compounding/cases/control'/period/scenario)
            result['compound_prior100_hold8/'+period+'/'+scenario]=str(V16/'compounding/cases/prior100_failure8'/period/scenario)
        for scenario in ('baseline','combined'):
            result['fixed_prior100/'+period+'/'+scenario]=str(V7/'cases/expanding_removal_prior100'/period/scenario)
            result['fixed_prior100_hold8/'+period+'/'+scenario]=str(V16/'fixed/cases/prior100_failure8'/period/scenario)
    return result


def register(root):
    root.mkdir(parents=True,exist_ok=False);(root/'temp').mkdir()
    shutil.copyfile(REPO/'docs/optimisation/kalcb/experiments/kalcb-exp-0029.md',root/'registered_design.md')
    profile=profiles();inputs=[bound(root/'registered_design.md')];source_files=[];seen=set()
    for name,item in profile.items():
        verify([item[k] for k in ('manifest','plan','policy')])
        inputs.extend(item[k] for k in ('manifest','plan','policy'))
        manifest=read(item['manifest']['path']); expected=manifest.pop('manifest_hash')
        if stable_signature(manifest)!=expected:
            raise ValueError('Frozen manifest identity mismatch')
        code=Path(item['manifest']['path']).parent/'files'
        for relative,sha in manifest['files'].items():
            path=code/relative;stat=path.stat();identity=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,sha)
            if identity not in seen:
                if digest(path)!=sha:raise ValueError('Changed frozen source '+str(path))
                seen.add(identity)
            source_files.append(dict(path=str(path),sha256=sha))
        plan=read(item['plan']['path'])
        if plan['fixed_allocation_reference']['mode']!='fixed' or plan['fixed_allocation_reference']['reference']!=100_000_000.:
            raise ValueError('Different allocation contract')
        status(root,'source_verified',profile=name,files=len(manifest['files']))
    cases=existing_cases()
    for name,folder in cases.items():
        folder=Path(folder)
        if not read(folder/'independent_account_verification.json')['passed']:
            raise ValueError('Unverified inherited account')
        for f in ('request.json','completion.json','result.json','execution_audit.json','trades.json','fills.json','equity.json','daily.json','orders.jsonl.gz','independent_account_verification.json'):
            inputs.append(bound(folder/f))
        if name.startswith('compound_prior100/'):
            _,period,scenario=name.split('/')
            central=BASE/'holding_account_20261013/study_v2/cases/control'/period/scenario/'equity.json'
            if read(folder/'equity.json')!=read(central):
                raise ValueError('Current incumbent differs from inherited comparator')
            inputs.append(bound(central))
    # The exact published native bridges are evidence, not replayed aliases.
    bridges=[]
    for p in sorted((V16/'identity').glob('**/identity_bridge.json')):
        if not read(p)['passed']:raise ValueError('Failed historical identity bridge')
        bridges.append(bound(p))
    if not bridges:raise ValueError('Missing prior native identity bridges')
    inputs.extend(bridges)
    write(root/'frozen_source_files.json',source_files)
    programs=[bound(p) for p in (Path(__file__),METRICS_CODE,
        REPO/'backtests/strategies/kalcb/retention_research_io_v1.py',
        REPO/'backtests/strategies/kalcb/qualified_artifacts.py',
        REPO/'backtests/strategies/kalcb/fixed_allocation_funding_v1.py',
        REPO/'scripts/kalcb_population_assessment_v1.py')]
    verify(inputs+programs)
    registration=dict(at_utc=datetime.now(timezone.utc).isoformat(),experiment='kalcb-exp-0029',criteria='C5',
                      profiles=profile,existing_cases=cases,source_inventory=bound(root/'frozen_source_files.json'),
                      inputs=inputs,programs=programs,fresh_cases=8,workers=1,
                      historical_bridges=len(bridges),historical_source_unique_files_checked=len(seen),
                      source_files=len(source_files),native_strategy_changes=0,untouched_validation=False,
                      command=f'{sys.executable} -B {Path(__file__).resolve()} --phase run --output {root}')
    with (root/'registration.json').open('x',encoding='utf-8') as f:json.dump(registration,f,indent=2)
    status(root,'registered',fresh_cases=8)


def run(root):
    import psutil
    registration=read(root/'registration.json');verify(registration['inputs']+registration['programs']+[registration['source_inventory']])
    for p in psutil.process_iter(['pid','cmdline']):
        if p.pid==os.getpid():continue
        args=p.info['cmdline'] or []
        if any(str(x).endswith(('kalcb_fixed_allocation_worker_v1.py','kalcb_holding_account_worker_v1.py',
                               'kalcb_holding_account_worker_v2.py','kalcb_population_worker_v1.py')) for x in args):
            raise ValueError('Another KALCB native worker is active')
    if (root/'cases').exists():raise ValueError('Preserve existing native generation')
    cases=dict(registration['existing_cases']);completed=[]
    for period in ('is','oos'):
        for name in ('fixed_prior100','fixed_prior100_hold8'):
            item=registration['profiles'][name+'/'+period]
            code=Path(item['manifest']['path']).parent/'files'
            for scenario in ('participation','cost'):
                key=name+'/'+period+'/'+scenario;folder=root/'cases'/name/period/scenario
                folder.parent.mkdir(parents=True,exist_ok=True)
                command=[sys.executable,'-B','-u',str(code/'scripts/kalcb_fixed_allocation_worker_v1.py'),
                         '--code',str(code),'--plan',item['plan']['path'],'--policies',str(Path(item['policy']['path']).parent),
                         '--audit-manifest',item['manifest']['path'],'--output',str(folder),
                         '--scenario',scenario,'--mode','fixed','--reference','100000000']
                verify([item[k] for k in ('manifest','plan','policy')])
                if psutil.virtual_memory().available<384*1024**2:
                    raise RuntimeError('Insufficient available memory for the next native worker')
                log=root/(key.replace('/','_')+'.log');t=time.monotonic()
                with log.open('x',encoding='utf-8') as stream:
                    child=subprocess.Popen(command,cwd=code,env={**env(code),'TEMP':str(root/'temp'),'TMP':str(root/'temp')},
                                           stdout=stream,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                    identity=dict(pid=child.pid,created_at=psutil.Process(child.pid).create_time(),command=command)
                    status(root,'native_running',case=key,child=identity,completed=len(completed))
                    while child.poll() is None:
                        time.sleep(10)
                if child.returncode:
                    raise RuntimeError(f'Native worker failed for {key}; preserve {log}')
                status(root,'independent_account_validation',case=key,completed=len(completed))
                result=validate_native(folder,item,scenario)
                funding=funding_review(folder);write(folder/'funding_review.json',funding)
                allocation=read(folder/'allocation_reference.json')
                if not allocation['enabled'] or allocation['ledger_cash_mutations'] or allocation['external_capital_flows']:
                    raise ValueError('Changed native allocation/funding contract')
                completed.append(dict(case=key,seconds=time.monotonic()-t,result=result,log=bound(log)))
                write(root/'completed_cases.json',completed);cases[key]=str(folder)
                status(root,'native_case_complete',case=key,completed=len(completed),metrics=result)
    write(root/'account_index.json',cases)
    assess(root,registration,cases)


def assess(root,registration,cases):
    status(root,'economic_assessment',accounts=len(cases))
    spec=importlib.util.spec_from_file_location('frontier_metrics',METRICS_CODE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    accounts={};detailed={};funding={};bindings=[]
    for key,path in cases.items():
        folder=Path(path);accounts[key]=module.metrics(folder);detailed[key]=account(folder)
        funding[key]=funding_review(folder);bindings.extend(detailed[key]['bindings']+funding[key]['bindings'])
    comparisons={};interactions={}
    for period in ('is','oos'):
        for scenario in SCENARIOS:
            suffix='/'+period+'/'+scenario
            for a,b in (('fixed_prior100','compound_prior100'),('fixed_prior100_hold8','compound_prior100_hold8'),
                        ('fixed_prior100_hold8','fixed_prior100'),('compound_prior100_hold8','compound_prior100')):
                k=a+'_vs_'+b+suffix
                comparisons[k]=paired_attribution(detailed[a+suffix],detailed[b+suffix])
                comparisons[k]['additional_delta']={f:accounts[a+suffix][f]-accounts[b+suffix][f] for f in
                    ('max_money_dd_krw','material_funded_entries','profitable_trades','buy_notional_krw',
                     'end_underwater_pct','end_underwater_sessions','min_cash_krw','expectancy_krw')}
            interactions[period+'/'+scenario]={f:(accounts['fixed_prior100_hold8'+suffix][f]-accounts['fixed_prior100'+suffix][f])-
                (accounts['compound_prior100_hold8'+suffix][f]-accounts['compound_prior100'+suffix][f])
                for f in ('return_pct','dd_pct','max_money_dd_krw','material_funded_entries','profitable_trades')}
    predictions={}
    for name in ('fixed_prior100','fixed_prior100_hold8'):
        other='compound_prior100_hold8' if name.endswith('hold8') else 'compound_prior100'
        deltas=[accounts[name+'/oos/'+s]['return_pct']-accounts[other+'/oos/'+s]['return_pct'] for s in ('participation','cost')]
        money=[accounts[name+'/is/'+s]['max_money_dd_krw']-accounts[other+'/is/'+s]['max_money_dd_krw'] for s in ('participation','cost')]
        predictions[name]=dict(new_OOS_return_deltas_pp=deltas,any_new_OOS_gain_ge_1pp=max(deltas)>=1,
                               new_IS_money_DD_deltas_krw=money,IS_money_DD_lower_in_both=all(x<0 for x in money))
    verify(registration['inputs']+registration['programs']+bindings)
    write(root/'assessment.json',dict(at_utc=datetime.now(timezone.utc).isoformat(),registration=bound(root/'registration.json'),
        accounts=accounts,standard_diagnostics=detailed,funding=funding,comparisons=comparisons,interactions=interactions,
        predictions=predictions,bindings=bindings,scope='Complete funded 2x2 frontier; 24 inherited and 8 new scenarios. All reused development. No isolated/additive PnL claim; unchanged original leverage limit.',economic_objective_achieved=False))
    write(root/'completion.json',dict(complete=True,fresh_native_cases=8,accounts=32,assessment=bound(root/'assessment.json'),
        independent_account_and_funding_validation=True,economic_decision_required=True,economic_objective_achieved=False))
    status(root,'assessment_complete',fresh_native_cases=8,predictions=predictions)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=('register','run'),required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    try:
        (register if args.phase=='register' else run)(args.output)
    except Exception as exc:
        if args.output.exists():write(args.output/('failure_'+args.phase+'.json'),dict(at_utc=datetime.now(timezone.utc).isoformat(),error=repr(exc)))
        raise
