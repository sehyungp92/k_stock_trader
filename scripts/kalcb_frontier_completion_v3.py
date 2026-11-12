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
from backtests.strategies.kalcb.retention_research_io_v1 import read, write, bound, verify, digest, env, link
from backtests.strategies.kalcb.fixed_allocation_funding_v1 import funding_review
from backtests.strategies.kalcb.late_artifact_validation_v2 import validate_native
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
    parent=BASE/'kalcb_frontier_completion_20261016/v2'
    original=read(parent/'registration.json')
    verify(original['inputs']+original['programs']+[original['source_inventory']])
    historical=read(V16/'protocol.json')
    validator=REPO/'backtests/strategies/kalcb/late_artifact_validation_v2.py'
    matched=[x for x in historical['programs'] if Path(x['path']).resolve()==validator.resolve()]
    if len(matched)!=1 or digest(validator)!=matched[0]['sha256']:raise ValueError('Different historical validator')
    tests=read(BASE/'kalcb_frontier_completion_20261016/scope_tests_r2_receipt.json')
    if tests['exit_code']!=0:raise ValueError('Validator tests failed')
    verify(tests['bindings'])
    root.mkdir(parents=True,exist_ok=False);(root/'temp').mkdir()
    shutil.copyfile(parent/'registered_design.md',root/'registered_design.md')
    shutil.copyfile(REPO/'docs/optimisation/kalcb/experiments/kalcb-exp-0029-repair-r2.md',root/'registered_repair.md')
    inputs=list(original['inputs'])+[bound(parent/'registration.json'),original['source_inventory'],bound(parent/'failure_run.json'),
        bound(root/'registered_design.md'),bound(root/'registered_repair.md'),bound(V16/'protocol.json'),
        bound(BASE/'kalcb_frontier_completion_20261016/scope_tests_r2_receipt.json')]+tests['bindings']
    profile=original['profiles'];cases=dict(original['existing_cases']);preflight={}
    for name,item in profile.items():
        policy,period=name.split('/')
        reference=V16/'fixed/cases/prior100_failure8'/period/'baseline' if policy.endswith('hold8') else V7/'cases/expanding_removal_prior100'/period/'baseline'
        preflight[name]=validate_native(reference,item,'baseline',save=False)
        for filename in ('request.json','result.json','execution_audit.json','completion.json','fills.json','trades.json','daily.json','equity.json'):
            inputs.append(bound(reference/filename))
    key='fixed_prior100_hold8/is/participation';source=parent/'cases'/key
    checked=validate_native(source,profile['fixed_prior100_hold8/is'],'participation',save=False)
    allocation=read(source/'allocation_reference.json')
    if not allocation['enabled'] or allocation['ledger_cash_mutations'] or allocation['external_capital_flows']:
        raise ValueError('Changed inherited allocation contract')
    target=root/'reused'/key;target.mkdir(parents=True,exist_ok=False)
    publications=[]
    for file in sorted(source.iterdir()):
        if file.is_file():
            if file.name in ('independent_account_verification.json','funding_review.json'):raise ValueError('Unexpected prior sidecar')
            link(file,target/file.name)
            origin=bound(file);published=bound(target/file.name)
            if origin['sha256']!=published['sha256']:raise ValueError('Changed publication')
            inputs.extend([origin,published]);publications.append(dict(original=origin,published=published))
    if validate_native(target,profile['fixed_prior100_hold8/is'],'participation')!=checked:raise ValueError('Reused account changed')
    write(target/'funding_review.json',funding_review(target))
    inputs.extend([bound(target/'independent_account_verification.json'),bound(target/'funding_review.json')]);cases[key]=str(target)
    write(root/'validator_preflight.json',dict(historical=preflight,reused_native=checked,publication=publications,scope='Exact historical policy-bound validator; no shortened source or accounting checks.'))
    inputs.append(bound(root/'validator_preflight.json'))
    programs=original['programs']+[bound(Path(__file__)),bound(validator),bound(REPO/'backtests/auto/shared/cache_keys.py')]
    verify(inputs+programs)
    registration=dict(at_utc=datetime.now(timezone.utc).isoformat(),experiment='kalcb-exp-0029',criteria='C5',repair='R2 use unchanged historical policy-bound validator',
        profiles=profile,existing_cases=cases,source_inventory=original['source_inventory'],inputs=inputs,programs=programs,
        fresh_cases=5,study_new_cases=8,inherited_study_cases=3,workers=1,native_strategy_changes=0,untouched_validation=False,
        parent=bound(parent/'registration.json'),full_source_authentication_inherited=True,
        command=f'{sys.executable} -B {Path(__file__).resolve()} --phase run --output {root}')
    with (root/'registration.json').open('x',encoding='utf-8') as stream:json.dump(registration,stream,indent=2)
    status(root,'registered',fresh_cases=5,inherited_study_cases=3)


def run(root):
    import psutil
    registration=read(root/'registration.json');verify(registration['inputs']+registration['programs']+[registration['source_inventory']])
    for p in psutil.process_iter(['pid','cmdline']):
        if p.pid==os.getpid():continue
        args=p.info['cmdline'] or []
        if any(str(x).endswith(('kalcb_fixed_allocation_worker_v1.py','kalcb_holding_account_worker_v1.py',
                               'kalcb_holding_account_worker_v2.py','kalcb_population_worker_v1.py','kalcb_exit_allocation_worker_v1.py')) for x in args):
            raise ValueError('Another KALCB native worker is active')
    if (root/'cases').exists():raise ValueError('Preserve existing native generation')
    cases=dict(registration['existing_cases']);completed=[]
    for period in ('is','oos'):
        for name in ('fixed_prior100','fixed_prior100_hold8'):
            item=registration['profiles'][name+'/'+period]
            code=Path(item['manifest']['path']).parent/'files'
            for scenario in ('participation','cost'):
                key=name+'/'+period+'/'+scenario
                if key in cases:continue
                folder=root/'cases'/name/period/scenario
                folder.parent.mkdir(parents=True,exist_ok=True)
                command=[sys.executable,'-B','-u',item['worker']['path'],
                         '--code',str(code),'--plan',item['plan']['path'],'--policies',str(Path(item['policy']['path']).parent),
                         '--audit-manifest',item['manifest']['path'],'--output',str(folder),
                         '--scenario',scenario,'--mode','fixed','--reference','100000000']
                verify([item[k] for k in ('manifest','plan','policy','worker')])
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
        predictions=predictions,bindings=bindings,scope='Complete funded 2x2 frontier; 24 historical, 2 inherited-v1, 1 authenticated-v2 and 5 fresh-v3 scenarios. All reused development. No isolated/additive PnL claim; unchanged original leverage limit.',economic_objective_achieved=False))
    write(root/'completion.json',dict(complete=True,fresh_native_cases=5,study_native_cases=8,inherited_study_cases=3,accounts=32,assessment=bound(root/'assessment.json'),
        independent_account_and_funding_validation=True,economic_decision_required=True,economic_objective_achieved=False))
    status(root,'assessment_complete',fresh_native_cases=5,study_native_cases=8,predictions=predictions)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=('register','run'),required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    try:
        (register if args.phase=='register' else run)(args.output)
    except Exception as exc:
        if args.output.exists():write(args.output/('failure_'+args.phase+'.json'),dict(at_utc=datetime.now(timezone.utc).isoformat(),error=repr(exc)))
        raise
