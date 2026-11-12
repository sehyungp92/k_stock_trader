"""Prospective exp0030: ten native paths, frozen old policies, matched capital."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
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
from backtests.strategies.kalcb import capital_sensitivity_diagnostics_v1 as diag
from backtests.strategies.kalcb.late_artifact_validation_v2 import validate_native
BASE=Path('E:/CodexResearch/k_stock_trader')
PREVIOUS=BASE/'kalcb_frontier_completion_20261016/v3'
WORKER='scripts/kalcb_capital_sensitivity_worker_v1.py'
DESIGN=REPO/'docs/optimisation/kalcb/experiments/kalcb-exp-0030.md'
SCENARIOS=('baseline','participation','cost','combined')
PROGRAMS=(WORKER,'scripts/kalcb_capital_sensitivity_v1.py',
    'backtests/strategies/kalcb/capital_sensitivity_diagnostics_v1.py',
    'backtests/strategies/kalcb/late_artifact_validation_v2.py',
    'backtests/strategies/kalcb/qualified_artifacts.py',
    'backtests/strategies/kalcb/retention_research_io_v1.py',
    'scripts/kalcb_frontier_activity_attribution_v1.py','scripts/kalcb_population_assessment_v1.py',
    'backtests/auto/shared/cache_keys.py','tests/backtests/strategies/test_kalcb_capital_sensitivity_v1.py')


def status(root,phase,**kwargs):
    import psutil
    row=dict(at_utc=datetime.now(timezone.utc).isoformat(),phase=phase,pid=os.getpid(),
        created_at=psutil.Process().create_time(),command=sys.argv,**kwargs)
    write(root/'status.tmp',row);os.replace(root/'status.tmp',root/'status.json');print(json.dumps(row),flush=True)


def no_other_producer():
    import psutil
    for process in psutil.process_iter(['pid','name','cmdline']):
        if process.pid==os.getpid() or 'python' not in (process.info['name'] or '').lower():continue
        args=process.info['cmdline'] or []
        if any('kalcb' in a.lower() and (a.endswith('_worker_v1.py') or a.endswith('kalcb_capital_sensitivity_v1.py') or
               a.endswith('kalcb_frontier_completion_v3.py') or a.endswith('kalcb_semantic_feature_ablation_v1.py')) for a in args):
            raise ValueError('Another KALCB producer is active: '+str(process.pid))


def register(root,tests):
    no_other_producer()
    if not read(PREVIOUS/'completion.json')['complete']:raise ValueError('Previous study incomplete')
    prior=read(PREVIOUS/'registration.json');verify(prior['programs']+prior['inputs']+[prior['source_inventory']])
    receipt=read(tests)
    if receipt['exit_code']!=0:raise ValueError('Tests failed')
    verify(receipt['bindings'])
    index=read(PREVIOUS/'account_index.json');parent=prior['profiles']['fixed_prior100/oos']
    validate_native(index['fixed_prior100/oos/baseline'],parent,'baseline',save=False)
    root.mkdir(parents=True,exist_ok=False);(root/'temp').mkdir();shutil.copyfile(DESIGN,root/'registered_design.md')
    cases=[];inputs=[bound(PREVIOUS/f) for f in ('registration.json','completion.json','assessment.json','account_index.json')]
    inputs += [bound(tests),bound(root/'registered_design.md')]+receipt['bindings']+[parent[k] for k in ('manifest','plan','policy')]
    for mode in ('compound','fixed'):
        bridge=index[mode+'_prior100/oos/baseline']
        cases.append(dict(id='bridge_'+mode,mode=mode,scenario='baseline',initial_equity=100_000_000.,reference=100_000_000.,bridge=bridge))
    capitals={}
    for scenario in SCENARIOS:
        folder=Path(index['compound_prior100/is/'+scenario]);daily=read(folder/'daily.json')
        if daily[-1]['day']!='2026-03-31' or any(d['inventory'] or d['pending'] for d in daily):
            raise ValueError('Prior earned-capital account not flat')
        if not read(folder/'independent_account_verification.json')['passed']:raise ValueError('Unverified prior account')
        initial=daily[-1]['equity'];capitals[scenario]=initial
        for mode in ('compound','fixed'):
            cases.append(dict(id=mode+'_'+scenario,mode=mode,scenario=scenario,initial_equity=initial,
                reference=100_000_000.,capital_source=bound(folder/'daily.json')))
    for folder in {Path(index[m+'_prior100/'+period+'/'+s]) for m in ('compound','fixed') for period in ('is','oos') for s in SCENARIOS}:
        inputs.extend(bound(folder/f) for f in ('result.json','daily.json','equity.json','fills.json','trades.json','independent_account_verification.json'))
    protocol=dict(at_utc=datetime.now(timezone.utc).isoformat(),experiment='kalcb-exp-0030',criteria='C6',
        cases=cases,parent=parent,capitals=capitals,programs=[bound(REPO/p) for p in PROGRAMS],inputs=inputs,
        historical_accounts=index,full_continuous_state_claim=False,untouched_validation=False,native_cases=10,
        scope='Matched earned-control-capital flat restart. Historical bar-participation model only. No parameter optimization.')
    verify(inputs+protocol['programs']);write(root/'protocol.json',protocol)
    status(root,'source_authentication',files_checked=0)
    pp=Path(parent['manifest']['path']);payload=read(pp);sig=payload.pop('manifest_hash')
    if stable_signature(payload)!=sig:raise ValueError('Parent manifest payload changed')
    code=root/'inputs/files';files={};seen=set();originals=[]
    for i,(relative,sha) in enumerate(payload['files'].items()):
        source=pp.parent/'files'/relative;stat=source.stat();key=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,sha)
        if key not in seen:
            if digest(source)!=sha:raise ValueError('Changed frozen input '+relative)
            seen.add(key)
        link(source,code/relative);files[relative]=sha;originals.append(dict(path=str(source),sha256=sha))
        if i%500==0:status(root,'source_authentication',files_checked=i+1)
    if WORKER in files:raise ValueError('Unexpected pre-existing capital worker')
    shutil.copyfile(REPO/WORKER,code/WORKER);files[WORKER]=digest(code/WORKER)
    oldplan=read(parent['plan']['path']);oldcontext=read(oldplan['audit_configuration'])
    qualification=read(code/payload['evidence']['qualification_path'])
    # New exact capital/scenario declarations preserve all original rules.
    declaration=qualification['capacity']
    if declaration['kind']!='declared_bar_participation_model' or declaration['phase_liquidity_verified'] is not False:
        raise ValueError('Unsupported capacity evidence type')
    for scenario in SCENARIOS:
        oldaudit=read(Path(index['compound_prior100/oos/'+scenario])/'execution_audit.json')
        policy=oldaudit['audit_receipt']['qualification']['capacity']['policy']
        rows=[r for r in declaration['registered_execution_policies'] if r['initial_equity']==100_000_000 and r['buying_power_leverage']==2 and r['policy']==policy]
        if len(rows)!=1:raise ValueError('Ambiguous original capacity tuple')
        row=deepcopy(rows[0]);row['initial_equity']=capitals[scenario]
        if row in declaration['registered_execution_policies']:raise ValueError('Unexpected prior capital sensitivity')
        declaration['registered_execution_policies'].append(row)
    qrel='research/capital_sensitivity/qualification.json';write(code/qrel,qualification);files[qrel]=digest(code/qrel)
    payload['evidence']['qualification_path']=qrel
    profiles={}
    for case in cases:
        folder=root/'plans'/case['id'];context={**oldcontext,'initial_equity':case['initial_equity'],
            'output_root':str(root/'cases'/case['id']),'promotion_status':'capital_sensitivity_research_only'}
        write(folder/'audit_configuration.json',context)
        plan=deepcopy(oldplan);plan['input_sha256'].pop(plan['audit_configuration'],None)
        plan['audit_configuration']=str(folder/'audit_configuration.json')
        plan.pop('fixed_allocation_reference',None)
        plan['capital_sensitivity']={**case,'protocol':bound(root/'protocol.json')}
        for path in (folder/'audit_configuration.json',root/'protocol.json'):
            plan['input_sha256'][str(path)]=digest(path)
        if case.get('capital_source'):plan['input_sha256'][case['capital_source']['path']]=case['capital_source']['sha256']
        write(folder/'plan.json',plan)
        for name in ('plan.json','audit_configuration.json'):
            rel='research/capital_sensitivity/'+case['id']+'/'+name;link(folder/name,code/rel);files[rel]=digest(folder/name)
        profiles[case['id']]=dict(name=case['id'],period='oos',folder=str(root),manifest=None,
            plan=bound(folder/'plan.json'),policy=bound(code/'research/kalcb_executed_value/policies/policies_manifest.json'))
    rel='research/capital_sensitivity/protocol.json';link(root/'protocol.json',code/rel);files[rel]=digest(root/'protocol.json')
    generation=source_hash(code);payload['source_generation_hash']=generation;payload['files']=files
    payload['evidence']['capital_sensitivity']=dict(protocol=bound(root/'protocol.json'),parent_manifest=parent['manifest'],
        full_continuous_state_claim=False,exact_capital_scenarios=capitals,core_engine_policy_changes=False)
    write(root/'inputs/manifest.json',{**payload,'manifest_hash':stable_signature(payload)})
    for item in profiles.values():item.update(manifest=bound(root/'inputs/manifest.json'),source_generation_hash=generation)
    if any(item['policy']['sha256']!=parent['policy']['sha256'] for item in profiles.values()):raise ValueError('Policy drift')
    preflight_command=[sys.executable,'-B','-c',
        "import json;from dataclasses import asdict;from pathlib import Path;from scripts.kalcb_executed_value_study import config_for;print(json.dumps(asdict(config_for(json.loads(Path("+repr(str(root/'plans/bridge_fixed/plan.json'))+").read_text()))),default=str))"]
    environment=env(code);environment.update(TEMP=str(root/'temp'),TMP=str(root/'temp'))
    config=json.loads(subprocess.check_output(preflight_command,cwd=code,env=environment,text=True).strip().splitlines()[-1])
    guards=('risk_dynamic_notional_enabled','risk_max_single_equity_pct','risk_max_gross_equity_pct',
            'risk_daily_vol_target','risk_drawdown_full_pct','frontier_shadow_enabled','frontier_rotation_enabled')
    if any(config[k] for k in guards):raise ValueError('Unsupported capital/state dependency')
    if config['carry_mode'] not in ('off','OFF'):raise ValueError('Overnight state unsupported')
    help_result=subprocess.run([sys.executable,'-B',str(code/WORKER),'--help'],cwd=code,env=environment,capture_output=True,text=True,check=True)
    write(root/'source_preflight.json',dict(original_files=len(originals),full_hash_passed=True,executing_generation=generation,
        effective_config=config,disabled_dependency_guards={k:config[k] for k in guards},worker_help=help_result.stdout,
        native_account_validator='unchanged late_artifact_validation_v2.validate_case',capital_declaration=bound(code/qrel)))
    write(root/'source_inventory.json',originals)
    verify(protocol['programs']+protocol['inputs'])
    registration=dict(at_utc=datetime.now(timezone.utc).isoformat(),protocol=bound(root/'protocol.json'),profiles=profiles,
        programs=protocol['programs'],source_preflight=bound(root/'source_preflight.json'),source_inventory=bound(root/'source_inventory.json'))
    write(root/'registration.json',registration);status(root,'registered',native_cases=10)


def run(root):
    import psutil,msvcrt
    no_other_producer();registration=read(root/'registration.json');protocol=read(root/'protocol.json')
    verify(registration['programs']+[registration['protocol'],registration['source_preflight'],registration['source_inventory']])
    if (root/'cases').exists():raise ValueError('Preserve existing generation')
    lock=(root/'run.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    code=root/'inputs/files';environment=env(code);environment.update(TEMP=str(root/'temp'),TMP=str(root/'temp'))
    completed=[]
    try:
        for case in protocol['cases']:
            if not case.get('bridge') and len(completed)<2:raise ValueError('Missing identity bridges')
            verify(registration['programs']);item=registration['profiles'][case['id']]
            target=root/'cases'/case['id'];target.parent.mkdir(exist_ok=True)
            command=[sys.executable,'-B','-u',str(code/WORKER),'--code',str(code),'--plan',item['plan']['path'],
                '--policies',str(Path(item['policy']['path']).parent),'--audit-manifest',item['manifest']['path'],'--output',str(target)]
            started=time.monotonic()
            with (root/(case['id']+'.log')).open('x',encoding='utf-8') as log:
                child=subprocess.Popen(command,cwd=code,env=environment,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                identity=dict(pid=child.pid,created_at=psutil.Process(child.pid).create_time(),command=command)
                status(root,'native',case=case['id'],child=identity,completed=len(completed))
                child.wait()
            if child.returncode:raise RuntimeError('Native case failed: '+case['id'])
            validation=diag.validate_capital_native(target,item)
            write(target/'funding_review.json',diag.funding_review(target,case['initial_equity']))
            if case.get('bridge') and not read(target/'identity_bridge.json')['passed']:raise ValueError('Identity failed')
            completed.append(dict(id=case['id'],seconds=time.monotonic()-started,validation=validation,result=bound(target/'result.json')))
            write(root/'completed_cases.json',completed);status(root,'case_validated',case=case['id'],completed=len(completed))
        assess(root,registration,protocol);status(root,'assessment_complete',completed=len(completed))
    finally:
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()


def assess(root,registration,protocol):
    if (root/'assessment.json').exists():raise ValueError('Preserve prior assessment')
    rows={};diagnostics={};activity={};funding={};comparisons={};inputs=[]
    for case in protocol['cases']:
        if case.get('bridge'):continue
        folder=root/'cases'/case['id'];initial=case['initial_equity']
        diag.validate_capital_native(folder,registration['profiles'][case['id']],save=False)
        rows[case['id']]=diag.metrics(folder,initial);rows[case['id']]['initial_equity_krw']=initial
        diagnostics[case['id']]=diag.account(folder,initial);activity[case['id']]=diag.opportunities(folder,initial)
        funding[case['id']]=diag.funding_review(folder,initial)
        inputs.extend(bound(folder/f) for f in ('result.json','execution_audit.json','completion.json','request.json',
            'trades.json','fills.json','daily.json','equity.json','allocation_reference.json','independent_account_verification.json'))
    for scenario in SCENARIOS:
        a='fixed_'+scenario;b='compound_'+scenario
        comparisons[scenario]=dict(path=diag.paired_attribution(diagnostics[a],diagnostics[b]),
            activity=diag.compare(activity[a],activity[b]),
            delta_money_dd_krw=rows[a]['max_money_dd_krw']-rows[b]['max_money_dd_krw'])
        comparisons[scenario]['capital_changes']={}
        for mode in ('compound','fixed'):
            old=Path(protocol['historical_accounts'][mode+'_prior100/oos/'+scenario]);key=mode+'_'+scenario
            comparisons[scenario]['capital_changes'][mode]=dict(
                activity=diag.compare(activity[key],diag.opportunities(old)),
                path=diag.paired_attribution(diagnostics[key],diag.account(old)))
    deltas=[comparisons[s]['activity']['net_pnl_delta_krw'] for s in SCENARIOS]
    predictions=dict(fixed_positive_stresses=sum(x>0 for x in deltas),combined_positive=deltas[3]>0,
        any_gain_ge_1m=any(x>=1_000_000 for x in deltas),
        lower_money_dd_stresses=sum(comparisons[s]['delta_money_dd_krw']<0 for s in SCENARIOS))
    verify(registration['programs']);write(root/'assessment.json',dict(at_utc=datetime.now(timezone.utc).isoformat(),
        registration=bound(root/'registration.json'),accounts=rows,diagnostics=diagnostics,activity=activity,
        funding=funding,comparisons=comparisons,predictions=predictions,bindings=inputs,
        scope=protocol['scope'],economic_objective_achieved=False))
    write(root/'completion.json',dict(complete=True,native_cases=10,bridges=2,sensitivity_accounts=8,
        assessment=bound(root/'assessment.json'),independent_account_and_funding_validation=True,
        economic_decision_required=True,economic_objective_achieved=False))


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
