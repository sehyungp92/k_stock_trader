"""Finite funded allocation benchmark through the existing shared PhaseRunner."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from backtests.strategies.kalcb.retention_research_io_v1 import read,write,bound,verify,env,validate_native,PERIODS
from backtests.strategies.kalcb.fixed_allocation_research_io_v1 import freeze_allocation,OVERLAY,WORKER
from scripts.kalcb_population_qualification_v1 import commit_available


def recipes():
    return {name:{'parent':parent,'prior':prior,'scenarios':['baseline','combined'],'config':{}}
        for name,parent,prior in [('control','expanding',None),('control_expanding_atr5','expanding_atr5',None),
            ('expanding_removal_prior100','expanding',100),('expanding_atr5_removal_prior100','expanding_atr5',100)]}


def register(root,qualification,assessment,tests):
    q=Path(qualification);old=read(q/'protocol.json');done=read(q/'stage_completion.json');test=read(tests)
    if not done['complete'] or done['fresh_native_cases']!=32 or test['exit_code']!=0:
        raise ValueError('Qualification or test gate incomplete')
    verify(old['programs']+old['bindings']+test['programs']+[test['log'],test['publication_preflight']])
    preflight=read(test['publication_preflight']['path'])
    if not preflight['static_passed'] or preflight['native_replay']:
        raise ValueError('Static publication preflight missing or misclassified')
    verify([preflight['input_profile'],preflight['check_log']])
    analysis=read(assessment)
    if len(analysis['trials'])!=8:raise ValueError('Missing finite-prior assessment')
    inputs=old['bindings']+[bound(q/f) for f in ('protocol.json','study.json','stage_completion.json','qualification_ledger.json')]
    inputs += [bound(assessment),bound(tests),test['log'],test['publication_preflight'],preflight['input_profile'],preflight['check_log']]
    profiles={};parents={(p['name'],p['period']):p for p in old['parent_profiles']}
    for name,spec in recipes().items():
        profiles[name]={}
        for period in ('is','oos'):
            item=(parents[spec['parent'],period] if spec['prior'] is None else
                read(Path(old['screening_root'])/'cases'/(spec['parent']+'_population_removals_only')/period/'input_profile.json'))
            profiles[name][period]=item
            inputs += [item[k] for k in ('manifest','plan','policy')]
            for scenario in ('baseline','combined'):
                folder=q/'cases'/name/period/scenario
                validate_native(folder,item,scenario,save=False)
                inputs += [bound(folder/f) for f in ('result.json','request.json','daily.json','equity.json','fills.json','independent_account_verification.json')]
    programs={p['path']:p for p in old['programs']}
    for relative in (OVERLAY,WORKER,'scripts/kalcb_fixed_allocation_study_v1.py',
        'scripts/kalcb_removal_qualification_assessment_v1.py','scripts/kalcb_population_assessment_v1.py',
        'backtests/strategies/kalcb/fixed_allocation_research_io_v1.py',
        'backtests/strategies/kalcb/fixed_allocation_funding_v1.py','backtests/strategies/kalcb/fixed_allocation_plugin_v1.py',
        'tests/backtests/strategies/test_kalcb_fixed_allocation_reference_v1.py',
        'tests/backtests/strategies/test_kalcb_fixed_allocation_packaging_v1.py'):
        p=REPO/relative;programs[str(p.resolve())]=bound(p)
    root.mkdir(parents=True,exist_ok=False)
    protocol={'registered_at_utc':datetime.now(timezone.utc).isoformat(),'qualification':str(q),
        'recipes':recipes(),'profiles':profiles,'programs':list(programs.values()),'bindings':inputs,
        'overlay_programs':{p:bound(REPO/p) for p in (OVERLAY,WORKER)},'tests':test,'periods':PERIODS,
        'reference_krw':100_000_000.,'native_cases':20,
        'design':'Four unchanged parent baseline identity bridges, then sixteen fixed-reference baseline/combined IS/OOS cases on expanding and ATR5 parents and their prior100 removals. This tests the already selected centers, not a newly optimized budget or prior.',
        'hypothesis':'Removal benefit persists when nominal allocation capacity cannot reinvest earlier profits. Compare candidate-parent deltas within each capital regime. Actual cash, borrow limits, inventory, risk state, next-bar fills and costs remain native.',
        'scope':'Fixed nominal allocation reference minus invested cost; no cash injection/reset, no externally replenished constant balance. Continuous market/model/label history and original fitting clocks; reused development only. No automatic promotion.',
        'remaining':'Joint assessment of all four center stresses and finite priors, source/account/funding diagnostics, concentration and conditional uncertainty; existing-script final report. Any follow-up requires finite evidence-supported registration.'}
    write(root/'protocol.json',protocol)
    write(root/'study.json',{'recipes':recipes(),'controls':{'control':'expanding','control_expanding_atr5':'expanding_atr5'},
        'code':str(REPO),'source_generation_hash':bound(root/'protocol.json')['sha256'],
        'global_IS_reference':{'baseline':73.1303145472,'combined':47.8121465627},
        'bound_inputs':[bound(root/'protocol.json')],'source_programs':list(programs.values())})
    print(json.dumps({'registered':str(root),'native_cases':20}),flush=True)


def run(root):
    import psutil,msvcrt
    from backtests.strategies.kalcb.fixed_allocation_plugin_v1 import FixedAllocationPlugin
    from backtests.strategies.kalcb.fixed_allocation_funding_v1 import funding_review
    from backtests.auto.shared.phase_runner import PhaseRunner
    from backtests.auto.shared.round_manager import RoundManager
    protocol=read(root/'protocol.json');verify(protocol['programs']+protocol['bindings'])
    for p in psutil.process_iter(['pid','cmdline']):
        args=p.info['cmdline'] or []
        if p.pid==os.getpid():continue
        workers=('kalcb_population_worker_v1.py','kalcb_population_removal_worker_v1.py','kalcb_retention_worker_v1.py','kalcb_fixed_allocation_worker_v1.py')
        supervisors=('kalcb_population_qualification_v1.py','kalcb_population_research_v1.py','kalcb_fixed_allocation_study_v1.py')
        if any(str(a).endswith(workers) for a in args) or 'scripts.kalcb_executed_value_replay' in args:
            raise ValueError('Another KALCB native worker is active')
        if 'run' in args and any(str(a).endswith(supervisors) for a in args):raise ValueError('Another KALCB supervisor is active')
    lock=(root/'run.lock').open('a+b');lock.seek(0)
    if not lock.read(1):lock.write(b'0');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    if (root/'progress.json').exists() or (root/'identity').exists():raise ValueError('Preserve existing run; use versioned successor')
    state={'pid':os.getpid(),'process_created':psutil.Process().create_time(),'command':sys.argv}
    progress=[];authenticated=set();bridges=[]
    def status(stage,**kwargs):
        state.update(stage=stage,updated_at_utc=datetime.now(timezone.utc).isoformat(),**kwargs);write(root/'status.json',state)
    def execute(name,period,scenario,mode):
        verify(protocol['programs']+protocol['bindings'])
        old=protocol['profiles'][name][period]
        bridge=Path(protocol['qualification'])/'cases'/name/period/'baseline' if mode=='identity' else None
        status('publishing_'+mode,case=name+'/'+period+'/'+scenario,child=None,completed=len(progress))
        item=freeze_allocation(root,name,period,old,protocol['overlay_programs'],authenticated,
            mode=mode,reference=protocol['reference_krw'],bridge=bridge)
        code=Path(item['folder'])/'inputs/files';target=Path(item['folder'])/scenario
        if target.exists():raise ValueError('Preserve existing native output')
        while psutil.virtual_memory().available<384*1024**2 or commit_available()<1280*1024**2:
            status('waiting_for_measured_headroom',physical_available=psutil.virtual_memory().available,commit_available=commit_available());time.sleep(15)
        command=[sys.executable,'-B','-u',str(code/WORKER),'--code',str(code),'--plan',item['plan']['path'],
            '--policies',str(Path(item['policy']['path']).parent),'--audit-manifest',item['manifest']['path'],
            '--output',str(target),'--scenario',scenario,'--mode',mode,'--reference',str(protocol['reference_krw'])]
        if bridge:command += ['--bridge',str(bridge)]
        log=target.with_suffix('.log');started=time.monotonic()
        with log.open('x',encoding='utf-8') as stream:
            child=subprocess.Popen(command,cwd=code,env=env(code),stdout=stream,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            identity={'pid':child.pid,'process_created':psutil.Process(child.pid).create_time(),'command':command}
            status('native_'+mode,case=name+'/'+period+'/'+scenario,child=identity)
            if not (root/'launch_receipt.json').exists():write(root/'launch_receipt.json',{'verified':True,'supervisor':dict(state),'tests':protocol['tests'],'study_complete':False})
            while child.poll() is None:
                try:
                    sample={'at_utc':datetime.now(timezone.utc).isoformat(),'identity':identity,
                        'memory':psutil.Process(child.pid).memory_info()._asdict(),'physical_available':psutil.virtual_memory().available,'commit_available':commit_available()}
                    with (root/'resource_observations.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(sample)+'\n')
                except psutil.NoSuchProcess:pass
                time.sleep(10)
        row={'name':name,'period':period,'scenario':scenario,'mode':mode,'seconds':time.monotonic()-started,'returncode':child.returncode,'log':bound(log)}
        if child.returncode:write(root/'failed_case.json',row);raise RuntimeError('Native fixed-allocation case failed')
        row['metrics']=validate_native(target,item,scenario)
        write(target/'funding_review.json',funding_review(target))
        if mode=='identity':
            if not read(target/'identity_bridge.json')['passed']:raise ValueError('Identity native bridge failed')
            bridges.append(bound(target/'identity_bridge.json'))
        row.update(result=bound(target/'result.json'),funding=bound(target/'funding_review.json'),allocation=bound(target/'allocation_reference.json'))
        progress.append(row);write(root/'progress.json',progress);status('case_completed',child=None,completed=len(progress));print(json.dumps(row),flush=True)
        return read(target/'result.json')
    def native(name,period,scenario):
        if not bridges:
            for parent in ('control','control_expanding_atr5'):
                for p in ('is','oos'):execute(parent,p,'baseline','identity')
        if len(bridges)!=4:raise ValueError('All four native identity bridges required')
        return execute(name,period,scenario,'fixed')
    try:
        plugin=FixedAllocationPlugin(root/'study.json',native)
        manager=RoundManager('stock',plugin.name,base_dir=root/'framework')
        PhaseRunner(plugin,manager.get_round_dir(1),round_name='Funded fixed allocation comparison',round_manager=manager,
            round_num=1,max_rounds=1,min_delta=.00001,max_retries=0,max_diagnostic_retries=0).run_all_phases()
        verify(protocol['programs']+protocol['bindings'])
        write(root/'stage_completion.json',{'complete':True,'native_cases':len(progress),'identity_bridges':bridges,'study_complete':False,'next':protocol['remaining']})
        status('awaiting_final_assessment',child=None)
    except Exception as error:status('attention_required',error=repr(error));raise
    finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('register','run'))
    for key in ('root','qualification','assessment','tests'):p.add_argument('--'+key,type=Path,required=key=='root')
    a=p.parse_args()
    if a.stage=='register':register(a.root,a.qualification,a.assessment,a.tests)
    else:run(a.root)
