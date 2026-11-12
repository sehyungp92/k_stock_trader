"""Finite opening-exit stress and funded-allocation qualification via PhaseRunner."""
import argparse
from datetime import datetime,timezone
import os
from pathlib import Path
import sys
import time
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from backtests.strategies.kalcb.retention_research_io_v1 import read,write,bound,digest,verify,link,PERIODS
from backtests.strategies.kalcb.qualified_artifacts import REPLAY_FILES
from backtests.strategies.kalcb.exit_allocation_io_v2 import freeze_allocation,OVERLAY,WORKER
from backtests.strategies.kalcb.research_process_v1 import completed_dependency,execute

BASE=Path('E:/CodexResearch/k_stock_trader/kalcb_late_entry_20261011')
SCREEN=BASE/'v12_mechanisms'
DEPENDENCY=BASE/'v15_incremental'
FUNDED=Path('E:/CodexResearch/k_stock_trader/kalcb_selective_recovery_20261010/v7_funded_qualification')
PROGRAMS=('scripts/kalcb_exit_qualification_study_v2.py',WORKER,OVERLAY,
 'backtests/strategies/kalcb/exit_allocation_io_v2.py',
 'backtests/strategies/kalcb/exit_qualification_plugin_v1.py',
 'backtests/strategies/kalcb/research_process_v1.py',
 'backtests/strategies/kalcb/fixed_allocation_funding_v1.py',
 'backtests/strategies/kalcb/late_artifact_validation_v2.py',
 'backtests/strategies/kalcb/late_artifact_validation_v1.py',
 'backtests/strategies/kalcb/retention_research_io_v1.py',
 'backtests/strategies/kalcb/population_research_io_v1.py',
 'backtests/strategies/kalcb/population_qualification_plugin_v1.py',
 'backtests/strategies/kalcb/selective_recovery_plugin_v1.py',
 'backtests/strategies/kalcb/oos_ablation.py',
 'backtests/strategies/kalcb/qualified_artifacts.py',
 'tests/backtests/strategies/test_kalcb_exit_qualification_v1.py')


def recipes(mode):
    if mode not in ('compounding','identity','fixed'):raise ValueError('Undeclared stage')
    holds=(0,6,8) if mode=='compounding' else (0,8)
    scenarios=['baseline','participation','cost','combined'] if mode=='compounding' else ['baseline','combined']
    return {('control' if h==0 else f'prior100_failure{h}'):
        dict(prior=100,hold=h,parent='expanding',scenarios=scenarios,config={}) for h in holds}


def register(root,tests):
    from backtests.strategies.kalcb.incremental_storage_reuse_v1 import require_closed
    import shutil
    oldroot=BASE/'v14_exit_qualification';require_closed(oldroot)
    old=read(oldroot/'protocol.json');test=read(tests)
    if read(oldroot/'status.json')['stage']!='attention_required' or (oldroot/'progress.json').exists():raise ValueError('Unexpected qualification state')
    if test['exit_code']!=0:raise ValueError('Tests failed')
    verify(test['programs']+[test['log']]+old['programs']+old['bindings'])
    if shutil.disk_usage(root.parent).free<10*1024**3:raise ValueError('External capacity unavailable')
    dep=BASE/'v15_incremental';identity=read(dep/'supervisor_start.json')
    from backtests.strategies.kalcb.research_process_v1 import exact_process
    if not exact_process(identity):raise ValueError('Successor dependency not healthy')
    if read(dep/'status.json').get('stage')=='attention_required':raise ValueError('Dependency failed')
    programs=old['programs']+[bound(REPO/x) for x in (
        'scripts/kalcb_exit_qualification_study_v2.py',
        'backtests/strategies/kalcb/exit_allocation_io_v2.py',
        'backtests/strategies/kalcb/late_entry_io_v5.py',
        'tests/backtests/strategies/test_kalcb_storage_successor_v1.py')]
    root.mkdir(parents=True,exist_ok=False)
    for f in ('economic_decision.json',):link(oldroot/f,root/f)
    old.update(version='kalcb-exit-qualification-storage-v2',registered_at_utc=datetime.now(timezone.utc).isoformat(),
        programs=programs,tests=test,runtime_root=str(root/'runtime'),
        dependency=dict(root=str(dep),identity=identity,native_cases=32),
        bindings=old['bindings']+[bound(oldroot/f) for f in ('protocol.json','status.json','background.err.log','supervisor_start.json')]+[bound(dep/'protocol.json'),bound(dep/'supervisor_start.json'),bound(tests)],
        storage_repair='V14 exited before any native case when V13 disk failure closed its dependency. Same40-cell qualification. V15 storage successor dependency; external runtime/content-cache/temp; no economic changes.')
    write(root/'protocol.json',old)
    study=dict(code=str(REPO),source_generation_hash=digest(root/'protocol.json'),bound_inputs=[bound(root/'protocol.json')],source_programs=programs,scope=old['remaining'],study_complete=False)
    write(root/'study.json',{**study,'stages':old['stages']})
    for mode,spec in old['stages'].items():write(root/mode/'study.json',{**study,'mode':mode,'recipes':spec})
    print(dict(registered=str(root),fresh=28,reused=12),flush=True)


def assert_no_other_native():
    import psutil
    for proc in psutil.process_iter(['pid','cmdline']):
        if proc.pid==os.getpid():continue
        for a in proc.info['cmdline'] or []:
            if a=='scripts.kalcb_executed_value_replay' or (a.startswith('scripts.kalcb_') and '_worker_v' in a and ' ' not in a) or (Path(a).name.startswith('kalcb_') and '_worker_v' in Path(a).name and Path(a).suffix=='.py'):
                raise ValueError('Another KALCB native worker exists')


def run(root):
    import psutil,msvcrt
    from backtests.strategies.kalcb.exit_qualification_plugin_v1 import ExitQualificationPlugin
    from backtests.strategies.kalcb.late_artifact_validation_v2 import validate_native
    from backtests.strategies.kalcb.fixed_allocation_funding_v1 import funding_review
    from backtests.strategies.kalcb.fixed_allocation_reference_v1 import verify_identity_bridge
    from backtests.auto.shared.phase_runner import PhaseRunner
    from backtests.auto.shared.round_manager import RoundManager
    p=read(root/'protocol.json');verify(p['programs']+p['bindings'])
    temp=root/'temporary';temp.mkdir(exist_ok=True);os.environ.update(TEMP=str(temp),TMP=str(temp))
    lock=(root/'run.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    if (root/'supervisor_start.json').exists():raise ValueError('Existing attempt must be preserved')
    state=dict(pid=os.getpid(),process_created=psutil.Process().create_time(),command=psutil.Process().cmdline())
    write(root/'supervisor_start.json',state)
    progress=[];authenticated=set()
    def status(stage,**kw):write(root/'status.json',{**state,'stage':stage,'completed':len(progress),'child':None,**kw})
    try:
        dependency=p['dependency'];status('waiting_exact_incremental_completion',dependency=dependency)
        while not completed_dependency(dependency['root'],dependency['identity'],32):time.sleep(15)
        write(root/'dependency_gate.json',dict(passed=True,completion=bound(Path(dependency['root'])/'stage_completion.json'),identity=dependency['identity']))
        for mode in ('compounding','identity','fixed'):
            def native(name,period,scenario):
                assert_no_other_native()
                old=p['profiles'][name][period];out=root/mode/'cases'/name/period/scenario
                if out.exists():raise ValueError('Existing case must be preserved')
                reused=mode=='compounding' and scenario in ('baseline','combined')
                if mode=='fixed' and sum(r['mode']=='identity' and r['exact_bridge'] for r in progress)!=8:
                    raise ValueError('Eight disabled-overlay bridges required')
                if mode=='compounding' and name!='control' and sum(r['mode']=='compounding' and r['name']=='control' for r in progress)!=8:
                    raise ValueError('Full parent stress qualification required first')
                status('authenticating_reuse' if reused else 'publishing_'+mode,case=[name,period,scenario])
                bridge=None
                if mode=='compounding':
                    item=old
                    if name=='control':bridge=Path(p['parent_stress_references'][period][scenario])
                else:
                    if mode=='identity':bridge=Path(p['reused'][name][period][scenario])
                    # Distinct scenario identity contracts prevent cache reuse against the wrong reference.
                    publication=name+'_'+scenario if mode=='identity' else name
                    item=freeze_allocation(root,publication,period,old,p['overlay_programs'],authenticated,
                        mode=mode,bridge=bridge)
                if reused:
                    source=Path(p['reused'][name][period][scenario]);validate_native(source,item,scenario,save=False)
                    for f in source.iterdir():
                        if f.is_file() and f.name!='independent_account_verification.json':link(f,out/f.name)
                    log=None
                else:
                    code=Path(item['manifest']['path']).parent/'files'
                    if mode=='compounding':
                        command=[sys.executable,'-B','-u','-m','scripts.kalcb_mechanism_worker_v2',
                            '--alternatives',str(code/'research/mechanism/alternatives.json')]
                    else:command=[sys.executable,'-B','-u',str(code/WORKER),'--code',str(code),'--mode',mode,'--reference','100000000.0']
                    command+=['--plan',item['plan']['path'],'--policies',str(Path(item['policy']['path']).parent),
                        '--audit-manifest',item['manifest']['path'],'--scenario',scenario,'--output',str(out)]
                    if bridge is not None:command+=['--bridge',str(bridge)]
                    log=execute(root,command,'native_'+mode+'_'+name+'_'+period+'_'+scenario,code,state)
                metrics=validate_native(out,item,scenario)
                exact=False
                if bridge is not None:
                    proof=verify_identity_bridge(out,bridge)
                    # Never overwrite a hardlinked original receipt.
                    write(out/'qualification_identity_bridge.json',proof);exact=proof['passed']
                if mode=='fixed' and name=='control':
                    proof=verify_identity_bridge(out,Path(p['fixed_parent_references'][period][scenario]))
                    write(out/'funded_parent_bridge.json',proof);exact=proof['passed']
                write(out/'qualification_input_profile.json',item)
                write(out/'funding_review.json',funding_review(out))
                row=dict(mode=mode,name=name,period=period,scenario=scenario,metrics=metrics,authenticated_reuse=reused,
                    exact_bridge=exact,log=log,result=bound(out/'result.json'),input_profile=bound(out/'qualification_input_profile.json'),
                    funding=bound(out/'funding_review.json'),evidence=bound(out/'mechanism_evidence.json'))
                progress.append(row);write(root/'progress.json',progress);status('case_audited',case=[mode,name,period,scenario])
                print(row,flush=True);return read(out/'result.json')
            plugin=ExitQualificationPlugin(root/mode/'study.json',native)
            manager=RoundManager('stock',plugin.name,base_dir=root/mode/'framework')
            PhaseRunner(plugin,manager.get_round_dir(1),round_name='Opening exit '+mode+' qualification',
                round_manager=manager,round_num=1,max_rounds=1,min_delta=.00001,max_retries=0,max_diagnostic_retries=0).run_all_phases()
        if len(progress)!=40 or sum(not r['authenticated_reuse'] for r in progress)!=28:raise ValueError('Incomplete finite design')
        verify(p['programs']+p['bindings'])
        write(root/'stage_completion.json',dict(complete=True,native_cases=40,fresh_native_cases=28,authenticated_reuse=12,
            study_complete=False,next=p['remaining']))
        status('native_qualification_complete_joint_assessment_pending')
    except Exception as error:status('attention_required',error=repr(error));raise
    finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('register','run'))
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--tests',type=Path)
    args=parser.parse_args()
    if args.stage=='register':register(args.root.resolve(),args.tests)
    else:run(args.root.resolve())
