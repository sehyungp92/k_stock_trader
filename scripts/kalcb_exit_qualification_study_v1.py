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
from backtests.strategies.kalcb.exit_allocation_io_v1 import freeze_allocation,OVERLAY,WORKER
from backtests.strategies.kalcb.research_process_v1 import completed_dependency,execute

BASE=Path('E:/CodexResearch/k_stock_trader/kalcb_late_entry_20261011')
SCREEN=BASE/'v12_mechanisms'
DEPENDENCY=BASE/'v13_incremental'
FUNDED=Path('E:/CodexResearch/k_stock_trader/kalcb_selective_recovery_20261010/v7_funded_qualification')
PROGRAMS=('scripts/kalcb_exit_qualification_study_v1.py',WORKER,OVERLAY,
 'backtests/strategies/kalcb/exit_allocation_io_v1.py',
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
    from backtests.strategies.kalcb.late_artifact_validation_v2 import validate_native
    test=read(tests)
    if test['exit_code']!=0:raise ValueError('Focused tests failed')
    verify(test['programs']+[test['log']])
    preflight=read(test['publication_preflight']['path']);verify([test['publication_preflight'],preflight['input_profile'],preflight['check_log']])
    if not preflight['static_passed'] or preflight['native_replay']:raise ValueError('Actual frozen publication preflight required')
    old=read(SCREEN/'protocol.json');assessment=SCREEN/'assessment_v1/assessment.json'
    if not read(SCREEN/'stage_completion.json')['complete'] or not read(SCREEN/'assessment_v1/completion.json')['complete']:
        raise ValueError('Complete screening and economic assessment required')
    verify(old['programs']+old['bindings'])
    analysis=read(assessment);verify(analysis['bindings'])
    if len(analysis['accounts'])!=24 or len(analysis['bridges'])!=8:raise ValueError('Incomplete screening review')
    profiles={};bindings=[bound(SCREEN/f) for f in ('protocol.json','stage_completion.json','progress.json','bridge_gate.json')]
    bindings += [bound(assessment),bound(SCREEN/'assessment_v1/completion.json'),bound(tests),test['log'],test['publication_preflight'],preflight['input_profile'],preflight['check_log']]
    bindings += [bound(DEPENDENCY/f) for f in ('protocol.json','supervisor_start.json','preparation_review_v1.json')]
    reused={};fixed_refs={};parent_refs={};competitor_refs={};selection={}
    prior=read(DEPENDENCY/'protocol.json')
    for name in recipes('compounding'):
        profiles[name]={};reused[name]={}
        for period in PERIODS:
            item=read(SCREEN/'cases'/name/period/'input_profile.json');profiles[name][period]=item
            bindings += [item[k] for k in ('manifest','plan','policy')]
            reused[name][period]={}
            for scenario in ('baseline','combined'):
                case=SCREEN/'cases'/name/period/scenario
                validate_native(case,item,scenario,save=False)
                reused[name][period][scenario]=str(case)
                bindings += [bound(case/f) for f in (*REPLAY_FILES,'independent_account_verification.json','mechanism_evidence.json')]
                selection[name+'/'+period+'/'+scenario]=read(case/'result.json')
    for period in PERIODS:
        fixed_refs[period]={};parent_refs[period]={};competitor_refs[period]={}
        for scenario in ('baseline','participation','cost','combined'):
            ref=Path(prior['profiles']['100'][period]['references'][scenario]).parent
            if not read(ref/'independent_account_verification.json')['passed']:raise ValueError('Unaudited stress reference')
            parent_refs[period][scenario]=str(ref)
            bindings += [bound(ref/f) for f in (*REPLAY_FILES,'independent_account_verification.json')]
            competitor=Path(prior['profiles']['125'][period]['references'][scenario]).parent
            if not read(competitor/'independent_account_verification.json')['passed']:raise ValueError('Unaudited competing parent')
            competitor_refs[period][scenario]=str(competitor)
            bindings += [bound(competitor/f) for f in (*REPLAY_FILES,'independent_account_verification.json')]
        for scenario in ('baseline','combined'):
            ref=FUNDED/'cases/expanding_removal_prior100'/period/scenario
            item=read(ref.parent/'input_profile.json') if (ref.parent/'input_profile.json').exists() else None
            if item is None:
                item=read(FUNDED/'published/fixed/expanding_removal_prior100'/period/'input_profile.json')
            validate_native(ref,item,scenario,save=False)
            fixed_refs[period][scenario]=str(ref)
            bindings += [bound(ref/f) for f in (*REPLAY_FILES,'independent_account_verification.json')]
            bindings += [item[k] for k in ('manifest','plan','policy')]
    programs=[bound(REPO/p) for p in PROGRAMS]+[bound(p) for p in (REPO/'backtests/auto/shared').glob('*.py')]
    root.mkdir(parents=True,exist_ok=False)
    decision=dict(assessment=bound(assessment),selected_center='prior100_failure8',backouts=['control','prior100_failure6'],
        competing_parent='prior125 unchanged; prior125 exit variants rejected for OOS deterioration',
        rationale='Prior100 hold8 improves all four baseline/combined returns, modestly reduces OOS MTM DD, retains IS right-tail contribution in aggregate. IS DD slightly higher; activity essentially unchanged; conditional intervals include deterioration. Provisional partial finalist only.',
        limits='This qualification cannot create new profitable entry opportunities. V13 zero-admission recipes remain separate unchanged-action evidence. No claim the joint objective is achieved.',
        followup='A distinct completed pullback/reclaim entry still needs finite causal design and fresh mature full-cost labels if supported; do not lower V13 registered thresholds.',
        cases=selection,study_complete=False,canonical_replaced=False)
    write(root/'economic_decision.json',decision);bindings.append(bound(root/'economic_decision.json'))
    protocol=dict(version='kalcb-exit-qualification-v1',registered_at_utc=datetime.now(timezone.utc).isoformat(),
        profiles=profiles,reused=reused,parent_stress_references=parent_refs,fixed_parent_references=fixed_refs,competing_prior125_references=competitor_refs,
        dependency=dict(root=str(DEPENDENCY),identity=read(DEPENDENCY/'supervisor_start.json'),native_cases=32),
        programs=programs,bindings=bindings,tests=test,overlay_programs={p:bound(REPO/p) for p in (OVERLAY,WORKER)},
        runtime_root=str(Path('C:/Users/sehyu/.codex/research/kalcb_late_entry_20261011')/root.name),
        stages={mode:recipes(mode) for mode in ('compounding','identity','fixed')},
        native_cases=40,fresh_native_cases=28,authenticated_reuse=12,periods=PERIODS,
        design='24 compounding cells (12 audited baseline/combined reuses and 12 fresh participation/cost); 8 fresh disabled-overlay bridges for parent and hold8 baseline/combined; 8 funded fixed-allocation baseline/combined cells. Shared PhaseRunner owns each stage.',
        reference_krw=100_000_000.,
        benchmark='Nominal allocation reference minus invested cost. Actual funded cash/equity, original 2x intraday funding, risk state, inventory, timing and fees remain unchanged; no cash reset, external flows or post-hoc rescaling.',
        requirements='All source/account checks, all8 disabled-overlay exact bridges before fixed cases; fixed parent all4 cells must reproduce earlier audited funded parent. Fresh compounding stress parents must reproduce earlier parent stress paths. Same complete parent exit cascade and direct actual filled support.',
        remaining='Joint stress and funded-regime candidate-parent deltas; compare competing prior125; retain hold6/hold0 stability backouts, right tails, support, month/issuer concentration, MTM episodes, paired circular5/10-session10000 seed20261011 conditional uncertainty. Truthful existing-engine full diagnostics remain required.',
        limits='Reused April-July development, no independent or multiplicity-adjusted validation; 2/5/10pp sacrifice reporting bands and DD8/10 preferences. No automatic promotion or live enablement.')
    write(root/'protocol.json',protocol)
    study=dict(code=str(REPO),source_generation_hash=digest(root/'protocol.json'),
        bound_inputs=[bound(root/'protocol.json')],source_programs=programs,
        scope=protocol['remaining'],study_complete=False)
    write(root/'study.json',{**study,'stages':protocol['stages']})
    for mode,spec in protocol['stages'].items():write(root/mode/'study.json',{**study,'mode':mode,'recipes':spec})
    print(dict(registered=str(root),fresh=28,reused=12),flush=True)


def assert_no_other_native():
    import psutil
    for proc in psutil.process_iter(['pid','cmdline']):
        if proc.pid==os.getpid():continue
        args=proc.info['cmdline'] or []
        if any('kalcb_' in a and ('worker_' in a or a=='scripts.kalcb_executed_value_replay') for a in args):
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
