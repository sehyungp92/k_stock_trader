"""Register a bounded action-evidence study; execute only through PhaseRunner."""
import argparse
from datetime import datetime,timezone
import os
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from backtests.strategies.kalcb.retention_research_io_v1 import read,write,bound,digest,verify
from backtests.strategies.kalcb.holding_account_io_v2 import OVERLAYS,freeze
from backtests.strategies.kalcb.holding_action_v1 import ValueContract,FEATURES
from backtests.strategies.kalcb.holding_account_model_v1 import MODEL_SPEC
from backtests.strategies.kalcb.late_artifact_validation_v2 import validate_native
from scripts.kalcb_opening_account_study_v1 import activity
from dataclasses import asdict
from backtests.strategies.kalcb.holding_account_recovery_v2 import (authenticate_predecessor,reuse_complete,verify_terminal_transition,check_successor_sources)

PARENT=Path('E:/CodexResearch/k_stock_trader/opening_account_20261013/study_v3')
SCENARIOS=('baseline','participation','cost','combined')
EXTRA=('backtests/strategies/kalcb/holding_account_io_v1.py','backtests/strategies/kalcb/holding_account_io_v2.py',
    'backtests/strategies/kalcb/holding_account_recovery_v2.py','tests/test_kalcb_holding_terminal_v2.py',
    'tests/test_kalcb_holding_recovery_v2.py',
    'backtests/strategies/kalcb/holding_account_plugin_v1.py','scripts/kalcb_holding_account_study_v2.py',
    'scripts/kalcb_holding_account_review_v1.py',
    'tests/test_kalcb_holding_account_v1.py','tests/test_kalcb_holding_model_v1.py',
    'tests/test_kalcb_holding_framework_v1.py')


def programs():
    files=set(OVERLAYS+EXTRA)
    files.update('backtests/strategies/kalcb/'+x+'.py' for x in ('account_decision_v1','account_history_v1',
        'pending_entry_account_pilot_v1','opening_account_plugin_v3','opening_account_plugin_v1','oos_ablation',
        'retention_research_io_v1','late_entry_io_v5','research_process_v1','late_artifact_validation_v2',
        'qualified_artifacts','population_research_io_v1'))
    files.add('scripts/kalcb_opening_account_study_v1.py')
    files.update(x.relative_to(REPO).as_posix() for x in (REPO/'backtests/auto/shared').glob('*.py'))
    return [bound(REPO/x) for x in sorted(files)]


def register(root,tests,preflight,predecessor):
    if root.exists():raise ValueError('Preserve previous research; use a versioned successor')
    recovery=authenticate_predecessor(predecessor)
    test=read(tests);pre=read(preflight)
    if test.get('exit_code')!=0 or test.get('tests',0)<46:raise ValueError('Complete passed validity tests required')
    verify(test['programs']+[test['log']])
    if not pre.get('passed') or not pre.get('terminal_audit_regression_passed') or pre.get('rows',0)<2:raise ValueError('Audited native branch/validator preflight required')
    verify(pre['bindings'])
    verify([pre['terminal_identity_diagnosis'],pre['source_change_evidence']])
    recovery.update(terminal_identity_diagnosis=pre['terminal_identity_diagnosis'],
                    source_change_evidence=pre['source_change_evidence'])
    # Only the exercised native services must be identical to this preflight.
    verify(pre['exercised_programs'])
    profiles={};references={};bindings=[bound(tests),bound(preflight)]
    for period in ('is','oos'):
        profiles[period]=read(PARENT/'cases/control'/period/'baseline/input_profile.json')
        verify([profiles[period][k] for k in ('manifest','plan','policy')]);references[period]={}
        for scenario in SCENARIOS:
            folder=PARENT/'cases/control'/period/scenario
            item=read(folder/'input_profile.json');validate_native(folder,item,scenario,save=False)
            references[period][scenario]=str(folder)
            bindings.extend(bound(folder/f) for f in ('input_profile.json','completion.json','result.json','fills.json','equity.json','daily.json','execution_audit.json'))
    root.mkdir(parents=True,exist_ok=False);overlays={}
    for rel in OVERLAYS:
        path=root/'overlays'/rel;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes((REPO/rel).read_bytes());overlays[rel]=bound(path)
    old=read(PARENT/'protocol.json')
    protocol={'version':'native-remaining-account-action-v2-terminal-audit-repair','recovery':recovery,'registered_at':datetime.now(timezone.utc).isoformat(),
        'overlays':overlays,'programs':programs(),'profiles':profiles,'references':references,'bindings':bindings,
        'databases':old['databases'],'database_bindings':old['database_bindings'],
        'recipes':['control','selective'],'scenarios':SCENARIOS,'native_accounts_min':8,'native_accounts_max':16,
        'sampling':'Each material funded opening holding every third completed holding bar, starting at bar2. '
            'No existing native exit, no nonmaterial/integer alias, continuous-session execution only. '
            'All dates and observed states; eligibility never uses future outcomes.',
        'actions':['hold','half_current_integer_quantity','full_current_integer_quantity'],
        'continuation':'One executable action then unchanged native parent through EOD. Pending buy remainder cancelled before selling. '
            'Actual cash/fees/funding/descendants retained; exact disabled EOD state checked. No sum of local labels as returns.',
        'value_contract':asdict(ValueContract()),'features':FEATURES,'model_spec':MODEL_SPEC,
        'learning':'Separate scenario and action targets; three leave-contiguous-session-block-out models. '
            'Equal total session weights, at least60 completed training sessions plus1 full embargo, refit every20. '
            'Expand IS only; freeze mature final IS fit for all reused OOS. '
            'Predict actual account wealth delta and inherited-peak path DD separately. '
            'Act only if all models predict positive wealth and nonworsening DD, within observed feature ranges '
            'and the95th percentile six-neighbor joint-state support distance. One action per lifetime and one per clock.',
        'readiness':'IS past-only out-of-time actions require20 distinct lifetimes on10 sessions in EACH stress; '
            'circular5-active-session block bootstrap(2000 draws,seed1301),5th percentile mean action wealth>0 '
            'and95th percentile mean action DD change<=0. Report unconditional half/full and largest harmful actions. '
            'This is an evidence gate, not a return estimate or independent holdout. Failed gate skips policy accounts.',
        'objective':{'minimum_IS_return_retention_each_stress':.98,'minimum_OOS_return_gain_pp_each_stress':1.,
            'maximum_DD_deterioration_pp':0.,'minimum_material_activity_retention_each_period_stress':.95},
        'coverage_limit':'Causal warmup preserves early IS, including early DD episodes. No claim to have repaired those '
            'without earlier mature native labels; OOS is reused development. No live/canonical modification.',
        'failure_policy':'No retries, silent overwrite, widened gate or parameter grid. Preserve technical failures; version repairs.'}
    write(root/'protocol.json',protocol);write(root/'progress.json',[])
    check_successor_sources(root)
    write(root/'study.json',{'code':str(REPO),'source_generation_hash':digest(root/'protocol.json'),
        'bound_inputs':[bound(root/'protocol.json')],'source_programs':protocol['programs'],
        'recipes':{n:{'holding_policy':n} for n in protocol['recipes']},'periods':{'is':{'sessions':156},'oos':{'sessions':69}}})


def run(root):
    import psutil
    from backtests.auto.shared.phase_runner import PhaseRunner
    from backtests.auto.shared.round_manager import RoundManager
    from backtests.strategies.kalcb.holding_account_plugin_v1 import HoldingAccountPlugin
    from backtests.strategies.kalcb.research_process_v1 import execute
    p=read(root/'protocol.json');verify(p['programs']+p['bindings']+p['database_bindings']);check_successor_sources(root)
    me=psutil.Process();write(root/'supervisor_start.json',{'pid':me.pid,'process_created':me.create_time(),'command':me.cmdline()})
    for other in psutil.process_iter(['pid','name','cmdline']):
        cmd=other.info['cmdline'] or []
        if other.pid!=me.pid and 'python' in (other.info['name'] or '').lower() and any('kalcb_' in x for x in cmd) and any(x in ('run','worker','pipeline','produce','labels') for x in cmd):
            raise ValueError('Another native research producer is active')
    (root/'temp').mkdir(exist_ok=True);os.environ.update(TEMP=str(root/'temp'),TMP=str(root/'temp'))
    def native_case(name,period,scenario):
        verify(p['programs']);out=root/'cases'/name/period/scenario
        key='/'.join((name,period,scenario))
        reused=p['recovery']['completed_cases'].get(key)
        if reused:
            reuse_complete(root,reused,out,scenario)
            item=read(out/'input_profile.json')
        else:
            item=freeze(root,period,p['profiles'][period])
            if out.exists():raise ValueError('Do not overwrite or reuse unregistered native output')
            code=Path(item['manifest']['path']).parent/'files'
            command=[sys.executable,'-B','-u','-m','scripts.kalcb_holding_account_worker_v2','--plan',item['plan']['path'],
                '--policies',str(Path(item['policy']['path']).parent),'--audit-manifest',item['manifest']['path'],
                '--output',str(out),'--scenario',scenario,'--mode','labels' if name=='control' else 'policy']
            if name=='control':command+=['--bridge',p['references'][period][scenario]]
            else:command+=['--models',str(root/'models')]
            execute(root,command,f'{name}_{period}_{scenario}',code,{'recipe':name,'period':period,'scenario':scenario,'study_complete':False})
            write(out/'input_profile.json',item);validate_native(out,item,scenario)
            verify_terminal_transition(out)
        if name=='control' and not read(out/'identity_bridge.json')['passed']:raise ValueError('Control bridge failed')
        result=read(out/'result.json');a=activity(out);write(out/'activity.json',a)
        result.update(a,result_binding=bound(out/'result.json'))
        rows=read(root/'progress.json');rows.append({'name':name,'period':period,'scenario':scenario,
            'result':bound(out/'result.json'),'return_pct':result['return_pct'],'drawdown_pct':result['drawdown_pct'],
            'holding':read(out/'holding_complete.json'),'activity':a});write(root/'progress.json',rows)
        print({'complete':name,'period':period,'scenario':scenario,'holding':read(out/'holding_complete.json')},flush=True)
        return result
    plugin=HoldingAccountPlugin(root/'study.json',native_case);manager=RoundManager('stock',plugin.name,base_dir=root/'framework')
    state=PhaseRunner(plugin,manager.get_round_dir(1),round_name='Executable remaining-account action value',
        round_manager=manager,round_num=1,max_rounds=1,min_delta=.00001,max_retries=0,max_diagnostic_retries=0).run_all_phases()
    if set(state.completed_phases)!={1}:raise ValueError('Shared framework did not finalize the finite study')
    verify(p['programs']+p['bindings']+p['database_bindings'])
    check_successor_sources(root)
    for record in read(root/'progress.json'):
        key='/'.join((record['name'],record['period'],record['scenario']))
        if key not in p['recovery']['completed_cases']:
            verify_terminal_transition(root/'cases'/record['name']/record['period']/record['scenario'])
    count=len(read(root/'progress.json'));ready=read(root/'models/manifest.json')['economic_readiness_pass']
    if count!=(16 if ready else 8):raise ValueError('Wrong complete native matrix')
    write(root/'completion.json',{'complete':True,'native_accounts':count,'economic_readiness':ready,
        'economic_objective':plugin.diagnostic()['objective_achieved_on_reused_development'],
        'independent_final_assessment_complete':False,'canonical_changed':False})
    write(root/'status.json',{'stage':'execution_complete_independent_assessment_pending','complete':True})


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('mode',choices=('register','run'))
    ap.add_argument('--root',type=Path,required=True);ap.add_argument('--tests',type=Path);ap.add_argument('--preflight',type=Path);ap.add_argument('--predecessor',type=Path)
    args=ap.parse_args()
    if args.mode=='register':register(args.root,args.tests,args.preflight,args.predecessor)
    else:
        try:run(args.root)
        except BaseException as error:
            write(args.root/'supervisor_failure.json',{'error':repr(error),'preserve_evidence':True})
            raise
