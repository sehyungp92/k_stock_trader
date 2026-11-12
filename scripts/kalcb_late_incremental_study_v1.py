"""Finite registered incremental selection experiment; one shared native lifecycle."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import sys
import time
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from backtests.strategies.kalcb.retention_research_io_v1 import read,write,bound,verify,digest,link,PERIODS
from scripts.kalcb_late_signal_study_v2 import execute
from scripts.kalcb_mechanism_assessment_v1 import exact_process,check_gate

BASE=Path('E:/CodexResearch/k_stock_trader/kalcb_late_entry_20261011')
PROGRAMS=('scripts/kalcb_late_incremental_study_v1.py','scripts/kalcb_late_incremental_producer_v1.py',
 'backtests/strategies/kalcb/late_incremental_v1.py','backtests/strategies/kalcb/late_incremental_validation_v1.py',
 'backtests/strategies/kalcb/late_incremental_plugin_v1.py',
 'tests/backtests/strategies/test_kalcb_late_incremental_v1.py',
 'scripts/kalcb_late_signal_study_v2.py','scripts/kalcb_mechanism_assessment_v1.py')


def recipes():
    result={n:dict(prior=p,quantity=0,flat_only=False,parent=p,config={},signal='none')
        for n,p in [('control',100),('control_prior125',125)]}
    for prior in (100,125):
        for kind in ('aligned48','reserve','aligned_reserve'):
            result[f'prior{prior}_{kind}']=dict(prior=prior,quantity=.25,flat_only=False,parent=prior,config={},signal=kind)
    return result


def dependency_gate(protocol):
    dep=protocol['dependency'];root=Path(dep['native_root']);out=Path(dep['assessment_root'])
    if not check_gate(root,dep['native_supervisor']):return False
    status=read(out/'status.json')
    if status.get('stage')=='attention_required':raise RuntimeError('Existing assessment failed; preserve and reassess')
    if exact_process(dep['assessment_supervisor']):return False
    if not (out/'completion.json').exists():raise RuntimeError('Assessment exited without completion')
    done=read(out/'completion.json')
    if not done.get('complete'):raise ValueError('Incomplete prerequisite assessment')
    verify([done['assessment']])
    return True


def register(root,tests):
    from backtests.strategies.kalcb.late_incremental_v1 import RULE
    old=read(BASE/'v10_signal/protocol.json');test=read(tests);active=BASE/'v12_mechanisms'
    if test['exit_code']!=0:raise ValueError('Tests must pass')
    verify(test['programs']+[test['log']]+old['programs']+old['input_bindings']+old['data_bindings'])
    if not read(BASE/'v10_signal/stage_completion.json')['complete']:raise ValueError('Old signals incomplete')
    progress=read(active/'progress.json');identities=[r for r in progress if r['is_identity']]
    if len(identities)!=8 or not read(active/'bridge_gate.json')['passed']:raise ValueError('Eight native observer bridges required')
    bindings=[];support=[]
    for row in identities:
        verify([row['evidence'],row['result']]);bindings.extend([row['evidence'],row['result']])
        e=read(row['evidence']['path']);probes=e['entry_probes']
        if not all(x['native_prefix_matched'] for x in probes):raise ValueError('Unbound feasibility preview')
        submitted=[x for x in probes if x.get('native_order_submitted')]
        support.append(dict(name=row['name'],period=row['period'],scenario=row['scenario'],
            observed_previews=len(probes),native_submissions=len(submitted),
            evidence=row['evidence'],not_fills_or_funded_pnl=True))
    for prior in ('control','control_prior125'):
        if not any(x['name']==prior and x['period']=='oos' and x['native_submissions']>0 for x in support):
            raise ValueError('No native OOS feasibility support')
    signal=BASE/'v10_signal';models=read(signal/'models/completion.json')
    bindings += [bound(signal/'models/completion.json'),bound(signal/'protocol.json')]
    bindings += [{'path':str(signal/'models'/f),'sha256':h} for f,h in models['files'].items()]
    runtime=read(signal/'cases/control/oos/input_profile.json')
    verify([runtime[k] for k in ('manifest','plan','policy')]+bindings)
    code=Path(runtime['folder'])/'inputs/files'
    manifest=read(runtime['manifest']['path'])
    if manifest['files'].get('backtests/strategies/kalcb/late_signal_v1.py')!=digest(REPO/'backtests/strategies/kalcb/late_signal_v1.py'):
        raise ValueError('Frozen predecessor model module absent')
    root.mkdir(parents=True,exist_ok=False)
    support_files={'research/incremental_v1/'+Path(s).name:bound(REPO/s) for s in PROGRAMS}
    for s in ('backtests/strategies/kalcb/late_incremental_v1.py','backtests/strategies/kalcb/late_signal_v1.py'):
        support_files[s]=bound(REPO/s)
    clean={k:old[k] for k in ('profiles','overrides','bindings','data_bindings','opening_prepared','databases',
        'watch_symbols','periods','checkpoint_route_budget','completed_input_reuse','label_root','input_bindings')}
    p={**clean,'version':'kalcb-incremental-selection-v1','registered_at_utc':datetime.now(timezone.utc).isoformat(),
        'programs':old['programs']+[bound(REPO/s) for s in PROGRAMS],
        'input_bindings':old['input_bindings']+bindings,'tests':test,'recipes':recipes(),
        'runtime_root':'C:/Users/sehyu/.codex/research/kalcb_late_entry_20261011/v13_incremental',
        'producer_runtime':runtime,'signal_root':str(signal),'incremental_rule':RULE,
        'overrides':{period:{**old['overrides'][period],**support_files} for period in PERIODS},
        'native_cases':32,'fresh_native_cases':32,'authenticated_screening_cases':0,
        'native_feasibility_support':support,
        'dependency':dict(native_root=str(active),assessment_root=str(active/'assessment_v1'),
            native_supervisor=read(active/'supervisor_start.json'),
            assessment_supervisor=read(active/'assessment_v1/launch_receipt.json')),
        'hypotheses':dict(aligned48='Exact-quarter net/notional target and known new-issuer prequential calibration; fixed48 safety reserve backout.',
            reserve='Inherited compact-affine forecasts; eligibility/positive-population session-cluster mean-error reserve only.',
            aligned_reserve='Single interaction of target/population alignment and the identical reserve estimator.'),
        'history':'Continuous286 source sessions,217 beforeApr1; original20-session clock and mature one-session embargo. Unknown pre-IS membership abstains for population maps. Parent history unchanged.',
        'scope':'IS156/OOS69; April-July reused development, not independent validation. No exit interaction in this wave.',
        'remaining':'All components/backouts, actual funded late fills, shared route/risk/funding constraints, tails, concentration, paired5/10-session10000repseed20261011; credible finalists frozen before4stresses/neighbors/funded comparisons and existing-engine full diagnostics. No promotion.',
        'economic_gates':'Local/global2/5/10pp IS bands, DD8/10preferences. No forced success, date/symbol exclusions, source bypass or new action interpretation.'}
    # The launch receipt can wrap identity; require the exact registered shape.
    dep=p['dependency']['assessment_supervisor']
    if 'pid' not in dep:
        raise ValueError('Inspect assessment receipt structure before registration')
    write(root/'protocol.json',p)
    write(root/'study.json',dict(recipes=p['recipes'],code=str(REPO),source_generation_hash=digest(root/'protocol.json'),
        periods={k:dict(sessions=v[2]) for k,v in PERIODS.items()},workers=1,policy_name='interaction_cost_margin',
        bound_inputs=[bound(root/'protocol.json')],source_programs=p['programs']))
    for f in (Path(p['label_root'])/'prepared').iterdir():
        if f.is_file():link(f,root/'prepared'/f.name)
    write(root/'assessment_inheritance.json',dict(contract=bound(BASE/'v5/assessment_protocol_v1.json'),
        units='net account returns/MTM drawdown/filled activity; model units explicitly net planned-notional bps',
        new_mechanisms=p['hypotheses'],study_complete=False))
    print(json.dumps(dict(registered=str(root),cases=32,dependency=p['dependency'])),flush=True)


def native(root,name,period,scenario,authenticated,state):
    from backtests.strategies.kalcb.late_entry_io_v4 import freeze
    from backtests.strategies.kalcb.late_incremental_validation_v1 import validate_native
    from backtests.strategies.kalcb.population_research_io_v1 import bridge_accounts
    p=read(root/'protocol.json');recipe=p['recipes'][name];old=p['profiles'][str(recipe['prior'])][period]
    if recipe['quantity'] and sum(r['is_identity'] for r in read(root/'progress.json'))!=8:
        raise ValueError('Eight fresh exact native controls required')
    item=freeze(root,name,period,old,root/'policies'/name/period,p['overrides'][period],authenticated)
    code=Path(item['folder'])/'inputs/files';out=root/'cases'/name/period/scenario
    command=[sys.executable,'-B','-u','-m','scripts.kalcb_executed_value_replay','--plan',item['plan']['path'],
        '--policies',str(code/'research/kalcb_executed_value/policies'),'--policy','interaction_cost_margin',
        '--audit-manifest',item['manifest']['path'],'--scenario',scenario,'--output',str(out)]
    log=execute(root,command,'native_'+name+'_'+period+'_'+scenario,code,state)
    metrics=validate_native(out,item,scenario)
    if not recipe['quantity']:
        bridge=bridge_accounts(out,Path(old['references'][scenario]).parent)
        if not bridge['passed']:raise ValueError('Native parent mismatch')
        write(out/'identity_bridge.json',bridge)
    progress=read(root/'progress.json') if (root/'progress.json').exists() else []
    row=dict(name=name,period=period,scenario=scenario,metrics=metrics,result=bound(out/'result.json'),
        log=log,authenticated_reuse=False,is_identity=recipe['quantity']==0)
    progress.append(row);write(root/'progress.json',progress);print(json.dumps(row),flush=True)
    return read(out/'result.json')


def run(root):
    import psutil,msvcrt
    from backtests.auto.shared.phase_runner import PhaseRunner
    from backtests.auto.shared.round_manager import RoundManager
    from backtests.strategies.kalcb.late_incremental_plugin_v1 import IncrementalPlugin
    from backtests.strategies.kalcb.late_incremental_validation_v1 import validate_first_hour_contract
    p=read(root/'protocol.json');verify(p['programs']+p['input_bindings']+p['data_bindings'])
    lock=(root/'run.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    if (root/'supervisor_start.json').exists():raise ValueError('Preserve previous attempt; versioned successor required')
    state=dict(pid=os.getpid(),process_created=psutil.Process().create_time(),command=psutil.Process().cmdline())
    write(root/'supervisor_start.json',state)
    try:
        # Small isolated model production uses measured headroom and no native
        # worker. Funded replay cannot begin until both exact predecessors exit.
        code=Path(p['producer_runtime']['folder'])/'inputs/files'
        execute(root,[sys.executable,'-B','-u',str(REPO/'scripts/kalcb_late_incremental_producer_v1.py'),
            '--root',str(root),'--code',str(code)],'produce',code,state)
        for phase in ('before','after'):
            proof=read(root/f'producer_imports_{phase}.json');verify([proof['runtime_manifest'],*proof['origins'].values()])
        if not read(root/'policies/completion.json')['complete']:raise ValueError('Unfinished policy production')
        for name in p['recipes']:
            for period in PERIODS:
                file=root/'policies'/name/period/'policies_manifest.json'
                validate_first_hour_contract(file,read(file))
        write(root/'policy_contract_preflight.json',dict(passed=True,policies=16))
        while not dependency_gate(p):
            write(root/'status.json',{**state,'stage':'waiting_exact_mechanism_native_and_assessment_exit','child':None})
            time.sleep(30)
        for proc in psutil.process_iter(['pid','cmdline']):
            args=proc.info['cmdline'] or []
            if proc.pid==os.getpid():continue
            if any(x in args for x in ('scripts.kalcb_executed_value_replay','scripts.kalcb_mechanism_worker_v2')):
                raise ValueError('Another native worker active')
        authenticated=set()
        plugin=IncrementalPlugin(root/'study.json',lambda n,p,s:native(root,n,p,s,authenticated,state))
        manager=RoundManager('stock',plugin.name,base_dir=root/'framework')
        PhaseRunner(plugin,manager.get_round_dir(1),round_name='Incremental target/population and uncertainty reserve',
            round_manager=manager,round_num=1,max_rounds=1,min_delta=.00001,max_retries=0,max_diagnostic_retries=0).run_all_phases()
        if len(read(root/'progress.json'))!=32:raise ValueError('Incomplete native cells')
        write(root/'stage_completion.json',dict(complete=True,native_cases=32,study_complete=False,next=p['remaining']))
        write(root/'status.json',{**state,'stage':'awaiting_economic_assessment_and_finalist_qualification','child':None})
    except Exception as e:
        write(root/'status.json',{**state,'stage':'attention_required','error':repr(e)});raise
    finally:
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',choices=('register','run'))
    ap.add_argument('--root',type=Path,required=True);ap.add_argument('--tests',type=Path)
    args=ap.parse_args()
    if args.stage=='register':register(args.root.resolve(),args.tests)
    else:run(args.root.resolve())
