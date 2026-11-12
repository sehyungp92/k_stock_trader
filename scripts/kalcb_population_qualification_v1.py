"""Registered finite removal qualification through the shared PhaseRunner."""
import argparse
import ctypes
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from backtests.strategies.kalcb.population_research_io_v1 import (
    read,write,bound,digest,verify,link,env,freeze_case,validate_native,native_support,PERIODS,POLICY)

BASE=Path('E:/CodexResearch/k_stock_trader/kalcb_selective_recovery_20261010')
OLD=BASE/'v5_population'
QUAL=Path('E:/CodexResearch/k_stock_trader/kalcb_recovery_qualification_20261010/v4/profiles')
WORKER=REPO/'scripts/kalcb_population_worker_v1.py'
OBSERVER_WORKER=REPO/'scripts/kalcb_population_removal_worker_v1.py'
OBSERVER=REPO/'backtests/strategies/kalcb/population_removal_observer_v1.py'
SCENARIOS=('baseline','participation','cost','combined')


def commit_available():
    class Info(ctypes.Structure):
        _fields_=[('cb',ctypes.c_ulong)]+[(k,ctypes.c_size_t) for k in
            ('CommitTotal','CommitLimit','CommitPeak','PhysicalTotal','PhysicalAvailable','SystemCache','KernelTotal','KernelPaged','KernelNonpaged','PageSize')]+[(k,ctypes.c_ulong) for k in ('HandleCount','ProcessCount','ThreadCount')]
    v=Info();v.cb=ctypes.sizeof(v)
    if not ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(v),v.cb):raise OSError('Commit inventory unavailable')
    return (v.CommitLimit-v.CommitTotal)*v.PageSize


def recipes():
    result={'control':{'parent':'expanding','prior':None,'config':{},'scenarios':SCENARIOS},
            'control_expanding_atr5':{'parent':'expanding_atr5','prior':None,'config':{},'scenarios':SCENARIOS}}
    for prior in (100,75,125):
        for parent in ('expanding','expanding_atr5'):
            result[parent+'_removal_prior'+str(prior)]={'parent':parent,'prior':prior,'config':{},
                 'scenarios':SCENARIOS if prior==100 else ('baseline','combined')}
    return result


def register(out,tests):
    done=read(OLD/'screening_completion.json');assessment=OLD/'assessment_v1/assessment.json'
    old=read(OLD/'protocol.json');test=read(tests)
    if not done['complete'] or done['native_cases']!=50 or test['exit_code']!=0:raise ValueError('Prelaunch gates incomplete')
    verify(old['programs']+old['bindings']+[test['log']])
    if len(read(assessment)['trials'])!=16:raise ValueError('Incomplete four-parent assessment')
    programs={b['path']:b for b in old['programs']}
    for p in (Path(__file__),OBSERVER,OBSERVER_WORKER,
              REPO/'backtests/strategies/kalcb/population_neighborhood_v1.py',
              REPO/'backtests/strategies/kalcb/population_qualification_plugin_v1.py',
              REPO/'tests/backtests/strategies/test_kalcb_population_removal_observer_v1.py',
              REPO/'tests/backtests/strategies/test_kalcb_population_qualification_v1.py'):
        programs[str(p.resolve())]=bound(p)
    inputs=old['bindings']+[bound(OLD/'protocol.json'),bound(OLD/'study.json'),bound(OLD/'progress.json'),
        bound(OLD/'screening_completion.json'),bound(assessment),bound(tests),test['log'],
        bound(BASE/'v6_qualification_setup/observer_preflight.json')]
    inputs += [bound(p) for p in (OLD/'cases').glob('*/record.json')]
    out.mkdir(parents=True,exist_ok=False)
    write(out/'protocol.json',{'registered_at_utc':datetime.now(timezone.utc).isoformat(),
        'screening_root':str(OLD),'calibration_priors':[75.,100.,125.],'recipes':recipes(),
        'parent_profiles':old['parent_profiles'],'programs':list(programs.values()),'bindings':inputs,'tests':test,
        'decision':'Qualify expanding removal-only as primary return/IS-preserving candidate and expanding+ATR5 as lower-DD tradeoff. Preserve all four-parent trials, including Decay partial stress benefits and IS sacrifices.',
        'hypothesis':'Test local shrinkage strength around prior100, retaining identical eligible population, support minima, sixty-session lookback, original fitting clock, label embargo, exact nonpositive removal and physical gate. No new admissions or rank substitutions.',
        'native_cases':'32 planned fresh cases:8 unchanged parent observer baseline/combined,8 center participation/cost,16 neighbor baseline/combined. Exact executable aliases authenticated and reused rather than replayed.16 other cells reuse audited controls or centers.',
        'causal_support':'Same-state opening gate support on actual filled parent orders; exact unchanged economic bridge plus independent account audit. No hypothetical profit, no checkpoint veto, no claims that descendants are independent.',
        'selection_limit':'Priors75/125 are fixed symmetric neighbors, not an open grid. Both centers receive all four native stresses. Extra variants require new evidence/registration.',
        'periods':PERIODS,'objective':old['objective'],'boundary':old['boundary'],'resources':old['resources'],
        'remaining_after_this_stage':'Funded native constant-capital comparison, complete stress/stability/concentration/paired uncertainty review and existing-script full final diagnostics. Qualify any replacement before changing canonical round1. No automatic promotion or live enablement.',
        'scope':'All April-July evidence is reused development. This stage cannot finish the study or establish independent expected returns.'})
    write(out/'study.json',{'recipes':recipes(),'controls':{'control':'expanding','control_expanding_atr5':'expanding_atr5'},
        'periods':{p:{'sessions':v[2]} for p,v in PERIODS.items()},'workers':1,'code':str(REPO),
        'policy_name':POLICY,'source_generation_hash':digest(out/'protocol.json'),
        'bound_inputs':[bound(out/'protocol.json')],'source_programs':list(programs.values())})
    print(json.dumps({'registered':str(out),'fresh_native_cases_before_aliases':32}),flush=True)


def run(out):
    import msvcrt,psutil
    from backtests.strategies.kalcb.population_neighborhood_v1 import build
    from backtests.strategies.kalcb.population_qualification_plugin_v1 import PopulationQualificationPlugin
    from backtests.strategies.kalcb.qualified_artifacts import REPLAY_FILES
    from backtests.auto.shared.phase_runner import PhaseRunner
    from backtests.auto.shared.round_manager import RoundManager
    protocol=read(out/'protocol.json');verify(protocol['programs']+protocol['bindings'])
    for p in psutil.process_iter(['pid','cmdline','create_time']):
        args=p.info['cmdline'] or []
        if p.pid==os.getpid():continue
        if 'scripts.kalcb_executed_value_replay' in args or any(str(a).endswith((
            'kalcb_population_worker_v1.py','kalcb_population_removal_worker_v1.py','kalcb_retention_worker_v1.py')) for a in args):
            raise ValueError('Another KALCB native worker is active')
        if any(str(a).endswith(('kalcb_population_research_v1.py','kalcb_population_qualification_v1.py')) for a in args) and 'run' in args:
            raise ValueError('Another KALCB supervisor is active')
    lock=(out/'run.lock').open('a+b');lock.seek(0)
    if not lock.read(1):lock.write(b'0');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    if (out/'progress.json').exists() or (out/'build').exists():raise ValueError('Preserve existing work; use versioned successor')
    state={'pid':os.getpid(),'process_created':psutil.Process().create_time(),'command':sys.argv,'child':None}
    def status(stage,**kw):
        state.update(stage=stage,updated_at_utc=datetime.now(timezone.utc).isoformat(),**kw)
        write(out/'status.tmp',state);os.replace(out/'status.tmp',out/'status.json')
    progress=[];authenticated=set();observers={}
    profiles={(p['name'],p['period']):p for p in protocol['parent_profiles']}
    spec=read(out/'study.json')['recipes']
    def reuse(target,source,item,scenario):
        validate_native(source,item,scenario,save=False)
        target.mkdir(parents=True,exist_ok=False)
        for f in (*REPLAY_FILES,'independent_account_verification.json'):
            link(source/f,target/f)
        for f in ('intervention_support.json','observer_bridge.json'):
            if (source/f).exists():link(source/f,target/f)
        write(target/'reuse.json',{'reference':str(source),'authenticated':True,'result':bound(source/'result.json')})
        return read(target/'result.json')
    def native_case(name,period,scenario):
        recipe=spec[name];parent=recipe['parent'];old=profiles[parent,period]
        target=out/'cases'/name/period/scenario;reference=QUAL/parent/period/scenario
        is_control=recipe['prior'] is None
        observe=is_control and scenario in ('baseline','combined')
        if is_control:item=old
        elif recipe['prior']==100:
            item=read(OLD/'cases'/(parent+'_population_removals_only')/period/'input_profile.json')
        else:
            alias=read(out/'build/completion.json')['aliases'].get(name)
            if alias:
                source=out/'cases'/alias/period/scenario
                # Center reuse retains original frozen policy lineage.
                source_item=(read(OLD/'cases'/(parent+'_population_removals_only')/period/'input_profile.json')
                             if spec[alias]['prior']==100 else read(out/'cases'/alias/period/'input_profile.json'))
                return reuse(target,source,source_item,scenario)
            status('freezing_case',case=name+'/'+period+'/'+scenario,child=None,completed=len(progress))
            item=freeze_case(out,name,period,recipe,old,out/'build/policies'/name/period,authenticated)
        if is_control and not observe:return reuse(target,reference,item,scenario)
        if recipe['prior']==100 and scenario in ('baseline','combined'):
            return reuse(target,OLD/'cases'/(parent+'_population_removals_only')/period/scenario,item,scenario)
        if not is_control and len(observers)!=8:raise ValueError('All eight unchanged observer bridges required')
        verify(protocol['programs']+protocol['bindings'])
        while psutil.virtual_memory().available<384*1024**2 or commit_available()<1280*1024**2:
            status('waiting_for_measured_headroom',physical_available=psutil.virtual_memory().available,
                   commit_available=commit_available());time.sleep(15)
        code=Path(item['folder'])/'inputs/files';policy=Path(item['policy']['path']).parent
        # Parent policy must be its audited in-bundle copy for the source audit.
        if is_control:policy=code/'research/kalcb_executed_value/policies'
        command=[sys.executable,'-B','-u',str(OBSERVER_WORKER if observe else WORKER),
            '--code',str(code),'--plan',item['plan']['path'],'--policies',str(policy),
            '--audit-manifest',item['manifest']['path'],'--output',str(target),'--scenario',scenario]
        if observe:
            gate=Path(read(OLD/'cases'/(parent+'_population_removals_only')/period/'input_profile.json')['policy']['path']).parent
            registration=out/'observer_registrations'/parent/period/(scenario+'.json')
            bindings=[bound(p) for p in (OBSERVER_WORKER,OBSERVER,Path(item['plan']['path']),Path(item['manifest']['path']),
                policy/'policies_manifest.json',gate/'policies_manifest.json')]
            bindings += [bound(reference/f) for f in ('result.json','daily.json','equity.json','fills.json','independent_account_verification.json')]
            write(registration,{'bindings':bindings,'scenario':scenario,'source_generation_hash':item['source_generation_hash'],
                                'protocol':bound(out/'protocol.json')})
            command+=['--gate-policies',str(gate),'--observer',str(OBSERVER),'--registration',str(registration),'--bridge',str(reference)]
        if target.exists():raise ValueError('Preserve existing native output')
        target.parent.mkdir(parents=True,exist_ok=True);log=target.with_suffix('.log');start=time.monotonic()
        with log.open('x',encoding='utf-8') as stream:
            child=subprocess.Popen(command,cwd=code,env=env(code),stdout=stream,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            identity={'pid':child.pid,'process_created':psutil.Process(child.pid).create_time(),'command':command}
            status('native_observer' if observe else 'native_replay',case=name+'/'+period+'/'+scenario,child=identity,completed=len(progress))
            if not (out/'launch_receipt.json').exists():write(out/'launch_receipt.json',{'verified':True,'supervisor':dict(state),'tests':protocol['tests'],'study_complete':False})
            while child.poll() is None:
                try:
                    sample={'at_utc':datetime.now(timezone.utc).isoformat(),'identity':identity,'memory':psutil.Process(child.pid).memory_info()._asdict(),
                            'physical_available':psutil.virtual_memory().available,'commit_available':commit_available()}
                    with (out/'resource_observations.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(sample)+'\n')
                except psutil.NoSuchProcess:pass
                time.sleep(10)
        row={'name':name,'period':period,'scenario':scenario,'returncode':child.returncode,'seconds':time.monotonic()-start,'log':bound(log)}
        if child.returncode:write(out/'failed_case.json',row);raise RuntimeError('Native case failed; preserve evidence')
        row['metrics']=validate_native(target,item,scenario);row['result']=bound(target/'result.json')
        if observe:
            if not read(target/'observer_bridge.json')['passed']:raise ValueError('Observer bridge failed')
            observers[name,period,scenario]=bound(target/'observer_bridge.json')
            row['support']=bound(target/'intervention_support.json')
        else:
            write(target/'intervention_support.json',native_support(target,reference,policy,Path(old['policy']['path']).parent))
            row['support']=bound(target/'intervention_support.json')
        progress.append(row);write(out/'progress.json',progress);status('case_completed',child=None,completed=len(progress))
        print(json.dumps(row),flush=True);return read(target/'result.json')
    try:
        status('building_neighborhood',case=None,completed=0)
        build(out/'build',out/'protocol.json')
        plugin=PopulationQualificationPlugin(out/'study.json',native_case)
        manager=RoundManager('stock',plugin.name,base_dir=out/'framework')
        PhaseRunner(plugin,manager.get_round_dir(1),round_name='Finite population-removal qualification',round_manager=manager,
            round_num=1,max_rounds=1,min_delta=.00001,max_retries=0,max_diagnostic_retries=0).run_all_phases()
        verify(protocol['programs']+protocol['bindings'])
        write(out/'stage_completion.json',{'complete':True,'fresh_native_cases':len(progress),'study_complete':False,
              'next':'Review all four stresses, local stability and actual blocked-order support. Complete funded native constant-capital comparisons and final qualification/diagnostics before any promotion.'})
        status('awaiting_funded_qualification_review',child=None)
    except Exception as error:
        status('attention_required',error=repr(error));raise
    finally:
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('register','run'))
    p.add_argument('--root',type=Path,required=True);p.add_argument('--tests',type=Path)
    a=p.parse_args()
    if a.stage=='register':register(a.root,a.tests)
    else:run(a.root)
