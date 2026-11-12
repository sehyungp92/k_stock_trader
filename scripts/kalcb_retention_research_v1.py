"""Finite retention follow-up across four fixed chronological KALCB selectors."""
import argparse
import ctypes
from datetime import datetime,timezone
import os
from pathlib import Path
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from backtests.strategies.kalcb.retention_research_io_v1 import (
    read,write,bound,digest,verify,link,env,freeze_case,validate_native,PERIODS,POLICY)

FIRST=Path('E:/CodexResearch/k_stock_trader/kalcb_selective_recovery_20261010/v3')
DEST=FIRST.with_name('v4_retention')
Q=Path('E:/CodexResearch/k_stock_trader/kalcb_recovery_qualification_20261010/v4')
P=Q.with_name('v3')
PARENTS=('expanding','expanding_atr5','decay60','decay60_atr5')
CONTROLS={'control':'expanding',**{'control_'+p:p for p in PARENTS[1:]}}
CHANGES={'maturity100':{'exit_profit_retention_start_atr':1.0},
         'maturity075':{'exit_profit_retention_start_atr':.75},
         'recent_confirmation':{'exit_profit_retention_confirmation':'recent_vwap'}}
RECIPES={p+'_'+k:{'parent':p,'config':v,'mechanism':k} for k,v in CHANGES.items() for p in PARENTS}
WORKER=REPO/'scripts/kalcb_retention_worker_v1.py'
OBSERVER=REPO/'backtests/strategies/kalcb/retention_intervention_v1.py'


def commit_available():
    class Info(ctypes.Structure):
        _fields_=[('cb',ctypes.c_ulong)]+[(k,ctypes.c_size_t) for k in (
            'CommitTotal','CommitLimit','CommitPeak','PhysicalTotal','PhysicalAvailable','SystemCache','KernelTotal','KernelPaged','KernelNonpaged','PageSize')]+[(k,ctypes.c_ulong) for k in ('HandleCount','ProcessCount','ThreadCount')]
    value=Info();value.cb=ctypes.sizeof(value)
    if not ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(value),value.cb):raise OSError('Windows commit inventory unavailable')
    return (value.CommitLimit-value.CommitTotal)*value.PageSize


def register(out,test_log):
    done=read(FIRST/'screening_completion.json')
    if not done['complete'] or done['cases']!=40:raise ValueError('Complete first-wave evidence required')
    verify([done['comparison']]);prior=read(FIRST/'protocol.json');verify(prior['programs'])
    if '28 passed' not in Path(test_log).read_text(encoding='utf-8'):raise ValueError('Required28 tests not passed')
    programs=prior['programs']+[bound(p) for p in (Path(__file__),WORKER,OBSERVER,
        REPO/'backtests/strategies/kalcb/retention_research_io_v1.py',
        REPO/'backtests/strategies/kalcb/retention_research_plugin_v1.py',
        REPO/'tests/backtests/strategies/test_kalcb_retention_intervention_v1.py')]
    bindings=[bound(FIRST/p) for p in ('protocol.json','screening_completion.json','first_wave_comparison.json',
        'observational_review_20261010_1200/cohort_transfer.json',
        'observational_review_20261010_1300/hybrid_effects.json','observational_review_20261010_1300/path_review.json',
        'observational_review_20261010_1300/findings.json')]
    bindings += [bound(P/'preparation.json'),bound(test_log),bound(REPO/'config/research/kalcb_intraday_advantage_20261007/selected_config.json')]
    profiles=read(P/'preparation.json')['profiles']
    for item in profiles:
        if item['name'] in PARENTS:
            bindings += [item[k] for k in ('manifest','plan','policy')]
            for s in ('baseline','combined'):
                bindings.append(bound(Q/'profiles'/item['name']/item['period']/s/'result.json'))
    verify(bindings);out.mkdir(parents=True,exist_ok=False)
    write(out/'first_wave_assessment.json',{'complete':True,'source':done['comparison'],
        'findings':'No first-wave variant combines material IS preservation with improved OOS returns, activity and DD. Conditional gains sacrifice20+pp global IS; hybrids add filled but weak-net-edge satellites; expanding quantity budget loses19pp IS. Decay quantity combined OOS improves locally, but global IS sacrifice remains22-25pp and OOS DD is not uniformly lower.',
        'retained_partial_tradeoff':'Decay quantity OOS .0586% baseline/-8.9892% combined,288/292trades,11.0641%/15.1206%MTM DD; not promoted.',
        'next_mechanisms':['earlier ATR maturity at1.0/.75 versus parent1.25, cost floor retained',
            'same1.25ATR maturity with six-bar recentVWAP confirmation instead of sessionVWAP'],
        'deferred':'Population-specific calibration support is a third possible mechanism; no unimplemented calibration change is claimed in this exit wave. Broad risk relaxation, larger satellites and unconditional quantity changes lack support.',
        'all_trials':read(FIRST/'first_wave_comparison.json')})
    bindings.append(bound(out/'first_wave_assessment.json'))
    protocol={'registered_at_utc':datetime.now(timezone.utc).isoformat(),'recipes':RECIPES,'parents':list(PARENTS),
        'bindings':bindings,'programs':programs,'parent_profiles':profiles,'tests':bound(test_log),
        'objective':'Maximize IS/OOS net return and meaningful activity while lowering MTM DD; preserve stronger global expanding IS73.1303%/47.8121%. Report2/5/10pp sacrifice bands; DD8/10% soft preferences.',
        'design':'Two mechanisms, three settings, four parents,48 new baseline/combined native accounts plus one unchanged observed control bridge. No crossed grid. Existing unmodified parents are backouts;1.0/.75ATR are a local activation neighborhood.',
        'selection':'Opening/checkpoint candidates and quantities, model clocks, mature training labels and calibrations remain exact authenticated parent bytes. This isolates management. New native full-cost account trade outcomes are saved, not fed into selection; a subsequent retraining needs new isolated management labels.',
        'invariants':'Completed-bar signals, two contemporaneousVWAP closes, unchanged four-times48bps cost floor, same hold requirement, gap/next-bar fills, physical risk gate, source audit, funding, continuous history and late outcomes. No symbol/date exclusions or live changes.',
        'support':'Same-position complete parent exit-reason cascade backout at the real native exit call; only actual retention signals reaching submitted and filled exits when the parent cascade has no same-time exit count as direct support. A different parent exit reason does not count. Prefer10issuer-days across8sessions per period/stress; report low support without inventing descendants. Reconcile all fills/cash/inventory/costs/MTM.',
        'resources':'One native worker; permit available RAM and Windows commit instead of blanket2GiB guard. Require384MiB physical system headroom and1.25GiB available commit. Sample actual peaks. No active study restarts.',
        'completion':'Research screening only. Full four-scenario, concentration, parameter/support stability, paired uncertainty and funded constant-capital qualification plus existing full diagnostics remain required. No automatic canonical promotion. April-July reused development.'}
    write(out/'protocol.json',protocol)
    write(out/'backout.json',{'exit_profit_retention_start_atr':1.25,'exit_profit_retention_confirmation':'session_vwap'})
    write(out/'study.json',{'recipes':{**{n:{'config':{},'parent':p} for n,p in CONTROLS.items()},**RECIPES},
        'controls':CONTROLS,'periods':{p:{'sessions':v[2]} for p,v in PERIODS.items()},
        'source_generation_hash':digest(out/'protocol.json'),'code':str(REPO),'workers':1,'policy_name':POLICY,
        'bound_inputs':[bound(out/'protocol.json'),bound(out/'backout.json')],'source_programs':programs})
    print({'registered':str(out),'native_variant_cases':48,'observer_bridge_cases':1},flush=True)


def run(out):
    import msvcrt,psutil
    from backtests.strategies.kalcb.retention_research_plugin_v1 import RetentionResearchPlugin
    from backtests.strategies.kalcb.qualified_artifacts import REPLAY_FILES
    from backtests.auto.shared.phase_runner import PhaseRunner
    from backtests.auto.shared.round_manager import RoundManager
    protocol=read(out/'protocol.json');verify(protocol['bindings']+protocol['programs'])
    if protocol['recipes']!=RECIPES:raise ValueError('Changed registered recipes')
    for process in psutil.process_iter(['pid','cmdline','create_time']):
        args=process.info['cmdline'] or []
        if process.pid==os.getpid():continue
        if any(str(a).endswith('kalcb_selective_recovery_v3.py') for a in args) and 'run' in args:
            raise ValueError('Preserve active predecessor supervisor')
        if 'scripts.kalcb_executed_value_replay' in args or any(str(a).endswith('kalcb_retention_worker_v1.py') for a in args):
            raise ValueError('Another native KALCB case is active; do not duplicate')
    lock=(out/'run.lock').open('a+b');lock.seek(0)
    if not lock.read(1):lock.write(b'0');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    if (out/'progress.json').exists():raise ValueError('Preserve prior run; require versioned successor')
    state={'pid':os.getpid(),'process_created':psutil.Process().create_time(),'command':sys.argv,'child':None}
    def status(stage,**changes):
        state.update(stage=stage,updated_at_utc=datetime.now(timezone.utc).isoformat(),**changes)
        write(out/'status.tmp',state);os.replace(out/'status.tmp',out/'status.json')
    profiles={(x['name'],x['period']):x for x in protocol['parent_profiles']}
    progress=[];authenticated=set()
    def native_case(name,period,scenario):
        recipe=read(out/'study.json')['recipes'][name];old=profiles[(recipe['parent'],period)]
        reference=Q/'profiles'/recipe['parent']/period/scenario
        target=out/'cases'/name/period/scenario
        observed_control=name=='control' and period=='is' and scenario=='baseline'
        if name in CONTROLS and not observed_control:
            validate_native(reference,old,scenario,save=False)
            if target.exists():raise ValueError('Preserve prior control export')
            target.mkdir(parents=True)
            for f in (*REPLAY_FILES,'independent_account_verification.json'):link(reference/f,target/f)
            write(target/'reuse.json',{'reference':str(reference),'source_audited':True})
            return read(target/'result.json')
        verify(protocol['bindings']+protocol['programs'])
        status('freezing_case',case=name+'/'+period+'/'+scenario,completed=len(progress),total=49,child=None)
        item=freeze_case(out,name,period,recipe,old,authenticated)
        while psutil.virtual_memory().available < 384*1024**2 or commit_available()<1280*1024**2:
            status('waiting_for_measured_headroom',case=name+'/'+period+'/'+scenario,child=None,
                   physical_available=psutil.virtual_memory().available,commit_available=commit_available())
            time.sleep(15)
        if target.exists():raise ValueError('Preserve incomplete native case')
        code=Path(item['folder'])/'inputs/files'
        command=[sys.executable,'-B','-u',str(WORKER),'--code',str(code),'--plan',item['plan']['path'],
            '--policies',str(Path(item['policy']['path']).parent),'--audit-manifest',item['manifest']['path'],
            '--observer',str(OBSERVER),'--backout',str(out/'backout.json'),'--output',str(target),'--scenario',scenario]
        if observed_control:command+=['--bridge',str(reference)]
        logfile=target.with_suffix('.log')
        with logfile.open('x',encoding='utf-8') as stream:
            child=subprocess.Popen(command,cwd=code,env=env(code),stdout=stream,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            identity={'pid':child.pid,'process_created':psutil.Process(child.pid).create_time(),'command':command}
            status('native_replay',child=identity)
            while child.poll() is None:
                try:
                    mem=psutil.Process(child.pid).memory_info()._asdict()
                    with (out/'resource_observations.jsonl').open('a',encoding='utf-8') as f:
                        import json
                        f.write(json.dumps({'at_utc':datetime.now(timezone.utc).isoformat(),'identity':identity,'memory':mem,
                                            'physical_available':psutil.virtual_memory().available,'commit_available':commit_available()})+'\n')
                except psutil.NoSuchProcess:pass
                time.sleep(10)
        row={'name':name,'period':period,'scenario':scenario,'returncode':child.returncode,'log':bound(logfile)}
        if child.returncode and not (target/'completion.json').exists():
            write(out/'failed_case.json',row);raise RuntimeError('Native worker failure; preserve all evidence')
        try:
            row['metrics']=validate_native(target,item,scenario);row['result']=bound(target/'result.json')
            row['support']=bound(target/'intervention_support.json')
        except Exception as error:
            native=read(target/'result.json')
            if native.get('audit_status')=='passed' and native.get('execution_validity_pass'):raise
            row.update(rejected=True,error=repr(error));write(target/'rejected_evidence.json',row)
        progress.append(row);write(out/'progress.json',progress);status('case_completed',child=None,completed=len(progress))
        print(row,flush=True)
        if observed_control:
            if not read(target/'observer_bridge.json')['passed'] or row.get('rejected'):raise ValueError('Observer bridge failed')
            write(out/'launch_receipt.json',{'verified':True,'at_utc':datetime.now(timezone.utc).isoformat(),
                'supervisor':state,'control_case':row,'observer_bridge':bound(target/'observer_bridge.json'),
                'source_and_account_audits_passed':True,'registered_programs':len(protocol['programs']),
                'tests':protocol['tests'],'study_complete':False})
        return {'research_rejected':True,'error':row['error']} if row.get('rejected') else read(target/'result.json')
    try:
        plugin=RetentionResearchPlugin(out/'study.json',native_case)
        manager=RoundManager('stock',plugin.name,base_dir=out/'framework')
        PhaseRunner(plugin,manager.get_round_dir(1),round_name='Four-parent retention perturbations',round_manager=manager,
            round_num=1,max_rounds=1,min_delta=.00001,max_retries=0,max_diagnostic_retries=0).run_all_phases()
        verify(protocol['bindings']+protocol['programs'])
        write(out/'screening_completion.json',{'complete':True,'native_cases':len(progress),'study_complete':False,
              'next':'Review direct intervention support and IS/OOS local/global tradeoffs. Qualify credible finalists in all four scenarios; retain component and parameter evidence. Continue authorized study.'})
        status('awaiting_retention_review',child=None)
    except Exception as error:status('attention_required',error=repr(error));raise
    finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['register','run'])
    parser.add_argument('--root',type=Path,default=DEST);parser.add_argument('--test-log',type=Path)
    args=parser.parse_args()
    if args.stage=='register':register(args.root,args.test_log)
    else:run(args.root)
