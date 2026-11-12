"""Eight missing-stress native cases plus eight authenticated V25 accounts."""
from datetime import datetime, timezone
import argparse
import os
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from backtests.strategies.kalcb.retention_research_io_v1 import read,write,bound,digest,verify,PERIODS

BASE=Path('E:/CodexResearch/k_stock_trader/kalcb_late_entry_20261011')
V25=BASE/'v25_pending_funding_screen'
RECIPES={'control':{'prior':100,'reservation':0},'pending_100':{'prior':100,'reservation':2}}
SCENARIOS=('baseline','participation','cost','combined')


def require_controls(rows):
    controls=[r for r in rows if r['name']=='control']
    expected={(p,s) for p in PERIODS for s in SCENARIOS}
    if len(controls)!=8 or {(r['period'],r['scenario']) for r in controls}!=expected:
        raise ValueError('Eight unique audited parent scenarios must gate candidate')
    if any(r['metrics']['audit_status']!='passed' or not r['identity_passed'] for r in controls):
        raise ValueError('Unverified exact parent')


def complete_grid(rows):
    expected={(n,p,s) for n in RECIPES for p in PERIODS for s in SCENARIOS}
    if len(rows)!=16 or {(r['name'],r['period'],r['scenario']) for r in rows}!=expected:
        raise ValueError('Incomplete or duplicate qualification grid')
    if sum(r['reused'] for r in rows)!=8 or any(r['metrics']['audit_status']!='passed' for r in rows):
        raise ValueError('Wrong reuse/fresh count or failed audit')
    require_controls(rows)


def register(root,tests):
    from backtests.strategies.kalcb.research_process_v1 import exact_process
    from backtests.strategies.kalcb.late_artifact_validation_v2 import validate_native
    from backtests.strategies.kalcb.pending_funding_assessment_v1 import completion_gate
    completion_gate(V25)
    for folder in (V25,V25/'assessment_v1',V25/'full_reports_v1'):
        identity=read(folder/'supervisor_start.json')
        if exact_process(identity):raise ValueError('Prerequisite still running')
    t=read(tests)
    if not t['passed']:raise ValueError('Focused qualification tests failed')
    verify(t['programs']+[t['log']])
    old=read(V25/'protocol.json')
    verify(list(old['overrides'].values())+[old['label_review']])
    profiles={'100':{}};reuse={};references={};bindings=[bound(tests),bound(V25/'stage_completion.json'),
        bound(V25/'assessment_v1/completion.json'),bound(V25/'economic_review_v1/economic_decision.json'),
        bound(V25/'full_reports_v1/verification_v1/full_report_verification.json'),
        bound(BASE/'v27_opening_conditional_value_v2/completion.json'),bound(BASE/'v27_opening_conditional_value_v2/assessment.json')]
    artifacts=('result.json','completion.json','execution_audit.json','independent_account_verification.json',
        'fills.json','daily.json','equity.json','trades.json')
    for period in PERIODS:
        profile=read(V25/'cases/control'/period/'input_profile.json');profiles['100'][period]=profile
        bindings.extend(profile[k] for k in ('manifest','plan','policy'))
        references[period]={}
        for scenario in ('participation','cost'):
            folder=BASE/'v16_exit_qualification/compounding/cases/control'/period/scenario
            item=read(folder/'qualification_input_profile.json');validate_native(folder,item,scenario,save=False)
            references[period][scenario]=str(folder)
            bindings.append(bound(folder/'qualification_input_profile.json'))
            bindings.extend(bound(folder/f) for f in artifacts)
        for name in RECIPES:
            for scenario in ('baseline','combined'):
                folder=V25/'cases'/name/period/scenario;profile_path=folder.parent/'input_profile.json'
                item=read(profile_path);metrics=validate_native(folder,item,scenario,save=False)
                if name=='control' and not read(folder/'identity_bridge.json')['passed']:raise ValueError('Missing V25 exact bridge')
                casebindings=[bound(folder/f) for f in artifacts+('independent_exposure_mtm.json','observer_evidence.json','request.json')]
                casebindings += [bound(profile_path)]+[item[k] for k in ('manifest','plan','policy')]
                if name=='control':casebindings.append(bound(folder/'identity_bridge.json'))
                verify(casebindings);bindings.extend(casebindings)
                reuse[name+'/'+period+'/'+scenario]={'folder':str(folder),'input_profile':bound(profile_path),
                    'metrics':metrics,'bindings':casebindings}
    root.mkdir(parents=True,exist_ok=False)
    overrides={}
    for rel,proof in old['overrides'].items():
        target=root/'overrides'/rel;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(Path(proof['path']).read_bytes());overrides[rel]=bound(target)
        if overrides[rel]['sha256']!=proof['sha256']:raise ValueError('Economic source changed')
    # The fresh bundle reuses the already-tested V25 core/worker byte for byte.
    paths=set((REPO/'backtests/auto/shared').glob('*.py'))|set((REPO/'backtests/strategies/kalcb').glob('*.py'))
    paths.update({Path(__file__),REPO/'tests/backtests/strategies/test_kalcb_pending_funding_qualification_v1.py'})
    programs=[bound(p) for p in sorted(paths)]
    p={'version':'pending-funding-qualification-v1','registered_at_utc':datetime.now(timezone.utc).isoformat(),
        'recipes':RECIPES,'scenarios':SCENARIOS,'profiles':profiles,'reused':reuse,'parent_stress_references':references,
        'programs':programs,'bindings':bindings,'overrides':overrides,'runtime_root':str(root/'runtime'),
        'databases':old['databases'],'label_review':old['label_review'],'tests':bound(tests),
        'fresh_cases':8,'authenticated_reuse':8,'accounted_cases':16,
        'mechanism':old['mechanism'],'economic_changes_from_v25':False,
        'runtime_estimate':'45-90 minutes from native supervisor launch for eight fresh stress cells, plus read-only assessment/reporting. No automatic extension.',
        'reason':'V25 prior100 provisional gains require isolated participation and cost stresses. V26/V27 opening-context correction failed readiness and is excluded. This qualification does not assert that funding fixes the broader alpha weakness.',
        'required_assessment':['all4scenarioIS/OOSnetreturnandMTMDD','meaningfulprofitableactivity','actualreducedsubmittedandfilledorders','zeroblocks','fundingrejectionsanddescendants','ISrighttails','monthissuerconcentration','costs','paired5/10session10000repseed20261011','effectivealiases','fullengine_reports'],
        'acceptance':'Joint assessment, not fewer funding rejects or one-share counts. Keep unchanged prior100 and125 as references. No automatic promotion or new optimization parent. Funded qualification only if joint stresses justify it.',
        'invariants':old['invariants'],'study_complete':False}
    write(root/'protocol.json',p)
    write(root/'study.json',{'recipes':RECIPES,'code':str(REPO),'source_generation_hash':digest(root/'protocol.json'),
        'periods':{k:{'sessions':v[2]} for k,v in PERIODS.items()},'workers':1,'policy_name':'interaction_cost_margin',
        'bound_inputs':[bound(root/'protocol.json')],'source_programs':programs})
    print({'registered':str(root),'fresh':8,'reuse':8},flush=True)


def run(root):
    import psutil,msvcrt
    from backtests.strategies.kalcb.pending_funding_io_v1 import freeze
    from backtests.strategies.kalcb.pending_funding_qualification_plugin_v1 import PendingFundingQualificationPlugin
    from backtests.strategies.kalcb.late_artifact_validation_v2 import validate_native
    from backtests.strategies.kalcb.research_process_v1 import execute
    from backtests.strategies.kalcb.reclaim_observer_assessment_v1 import exposure_positions,decision_accounts,reconstruct_mtm
    from backtests.auto.shared.phase_runner import PhaseRunner
    from backtests.auto.shared.round_manager import RoundManager
    p=read(root/'protocol.json');verify(p['programs']+p['bindings'])
    for proc in psutil.process_iter(['pid','cmdline']):
        args=proc.info['cmdline'] or []
        if proc.pid!=os.getpid() and any(a.startswith('scripts.kalcb_') and ('worker' in a or a=='scripts.kalcb_executed_value_replay') for a in args):raise ValueError('Another native worker exists')
    lock=(root/'run.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    if (root/'supervisor_start.json').exists():raise ValueError('Preserve prior launch')
    temp=root/'temp';temp.mkdir(exist_ok=True);os.environ['TEMP']=os.environ['TMP']=str(temp)
    state={'pid':os.getpid(),'process_created':psutil.Process().create_time(),'command':psutil.Process().cmdline()}
    write(root/'supervisor_start.json',state)
    def native(name,period,scenario):
        rows=read(root/'progress.json') if (root/'progress.json').exists() else []
        if any((r['name'],r['period'],r['scenario'])==(name,period,scenario) for r in rows):raise ValueError('Duplicate dispatch')
        recipe=p['recipes'][name]
        if recipe['reservation']:require_controls(rows)
        key=name+'/'+period+'/'+scenario
        if key in p['reused']:
            previous=p['reused'][key];verify(previous['bindings']);out=Path(previous['folder']);profile=read(previous['input_profile']['path'])
            metrics=validate_native(out,profile,scenario,save=False)
            if metrics!=previous['metrics']:raise ValueError('Reused metric drift')
            profile_binding=previous['input_profile'];log=None;is_reused=True
        else:
            if scenario not in ('participation','cost'):raise ValueError('Undeclared fresh scenario')
            write(root/'status.json',{**state,'stage':'freezing_case','case':[name,period,scenario],'completed':len(rows),'child':None})
            profile=freeze(root,name,period,p['profiles']['100'][period]);code=Path(profile['folder'])/'inputs/files'
            out=root/'cases'/name/period/scenario;profile_binding=bound(out.parent/'input_profile.json')
            command=[sys.executable,'-B','-u','-m','scripts.kalcb_pending_funding_worker_v1','--plan',profile['plan']['path'],
                '--policies',str(Path(profile['policy']['path']).parent),'--audit-manifest',profile['manifest']['path'],
                '--scenario',scenario,'--output',str(out)]
            if not recipe['reservation']:command+=['--bridge',p['parent_stress_references'][period][scenario]]
            log=execute(root,command,'native_'+name+'_'+period+'_'+scenario,code,state)
            metrics=validate_native(out,profile,scenario)
            evidence=read(out/'observer_evidence.json');fills=read(out/'fills.json');trades=read(out/'trades.json')
            positions,mapping=exposure_positions(evidence,fills,trades)
            decisions=decision_accounts(evidence,fills);mtm=reconstruct_mtm(out,positions,mapping,p['databases'],[])
            write(out/'independent_exposure_mtm.json',{'passed':True,'positions':positions,'decision_accounts':decisions,'mtm':mtm})
            is_reused=False
        identity=read(out/'identity_bridge.json')['passed'] if name=='control' else None
        if name=='control' and not identity:raise ValueError('Exact parent stress bridge failed')
        rows.append({'name':name,'period':period,'scenario':scenario,'reused':is_reused,'is_identity':name=='control',
            'identity_passed':identity,'folder':str(out),'input_profile':profile_binding,'metrics':metrics,
            'result':bound(out/'result.json'),'exposure_mtm':bound(out/'independent_exposure_mtm.json'),'log':log})
        write(root/'progress.json',rows);write(root/'status.json',{**state,'stage':'case_audited','completed':len(rows),'fresh_completed':sum(not r['reused'] for r in rows),'child':None})
        print({'accounted':len(rows),'case':key,'reused':is_reused,'metrics':metrics},flush=True)
        return read(out/'result.json')
    try:
        plugin=PendingFundingQualificationPlugin(root/'study.json',native);manager=RoundManager('stock',plugin.name,base_dir=root/'framework')
        PhaseRunner(plugin,manager.get_round_dir(1),round_name='Opening reservation four-scenario qualification',round_manager=manager,
            round_num=1,max_rounds=1,min_delta=.00001,max_retries=0,max_diagnostic_retries=0).run_all_phases()
        rows=read(root/'progress.json');complete_grid(rows);verify(p['programs']+p['bindings'])
        write(root/'stage_completion.json',{'complete':True,'accounted_cases':16,'fresh_cases':8,'authenticated_reuse':8,'exact_parent_bridges':8,'study_complete':False})
        write(root/'status.json',{**state,'stage':'awaiting_joint_assessment','completed':16,'fresh_completed':8,'child':None})
    except Exception as error:
        write(root/'status.json',{**state,'stage':'attention_required','error':repr(error)});raise
    finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',choices=('register','run'))
    ap.add_argument('--root',type=Path,required=True);ap.add_argument('--tests',type=Path)
    args=ap.parse_args();register(args.root,args.tests) if args.stage=='register' else run(args.root)
