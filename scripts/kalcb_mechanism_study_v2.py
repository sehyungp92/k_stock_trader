"""Registered two-parent opening-loss ablation and funded opportunity observers."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from backtests.strategies.kalcb.retention_research_io_v1 import read,write,bound,digest,verify,PERIODS

PARENT=Path('E:/CodexResearch/k_stock_trader/kalcb_late_entry_20261011/v10_signal')
PROGRAMS=('scripts/kalcb_mechanism_study_v2.py','scripts/kalcb_mechanism_worker_v2.py',
 'strategy_kalcb/opening_failure_v1.py','backtests/strategies/kalcb/mechanism_observer_v2.py',
 'backtests/strategies/kalcb/mechanism_io_v1.py','backtests/strategies/kalcb/mechanism_plugin_v1.py',
 'tests/backtests/strategies/test_kalcb_mechanism_v2.py')


def recipes():
    result={'control':{'prior':100,'hold':0},'control_prior125':{'prior':125,'hold':0}}
    for prior in (100,125):
        for hold in (6,8):result[f'prior{prior}_failure{hold}']={'prior':prior,'hold':hold}
    return result


def build_alternatives():
    from collections import defaultdict
    from scripts.kalcb_action_quantity_labels import scaled_row
    label_root=PARENT.parent/'v6'
    rows={r['opportunity_id']:r for r in map(json.loads,(label_root/'prepared/opportunities.jsonl').read_text(encoding='utf-8').splitlines())}
    collected=defaultdict(list);bindings=[bound(label_root/'prepared/opportunities.jsonl')]
    for kind in ('raw_affine','compact','compact_affine'):
        path=PARENT/'models'/(kind+'.jsonl');bindings.append(bound(path))
        for score in map(json.loads,path.read_text(encoding='utf-8').splitlines()):
            if not (score['calibration_supported'] and score['capacity_pass'] and score['failure_probability']<.001
                    and min(score['baseline'],score['combined'])>0):continue
            if not score['opportunity_id'].endswith('/q0.25'):raise ValueError('Wrong action')
            collected[score['opportunity_id']].append(score)
    alternatives=[]
    for key,scores in sorted(collected.items()):
        row=scaled_row(rows[key.removesuffix('/q0.25')],.25)
        if not row['action']['ready'] or not row['guard_at_rank1']:raise ValueError('Unconfirmed alternative')
        alternatives.append([row,scores])
    return {'alternatives':alternatives,'bindings':bindings,
        'scope':'All positive-net, risk/capacity-supported forecasts across three completed families; no outcomes read or threshold fitted. Incumbents and slot competition retained.'}


def register(root,tests):
    from backtests.strategies.kalcb.mechanism_io_v1 import make_overrides
    from backtests.strategies.kalcb.late_artifact_validation_v2 import validate_native
    from backtests.strategies.kalcb.late_signal_reuse_v1 import seed_verified_cache
    test=read(tests)
    if test['exit_code']!=0:raise ValueError('Focused tests must pass')
    verify(test['programs']+[test['log']])
    profiles={};bindings=[]
    for prior,name in ((100,'control'),(125,'control_prior125')):
        profiles[str(prior)]={}
        for period in PERIODS:
            item=read(PARENT/'cases'/name/period/'input_profile.json')
            for scenario in ('baseline','combined'):
                case=PARENT/'cases'/name/period/scenario
                validate_native(case,item,scenario,save=False)
                for f in ('result.json','completion.json','request.json','execution_audit.json','daily.json','equity.json','fills.json'):
                    bindings.append(bound(case/f))
            item['references']={s:str(PARENT/'cases'/name/period/s) for s in ('baseline','combined')}
            profiles[str(prior)][period]=item
            bindings += [item[k] for k in ('manifest','plan','policy')]
    diagnosis=PARENT/'mechanism_review_v2'
    for name in ('report.txt','next_steps.json','analysis.json','population_and_paths.json','verification.json','full_report_verification.json'):
        bindings.append(bound(diagnosis/name))
    root.mkdir(parents=True,exist_ok=False)
    alternatives=build_alternatives();write(root/'alternatives.json',alternatives)
    bindings+=alternatives['bindings']+[bound(root/'alternatives.json')]
    failed=PARENT.parent/'v11_mechanisms'
    for name in ('protocol.json','status.json','failed_stage.json','native_control_is_baseline.log'):
        bindings.append(bound(failed/name))
    published={p:bound(REPO/p) for p in PROGRAMS if not p.startswith('tests/')}
    overrides={}
    for period in PERIODS:
        code=Path(profiles['100'][period]['manifest']['path']).parent/'files'
        overrides[period]=make_overrides(code,root/'overrides'/period,published)
    support=[*PROGRAMS,'scripts/kalcb_late_signal_study_v3.py','scripts/kalcb_action_quantity_labels.py',
        'backtests/strategies/kalcb/retention_research_io_v1.py','backtests/strategies/kalcb/late_entry_io_v4.py',
        'backtests/strategies/kalcb/late_artifact_validation_v2.py','backtests/strategies/kalcb/qualified_artifacts.py',
        'backtests/strategies/kalcb/selective_recovery_plugin_v1.py','backtests/strategies/kalcb/oos_ablation.py',
        'backtests/strategies/kalcb/late_signal_reuse_v1.py']
    programs=[bound(REPO/f) for f in support]+[bound(f) for f in (REPO/'backtests/auto/shared').glob('*.py')]
    runtime=Path('C:/Users/sehyu/.codex/research/kalcb_late_entry_20261011')/root.name
    protocol={'version':'kalcb-opening-mechanisms-v2','registered_at_utc':datetime.now(timezone.utc).isoformat(),
        'successor':'Observer-copy repair only: preserve completed TradeOutcome mappings when cloning actual broker state. V11 failed first control before scoring; no failed results reused. Same 24-case economic design.',
        'recipes':recipes(),'profiles':profiles,'overrides':overrides,'programs':programs,'bindings':bindings,
        'runtime_root':str(runtime),'periods':PERIODS,'tests':test,'native_cases':24,'identity_bridges':8,
        'predicate':'Only first30_open_anchor; gross losing (strictly protects gross winners); min6/8heldbars; two consecutive completed post-entry closes below each own sessionVWAP and original first-six-bar ORmid. Parent cascade always wins. No future MFE.',
        'support_gate':'All8 source/account exact parent bridges pass before variants; at least one same-state no-parent-exit opportunity per period on the local parent/hold across baseline or combined, otherwise do not run unsupported family.',
        'selection':'Exact parent opening/45-minute policies, models, quantities and training clocks remain unchanged.',
        'observer':'Real parent pre-bar account with pending fills/orders is copied. Native adapter syncs fills, activates single alternative and submits in copy only. No future fill or PnL claim; alternatives compete.',
        'objective':'Joint IS/OOS net returns, meaningful filled activity and lower MTM DD. GlobalIS73.1303145472/47.8121465627; local/global2/5/10pp reporting bands; DD8/10preferences.',
        'qualification':'Screening is not completion. Full economic attribution, incremental target/eligibility/reserve pipeline, justified combinations, all4stress, finite neighbors, funded allocation, paired5/10session10000 seed20261011, source/accounts and full reports remain.',
        'resources':'One native worker;384MiB physical and1280MiB commit headroom. No cash reset or live changes.',
        'independence':'April-July reused development; no independent validation; no automatic promotion'}
    write(root/'protocol.json',protocol)
    write(root/'study.json',{'recipes':recipes(),'code':str(REPO),'source_generation_hash':digest(root/'protocol.json'),
        'periods':{p:{'sessions':v[2]} for p,v in PERIODS.items()},'workers':1,'policy_name':'interaction_cost_margin',
        'bound_inputs':[bound(root/'protocol.json')],'source_programs':programs})
    write(root/'cache_reuse.json',seed_verified_cache(Path(read(PARENT/'protocol.json')['runtime_root'])/'content_cache',runtime/'content_cache'))
    print(json.dumps({'registered':str(root),'cases':24,'alternatives':len(alternatives['alternatives'])}),flush=True)


def run(root):
    import psutil,msvcrt
    from backtests.strategies.kalcb.mechanism_io_v1 import freeze
    from backtests.strategies.kalcb.late_artifact_validation_v2 import validate_native
    from backtests.strategies.kalcb.mechanism_plugin_v1 import MechanismPlugin
    from scripts.kalcb_late_signal_study_v3 import execute
    from backtests.auto.shared.phase_runner import PhaseRunner
    from backtests.auto.shared.round_manager import RoundManager
    p=read(root/'protocol.json');verify(p['programs']+p['bindings'])
    for proc in psutil.process_iter(['pid','cmdline','create_time']):
        args=proc.info['cmdline'] or []
        if proc.pid==os.getpid():continue
        if any('kalcb_mechanism_study' in a for a in args) and 'run' in args:raise ValueError('Existing mechanism supervisor')
        if any(a=='scripts.kalcb_executed_value_replay' or a.startswith('scripts.kalcb_mechanism_worker_v') for a in args):raise ValueError('Existing native worker')
    lock=(root/'run.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    if (root/'supervisor_start.json').exists():raise ValueError('Preserve previous attempt')
    state={'pid':os.getpid(),'process_created':psutil.Process().create_time(),'command':psutil.Process().cmdline()}
    write(root/'supervisor_start.json',state)
    def native(name,period,scenario):
        recipe=p['recipes'][name];old=p['profiles'][str(recipe['prior'])][period]
        progress=read(root/'progress.json') if (root/'progress.json').exists() else []
        if recipe['hold']:
            if sum(r['is_identity'] for r in progress)!=8:raise ValueError('Eight exact bridges required')
            cname='control' if recipe['prior']==100 else 'control_prior125'
            for per in PERIODS:
                supported=sum(read(root/'cases'/cname/per/s/'mechanism_evidence.json')['summary']['observed_opportunities'][str(recipe['hold'])]['parent_no_exit_positions'] for s in ('baseline','combined'))
                if not supported:
                    write(root/'unsupported_family.json',{'recipe':name,'period':per,'study_complete':False})
                    raise ValueError('No direct observational support for family; reassess finite design')
        write(root/'status.json',{**state,'stage':'freezing_case','case':[name,period,scenario],'completed':len(progress),'child':None})
        item=freeze(root,name,period,old);code=Path(item['folder'])/'inputs/files'
        out=root/'cases'/name/period/scenario
        command=[sys.executable,'-B','-u','-m','scripts.kalcb_mechanism_worker_v2','--plan',item['plan']['path'],
            '--policies',str(Path(item['policy']['path']).parent),'--audit-manifest',item['manifest']['path'],
            '--scenario',scenario,'--output',str(out),'--alternatives',str(code/'research/mechanism/alternatives.json')]
        if not recipe['hold']:command+=['--bridge',old['references'][scenario]]
        log=execute(root,command,'native_'+name+'_'+period+'_'+scenario,code,state)
        metrics=validate_native(out,item,scenario)
        if not recipe['hold'] and not read(out/'identity_bridge.json')['passed']:raise ValueError('Parent bridge failed')
        row={'name':name,'period':period,'scenario':scenario,'metrics':metrics,'result':bound(out/'result.json'),
             'is_identity':not recipe['hold'],'log':log,'evidence':bound(out/'mechanism_evidence.json')}
        progress.append(row);write(root/'progress.json',progress)
        write(root/'status.json',{**state,'stage':'case_audited','case':[name,period,scenario],'completed':len(progress),'child':None})
        if sum(r['is_identity'] for r in progress)==8 and not (root/'bridge_gate.json').exists():
            write(root/'bridge_gate.json',{'passed':True,'exact_source_account_bridges':8,'cases':progress,'study_complete':False})
        print(json.dumps(row),flush=True);return read(out/'result.json')
    try:
        plugin=MechanismPlugin(root/'study.json',native)
        manager=RoundManager('stock',plugin.name,base_dir=root/'framework')
        PhaseRunner(plugin,manager.get_round_dir(1),round_name='Opening failure and funded opportunity mechanisms',
            round_manager=manager,round_num=1,max_rounds=1,min_delta=.00001,max_retries=0,max_diagnostic_retries=0).run_all_phases()
        if len(read(root/'progress.json'))!=24:raise ValueError('Missing registered cases')
        verify(p['programs']+p['bindings'])
        write(root/'stage_completion.json',{'complete':True,'native_cases':24,'study_complete':False,'next':p['qualification']})
        write(root/'status.json',{**state,'stage':'awaiting_mechanism_assessment_and_incremental_pipeline','child':None})
    except Exception as error:
        write(root/'status.json',{**state,'stage':'attention_required','error':repr(error)});raise
    finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',choices=('register','run'))
    ap.add_argument('--root',type=Path,required=True);ap.add_argument('--tests',type=Path)
    args=ap.parse_args()
    if args.stage=='register':register(args.root.resolve(),args.tests)
    else:run(args.root.resolve())
