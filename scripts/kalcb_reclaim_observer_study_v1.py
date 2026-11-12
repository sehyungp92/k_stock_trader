"""Eight unchanged parent native bridges and finite account/exposure census."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from backtests.strategies.kalcb.retention_research_io_v1 import read,write,bound,digest,verify,PERIODS
BASE=Path('E:/CodexResearch/k_stock_trader/kalcb_late_entry_20261011')
PROGRAMS=('scripts/kalcb_reclaim_observer_study_v1.py','scripts/kalcb_reclaim_observer_worker_v1.py',
    'backtests/strategies/kalcb/reclaim_observer_v1.py','backtests/strategies/kalcb/reclaim_observer_io_v1.py',
    'backtests/strategies/kalcb/reclaim_observer_plugin_v1.py','tests/backtests/strategies/test_kalcb_reclaim_observer_v1.py')


def build_opportunities():
    from collections import defaultdict
    from scripts.kalcb_action_quantity_labels import scaled_row
    original=read(BASE/'v18_reclaim_labels/protocol.json');scores=defaultdict(list);bindings=[]
    for prior in (100,125):
        for recipe in ('raw48','affine48'):
            path=BASE/'v20_reclaim_models/models'/f'{prior}_{recipe}.jsonl';bindings.append(bound(path))
            for line in path.read_text(encoding='utf-8').splitlines():
                r=json.loads(line);scores[r['opportunity_id']].append(r)
    path=BASE/'v18_reclaim_labels/prepared/opportunities.jsonl';rows=[]
    for line in path.read_text(encoding='utf-8').splitlines():
        r=json.loads(line)
        if '2025-08-07'<=r['trade_date']<='2026-07-10' and r['guard_at_rank1'] and r['quarter_preview']['ready']:
            scaled=scaled_row(r,.25)
            if scaled['action']['quantity']!=r['quarter_preview']['quantity']:raise ValueError('Quarter action mismatch')
            rows.append([scaled,scores[scaled['opportunity_id']]])
    if len(rows)!=235:raise ValueError('Changed preregistered complete population')
    opening=[str(Path(p)/'opportunities.jsonl') for p in original['opening_prepared']]
    bindings += [bound(path),bound(BASE/'v18_reclaim_labels/prepared/source_population_audit.json')]
    bindings += [bound(p) for p in [*opening,*original['databases']]]
    return {'opportunities':rows,'prepared':bound(path),'opening_prepared':opening,'databases':original['databases'],
        'bindings':bindings,'scope':'All235geometry+quarter-ready IS/OOS issuer-days, no outcome selection. Forecasts label observations only, not eligibility overrides.'}


def register(root,tests):
    from backtests.strategies.kalcb.late_artifact_validation_v2 import validate_native
    t=read(tests)
    if t['exit_code']!=0:raise ValueError('Focused tests must pass')
    verify(t['programs']+[t['log']]);profiles={};bindings=[]
    for prior,name in ((100,'control'),(125,'control_prior125')):
        profiles[str(prior)]={}
        for period in PERIODS:
            parent=BASE/'v12_mechanisms/cases'/name/period;item=read(parent/'input_profile.json')
            for scenario in ('baseline','combined'):
                validate_native(parent/scenario,item,scenario,save=False)
                for f in ('result.json','completion.json','request.json','execution_audit.json','daily.json','equity.json','fills.json'):
                    bindings.append(bound(parent/scenario/f))
            item['references']={s:str(parent/s) for s in ('baseline','combined')};profiles[str(prior)][period]=item
            bindings.extend(item[k] for k in ('manifest','plan','policy'))
    spec=build_opportunities();bindings+=spec['bindings']
    for relative in ('v19_reclaim_risk_population/review_v3/review.json','v20_reclaim_models/review_v1/review.json',
        'v20_reclaim_models/protocol.json','v20_reclaim_models/models/completion.json',
        'v16_exit_qualification/full_reports_v2/verification_v1/post_report_judgment.json'):
        bindings.append(bound(BASE/relative))
    root.mkdir(parents=True,exist_ok=False);write(root/'opportunities.json',spec);bindings.append(bound(root/'opportunities.json'))
    support=[*PROGRAMS,'backtests/strategies/kalcb/reclaim_source_audit_v1.py',
        'backtests/strategies/kalcb/mechanism_observer_v2.py','backtests/strategies/kalcb/late_entry_io_v5.py',
        'backtests/strategies/kalcb/retention_research_io_v1.py','backtests/strategies/kalcb/research_process_v1.py',
        'backtests/strategies/kalcb/late_artifact_validation_v2.py','backtests/strategies/kalcb/qualified_artifacts.py',
        'backtests/strategies/kalcb/selective_recovery_plugin_v1.py','backtests/strategies/kalcb/oos_ablation.py',
        'backtests/strategies/kalcb/population_research_io_v1.py','scripts/kalcb_action_quantity_labels.py']
    programs=[bound(REPO/f) for f in support]+[bound(f) for f in (REPO/'backtests/auto/shared').glob('*.py')]
    overrides={}
    for rel in (PROGRAMS[1],PROGRAMS[2],'backtests/strategies/kalcb/reclaim_source_audit_v1.py'):
        target=root/'overrides'/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((REPO/rel).read_bytes());overrides[rel]=bound(target)
    recipes={'control':{'prior':100},'control_prior125':{'prior':125}}
    p={'version':'kalcb-reclaim-parent-observer-v1','registered_at_utc':datetime.now(timezone.utc).isoformat(),
        'recipes':recipes,'profiles':profiles,'periods':PERIODS,'programs':programs,'bindings':bindings,'overrides':overrides,
        'runtime_root':str(root/'runtime'),'tests':t,'native_cases':8,'identity_bridges':8,'new_economic_variants':0,
        'observer_boundary':'Before the real minute90 timer, after actual parent bar decisions. Each separate copy appends one new issuer last-ranked. Missing inactive watch bars are warmed only in the copy from independently audited complete90minute source. The original feed and transitions are unchanged. Missing native clock is recorded, never fabricated.',
        'selection':'All235geometry+native quarter ready issuer-days, independent of outcomes and forecasts. Scores label readiness only. Never sum mutually exclusive previews or call copied submissions future fills/funded PnL.',
        'exposure':'Every actual native SubmitEntry captures pre-submission funded cash/equity/buying power/positions/sector/ATR/pending/state and joins only unique physical entry order IDs. No rescaled portfolio PnL.',
        'invariants':'Original policy/config/45minute source obligations, clocks, risk/physical gates, quantities, management/cost/funding and one successful checkpoint remain unchanged. No canonical/live changes. April-July reused development.',
        'next':'Authenticate all8native/source/account identities and90prefix proofs; assess funding/checkpoint/rank and opening exposure attribution before freezing any finite follow-up. No new budget/threshold grid or automatic promotion.'}
    write(root/'protocol.json',p)
    write(root/'study.json',{'recipes':recipes,'code':str(REPO),'source_generation_hash':digest(root/'protocol.json'),
        'periods':{key:{'sessions':v[2]} for key,v in PERIODS.items()},'workers':1,'policy_name':'interaction_cost_margin',
        'bound_inputs':[bound(root/'protocol.json')],'source_programs':programs})
    print(json.dumps({'registered':str(root),'cases':8,'opportunities':235}),flush=True)


def run(root):
    import psutil,msvcrt
    from backtests.strategies.kalcb.reclaim_observer_io_v1 import freeze
    from backtests.strategies.kalcb.reclaim_observer_plugin_v1 import ReclaimObserverPlugin
    from backtests.strategies.kalcb.late_artifact_validation_v2 import validate_native
    from backtests.strategies.kalcb.research_process_v1 import execute
    from backtests.auto.shared.phase_runner import PhaseRunner
    from backtests.auto.shared.round_manager import RoundManager
    p=read(root/'protocol.json');verify(p['programs']+p['bindings'])
    for proc in psutil.process_iter(['pid','cmdline']):
        if proc.pid==os.getpid():continue
        args=proc.info['cmdline'] or []
        native_worker=any(a.startswith('scripts.kalcb_') and ('worker' in a or a=='scripts.kalcb_executed_value_replay') for a in args)
        if native_worker:raise ValueError('Existing native worker')
    lock=(root/'run.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    if (root/'supervisor_start.json').exists():raise ValueError('Preserve previous attempt')
    temp=root/'temp';temp.mkdir(exist_ok=True);os.environ['TEMP']=os.environ['TMP']=str(temp)
    state={'pid':os.getpid(),'process_created':psutil.Process().create_time(),'command':psutil.Process().cmdline()}
    write(root/'supervisor_start.json',state)
    def native(name,period,scenario):
        prior=p['recipes'][name]['prior'];old=p['profiles'][str(prior)][period]
        progress=read(root/'progress.json') if (root/'progress.json').exists() else []
        write(root/'status.json',{**state,'stage':'freezing_case','case':[name,period,scenario],'completed':len(progress),'child':None})
        item=freeze(root,name,period,old);code=Path(item['folder'])/'inputs/files';out=root/'cases'/name/period/scenario
        command=[sys.executable,'-B','-u','-m','scripts.kalcb_reclaim_observer_worker_v1','--plan',item['plan']['path'],
            '--policies',str(Path(item['policy']['path']).parent),'--audit-manifest',item['manifest']['path'],
            '--scenario',scenario,'--output',str(out),'--opportunities',str(code/'research/reclaim_observer/opportunities.json'),
            '--prior',str(prior),'--bridge',old['references'][scenario]]
        log=execute(root,command,'native_'+name+'_'+period+'_'+scenario,code,state)
        metrics=validate_native(out,item,scenario)
        if not read(out/'identity_bridge.json')['passed']:raise ValueError('Parent bridge failed')
        progress.append({'name':name,'period':period,'scenario':scenario,'metrics':metrics,'is_identity':True,
            'log':log,'result':bound(out/'result.json'),'evidence':bound(out/'observer_evidence.json')})
        write(root/'progress.json',progress);write(root/'status.json',{**state,'stage':'case_audited','completed':len(progress),'child':None})
        print(json.dumps({'audited':len(progress),'case':[name,period,scenario]}),flush=True)
        return read(out/'result.json')
    try:
        plugin=ReclaimObserverPlugin(root/'study.json',native);manager=RoundManager('stock',plugin.name,base_dir=root/'framework')
        PhaseRunner(plugin,manager.get_round_dir(1),round_name='Unchanged parent reclaim readiness and exposure census',round_manager=manager,
            round_num=1,max_rounds=1,min_delta=.00001,max_retries=0,max_diagnostic_retries=0).run_all_phases()
        rows=read(root/'progress.json')
        if len(rows)!=8 or len({(r['name'],r['period'],r['scenario']) for r in rows})!=8:raise ValueError('Incomplete cases')
        verify(p['programs']+p['bindings'])
        write(root/'stage_completion.json',{'complete':True,'native_cases':8,'exact_parent_bridges':8,'study_complete':False,'next':p['next']})
        write(root/'status.json',{**state,'stage':'awaiting_observer_assessment','child':None})
    except Exception as error:
        write(root/'status.json',{**state,'stage':'attention_required','error':repr(error)});raise
    finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',choices=('register','run'))
    ap.add_argument('--root',type=Path,required=True);ap.add_argument('--tests',type=Path)
    a=ap.parse_args();register(a.root.resolve(),a.tests) if a.stage=='register' else run(a.root.resolve())
