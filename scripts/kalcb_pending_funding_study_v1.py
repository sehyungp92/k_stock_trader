"""Sixteen finite native opening-reservation cases, owned by shared PhaseRunner."""
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
FILES=('scripts/kalcb_pending_funding_study_v1.py','scripts/kalcb_pending_funding_worker_v1.py',
 'scripts/kalcb_pending_funding_scope_preflight_v1.py','backtests/strategies/kalcb/pending_funding_io_v1.py',
 'backtests/strategies/kalcb/pending_funding_plugin_v1.py','tests/backtests/strategies/test_kalcb_pending_funding_study_v1.py')


def require_controls(root,recipes):
    for name,r in recipes.items():
        if r['reservation']:continue
        for period in PERIODS:
            for scenario in ('baseline','combined'):
                folder=root/'cases'/name/period/scenario
                if not read(folder/'identity_bridge.json')['passed'] or not read(folder/'independent_account_verification.json')['passed']:
                    raise ValueError('All eight exact source/account bridges must gate variants')


def register(root,tests,preflight):
    from backtests.strategies.kalcb.pending_funding_io_v1 import patch_core
    from backtests.strategies.kalcb.research_process_v1 import exact_process
    from backtests.strategies.kalcb.late_artifact_validation_v2 import validate_native
    for name in ('v21_parent_observers','v23_pending_funding_preflight','v24_pending_funding_labels'):
        identity=read(BASE/name/'supervisor_start.json')
        if exact_process(identity):raise ValueError('Prerequisite still running')
    label=BASE/'v24_pending_funding_labels';review=read(label/'review_v2/review.json');done=read(label/'labels/completion.json')
    if not review['complete'] or review['audited_episodes']!=92 or not review['all_cash_fee_fill_inventory_mtm_reconciled']:
        raise ValueError('Fresh quantity labels are not independently audited')
    verify([read(label/'review_v2/completion.json')['review']])
    for output in done['outputs']:verify(list(output.values()))
    t=read(tests);pre=read(preflight)
    if t['exit_code']!=0 or not pre['complete'] or pre['batches']!=544 or not pre['controls_exact'] or not pre['opening_quantities_exact']:
        raise ValueError('Tests and full mechanical scope correspondence required')
    verify(t['programs']+[t['log']]+pre['programs'])
    publication=BASE/'v25_pending_funding_publication_preflight/completion.json'
    pub=read(publication);verify(pub['programs']+[pub['core']])
    if not pub['complete'] or {x['check']['gross_config'] for x in pub['checks']}!={0,2}:raise ValueError('Static publication and effective config proof missing')
    profiles={};bindings=[bound(tests),bound(preflight),bound(publication),bound(label/'protocol.json'),bound(label/'labels/completion.json'),bound(label/'review_v2/review.json'),
        bound(BASE/'v21_parent_observers/assessment_v1/assessment.json'),bound(BASE/'v21_parent_observers/economic_checkpoint_v1/review.json')]
    bindings.extend(b for o in done['outputs'] for b in o.values())
    for prior,name in ((100,'control'),(125,'control_prior125')):
        profiles[str(prior)]={}
        for period in PERIODS:
            folder=BASE/'v21_parent_observers/cases'/name/period;item=read(folder/'input_profile.json')
            item['references']={s:str(folder/s) for s in ('baseline','combined')}
            for scenario in ('baseline','combined'):
                validate_native(folder/scenario,item,scenario,save=False)
                for f in ('result.json','completion.json','execution_audit.json','daily.json','equity.json','fills.json','independent_account_verification.json'):
                    bindings.append(bound(folder/scenario/f))
            profiles[str(prior)][period]=item;bindings.extend(item[k] for k in ('manifest','plan','policy'))
    data=read(BASE/'v18_reclaim_labels/protocol.json')['databases'];bindings += [bound(x) for x in data]
    root.mkdir(parents=True,exist_ok=False)
    overrides={}
    for rel in ('scripts/kalcb_pending_funding_worker_v1.py',):
        target=root/'overrides'/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((REPO/rel).read_bytes());overrides[rel]=bound(target)
    original=Path(profiles['100']['oos']['manifest']['path']).parent/'files/strategy_kalcb/core/logic.py'
    target=root/'overrides/strategy_kalcb/core/logic.py';target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(patch_core(original.read_text(encoding='utf-8')),encoding='utf-8');overrides['strategy_kalcb/core/logic.py']=bound(target)
    if digest(target)!=pre['scoped_core_sha256']:raise ValueError('Preflight economic source mismatch')
    # Only KALCB/shared analysis dependencies; unrelated OLR modules are excluded.
    paths={REPO/f for f in FILES}
    paths.update((REPO/'backtests/auto/shared').glob('*.py'))
    paths.update((REPO/'backtests/strategies/kalcb').glob('*.py'))
    programs=[bound(x) for x in sorted(paths)]
    recipes={'control':{'prior':100,'reservation':0},'control_prior125':{'prior':125,'reservation':0},
        'pending_100':{'prior':100,'reservation':2},'pending_125':{'prior':125,'reservation':2}}
    p={'version':'opening-pending-funding-screening-v1','registered_at_utc':datetime.now(timezone.utc).isoformat(),
        'recipes':recipes,'profiles':profiles,'programs':programs,'bindings':bindings,'overrides':overrides,
        'runtime_root':str(root/'runtime'),'databases':data,'native_cases':16,'fresh_cases':16,'identity_bridges':8,
        'label_review':bound(label/'review_v2/review.json'),'tests':bound(tests),'scope_preflight':bound(preflight),
        'mechanism':'Only first30_open_anchor: calculate unchanged parent quantity first, then min(parent_qty,floor(max(0,2*actual_equity-governed_actual_and_remaining_pending_notional)/completed_close)). Keep parent decision order; zero abstains. Original checkpoint sizing and all management remain unchanged.',
        'evidence':'Actual parent opening requests exceeded2x equity in3IS and8OOS sessions per account; actual broker rejections bound in V21. Broad size/ATR cuts rejected because they carry IS right tails. V23 exact544 controls; V24 fresh exact-quantity full-cost labels, losses and aliases retained. Isolated outcomes are not portfolio gains.',
        'scope_narrowing':'V23 general cap only previewed opening batches. This version explicitly restricts reservation to the observed opening mechanism; all544 opening quantity outputs match V23 and nonopening sizing is tested unchanged. No new timing/quantity grid, forecast or refit.',
        'runtime_estimate':'3-4 hours for16 cases including audits/publication, from V21 eight cases totaling88 measured native minutes plus packaging; memory/IO variability can extend this. Stop for assessment after16.',
        'assessment_contract':{'compare':'local unchanged prior100/prior125 and global IS73.1303145472/47.8121465627; descriptive2/5/10pp bands, DD8/10 preferences',
            'support':'Actual submitted AND filled reduced orders, zero blocks, one-share/low-notional activity separately; original broker funding rejections versus candidate. Unique order/position lifetimes, not partial/retry independence.',
            'required':['net_return','profitable_positions','meaningful_notional_activity','costs','funding_interventions','IS_right_tails','month_issuer_concentration','MTM_episodes','path_aliases','paired_5_10session_10000rep_seed20261011'],
            'uncertainty':'Conditional on reused development and prior selection, not independent or multiplicity adjusted.',
            'next':'No automatic continuation or promotion. Full read-only assessment and existing-engine reports; credible finalists only then supported stress/funded qualification.'},
        'invariants':'IS156/OOS69full sessions;217 prior sessions beforeApril1, continuous history/clocks/embargo. Original policies, physical.001,48bps reserve, heat/capacity/onecheckpoint, cost/nextbar/gap/2xbroker funding. No cashreset/replenishment/rescaling, no live/canonical/OLR change. April-July reused development.'}
    write(root/'protocol.json',p)
    write(root/'study.json',{'recipes':recipes,'code':str(REPO),'source_generation_hash':digest(root/'protocol.json'),
        'periods':{k:{'sessions':v[2]} for k,v in PERIODS.items()},'workers':1,'policy_name':'interaction_cost_margin',
        'bound_inputs':[bound(root/'protocol.json')],'source_programs':programs})
    print(json.dumps({'registered':str(root),'cases':16,'bridges':8}),flush=True)


def run(root):
    import psutil,msvcrt
    from backtests.strategies.kalcb.pending_funding_io_v1 import freeze
    from backtests.strategies.kalcb.pending_funding_plugin_v1 import PendingFundingPlugin
    from backtests.strategies.kalcb.late_artifact_validation_v2 import validate_native
    from backtests.strategies.kalcb.research_process_v1 import execute
    from backtests.strategies.kalcb.reclaim_observer_assessment_v1 import exposure_positions,decision_accounts,reconstruct_mtm
    from backtests.auto.shared.phase_runner import PhaseRunner
    from backtests.auto.shared.round_manager import RoundManager
    p=read(root/'protocol.json');verify(p['programs']+p['bindings'])
    for proc in psutil.process_iter(['pid','cmdline']):
        if proc.pid==os.getpid():continue
        args=proc.info['cmdline'] or []
        if any(a.startswith('scripts.kalcb_') and ('worker' in a or a=='scripts.kalcb_executed_value_replay') for a in args):raise ValueError('Another native worker exists')
    lock=(root/'run.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    if (root/'supervisor_start.json').exists():raise ValueError('Preserve prior attempt')
    temp=root/'temp';temp.mkdir(exist_ok=True);os.environ['TEMP']=os.environ['TMP']=str(temp)
    state={'pid':os.getpid(),'process_created':psutil.Process().create_time(),'command':psutil.Process().cmdline()};write(root/'supervisor_start.json',state)
    def native(name,period,scenario):
        recipe=p['recipes'][name];old=p['profiles'][str(recipe['prior'])][period]
        if recipe['reservation']:require_controls(root,p['recipes'])
        rows=read(root/'progress.json') if (root/'progress.json').exists() else []
        write(root/'status.json',{**state,'stage':'freezing_case','case':[name,period,scenario],'completed':len(rows),'child':None})
        item=freeze(root,name,period,old);code=Path(item['folder'])/'inputs/files';out=root/'cases'/name/period/scenario
        command=[sys.executable,'-B','-u','-m','scripts.kalcb_pending_funding_worker_v1','--plan',item['plan']['path'],
            '--policies',str(Path(item['policy']['path']).parent),'--audit-manifest',item['manifest']['path'],
            '--scenario',scenario,'--output',str(out)]
        if not recipe['reservation']:command+=['--bridge',old['references'][scenario]]
        log=execute(root,command,'native_'+name+'_'+period+'_'+scenario,code,state)
        metrics=validate_native(out,item,scenario)
        evidence=read(out/'observer_evidence.json');fills=read(out/'fills.json');trades=read(out/'trades.json')
        positions,fill_positions=exposure_positions(evidence,fills,trades)
        accounts=decision_accounts(evidence,fills);mtm=reconstruct_mtm(out,positions,fill_positions,p['databases'],[])
        write(out/'independent_exposure_mtm.json',{'passed':True,'positions':positions,'decision_accounts':accounts,'mtm':mtm})
        if not recipe['reservation'] and not read(out/'identity_bridge.json')['passed']:raise ValueError('Parent bridge failed')
        rows.append({'name':name,'period':period,'scenario':scenario,'is_identity':not bool(recipe['reservation']),
            'metrics':metrics,'log':log,'result':bound(out/'result.json'),'exposure_mtm':bound(out/'independent_exposure_mtm.json')})
        write(root/'progress.json',rows);write(root/'status.json',{**state,'stage':'case_audited','completed':len(rows),'child':None})
        print(json.dumps({'audited':len(rows),'case':[name,period,scenario],'metrics':metrics}),flush=True)
        return read(out/'result.json')
    try:
        plugin=PendingFundingPlugin(root/'study.json',native);manager=RoundManager('stock',plugin.name,base_dir=root/'framework')
        PhaseRunner(plugin,manager.get_round_dir(1),round_name='Finite opening pending-funding reservation screening',round_manager=manager,
            round_num=1,max_rounds=1,min_delta=.00001,max_retries=0,max_diagnostic_retries=0).run_all_phases()
        rows=read(root/'progress.json')
        if len(rows)!=16 or len({(r['name'],r['period'],r['scenario']) for r in rows})!=16:raise ValueError('Incomplete finite grid')
        require_controls(root,p['recipes']);verify(p['programs']+p['bindings'])
        write(root/'stage_completion.json',{'complete':True,'native_cases':16,'exact_parent_bridges':8,'study_complete':False})
        write(root/'status.json',{**state,'stage':'awaiting_full_economic_assessment','completed':16,'child':None})
    except Exception as error:
        write(root/'status.json',{**state,'stage':'attention_required','error':repr(error)});raise
    finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',choices=('register','run'))
    ap.add_argument('--root',type=Path,required=True);ap.add_argument('--tests',type=Path);ap.add_argument('--preflight',type=Path)
    a=ap.parse_args();register(a.root.resolve(),a.tests,a.preflight) if a.stage=='register' else run(a.root.resolve())
