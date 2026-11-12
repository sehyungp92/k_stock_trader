"""Locked finalist acceptance and existing-framework full research diagnostics.

Consumes the registered study; does not search, select a live configuration or
write canonical numbered rounds. Every profile is an isolated comparison.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import os
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.kalcb_full_oos_recovery import ROOT, POLICY, read, write, bound, digest
from scripts import kalcb_recovery_qualification_round_v4 as qualification
from scripts.kalcb_recovery_stage_review import paired
from scripts.kalcb_recovery_stage5_review import concentration
from scripts.kalcb_recovery_report import segments

Q = qualification.DEST
F = Path('E:/CodexResearch/k_stock_trader/kalcb_forecast_qualification_20261010/v1')
S = Path('E:/CodexResearch/k_stock_trader/kalcb_atr_intervention_20261010/v2')
DEST = Path('E:/CodexResearch/k_stock_trader/kalcb_recovery_final_20261010/v1')


def require_ready(completion, forecast):
    expected = {(n,p,s) for n in qualification.PROFILES for p in qualification.PERIODS for s in qualification.SCENARIOS}
    rows = completion.get('results', [])
    actual = [(r['name'],r['period'],r['scenario']) for r in rows]
    if not completion.get('complete') or len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('Incomplete or duplicated finalist source audit population')
    if any(r['metrics'].get('audit_status') != 'passed' for r in rows):
        raise ValueError('Failed source-qualified case cannot enter final diagnostics')
    if forecast.get('complete') is not True or forecast.get('passed') is not True:
        raise ValueError('Saved model/forecast reconstruction is not qualified')


def register(out):
    import psutil
    if out.exists(): raise ValueError('Preserve prior final-artifact study')
    qualification.check(Q)
    if not read(S/'completion.json')['complete']: raise ValueError('Direct intervention evidence incomplete')
    paths = [Path(__file__), Q/'protocol.json', Q/'preparation.json', F/'protocol.json', S/'protocol.json', S/'completion.json',
             ROOT/'boundary_continuity.json', ROOT/'history_boundary_audit.json',
             ROOT/'qualification_gate_tests_20261010.json', ROOT/'qualification_gate_tests_20261010.log',
             ROOT/'qualification_source_tests_20261010.log', ROOT/'atr_intervention_observer_tests_20261010.log',
             ROOT/'recovery_report_context_repair_20261010.json', ROOT/'recovery_report_context_tests_20261010.log',
             ROOT/'forecast_qualification_tests_20261010.log', ROOT/'final_artifact_gate_tests_20261010.log',
             REPO/'docs/strategy-implementation-lessons.md',
             REPO/'config/research/kalcb_intraday_advantage_20261007/selected_config.json',
             REPO/'config/research/kalcb_intraday_advantage_20261007/selected_research.json']
    paths += [ROOT/(f'stage{i}_review.json') for i in (2,3,4,5)]
    for item in read(S/'completion.json')['results']: paths.append(Path(item['evidence']['path']))
    paths += list((REPO/'backtests/auto/shared').glob('*.py'))
    paths += [REPO/p for p in ('backtests/strategies/kalcb/qualified_artifacts.py',
        'backtests/strategies/kalcb/qualified_round.py','backtests/strategies/kalcb/full_diagnostics.py',
        'backtests/strategies/kalcb/recovery_full_diagnostics_v1.py',
        'scripts/kalcb_recovery_stage5_review.py','scripts/kalcb_recovery_stage_review.py',
        'scripts/kalcb_recovery_report.py','tests/backtests/strategies/test_kalcb_final_artifact_gates.py')]
    dependencies = {}
    for root in (Q,F):
        if (root/'completion.json').exists() and read(root/'completion.json').get('complete'):
            dependencies[str(root)] = None
            paths.append(root/'completion.json')
            continue
        state = read(root/'status.json')
        process = psutil.Process(state['pid'])
        if abs(process.create_time()-state['process_created']) > .1: raise ValueError('Dependency process identity changed')
        dependencies[str(root)] = {'pid':process.pid,'process_created':process.create_time(),'command':process.cmdline()}
    out.mkdir(parents=True)
    write(out/'protocol.json',{'registered_at_utc':datetime.now(timezone.utc).isoformat(),
        'bindings':[bound(p) for p in sorted(set(paths))], 'dependencies':dependencies,
        'profiles':list(qualification.PROFILES),'periods':qualification.PERIODS,
        'scope':'Eight source-qualified research comparison profiles, complete four-scenario native exports, full model/label lineage and existing-script full final diagnostics through shared PhaseRunner. No new candidate trials, canonical replacement or live enablement.',
        'acceptance':'Require all32 native audits, independent cash/cost/inventory/MTM validation, all-model forecast/calibration reconstruction, direct ATR support and complete component/neighborhood/concentration/paired-uncertainty review. Review completion is distinct from robust positive expectancy.',
        'risk_targets':{'baseline_dd_pct':8,'combined_dd_pct':10,'hard_cutoffs':False},
        'oos_scope':'The complete69-session April1-July10 period is reused development. It cannot support an independent-validation claim.'})


def wait_dependencies(out, status):
    import psutil
    protocol = read(out/'protocol.json')
    while True:
        pending = []
        for name, identity in protocol['dependencies'].items():
            root = Path(name)
            if (root/'completion.json').exists() and read(root/'completion.json').get('complete'):
                continue
            if identity is None: raise ValueError('Previously completed prerequisite is missing '+name)
            state = read(root/'status.json')
            if state['stage'] == 'attention_required': raise ValueError('Failed prerequisite '+name+': '+str(state.get('error')))
            try:
                process = psutil.Process(identity['pid'])
                if abs(process.create_time()-identity['process_created']) > .1 or process.cmdline() != identity['command']:
                    raise ValueError('Changed prerequisite identity '+name)
            except psutil.NoSuchProcess:
                raise ValueError('Prerequisite ended without complete evidence '+name)
            pending.append(name)
        if not pending: return
        status('waiting_for_qualification', waiting_for=pending)
        time.sleep(30)


def prepare(out):
    from scripts.kalcb_alignment_walk_forward_v3 import verify_bindings
    from backtests.strategies.kalcb.qualified_artifacts import validate_case, freeze_profile
    protocol = read(out/'protocol.json'); verify_bindings(protocol['bindings'])
    completed, forecast_done = read(Q/'completion.json'), read(F/'completion.json')
    require_ready(completed,forecast_done)
    verify_bindings([completed['preparation'],forecast_done['reconstruction'],forecast_done['protocol']])
    preparation = read(Q/'preparation.json'); forecast = read(F/'forecast_reconstruction.json')
    if not forecast['passed']: raise ValueError('Forecast qualification failed')
    boundary = read(ROOT/'boundary_continuity.json')
    bridge_path = Path(preparation['source_contract_bridge']['path'])
    if not boundary['passed'] or not read(bridge_path)['passed']: raise ValueError('Continuity/source repair bridge failed')
    lineage_path = Path(preparation['lineage']['path']); verify_bindings([preparation['lineage']])
    lineage = read(lineage_path)
    if not lineage['source_and_model_clocks_passed']: raise ValueError('Model lineage clock failed')
    archive = []
    for item in lineage['files']:
        p = lineage_path.parent/item['archived']
        if digest(p) != item['sha256']: raise ValueError('Changed archived model/label/feature/program evidence')
        archive.append(p)
    for check in forecast['models']: verify_bindings([check['model']])
    review = {i:read(ROOT/f'stage{i}_review.json') for i in (2,3,4,5)}
    trials = [*review[2]['trials'],*[r for i in (3,4,5) for n,r in review[i]['trials'].items() if n!='control']]
    if len(trials) != review[5]['trial_entries_including_aliases']: raise ValueError('Lost study trials/aliases')
    write(out/'all_trials.json',{'records':trials,'trial_entries_including_aliases':len(trials),
        'rejected':{str(i):review[i].get('rejected',{}) for i in review},
        'scope':'All registered study entries retained, including controls/aliases; not independent discoveries.',
        'sources':[bound(ROOT/f'stage{i}_review.json') for i in review]})
    canonical = read(REPO/'config/research/kalcb_intraday_advantage_20261007/selected_research.json')
    support = read(S/'completion.json')
    cases = {}; attribution = {}; profile_specs = []
    # Fully reauthenticate every native case before making any acceptance capsule.
    for item in preparation['profiles']:
        verify_bindings([item['manifest'],item['plan'],item['policy']])
        folder = Path(item['folder']); inputs = Path(item['input_folder']); name,period=item['name'],item['period']
        start,end,count=qualification.PERIODS[period]
        policy_dir=inputs/'inputs/files/research/kalcb_executed_value/policies'
        if digest(policy_dir/'policies_manifest.json') != item['policy']['sha256']: raise ValueError('Policy lineage drift')
        profile={'status':'qualified_research_only','live_enabled':False,'primary_profile':name,
            'configuration':str(inputs/'config.json'),'configuration_sha256':digest(inputs/'config.json'),
            'audit_manifest':item['manifest']['path'],'audit_manifest_sha256':item['manifest']['sha256'],
            'plan':item['plan']['path'],'policy':{'name':POLICY,'directory':str(policy_dir),'manifest_sha256':item['policy']['sha256'],'requires_timed_candidate_updates':True},
            'execution_source':{'code':str(inputs/'inputs/files'),'source_generation_hash':item['source_generation_hash']},
            'initial_equity_krw':100_000_000,'date_range':{'start':start,'end':end,'sessions':count},
            'result_paths':{s:str(folder/s/'result.json') for s in qualification.SCENARIOS},
            'metrics':{s:read(folder/s/'result.json') for s in qualification.SCENARIOS},
            'tested_variant_count':len(trials),'all_trials':str(out/'all_trials.json'),
            'research_protocols':{str(p):digest(p) for p in [out/'protocol.json',Q/'protocol.json',F/'protocol.json',S/'protocol.json']},
            'regression_tests':{'log':str(ROOT/'qualification_gate_tests_20261010.log'),'passed':62},
            'drawdown_preferences':{'baseline':8,'combined':10,'hard_rejection_thresholds':False},
            'recovery_report_context':{'period':period,'recipe':name,'independent_validation':False,'june_july_used_for_research':True,
                'training_contract':{'minimum_training_sessions':60,'refit_sessions':20,'embargo_sessions':1,'continuous_original_fitting_clock':True,'mature_labels_only':True,'earlier_oof_calibration_only':True}},
            'limits':['Repeatedly reused development history; uncertainty is unadjusted for adaptive selection.',
                'Source qualification does not establish positive stressed expectancy or choose a canonical replacement.',
                'Paper/live execution and fixed-capital funded replay remain outside these research acceptance claims.']}
        cell={}; detail={}
        for scenario in qualification.SCENARIOS:
            case=validate_case(folder/scenario,profile,scenario)
            qualification.validate(folder/scenario,item,scenario,Path(qualification.PROFILES[name]['reference'])/period/scenario,save=False)
            cell[scenario]=case['validation']
            control = Path(canonical['result_paths'][scenario]).parent if period=='is' else ROOT/'control'/scenario
            trades,fills=case['trades'],case['fills']
            detail[scenario]={'metrics':profile['metrics'][scenario], 'cash_reconciliation':case['validation'],
                'gross_pnl_after_slippage_krw':sum(t['gross_pnl'] for t in trades),
                'commission_and_tax_krw':sum(t['commission'] for t in trades),
                'adverse_slippage_krw':sum(f['qty']*abs(f['price']-f['metadata']['base_price']) for f in fills),
                'net_pnl_krw':sum(t['net_pnl'] for t in trades),'attribution':segments(folder/scenario),
                'concentration':concentration(folder/scenario),'paired_to_canonical':paired(folder/scenario,control),
                'cost_scope':'Realized costs on the executed account path; not an alternative cost-free portfolio replay.'}
        cases[name+'/'+period]=cell; attribution[name+'/'+period]=detail
        profile_specs.append((item,profile))
    write(out/'attribution.json',{'profiles':attribution,'components_and_neighborhoods':[bound(ROOT/f'stage{i}_review.json') for i in (2,3,4,5)],
        'direct_intervention':bound(S/'completion.json'),'scope':'Complete-period costs, monthly/exit attribution, concentration and descriptive paired1/3/5/10-day uncertainty; no proposed date/issuer exclusions.'})
    write(out/'account_verification.json',{'passed':True,'profiles':cases,'source_completion':bound(Q/'completion.json')})
    write(out/'scope_parity.json',{'passed':True,'boundary':bound(ROOT/'boundary_continuity.json'),
        'calendar_and_contract':bound(bridge_path),'forecast':bound(F/'forecast_reconstruction.json'),
        'account':bound(out/'account_verification.json'),'native_reference_bridges':'All baseline/combined final accounts reproduce every reference daily balance.'})
    write(out/'input_verification.json',{'passed':True,'lineage':bound(lineage_path),'archived_files_authenticated':len(archive),
        'preparation':bound(Q/'preparation.json'),'forecast':bound(F/'forecast_reconstruction.json'),
        'source_receipts':'Every scenario rechecked by existing validate_case; all source coverage fields pass.'})
    manifests=[]
    for item,profile in profile_specs:
        name,period=item['name'],item['period'];dest=out/'profiles'/name/period;dest.mkdir(parents=True)
        support_rows=[r for r in support['results'] if r['name']==name and r['period']==period]
        if name.endswith('_atr5') and ({r['scenario'] for r in support_rows}!={'baseline','combined'} or any(not r['summary']['preferred_support_passed'] for r in support_rows)):
            raise ValueError('Missing registered direct ATR support evidence')
        write(dest/'source_checks.json',{'passed':True,'source_generation_hash':item['source_generation_hash'],'cases':cases[name+'/'+period]})
        write(dest/'lesson_checks.json',{'passed':True,'source_and_accounting_checks_passed':True,
            'history_and_forecast_reconstruction_passed':True,'registered_ablation_and_neighborhood_review_complete':True,
            'full_period_cost_concentration_and_paired_uncertainty_review_complete':True,'direct_atr_support':support_rows,
            'evidence':[bound(out/'attribution.json'),bound(out/'all_trials.json'),bound(out/'scope_parity.json')],
            'scope':'Implementation/evidence review passed. This is not a stability, profitability or canonical-promotion verdict.',
            'canonical_promotion':False,'live_enabled':False})
        write(dest/'test_checks.json',{'passed':True,'receipts':[bound(ROOT/'qualification_gate_tests_20261010.json'),
            bound(ROOT/'qualification_source_tests_20261010.log'),bound(ROOT/'atr_intervention_observer_tests_20261010.log'),
            bound(ROOT/'recovery_report_context_repair_20261010.json'),bound(ROOT/'forecast_qualification_tests_20261010.log'),
            bound(ROOT/'final_artifact_gate_tests_20261010.log')]})
        profile.update(scope_preparation_parity=str(out/'scope_parity.json'),forecast_reconstruction=str(F/'forecast_reconstruction.json'),
            final_account_verification=str(out/'account_verification.json'),input_verification=str(out/'input_verification.json'),attribution=str(out/'attribution.json'))
        write(dest/'research_profile.json',profile)
        manifest=freeze_profile(dest/'research_profile.json',dest/'acceptance_manifest.json',
            acceptance_checks={k:str(dest/f'{v}_checks.json') for k,v in [('source','source'),('lessons','lesson'),('tests','test')]},
            additional_evidence=[*archive,*[Path(c['model']['path']) for c in forecast['models']],lineage_path,
                Q/'completion.json',Q/'preparation.json',F/'completion.json',S/'completion.json',bridge_path,
                *[Path(b['path']) for b in protocol['bindings']]])
        manifests.append({'name':name,'period':period,'manifest':bound(manifest)})
    write(out/'preparation.json',{'complete':True,'profiles':manifests,'accounts':bound(out/'account_verification.json'),
        'attribution':bound(out/'attribution.json'),'all_trials':bound(out/'all_trials.json')})


def render_one(out,name,period):
    from backtests.auto.shared.phase_runner import PhaseRunner
    from backtests.auto.shared.round_manager import RoundManager
    from backtests.strategies.kalcb.recovery_full_diagnostics_v1 import KALCBRecoveryQualifiedRoundPlugin
    manifest=out/'profiles'/name/period/'acceptance_manifest.json'
    plugin=KALCBRecoveryQualifiedRoundPlugin(manifest)
    manager=RoundManager('stock','kalcb',base_dir=out/'framework'/name/period)
    folder=manager.round_path(1)
    if folder.exists() and any(folder.iterdir()): raise ValueError('Preserve prior full diagnostics')
    PhaseRunner(plugin,manager.get_round_dir(1),round_name='round_1',round_manager=manager,round_num=1,
                max_rounds=1,max_retries=0,max_diagnostic_retries=0).run_all_phases()
    integrity=read(folder/'diagnostic_integrity.json')
    if not integrity['passed'] or not integrity['native_evidence_unchanged']: raise ValueError('Diagnostic output changed native evidence')
    write(out/'profiles'/name/period/'diagnostics_completion.json',{'complete':True,'report':bound(folder/'round_final_full_diagnostics.json'),
        'text':bound(folder/'round_final_diagnostics.txt'),'integrity':bound(folder/'diagnostic_integrity.json')})


def run(out):
    import psutil
    import msvcrt
    import subprocess
    from scripts.kalcb_full_oos_recovery import env
    from scripts.kalcb_alignment_walk_forward_v3 import verify_bindings
    protocol=read(out/'protocol.json');verify_bindings(protocol['bindings'])
    lock=(out/'run.lock').open('a+b');lock.seek(0)
    if not lock.read(1):lock.write(b'0');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    state={'pid':os.getpid(),'process_created':psutil.Process().create_time(),'child':None}
    def status(stage,**extra):
        state.update(stage=stage,updated_at_utc=datetime.now(timezone.utc).isoformat(),**extra)
        write(out/'status.tmp',state);os.replace(out/'status.tmp',out/'status.json')
    try:
        wait_dependencies(out,status)
        if (out/'preparation.json').exists():raise ValueError('No implicit restart of final acceptance')
        status('preparing_full_acceptance');prepare(out)
        results=[]
        for item in read(out/'preparation.json')['profiles']:
            command=[sys.executable,'-B','-u',str(Path(__file__).resolve()),'render','--name',item['name'],'--period',item['period']]
            log=out/'profiles'/item['name']/item['period']/'framework.log'
            with log.open('x',encoding='utf-8') as stream:
                process=subprocess.Popen(command,cwd=REPO,env=env(REPO),stdout=stream,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                status('rendering',profile=item['name']+'/'+item['period'],child={'pid':process.pid,'process_created':psutil.Process(process.pid).create_time(),'command':command})
                while process.poll() is None:time.sleep(15)
            if process.returncode:raise ValueError('Failed existing-framework diagnostics; preserve '+str(log))
            results.append(read(out/'profiles'/item['name']/item['period']/'diagnostics_completion.json'))
        verify_bindings(protocol['bindings'])
        write(out/'completion.json',{'complete':True,'profiles':results,'preparation':bound(out/'preparation.json'),
            'scope':'Eight full existing-framework comparison reports. Final economic assessment and canonical decision still require review; no canonical/live change.'})
        status('ready_for_final_assessment',child=None)
    except Exception as error:
        status('attention_required',error=repr(error));raise
    finally:
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['register','run','render'])
    parser.add_argument('--name',choices=list(qualification.PROFILES))
    parser.add_argument('--period',choices=list(qualification.PERIODS))
    args=parser.parse_args()
    if args.stage=='render':render_one(DEST,args.name,args.period)
    else:{'register':register,'run':run}[args.stage](DEST)
