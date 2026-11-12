"""Bind completed population qualification and render existing full diagnostics.

Report-only shared PhaseRunner lifecycle. No trading mutations or canonical writes.
"""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from backtests.strategies.kalcb.retention_research_io_v1 import read,write,bound,verify,digest,PERIODS
from backtests.strategies.kalcb.qualified_artifacts import freeze_profile,load_qualified_profile,validate_case
from scripts.kalcb_funded_qualification_assessment_v1 import require_complete

PARENTS=Path('E:/CodexResearch/k_stock_trader/kalcb_recovery_final_20261010/v2')
REPORT_PROGRAMS=('scripts/kalcb_population_final_artifacts_v1.py',
    'scripts/kalcb_funded_qualification_assessment_v1.py',
    'backtests/strategies/kalcb/population_full_diagnostics_v1.py',
    'tests/backtests/strategies/test_kalcb_population_final_reports_v1.py')


def candidates(protocol):
    return {n:r for n,r in protocol['recipes'].items() if r['prior'] in (100,125)}


def authenticate_capsule(path):
    from backtests.auto.shared.cache_keys import stable_signature
    capsule=read(path);payload={k:v for k,v in capsule.items() if k!='manifest_hash'}
    if stable_signature(payload)!=capsule['manifest_hash']:
        raise ValueError('Changed inherited acceptance capsule')
    verify([{'path':p,'sha256':h} for p,h in capsule['files'].items()])
    return capsule


def prepare(root,out,assessment,tests):
    if out.exists():raise ValueError('Preserve previous final artifacts')
    require_complete(root)
    p=read(root/'protocol.json');verify(p['programs']+p['bindings'])
    a=read(assessment);test=read(tests)
    if not a['all_cases_source_account_passed'] or a['fresh_native_cases']!=36 or a['fixed_cases']!=24 or a['compounding_cases']!=56:
        raise ValueError('Incomplete final assessment')
    verify(a['inputs']+[a['program']]+test['programs']+[test['log']])
    if test['exit_code']!=0:raise ValueError('Failed final-report tests')
    records={r['recipe']:r for r in a['trials']}
    selected=candidates(p)
    if len(selected)!=4 or len(a['identity_bridges'])!=4:raise ValueError('Wrong finalist or bridge population')
    old_capsules={};old_profiles={};archive={}
    for family in ('expanding','expanding_atr5'):
        for period in ('is','oos'):
            path=PARENTS/'profiles'/family/period/'acceptance_manifest.json'
            capsule=authenticate_capsule(path)
            profile=read(capsule['profile'])
            old_capsules[family,period]=bound(path);old_profiles[family,period]=profile
            archive.update(capsule['files'])
    # Bind all earlier finite trials and rejected entries; this report does not
    # turn separate research stages into independent discoveries.
    trial_sources=[Path(old_profiles['expanding','is']['all_trials'])]
    trial_sources += [root.parent/s/'ablation_ledger.json' for s in ('v3','v4_retention','v5_population')]
    trial_sources += [Path(p['qualification'])/'qualification_ledger.json',root/'compounding/qualification_ledger.json',root/'allocation_ledger.json']
    trial_records=[]
    for path in trial_sources:
        source=read(path);rows=source.get('trials',source.get('records',[]))
        if not rows:raise ValueError('Lost earlier trial records '+str(path))
        trial_records.append({'stage':str(path.parent),'binding':bound(path),'records':rows})
    out.mkdir(parents=True)
    inputs=[bound(root/f) for f in ('protocol.json','study.json','stage_completion.json','progress.json','prior125_support_transfer.json')]
    inputs += [bound(assessment),bound(tests),test['log']]+test['programs']+[bound(x) for x in trial_sources]+list(old_capsules.values())
    inputs += p['programs']+p['bindings']
    write(out/'protocol.json',{'registered_at_utc':datetime.now(timezone.utc).isoformat(),
        'programs':[bound(REPO/s) for s in REPORT_PROGRAMS],'bindings':inputs,
        'profiles':selected,'periods':PERIODS,'assessment':bound(assessment),'tests':test,
        'scope':'Eight report-only isolated comparison profiles via shared PhaseRunner and existing full diagnostic engine. No new policy trials, canonical replacement or live enablement.'})
    write(out/'all_trials.json',{'stages':trial_records,'trial_entries_including_controls_aliases_and_repeated_qualifications':sum(len(x['records']) for x in trial_records),
        'scope':'Complete earlier trial records and all current stress/funded comparisons retained. Repeated cells/controls are dependent evidence, not additional discoveries.'})
    # Population labels/model/fit proofs are already bound in the v5/v6/v7
    # protocols. Verify those bindings and carry the exact files into each capsule.
    lineage=p['bindings']+p['programs']
    q=read(Path(p['qualification'])/'protocol.json')
    for path in (Path(p['qualification'])/'build/completion.json',Path(p['qualification'])/'build/proofs.json'):
        if not path.exists():raise ValueError('Missing population calibration lineage '+str(path))
        lineage.append(bound(path))
    verify(lineage)
    write(out/'population_lineage.json',{'passed':True,'bindings':lineage,
        'inherited_parent_capsules':list(old_capsules.values()),
        'scope':'Authenticated inherited feature/model/label reconstruction plus fresh mature population calibration proofs and original-clock policy lineage. No future outcomes introduced.'})
    write(out/'account_verification.json',{'passed':True,'assessment':bound(assessment),'identities':a['identity_bridges'],
        'compounding_cases':56,'funded_fixed_cases':24})
    write(out/'input_verification.json',{'passed':True,'lineage':bound(out/'population_lineage.json'),
        'prior_acceptance_capsules':list(old_capsules.values()),'inherited_files_authenticated':len(archive)})
    manifests=[]
    for name,spec in selected.items():
        for period in ('is','oos'):
            family=spec['parent'];row=records[name];item=p['profiles'][name][period]
            dest=out/'profiles'/name/period;dest.mkdir(parents=True)
            inherited=old_profiles[family,period]
            if inherited['execution_source']['source_generation_hash']!=item['source_generation_hash']:
                raise ValueError('Unexpected parent economic source lineage')
            plan=read(item['plan']['path']);config=Path(plan['config'])
            baseline=root/'compounding/cases'/name/period
            funding_bindings=[];funded_cases={}
            parent='control' if family=='expanding' else 'control_expanding_atr5'
            for scenario in ('baseline','combined'):
                f=row['funded_fixed'][period+'/'+scenario];funded_cases[scenario]={
                    'metrics':f['account']['metrics'],'vs_local_parent':f['vs_local_parent'],
                    'policy_delta_fixed_minus_compounding':f['policy_delta_fixed_minus_compounding'],
                    'funding':f['account']['funding']}
                for recipe in (name,parent):
                    folder=root/'cases'/recipe/period/scenario
                    funding_bindings += [bound(folder/x) for x in ('result.json','daily.json','trades.json','fills.json','equity.json',
                        'orders.jsonl.gz','execution_audit.json','independent_account_verification.json','allocation_reference.json','funding_review.json')]
                    fp=read(folder.parent/'input_profile.json')
                    funding_bindings += [fp[k] for k in ('manifest','plan','policy')]
            for b in a['identity_bridges']:
                funding_bindings += [b['bridge'],b['account']]
            funding_bindings += [bound(root/'stage_completion.json'),bound(assessment)]
            write(dest/'funded_comparison.json',{'passed':True,'recipe':name,'period':period,'reference_krw':100_000_000.,
                'external_capital_flows':0,'cash_resets':0,'four_identity_bridges_passed':True,
                'cases':funded_cases,'bindings':funding_bindings,
                'scope':'Actual native funded fixed allocation reference, not replenished constant balance. No assertion of positive expectancy or isolated causal PnL.'})
            profile=deepcopy(inherited)
            profile.update(primary_profile=name,configuration=str(config),configuration_sha256=digest(config),
                audit_manifest=item['manifest']['path'],audit_manifest_sha256=item['manifest']['sha256'],plan=item['plan']['path'],
                policy={'name':'interaction_cost_margin','directory':str(Path(item['policy']['path']).parent),
                        'manifest_sha256':item['policy']['sha256'],'requires_timed_candidate_updates':True},
                execution_source={'code':str(Path(item['folder'])/'inputs/files'),'source_generation_hash':item['source_generation_hash']},
                result_paths={s:str(baseline/s/'result.json') for s in ('baseline','participation','cost','combined')},
                metrics={s:read(baseline/s/'result.json') for s in ('baseline','participation','cost','combined')},
                tested_variant_count=sum(len(x['records']) for x in trial_records),all_trials=str(out/'all_trials.json'),
                final_account_verification=str(out/'account_verification.json'),input_verification=str(out/'input_verification.json'),
                attribution=str(assessment),population_lineage=bound(out/'population_lineage.json'),
                funded_allocation_evidence=bound(dest/'funded_comparison.json'),
                regression_tests={'log':test['log']['path'],'passed':test['tests']})
            profile['research_protocols'].update({str(out/'protocol.json'):digest(out/'protocol.json'),
                str(root/'protocol.json'):digest(root/'protocol.json'),str(Path(p['qualification'])/'protocol.json'):digest(Path(p['qualification'])/'protocol.json')})
            profile['population_report_context']={'recipe':name,'parent':family,'prior':spec['prior'],
                'calibration_contract':{'lookback_sessions':60,'refit_sessions':5,'embargo_sessions':1,
                    'minimum_count':20,'minimum_sessions':8,'local_minimum_count':8,'local_minimum_sessions':5,
                    'unsupported_identity':True,'mature_labels_only':True,'checkpoint_readmission':True,
                    'selection':'exact_supported_nonpositive_removal_only'}}
            profile['limits']=['April-July is repeatedly reused development; paired uncertainty is not selection-adjusted.',
                'Source/account qualification does not establish positive stressed expectancy or authorize promotion.',
                'Fixed allocation benchmark preserves actual funding and risk; it is not a replenished constant-balance account.',
                'Paper/live latency, order/fill/restart parity and independent forward validation remain outstanding.']
            validations={s:validate_case(baseline/s,profile,s)['validation'] for s in profile['result_paths']}
            write(dest/'source_checks.json',{'passed':True,'source_generation_hash':item['source_generation_hash'],'cases':validations})
            write(dest/'lesson_checks.json',{'passed':True,'source_account_model_lineage':True,
                'all_trials_and_finite_neighborhoods_retained':True,'native_fixed_allocation_comparison_complete':True,
                'evidence':[bound(assessment),bound(out/'all_trials.json'),bound(out/'population_lineage.json'),bound(root/'prior125_support_transfer.json')],
                'scope':'Implementation and evidence checks only; not a profitability or canonical-promotion verdict.','canonical_promotion':False})
            write(dest/'test_checks.json',{'passed':True,'receipts':[bound(tests),p['tests']['log'],test['log']]})
            write(dest/'research_profile.json',profile)
            extra=[Path(x) for x in archive]+[Path(b['path']) for b in inputs+lineage+funding_bindings]
            extra += [out/'population_lineage.json',dest/'funded_comparison.json',root/'prior125_support_transfer.json']
            extra += [REPO/s for s in REPORT_PROGRAMS]+[REPO/'docs/strategy-implementation-lessons.md']
            manifest=freeze_profile(dest/'research_profile.json',dest/'acceptance_manifest.json',
                acceptance_checks={k:str(dest/f'{v}_checks.json') for k,v in [('source','source'),('lessons','lesson'),('tests','test')]},
                additional_evidence=sorted(set(extra)))
            manifests.append({'name':name,'period':period,'manifest':bound(manifest)})
    verify(p['programs']+p['bindings'])
    write(out/'preparation.json',{'complete':True,'profiles':manifests,'study_complete':False,'final_economic_decision_pending':True})
    print(str(out/'preparation.json'),flush=True)


def render(out,name,period):
    from backtests.auto.shared.phase_runner import PhaseRunner
    from backtests.auto.shared.round_manager import RoundManager
    from backtests.strategies.kalcb.population_full_diagnostics_v1 import KALCBPopulationQualifiedRoundPlugin
    protocol=read(out/'protocol.json');verify(protocol['programs']+protocol['bindings'])
    if name not in protocol['profiles'] or period not in PERIODS:raise ValueError('Unregistered report')
    prepared=read(out/'preparation.json');verify([r['manifest'] for r in prepared['profiles']])
    dest=out/'profiles'/name/period
    plugin=KALCBPopulationQualifiedRoundPlugin(dest/'acceptance_manifest.json')
    manager=RoundManager('stock','kalcb',base_dir=out/'framework'/name/period)
    folder=manager.round_path(1)
    if folder.exists() and any(folder.iterdir()):raise ValueError('Preserve prior full diagnostics')
    PhaseRunner(plugin,manager.get_round_dir(1),round_name=name,round_manager=manager,round_num=1,
        max_rounds=1,max_retries=0,max_diagnostic_retries=0).run_all_phases()
    integrity=read(folder/'diagnostic_integrity.json')
    if not integrity['passed'] or not integrity['native_evidence_unchanged']:raise ValueError('Report changed native evidence')
    write(dest/'diagnostics_completion.json',{'complete':True,'report':bound(folder/'round_final_full_diagnostics.json'),
        'text':bound(folder/'round_final_diagnostics.txt'),'integrity':bound(folder/'diagnostic_integrity.json')})
    print(str(dest/'diagnostics_completion.json'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('prepare','render'))
    for key in ('root','output','assessment','tests'):p.add_argument('--'+key,type=Path,required=key=='output')
    p.add_argument('--name');p.add_argument('--period');a=p.parse_args()
    if a.stage=='prepare':prepare(a.root,a.output,a.assessment,a.tests)
    else:render(a.output,a.name,a.period)
