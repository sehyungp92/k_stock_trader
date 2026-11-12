"""Population-removal research context around the existing full report engine."""
import json
from pathlib import Path
import shutil
from .qualified_artifacts import read, digest, require, close, diagnostic_result, load_qualified_profile
from .recovery_full_diagnostics_v1 import recovery_report_context
from .recovery_full_diagnostics_v2 import KALCBRecoveryQualifiedRoundPlugin
from .full_diagnostics import (analyze_kalcb_result, render_kalcb_diagnostics_report,
    _atomic_write_text, _atomic_write_json, _atomic_write_jsonl, _artifact_paths, _index_payload)


def population_report_context(profile, capsule):
    context = recovery_report_context(profile)
    variant = profile['population_report_context']
    parent, prior = variant['parent'], variant['prior']
    require(parent in ('expanding','expanding_atr5') and prior in (100,125), 'Unsupported final comparison')
    require(context['recipe']==parent, 'Parent training family mismatch')
    require(variant['recipe']==parent+'_removal_prior'+str(prior)==profile['primary_profile'], 'Mislabelled population variant')
    required = {'lookback_sessions':60,'refit_sessions':5,'embargo_sessions':1,
        'minimum_count':20,'minimum_sessions':8,'local_minimum_count':8,'local_minimum_sessions':5,
        'unsupported_identity':True,'mature_labels_only':True,'checkpoint_readmission':True,
        'selection':'exact_supported_nonpositive_removal_only'}
    require(variant['calibration_contract']==required,'Changed population contract')
    reference = profile['funded_allocation_evidence']
    path = str(Path(reference['path']).resolve())
    require(capsule['files'].get(path)==reference['sha256']==digest(path), 'Unbound funded benchmark')
    evidence = read(path)
    require(evidence.get('passed') is True and evidence['recipe']==variant['recipe'] and
        evidence['period']==context['period'], 'Wrong or failed funded benchmark')
    require(evidence['reference_krw']==100_000_000. and evidence['external_capital_flows']==0 and
        evidence['cash_resets']==0 and evidence['four_identity_bridges_passed'] is True, 'Unsupported benchmark claim')
    require(set(evidence['cases'])=={'baseline','combined'},'Missing funded benchmark scenarios')
    for b in evidence['bindings']:
        p=str(Path(b['path']).resolve())
        require(capsule['files'].get(p)==b['sha256']==digest(p),'Changed funded benchmark evidence')
    return {**context,'parent_recipe':parent,'recipe':variant['recipe'],'population_calibration':variant,
        'training':context['training']+f' Population residual calibration uses a trailing60-session mature history, original5-session update clock, one-session embargo and prior count{prior}; pool support20issuer-days/8sessions and local support8/5. Unsupported cells retain the parent action. Remove only an incumbent exact supported nonpositive action; independent checkpoints may readmit.',
        'funded_allocation_comparison':evidence,
        'funded_scope':'Native fixed100mKRW allocation cash/heat reference minus invested cost. Actual cash, borrowing, equity, risk state and fills retained. No replenishment or post-hoc rescaling. Fixed allocation is a benchmark, not the reported production policy.'}


def write_population_full_diagnostics(bundle, config, mutations, output_dir, *, round_num=1, round_name=''):
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    profile=bundle['profile'];context=population_report_context(profile,read(bundle['manifest']))
    result=diagnostic_result(bundle)
    analysis=analyze_kalcb_result(result,config=config,mutations=mutations)
    analysis.update(round=round_num,round_name=round_name,promotion_status='qualified_research_comparison_not_canonical')
    analysis['gate_result']={'status':'source_qualified_research_comparison','checks':{
        'four_source_audits':True,'frozen_inputs':True,'cash_cost_trade_mtm_reconciliation':True,
        'funded_fixed_allocation_comparison':True},'production_ready':False}
    analysis['live_parity_audit'].update(status='shared_core_contract_only_pending_paper_execution',
        observation_watch_count=103,observation_window_minutes=45,
        active_symbol_count_basis='Admitted candidate count; not verified live subscription occupancy',
        paper_promotion_blockers=['Full103-name prefix and model update delivery latency remains unqualified live.',
            'KIS paper order/fill/restart parity remains required.','Independent forward validation remains required.'])
    analysis['diagnostics_summary']['live_parity_audit']=analysis['live_parity_audit']
    scenarios={s:c['result'] for s,c in bundle['cases'].items()}
    supplemental={'acceptance_manifest':bundle['manifest'],'acceptance_manifest_hash':bundle['manifest_hash'],
        'scenarios':{s:{k:r[k] for k in ('return_pct','drawdown_pct','trades','active_days','audit_status')} for s,r in scenarios.items()},
        'scenario_checks':{s:c['validation'] for s,c in bundle['cases'].items()},
        'date_range':profile['date_range'],'initial_equity_krw':profile['initial_equity_krw'],
        'training':context['training'],'validation_scope':context['validation_scope'],'population_context':context,
        'policy':profile['policy'],'audit_manifest':profile['audit_manifest'],
        'forecast_reconstruction':profile['forecast_reconstruction'],'population_lineage':profile['population_lineage'],
        'drawdown_targets_are_soft':True,'original_trials':profile['all_trials'],'new_search_trials':0,
        'production_enabled':False,'canonical_replaced':False}
    close(analysis['final_metrics']['net_return_pct']*100,scenarios['baseline']['return_pct'],'Diagnostic return drift')
    close(analysis['final_metrics']['max_drawdown_pct']*100,scenarios['baseline']['drawdown_pct'],'Diagnostic drawdown drift')
    close(analysis['final_metrics']['profit_factor'],bundle['cases']['baseline']['validation']['net_profit_factor'],'Diagnostic PF drift')
    report=render_kalcb_diagnostics_report(analysis)
    report=report.replace('Full-universe alpha','Non-selected outcome evidence').replace('Intraday Alpha Curve','Intraday realized PnL attribution')
    report+='\n\nQUALIFIED POPULATION REMOVAL AND FUNDED ALLOCATION EVIDENCE\n'+json.dumps(supplemental,indent=2,default=str)+'\n'
    report+=('\nSortino uses exchange-session close-to-close returns,252sessions/year and zero downside target.\n'
        'Calmar uses annualized session return and timestamped bar-level MTM drawdown.\n'
        'MFE alternatives are hindsight path descriptions. Native funding comparisons retain actual account feedback; they do not identify isolated causal PnL.\n'
        'April-July2026 is reused development, not independent validation. Source/account qualification does not establish positive stressed expectancy or live readiness.\n')
    analysis['diagnostics_summary']['qualified_native_replay']=supplemental
    _atomic_write_text(output_dir/'round_final_diagnostics.txt',report)
    for name,value in [('diagnostics_summary.json',analysis['diagnostics_summary']),('candidate_frontier.json',analysis['candidate_frontier']),
        ('live_parity_audit.json',analysis['live_parity_audit']),('round_final_full_diagnostics.json',{k:v for k,v in analysis.items() if k not in ('trades','decisions')}),
        ('qualified_replay_diagnostics.json',supplemental),('selected_research_profile.json',profile)]:
        _atomic_write_json(value,output_dir/name)
    _atomic_write_jsonl(output_dir/'trade_events.jsonl',analysis['trade_events'])
    _atomic_write_jsonl(output_dir/'decision_events.jsonl',[d.to_json_dict() for d in result.decisions])
    shutil.copy2(profile['configuration'],output_dir/'strategy_config.json')
    shutil.copy2(bundle['manifest'],output_dir/'acceptance_manifest.json')
    paths=_artifact_paths(output_dir)
    for name in ('round_final_full_diagnostics.json','qualified_replay_diagnostics.json','decision_events.jsonl',
        'selected_research_profile.json','strategy_config.json','acceptance_manifest.json','diagnostic_integrity.json'):
        paths[name]=str(output_dir/name)
    _atomic_write_json(_index_payload(analysis,paths,{'qualified_profile':bundle['manifest'],'native_audit':profile['audit_manifest']}),output_dir/'full_diagnostics_index.json')
    load_qualified_profile(bundle['manifest'])
    population_report_context(profile,read(bundle['manifest']))
    _atomic_write_json({'passed':True,'native_evidence_unchanged':True,'report_sha256':digest(output_dir/'round_final_diagnostics.txt'),
        'metric_units':'Fractions for canonical return/drawdown; explicit percent in scenario tables'},output_dir/'diagnostic_integrity.json')
    return analysis['diagnostics_summary']


class KALCBPopulationQualifiedRoundPlugin(KALCBRecoveryQualifiedRoundPlugin):
    """Inherit locked shared-phase acceptance and feature-lineage adaptation."""
    def __init__(self, acceptance_manifest):
        super().__init__(acceptance_manifest)
        self.population_context=population_report_context(self.profile,read(acceptance_manifest))
        self.config['population_variant']=self.profile['primary_profile']
        self.execution_context['population_variant']=self.profile['primary_profile']

    def write_full_diagnostics(self,state,output_dir,**kwargs):
        return write_population_full_diagnostics(self.bundle,self.config,state.cumulative_mutations,output_dir,**kwargs)
