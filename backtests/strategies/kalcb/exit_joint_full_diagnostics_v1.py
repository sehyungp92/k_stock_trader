"""Read-only exit qualification reports using the existing full diagnostic engine.

All stress, funding, backout and rejected-prior contexts remain explicit. This
adapter never changes native inputs, qualifies live deployment or forces a winner.
"""
from dataclasses import asdict
from datetime import time
from pathlib import Path
import json
import shutil

from backtests.auto.shared.plugin import PhaseSpec, PhaseAnalysisPolicy
from backtests.auto.shared.types import GateCriterion, EndOfRoundArtifacts
from backtests.strategies.common.plugin_base import attach_official_metric_contract
from strategy_kalcb.config import KALCBConfig
from .qualified_round import KALCBQualifiedRoundPlugin
from .qualified_artifacts import diagnostic_result, require, close
from .late_artifact_validation_v2 import validate_native, validate_case
from .late_entry_v1 import cumulative_timed_policy
from .retention_research_io_v1 import read, write, bound, verify, digest, PERIODS
from .full_diagnostics import (analyze_kalcb_result, render_kalcb_diagnostics_report,
    _final_metrics, _atomic_write_text, _atomic_write_json, _atomic_write_jsonl,
    _artifact_paths, _index_payload)


def authenticated_protocol(output):
    protocol = read(Path(output) / 'protocol.json')
    require(protocol['version'] == 'kalcb-exit-joint-full-reports-v1', 'Wrong report protocol')
    verify(protocol['programs'] + protocol['bindings'])
    require(read(protocol['assessment_completion']['path'])['complete'] is True, 'Incomplete assessment')
    verify([protocol['assessment_completion']])
    require(protocol['native_case_count'] == 40 and protocol['report_count'] == sum(len(cases) for periods in protocol['profiles'].values() for cases in periods.values()),
            'Incomplete screening or report scope')
    require(protocol['report_count']==48 and set(protocol['recipes'])==set(protocol['profiles']),
            'Wrong joint report design')
    return protocol


def load_screening_bundle(output, name, period, scenario):
    protocol=authenticated_protocol(output)
    entry=protocol['profiles'][name][period][scenario]
    folder=Path(entry['folder']);item=read(entry['input_profile']['path']) if 'input_profile' in entry else entry['item']
    verify([item[k] for k in ('manifest','plan','policy')])
    validate_native(folder,item,scenario,save=False)
    start,end,sessions=PERIODS[period];plan=read(item['plan']['path']);policy_dir=Path(item['policy']['path']).parent
    from scripts.kalcb_executed_value_replay import verify_policy
    policy,opening=verify_policy(policy_dir,'interaction_cost_margin')
    updates=cumulative_timed_policy(policy_dir,policy,opening)
    profile={'primary_profile':name,'period':period,'recipe':protocol['recipes'][name],
        'status':'audited_exit_qualification_evidence_only','live_enabled':False,'initial_equity_krw':100000000.,
        'date_range':{'start':start,'end':end,'sessions':sessions},'configuration':plan['config'],
        'audit_manifest':item['manifest']['path'],'execution_source':{'source_generation_hash':item['source_generation_hash']},
        'policy':{'name':'interaction_cost_margin','manifest_sha256':item['policy']['sha256'],'directory':str(policy_dir)},
        'result_paths':{scenario:str(folder/'result.json')},'metrics':{scenario:read(folder/'result.json')}}
    cases={scenario:validate_case(folder,profile,scenario)}
    lineage={'basis':'Registered native source/config/policy/model/label chain, actual allocation overlay if applicable; no new model or refit',
        'input_profile':entry.get('input_profile',item),'assessment':protocol['assessment'],'native_protocol':protocol['native_protocol']}
    from backtests.auto.shared.cache_keys import stable_signature
    return {'profile':profile,'cases':cases,'opening':opening,'updates':updates,
        'policy':{**policy,'prepared_sha256':stable_signature(lineage)},'feature_lineage':lineage,
        'manifest':str(Path(output)/'protocol.json'),'manifest_hash':digest(Path(output)/'protocol.json'),'case_entry':entry}


class ExitJointReportPlugin(KALCBQualifiedRoundPlugin):
    """Only the inherited locked evaluator is reused; no qualification loader."""
    def __init__(self, output, name, period, scenario):
        require(scenario in ('baseline','participation','cost','combined'), 'Unscored stress scenario')
        self.output, self.recipe, self.period, self.scenario = Path(output), name, period, scenario
        self.bundle = load_screening_bundle(output, name, period, scenario)
        self.profile = self.bundle['profile']
        self.acceptance_manifest = Path(self.bundle['manifest'])
        self.source_fingerprint = self.profile['execution_source']['source_generation_hash']
        effective = KALCBConfig.from_mapping(read(self.profile['configuration']))
        self.initial_mutations = json.loads(json.dumps(asdict(effective), default=lambda x: x.isoformat() if isinstance(x, time) else x.value))
        self.config = {'initial_equity': 100000000., 'date_range': self.profile['date_range'],
            'promotion_status': 'audited_exit_qualification_economic_review_only',
            'artifact_promotion_policy': 'no_canonical_or_live_promotion',
            'capability_level': 'real_replay', 'risk_basis': 'native_bar_level_mark_to_market',
            'selected_candidate': name, 'scenario': scenario, 'initial_mutations': self.initial_mutations,
            'live_parity_fill_timing': 'next_5m_open', 'auction_mode': 'non_auction_continuous',
            'report_protocol': str(self.acceptance_manifest), 'required_candidate_policy': self.profile['policy']}
        self.execution_context = {**self.config, 'source_fingerprint': self.source_fingerprint,
            'feature_manifest_hash': self.bundle['policy']['prepared_sha256'],
            'feature_manifest_hash_basis': self.bundle['feature_lineage']['basis'],
            'candidate_snapshot_hash': self.profile['policy']['manifest_sha256'],
            'replay_mode': 'audited_native_timed_candidates', 'expected_universe_size': 103,
            'train_start': self.profile['date_range']['start'], 'train_end': self.profile['date_range']['end']}
        self.result = diagnostic_result(self.bundle, scenario)
        metrics = _final_metrics(self.result.metrics, self.result.trades, self.config,
            self.result.replay_result.equity_curve, timestamps=self.result.replay_result.timestamps)
        self.metrics = attach_official_metric_contract(metrics, requires_audit_pass=True, audit_pass=True,
            execution_contract={**self.execution_context, 'production_enabled': False,
                'metric_units': 'return and drawdown fractions; currency KRW'})

    def get_phase_spec(self, phase, state):
        return PhaseSpec('Locked source/account screening report, no economic promotion', [],
            lambda m: [GateCriterion('screening_source_account_integrity', 1, int(m['audit_pass']), bool(m['audit_pass']))],
            None, {}, PhaseAnalysisPolicy(focus_metrics=['official_mtm_net_return_pct', 'max_drawdown_pct']),
            max_rounds=1, phase_metric_basis='audited_source_replay',
            primary_promotion_metric='official_mtm_net_return_pct',
            official_metric_keys=('official_mtm_net_return_pct', 'max_drawdown_pct'), promotion_requires_audit_pass=True)

    def run_phase_diagnostics(self, phase, state, metrics, greedy_result):
        return json.dumps({'recipe': self.recipe, 'scenario': self.scenario,
            'checks': {s: c['validation'] for s, c in self.bundle['cases'].items()},
            'scope': 'Registered exit stress/backout/funded evidence plus retained prior125 alternatives. Economic judgment remains separately bound; no live qualification.',
            'study_complete': False, 'new_trials': 0}, indent=2)

    run_enhanced_diagnostics = run_phase_diagnostics

    def build_end_of_round_artifacts(self, state):
        self.canonicalize_mutations(state.cumulative_mutations)
        authenticated_protocol(self.output)
        report = self.run_phase_diagnostics(1, state, self.metrics, None)
        return EndOfRoundArtifacts(report, {'screening_integrity': report},
                                   'Exit qualification evidence archived; no canonical or live promotion')

    def write_full_diagnostics(self, state, output_dir, *, round_num=1, round_name=''):
        output_dir = Path(output_dir)
        analysis = analyze_kalcb_result(self.result, config=self.config, mutations=state.cumulative_mutations)
        native = self.bundle['cases'][self.scenario]
        close(analysis['final_metrics']['net_return_pct'] * 100, native['result']['return_pct'], 'Report return drift')
        close(analysis['final_metrics']['max_drawdown_pct'] * 100, native['result']['drawdown_pct'], 'Report DD drift')
        close(analysis['final_metrics']['profit_factor'], native['validation']['net_profit_factor'], 'Report PF drift')
        analysis.update(round=round_num, round_name=round_name, promotion_status='audited_screening_only_no_promotion')
        analysis['gate_result'] = {'status': 'source_account_passed_joint_economic_judgment_bound',
            'production_ready': False, 'four_stress_finalist_qualification': False}
        analysis['live_parity_audit'].update(status='shared_core_only_no_live_qualification',
            observation_watch_count=103, observation_window_minutes=self.bundle['policy'].get('observation_minutes',45),
            active_symbol_count_basis='Policy admission count, not live subscription occupancy',
            paper_promotion_blockers=['Joint return/activity/DD objective and robust dominance are not demonstrated.', 'No independent validation.', 'Paper/live latency and order/restart parity remain unqualified.'])
        analysis['diagnostics_summary']['live_parity_audit'] = analysis['live_parity_audit']
        p = read(self.output / 'protocol.json')
        assessment=read(p['assessment']['path']);entry=self.bundle['case_entry'];key=entry['assessment_key']
        if entry['assessment_kind']=='joint':
            account=assessment['accounts'][key]
        else:
            account=read(p['retained_screening_assessment']['path'])['accounts'][key]
        supplement={'profile':self.profile,'scenario':self.scenario,'full_assessment':p['assessment'],
            'economic_decision':p['economic_decision'],'feature_lineage':self.bundle['feature_lineage'],
            'profile_role':self.profile['recipe']['role'],'native_source_account_checks':{s:c['validation'] for s,c in self.bundle['cases'].items()},
            'actual_intervention_support':account['support'],'profitable_activity':account['activity'],
            'funding':account.get('funding',{'scope':'Retained rejected V12 case; source/account reconciled. No new funded benchmark was run for this rejected prior125 exit.'}),
            'allocation':account.get('allocation'),
            'scope':'All four compounding stresses and hold6/hold0 backouts for prior100; funded hold0/hold8 baseline/combined; unchanged prior125 four-stress alternative and every rejected prior125 exit baseline/combined retained. Identity bridges covered by joint assessment, not independent reports.',
            'management':'Opening-thesis exit only final fallthrough of complete parent cascade. Current gross losers require6/8heldbars and two completed confirmations. Currently profitable positions retain original cascade, but later winners can still be sacrificed.',
            'history':'April1 retains217prior intraday sessions, inherited fits and original embargo/clock. April-July reused development, not independent validation.',
            'limitations':['Same-state parent-no-exit probes plus actual submitted/filled new exits establish intervention support, not isolated saved-loss PnL.',
                'Partial fills, retries, account descendants and reused selection are dependent; paired5/10session10000repseed20261011 sensitivity is conditional and not multiplicity adjusted.',
                'Fixed benchmark preserves actual funding/cash/equity and2x intraday leverage; no reset/replenishment/external flow or post-hoc scaling. Nominal100m allocation reference is not a constant-balance account.',
                'Allocation mark timestamps are not decision-clock funding proof. Retrospective MFE is not recoverable PnL.'],
            'production_enabled':False,'canonical_replaced':False}
        analysis['diagnostics_summary']['audited_exit_qualification_context']=supplement
        report = render_kalcb_diagnostics_report(analysis)
        report = report.replace('Full-universe alpha', 'Non-selected outcome evidence').replace('Intraday Alpha Curve', 'Intraday realized PnL attribution')
        report += '\n\nAUDITED EXIT JOINT QUALIFICATION CONTEXT — NO ECONOMIC PROMOTION\n' + json.dumps(supplement, indent=2, default=str) + '\n'
        _atomic_write_text(output_dir / 'round_final_diagnostics.txt', report)
        for filename, data in [('diagnostics_summary.json', analysis['diagnostics_summary']),
            ('candidate_frontier.json', analysis['candidate_frontier']), ('live_parity_audit.json', analysis['live_parity_audit']),
            ('round_final_full_diagnostics.json', {k: v for k, v in analysis.items() if k not in ('trades', 'decisions')}),
            ('qualification_context.json', supplement)]:
            _atomic_write_json(data, output_dir / filename)
        _atomic_write_jsonl(output_dir / 'trade_events.jsonl', analysis['trade_events'])
        _atomic_write_jsonl(output_dir / 'decision_events.jsonl', [d.to_json_dict() for d in self.result.decisions])
        shutil.copy2(self.profile['configuration'], output_dir / 'strategy_config.json')
        paths = _artifact_paths(output_dir)
        paths.update({n: str(output_dir / n) for n in ('round_final_full_diagnostics.json', 'qualification_context.json', 'decision_events.jsonl', 'strategy_config.json', 'diagnostic_integrity.json')})
        _atomic_write_json(_index_payload(analysis, paths, {'screening_protocol': str(self.acceptance_manifest)}), output_dir / 'full_diagnostics_index.json')
        authenticated_protocol(self.output)
        write(output_dir / 'diagnostic_integrity.json', {'passed': True, 'native_evidence_unchanged': True,
            'scenario': self.scenario, 'report': bound(output_dir / 'round_final_diagnostics.txt'),
            'native_return_pct': native['result']['return_pct'], 'native_mtm_drawdown_pct': native['result']['drawdown_pct'],
            'metric_units': 'Canonical return/DD fractions, native table percent; KRW currency', 'economic_promotion': False})
        return analysis['diagnostics_summary']
