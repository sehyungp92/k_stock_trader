"""Read-only screening reports through the existing engine and shared lifecycle.

This adapter never qualifies a rejected candidate, launches a native replay, or
changes an economic input. Every scenario gets its own complete engine report.
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
from .late_incremental_validation_v1 import validate_native, validate_case
from .late_entry_v1 import cumulative_timed_policy
from .retention_research_io_v1 import read, write, bound, verify, digest, PERIODS
from .full_diagnostics import (analyze_kalcb_result, render_kalcb_diagnostics_report,
    _final_metrics, _atomic_write_text, _atomic_write_json, _atomic_write_jsonl,
    _artifact_paths, _index_payload)


def authenticated_protocol(output):
    protocol = read(Path(output) / 'protocol.json')
    require(protocol['version'] == 'kalcb-late-incremental-full-reports-v1', 'Wrong report protocol')
    verify(protocol['programs'] + protocol['bindings'])
    require(read(protocol['assessment_completion']['path'])['complete'] is True, 'Incomplete assessment')
    verify([protocol['assessment_completion']])
    require(protocol['native_case_count'] == 32 and protocol['report_count'] == 32,
            'Incomplete screening or report scope')
    return protocol


def load_screening_bundle(output, name, period):
    protocol = authenticated_protocol(output)
    root = Path(protocol['native_root'])
    require(name in protocol['recipes'] and period in PERIODS, 'Unregistered report profile')
    folder = root / 'cases' / name / period
    item = read(folder / 'input_profile.json')
    for scenario in ('baseline', 'combined'):
        validate_native(folder / scenario, item, scenario, save=False)
    start, end, sessions = PERIODS[period]
    plan = read(item['plan']['path'])
    policy_dir = Path(item['policy']['path']).parent
    from scripts.kalcb_executed_value_replay import verify_policy
    policy, opening = verify_policy(policy_dir, 'interaction_cost_margin')
    updates = cumulative_timed_policy(policy_dir, policy, opening)
    profile = {'primary_profile': name, 'period': period, 'recipe': protocol['recipes'][name],
        'status': 'audited_screening_only', 'live_enabled': False,
        'initial_equity_krw': 100000000., 'date_range': {'start': start, 'end': end, 'sessions': sessions},
        'configuration': plan['config'], 'audit_manifest': item['manifest']['path'],
        'execution_source': {'source_generation_hash': item['source_generation_hash']},
        'policy': {'name': 'interaction_cost_margin', 'manifest_sha256': item['policy']['sha256'],
                   'directory': str(policy_dir)},
        'result_paths': {s: str(folder / s / 'result.json') for s in ('baseline', 'combined')},
        'metrics': {s: read(folder / s / 'result.json') for s in ('baseline', 'combined')}}
    cases = {s: validate_case(folder / s, profile, s) for s in ('baseline', 'combined')}
    # Explicit compatibility metadata for the old report slot, never native mutation.
    lineage = {'basis': 'Aggregate registered source, model, label and policy evidence; not a single historical preparation',
               'input_profile': bound(folder / 'input_profile.json'),
               'assessment': protocol['assessment'], 'native_protocol': bound(root / 'protocol.json')}
    from backtests.auto.shared.cache_keys import stable_signature
    feature_hash = stable_signature(lineage)
    return {'profile': profile, 'cases': cases, 'opening': opening, 'updates': updates,
            'policy': {**policy, 'prepared_sha256': feature_hash}, 'feature_lineage': lineage,
            'manifest': str(Path(output) / 'protocol.json'), 'manifest_hash': digest(Path(output) / 'protocol.json')}


class LateIncrementalScreeningReportPlugin(KALCBQualifiedRoundPlugin):
    """Only the inherited locked evaluator is reused; no qualification loader."""
    def __init__(self, output, name, period, scenario):
        require(scenario in ('baseline', 'combined'), 'Unscored stress scenario')
        self.output, self.recipe, self.period, self.scenario = Path(output), name, period, scenario
        self.bundle = load_screening_bundle(output, name, period)
        self.profile = self.bundle['profile']
        self.acceptance_manifest = Path(self.bundle['manifest'])
        self.source_fingerprint = self.profile['execution_source']['source_generation_hash']
        effective = KALCBConfig.from_mapping(read(self.profile['configuration']))
        self.initial_mutations = json.loads(json.dumps(asdict(effective), default=lambda x: x.isoformat() if isinstance(x, time) else x.value))
        self.config = {'initial_equity': 100000000., 'date_range': self.profile['date_range'],
            'promotion_status': 'audited_screening_rejected_for_oos_objective',
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
            'scope': 'Completed baseline/combined screening. No new finalist; no four-stress or live qualification claim.',
            'study_complete': False, 'new_trials': 0}, indent=2)

    run_enhanced_diagnostics = run_phase_diagnostics

    def build_end_of_round_artifacts(self, state):
        self.canonicalize_mutations(state.cumulative_mutations)
        authenticated_protocol(self.output)
        report = self.run_phase_diagnostics(1, state, self.metrics, None)
        return EndOfRoundArtifacts(report, {'screening_integrity': report},
                                   'Screening evidence archived; no canonical or live promotion')

    def write_full_diagnostics(self, state, output_dir, *, round_num=1, round_name=''):
        output_dir = Path(output_dir)
        analysis = analyze_kalcb_result(self.result, config=self.config, mutations=state.cumulative_mutations)
        native = self.bundle['cases'][self.scenario]
        close(analysis['final_metrics']['net_return_pct'] * 100, native['result']['return_pct'], 'Report return drift')
        close(analysis['final_metrics']['max_drawdown_pct'] * 100, native['result']['drawdown_pct'], 'Report DD drift')
        close(analysis['final_metrics']['profit_factor'], native['validation']['net_profit_factor'], 'Report PF drift')
        analysis.update(round=round_num, round_name=round_name, promotion_status='audited_screening_only_no_promotion')
        analysis['gate_result'] = {'status': 'source_account_screening_passed_economic_objective_not_achieved',
            'production_ready': False, 'four_stress_finalist_qualification': False}
        analysis['live_parity_audit'].update(status='shared_core_only_no_live_qualification',
            observation_watch_count=103, observation_window_minutes=60,
            active_symbol_count_basis='Policy admission count, not live subscription occupancy',
            paper_promotion_blockers=['No new OOS improvement demonstrated.', 'No independent validation.', 'Paper/live latency and order/restart parity remain unqualified.'])
        analysis['diagnostics_summary']['live_parity_audit'] = analysis['live_parity_audit']
        p = read(self.output / 'protocol.json')
        assessment = read(p['assessment']['path'])
        component = [x for x in assessment['observations'] if x['name'] == self.recipe
            and x['period'] == self.period and x['scenario'] == self.scenario]
        funnel = [x for x in assessment['label_calibration']['selection_funnels']
            if x['name'] == self.recipe and x['period'] == self.period]
        supplement = {'profile': self.profile, 'scenario': self.scenario,
            'full_assessment': p['assessment'], 'economic_decision': p['economic_decision'],
            'profile_role': 'unchanged competing parent' if self.recipe.startswith('control') else 'incremental component trial',
            'incremental_component_evidence': component,
            'forecast_admission_funnel': funnel,
            'feature_lineage': self.bundle['feature_lineage'],
            'native_source_account_checks': {s: c['validation'] for s, c in self.bundle['cases'].items()},
            'scope': 'Baseline and combined screening only. All rejected trials retained. No new finalist or canonical promotion.',
            'incremental_context': {'model_lineage':'Original registered V13 models/policies, authenticated by storage-only V15 successor','targets':'Exact quarter isolated full-cost labels; equity-bps ranking, planned-notional-bps reserve','unsupported':'Maps abstain; unknown pre-IS snapshot membership abstains','aliases':'Zero policy admissions require authenticated native path equality; aliases are not independent trials','scope':'New-issuer snapshot membership is not proof of actual funded capacity'},
            'history': 'Continuous source/model history: April1 has217 prior sessions. No OOS warm-up reset. April-July is reused development.',
            'limitations': ['Paired block uncertainty is conditional on reused selection, not multiplicity-adjusted.',
                'Admissions are not fills; dependent quantity, retry and funding descendants are not independent evidence.',
                'MFE is hindsight, not recoverable PnL. Slippage is already embedded in net equity.',
                'No new funded or four-stress qualification was run for inactive incremental component variants. Parent qualification remains separately bound evidence.'],
            'production_enabled': False, 'canonical_replaced': False}
        analysis['diagnostics_summary']['audited_screening_context'] = supplement
        report = render_kalcb_diagnostics_report(analysis)
        report = report.replace('Full-universe alpha', 'Non-selected outcome evidence').replace('Intraday Alpha Curve', 'Intraday realized PnL attribution')
        report += '\n\nAUDITED LATE INCREMENTAL SCREENING CONTEXT — NO ECONOMIC PROMOTION\n' + json.dumps(supplement, indent=2, default=str) + '\n'
        _atomic_write_text(output_dir / 'round_final_diagnostics.txt', report)
        for filename, data in [('diagnostics_summary.json', analysis['diagnostics_summary']),
            ('candidate_frontier.json', analysis['candidate_frontier']), ('live_parity_audit.json', analysis['live_parity_audit']),
            ('round_final_full_diagnostics.json', {k: v for k, v in analysis.items() if k not in ('trades', 'decisions')}),
            ('screening_context.json', supplement)]:
            _atomic_write_json(data, output_dir / filename)
        _atomic_write_jsonl(output_dir / 'trade_events.jsonl', analysis['trade_events'])
        _atomic_write_jsonl(output_dir / 'decision_events.jsonl', [d.to_json_dict() for d in self.result.decisions])
        shutil.copy2(self.profile['configuration'], output_dir / 'strategy_config.json')
        paths = _artifact_paths(output_dir)
        paths.update({n: str(output_dir / n) for n in ('round_final_full_diagnostics.json', 'screening_context.json', 'decision_events.jsonl', 'strategy_config.json', 'diagnostic_integrity.json')})
        _atomic_write_json(_index_payload(analysis, paths, {'screening_protocol': str(self.acceptance_manifest)}), output_dir / 'full_diagnostics_index.json')
        authenticated_protocol(self.output)
        write(output_dir / 'diagnostic_integrity.json', {'passed': True, 'native_evidence_unchanged': True,
            'scenario': self.scenario, 'report': bound(output_dir / 'round_final_diagnostics.txt'),
            'native_return_pct': native['result']['return_pct'], 'native_mtm_drawdown_pct': native['result']['drawdown_pct'],
            'metric_units': 'Canonical return/DD fractions, native table percent; KRW currency', 'economic_promotion': False})
        return analysis['diagnostics_summary']
