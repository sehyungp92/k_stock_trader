"""Qualify frozen native capacity evidence through the shared phase lifecycle.

Exploratory case files are retained as development evidence. Only an independently
audited finalist can be scored against the audited safe control here.
"""
from pathlib import Path
import json

from backtests.auto.shared.plugin import PhaseSpec, PhaseAnalysisPolicy
from backtests.auto.shared.types import Experiment, ScoredCandidate, GateCriterion, EndOfRoundArtifacts
from backtests.engine.contracts import AUDIT_VERSION, METRIC_VERSION, EXECUTION_VERSION
from backtests.engine.integrity import REQUIRED_COVERAGE, stable_signature
from backtests.auto.shared.cache_keys import source_generation_hash
from backtests.strategies.kalcb.executed_value import digest, signature, write_json


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def paired_failures(candidate, control):
    failures = []
    if candidate['return_pct'] <= control['return_pct']:
        failures.append('return')
    if candidate['drawdown_pct'] > control['drawdown_pct'] + 1e-10:
        failures.append('drawdown')
    if candidate['trades'] < .95 * control['trades']:
        failures.append('frequency')
    if (not candidate['execution_validity_pass'] or candidate['terminal_inventory']
            or candidate['terminal_pending'] or candidate.get('terminal_deferred_intents', 0)):
        failures.append('execution_or_unfinished_inventory')
    return failures


def checked_case(path):
    path = Path(path)
    completion = read(path/'completion.json')
    result, audit, request = [read(path/name) for name in
                              ('result.json', 'execution_audit.json', 'request.json')]
    study = read(request['arguments']['study'])
    if (not completion['complete']
            or digest(path/'result.json') != completion['result_sha256']
            or digest(path/'execution_audit.json') != completion['audit_sha256']
            or digest(Path(request['arguments']['study'])) != request['study_hash']
            or result['source_generation_hash'] != study['source_generation_hash']
            or result['recipe'] != request['arguments']['recipe']
            or result['scenario'] != request['arguments']['scenario']):
        raise ValueError('Changed or mismatched native replay evidence: '+str(path))
    return result, audit


def native_audit_valid(audit, path):
    """Validate the native study identity without rewriting its audit receipt.

    The generic optimizer expects identity.request/mutations; native replay binds
    study_hash/recipe/configuration instead. Enforce equivalent source, coverage,
    signature and exact request binding through an explicit phase gate.
    """
    request = read(Path(path)/'request.json')
    study = read(request['arguments']['study'])
    args = request['arguments']
    manifest = read(study['manifest'])
    manifest_hash = manifest.pop('manifest_hash')
    receipt = dict(audit.get('audit_receipt') or {})
    receipt_hash = receipt.pop('receipt_hash', None)
    identity = receipt.get('identity', {})
    configuration = dict(study['configuration'])
    if args['standalone']:
        configuration['source_dependency_scopes'] = {
            args['standalone']: configuration['source_dependency_scopes'][args['standalone']]}
    expected = {'study_hash':request['study_hash'], 'recipe':request['recipe'],
        'scenario':args['scenario'], 'standalone':args['standalone'],
        'absolute_caps':args['absolute_caps'], 'no_upward_compounding':args['no_upward_compounding'],
        'configuration':configuration}
    return (audit.get('audit_pass') is True and audit.get('execution_validity_pass') is True
        and audit.get('audit_status') == 'passed' and args['full_audit'] is True
        and bool(receipt_hash) and receipt_hash == stable_signature(receipt)
        and all(identity.get(k) == v for k,v in expected.items())
        and manifest_hash == signature(manifest)
        and identity.get('input_manifest', {}).get('manifest_hash') == manifest_hash
        and identity.get('input_manifest', {}).get('source_generation_hash') == study['source_generation_hash']
        and bool(identity.get('qualification_snapshot_hash'))
        and receipt.get('audit_version') == AUDIT_VERSION
        and receipt.get('metric_version') == METRIC_VERSION
        and receipt.get('execution_version') == EXECUTION_VERSION
        and receipt.get('source_generation_hash') == study['source_generation_hash'] == source_generation_hash()
        and not receipt.get('errors') and set(receipt.get('coverage', {})) == REQUIRED_COVERAGE
        and all(v in ('pass', 'not_applicable') for v in receipt.get('coverage', {}).values()))


class OLRCapacityQualificationPlugin:
    name = 'olr_capacity_qualification'
    num_phases = 1
    ultimate_targets = {}
    initial_mutations = {'recipe': 'control'}

    def __init__(self, root, screening):
        self.root, self.screening = Path(root), Path(screening)
        self.study = read(self.root/'study_100000000.json')
        self.registration = read(self.screening/'registration.json')
        self.verdict = read(self.screening/'qualification_verdict.json')
        if not self.verdict['complete']:
            raise ValueError('Qualification evidence is incomplete')
        self.candidate = self.verdict['selected_profile']
        self.references = read(self.screening/'case_references.json')
        self.mechanisms = read(self.screening/'mechanism_results.json')
        self.config = {'artifact_promotion_policy': 'historical_high_capital_research_profile_only'}
        self.execution_context = {'source_fingerprint': self.study['source_generation_hash'],
            'shared_decision_core': 'live_shared_core', 'live_parity_fill_timing': 'next_5m_open',
            'capability_level': 'real_replay', **self.config}

    def canonicalize_mutations(self, mutations):
        if set(mutations) - {'recipe'}:
            raise ValueError('Unregistered capacity mutation')
        recipe = mutations.get('recipe', 'control')
        if recipe not in ('control', self.candidate):
            raise ValueError('Only frozen, fully qualified finalists can be selected')
        return {'recipe': recipe}

    def get_phase_spec(self, phase, state):
        return PhaseSpec(focus='Audited OLR capacity recovery with paired capital and stress qualification',
            candidates=[Experiment(self.candidate, {'recipe': self.candidate})] if self.candidate else [],
            gate_criteria_fn=lambda m: [
                GateCriterion('native_source_execution_receipts', 1, float(m['native_audits_valid']), m['native_audits_valid']),
                GateCriterion('all_qualification_checks', 1, float(m['qualified']), m['qualified'])],
            scoring_weights=None, hard_rejects={}, analysis_policy=PhaseAnalysisPolicy(
                focus_metrics=['baseline_return', 'combined_return', 'baseline_dd', 'combined_dd'],
                min_effective_score_delta_pct=0), max_rounds=1,
            phase_metric_basis='native_frozen_historical_replay', primary_promotion_metric='combined_return',
            # The explicit native receipt gate above replaces the generic
            # request/mutations-schema gate; audit validation is still mandatory.
            official_metric_keys=('baseline_return', 'combined_return'), promotion_requires_audit_pass=False)

    def reference(self, name, capital, scenario):
        key = f'{name}_{capital}_{scenario}'
        local = self.screening/'cases'/key
        return local if (local/'completion.json').exists() else Path(self.references[key])

    def evaluate(self, mutations):
        name = self.canonicalize_mutations(mutations)['recipe']
        failures, results, audits, comparisons = [], {}, {}, []
        native_audits_valid = True
        for scenario in ('baseline', 'combined'):
            path = self.root/'qualified'/f'{name}_100000000_{scenario}'
            results[scenario], audits[scenario] = checked_case(path)
            if not native_audit_valid(audits[scenario], path):
                native_audits_valid = False
                failures.append(scenario+'_audit')
            if results[scenario]['source_generation_hash'] != self.study['source_generation_hash']:
                failures.append(scenario+'_source')
            reference = self.reference(name, 100_000_000, scenario)
            equal = {f: read(path/f) == read(reference/f) for f in ('trades.json','equity.json','daily.json')}
            if not all(equal.values()):
                failures.append(scenario+'_economic_repeat')
            comparisons.append({'scenario': scenario, 'parsed_economic_outputs_equal': equal})
        if name != 'control':
            for capital in (90_000_000, 100_000_000, 110_000_000):
                for scenario in ('baseline', 'combined'):
                    candidate, _ = checked_case(self.reference(name, capital, scenario))
                    control, _ = checked_case(self.reference('control', capital, scenario))
                    errors = paired_failures(candidate, control)
                    failures.extend(f'{capital}_{scenario}_{error}' for error in errors)
                    comparisons.append({'capital': capital, 'scenario': scenario, 'failures': errors,
                        'candidate': {k:candidate[k] for k in ('return_pct','drawdown_pct','trades')},
                        'control': {k:control[k] for k in ('return_pct','drawdown_pct','trades')}})
            if self.mechanisms[f'{name}_100000000_baseline']['intervention_issuer_days'] < 10:
                failures.append('intervention_support')
        metrics = dict(audits['baseline'])
        metrics.update(qualified=not failures, native_audits_valid=native_audits_valid, research_only=True,
            official_metric_basis='SimBroker.equity_curve_bar_level_mtm',
            source_fingerprint=self.study['source_generation_hash'],
            feature_manifest_hash=read(self.study['manifest'])['manifest_hash'],
            candidate_snapshot_hash=audits['baseline']['audit_receipt']['identity']['qualification_snapshot_hash'],
            capability_level='real_replay', end_open_position_count=len(results['baseline']['terminal_inventory']))
        for scenario, result in results.items():
            metrics.update({scenario+'_'+k: result[v] for k,v in
                (('return','return_pct'),('dd','drawdown_pct'),('trades','trades'))})
        return {'recipe': name, 'metrics': metrics, 'failures': failures,
            'score': 100 + metrics['combined_return'], 'comparisons': comparisons}

    def create_evaluate_batch(self, phase, cumulative_mutations, **kwargs):
        def evaluate(candidates, current):
            scored = []
            for candidate in candidates:
                record = self.evaluate({**current, **candidate.mutations})
                scored.append(ScoredCandidate(candidate.name, record['score'],
                    bool(record['failures']), ','.join(record['failures']), record['metrics']))
            return scored
        return evaluate

    def compute_final_metrics(self, mutations):
        return self.evaluate(mutations)['metrics']

    def run_phase_diagnostics(self, phase, state, metrics, greedy_result):
        return json.dumps(self.evaluate(greedy_result.final_mutations), indent=2)

    run_enhanced_diagnostics = run_phase_diagnostics

    def build_end_of_round_artifacts(self, state):
        record = self.evaluate(state.cumulative_mutations)
        if record['failures']:
            raise ValueError('Cannot save unqualified capacity policy')
        write_json(self.root/'accepted_profile.json', record)
        text = json.dumps(record, indent=2)
        return EndOfRoundArtifacts(text, {'capacity_and_execution': text},
            'Historical high-capital qualification complete; independent market/queue evidence not established')
