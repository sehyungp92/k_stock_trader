"""Locked C12-B research-round acceptance; no search or alternative execution.

The existing full diagnostic engine consumes the complete authenticated native
accounts. The frozen worker/map/config, rather than a bare config alias, define
the selected policy. Source audits are inherited; account/behaviour checks are
fresh. Live execution and independent predictive validity are not certified.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, time, timezone
from pathlib import Path
import gc
import gzip
import json
import shutil

from backtests.auto.shared.cache_keys import stable_signature
from backtests.auto.shared.types import EndOfRoundArtifacts
from backtests.strategies.common.plugin_base import attach_official_metric_contract
from strategy_kalcb.config import KALCBConfig
from .qualified_round import KALCBQualifiedRoundPlugin
from .qualified_artifacts import read, digest, require, close, diagnostic_result, REPLAY_FILES, SCENARIOS
from .late_artifact_validation_v2 import validate_native, validate_case
from .retention_research_io_v1 import bound, write, PERIODS
from .fixed_allocation_funding_v1 import funding_review
from .fixed_allocation_reference_v1 import verify_identity_bridge
from .funded_horizon_v1 import enabled_signal
from .late_entry_v1 import cumulative_timed_policy

VERSION = 'kalcb-veto-cap-round-v1'
STATUS = 'user_selected_research_round_joint_objective_unmet'
LIMITS = [
    'All 225 sessions are adaptively reused development evidence, not untouched validation.',
    'Useful participation qualification remains failed; avoided losses and larger quantities are not additional frequency.',
    'May concentration and adverse June/July periods limit claims of persistent expected returns.',
    'The cap acts on only 3 IS and 7 OOS dates; scenario reruns are not independent episodes.',
    'This exact combination lacks larger-capital and continuous-state capital qualification.',
    'Physical queue, latency, paper/live order/fill/restart parity and live VETO delivery are unqualified.',
    'The frozen historical shared-core research controller is required; mutable current core/config alone is not this policy.',
]


def lines(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def check_signal_map(signals):
    for key, row in signals.items():
        require(key == row['day'] + '/' + row['symbol'], 'Signal identity mismatch')
        at = datetime.fromisoformat(row['available_at'])
        require(at.tzinfo is not None and at.isoformat().endswith('+09:00') and
                at.date().isoformat() == row['day'] and at.time().replace(tzinfo=None) == time(9, 30),
                'Wrong opening information availability')
        require(row['observed_through'] < row['day'] and row['origin'] <= row['day'],
                'Future training support')
        enabled_signal(row, at)


def check_activation(folder, case, signals):
    """Read-only C12 action reconciliation; no writes to historical receipts."""
    planned, submitted, capped, vetoed = [], [], [], []
    for d in lines(Path(folder) / 'decisions.jsonl.gz'):
        m = d.get('metadata', {}); cap = m.get('pending_funding_reservation')
        if cap:
            require(case['cap'] == 2 and m.get('entry_route') == 'first30_open_anchor' and cap['limit'] == 2,
                    'Wrong cap scope')
            expected = min(cap['original_qty'], int(max(0., 2 * cap['equity'] - cap['reserved_gross']) / cap['completed_close']))
            require(expected == cap['capped_qty'] and 0 <= expected <= cap['original_qty'], 'Cap arithmetic mismatch')
            capped.append((d['timestamp'][:10], expected < cap['original_qty']))
        for action in d.get('actions', []):
            if action['action_type'] == 'SubmitEntry':
                planned.append((d['timestamp'], action['symbol'], action['qty']))
    for row in lines(Path(folder) / 'orders.jsonl.gz'):
        o = row.get('order', {}); m = o.get('metadata', {})
        if row['event'] != 'submit' or o.get('side') != 'BUY':
            continue
        submitted.append((m['signal_bar'], o['symbol'], o['qty']))
        if m.get('entry_route') == 'first30_open_anchor':
            if case['cap']:
                require(m.get('pending_funding_reservation') and o['qty'] == m['pending_funding_reservation']['capped_qty'],
                        'Submitted opening cap mismatch')
            signal = signals.get(m['signal_bar'][:10] + '/' + o['symbol'])
            require(case['mode'] != 'veto' or not enabled_signal(signal, datetime.fromisoformat(m['decision_available_at'])),
                    'Vetoed opening was submitted')
        else:
            require(not m.get('pending_funding_reservation'), 'Later route incorrectly capped')
    require(sorted(planned) == sorted(submitted), 'Core proposals differ from broker submissions')
    for row in read(Path(folder) / 'horizon_actions.json'):
        require(case['mode'] == 'veto' and row['kind'] == 'veto', 'Wrong VETO action')
        require(row['signal'] == signals[row['key']], 'Changed executed prediction')
        require(enabled_signal(row['signal'], datetime.fromisoformat(row['available_at'])), 'Noncausal VETO')
        vetoed.append(row['key'])
    require(case['cap'] != 0 or not capped, 'Disabled cap acted')
    result = dict(passed=True, cap_decisions=len(capped), trimmed=sum(x[1] for x in capped),
                  distinct_cap_dates=sorted({d for d, changed in capped if changed}), veto_issuer_days=sorted(set(vetoed)))
    old = read(Path(folder) / 'activation.json')
    require(all(old[k] == v for k, v in result.items()), 'Historical activation disagreement')
    return result


def authenticate_files(bindings, progress=None):
    """Bounded I/O concurrency; every distinct path checked against its hash."""
    unique = {}
    for b in bindings:
        p = str(Path(b['path']).resolve()); old = unique.setdefault(p, b['sha256'])
        require(old == b['sha256'], 'Conflicting source identity: ' + p)
    def check(pair):
        p, sha = pair
        require(digest(p) == sha, 'Changed bound artifact: ' + p)
        return p
    with ThreadPoolExecutor(max_workers=8) as pool:
        for count, _ in enumerate(pool.map(check, unique.items()), 1):
            if progress and (count % 500 == 0 or count == len(unique)):
                write(progress, dict(stage='source_authentication', checked=count, total=len(unique)))
                print(f'Authenticated {count}/{len(unique)} source paths', flush=True)
    return [{'path': p, 'sha256': sha} for p, sha in unique.items()]


def profile_for(source, case, item):
    start, end, sessions = PERIODS[case['period']]
    plan = read(item['plan']['path']); folder = Path(source) / 'cases' / case['id']
    return dict(primary_profile='C12-B_VETO_plus_cap', period=case['period'], status=STATUS, live_enabled=False,
        initial_equity_krw=100_000_000., date_range=dict(start=start, end=end, sessions=sessions),
        configuration=plan['config'], audit_manifest=item['manifest']['path'],
        execution_source=dict(source_generation_hash=item['source_generation_hash']),
        policy=dict(name='interaction_cost_margin', manifest_sha256=item['policy']['sha256'], directory=str(Path(item['policy']['path']).parent)),
        result_paths={case['scenario']: str(folder / 'result.json')}, metrics={case['scenario']: read(folder / 'result.json')})


def validate_selected_account(source, case, item, signals):
    folder = Path(source) / 'cases' / case['id']
    request, done = read(folder / 'request.json'), read(folder / 'completion.json')
    require(done['mode'] == request['arguments']['mode'] == case['mode'] and
            done['cap'] == request['cap'] == case['cap'], 'Wrong executed VETO/cap policy')
    require(request['signal_map']['sha256'] == stable_map_digest(signals, request['signal_map']['path']), 'Wrong executed signal map')
    require(done['origins_before'] == request['import_bindings'], 'Request import identity changed')
    before = {b['path']: b['sha256'] for b in done['origins_before']}
    after = {b['path']: b['sha256'] for b in done['origins_after']}
    require(all(after.get(p) == sha for p, sha in before.items()), 'Executing imports changed')
    code = Path(item['manifest']['path']).resolve().parent / 'files'
    source_files = read(item['manifest']['path'])['files']
    for path, sha in after.items():
        origin = Path(path).resolve()
        require(origin.is_relative_to(code), 'Unfrozen executing import')
        require(source_files.get(origin.relative_to(code).as_posix()) == sha, 'Unbound executing import')
    validate_native(folder, item, case['scenario'], save=False)
    checked = validate_case(folder, profile_for(source, case, item), case['scenario'])
    activation = check_activation(folder, case, signals)
    funding = funding_review(folder)
    require(funding['verified'], 'Failed funding review')
    for k in ('terminal_cash_krw', 'minimum_funded_cash_krw', 'submitted_buy_orders', 'physically_filled_buy_orders'):
        close(funding[k], read(folder / 'funding_review.json')[k], 'Funding review drift: ' + k)
    return dict(account=checked['validation'], activation=activation, funding=funding)


def stable_map_digest(signals, path):
    require(read(path) == signals, 'Executed signal map differs')
    return digest(path)


def prepare_acceptance(source, output, tests):
    """Authenticate existing accounts, then publish an immutable acceptance capsule."""
    source, output = Path(source).resolve(), Path(output).resolve()
    require(not output.exists(), 'Preserve prior acceptance generation')
    output.mkdir(parents=True)
    try:
        p, reg, done, ledger = (read(source / n) for n in ('protocol.json', 'registration.json', 'completion.json', 'completed_cases.json'))
        require(p['experiment'] == 'kalcb-exp-0036' and p['criteria'] == 'C12-B diagnostic' and
                done['complete'] and done['native_accounts'] == 12 and len(ledger) == 12, 'Wrong/incomplete selected study')
        require([r['case'] for r in ledger] == p['cases'] and all(r['verdict']['qualified'] for r in ledger), 'Unqualified or incomplete matrix')
        require(read(tests)['passed'] is True, 'Readiness tests failed')
        source_bindings = p['inputs'] + p['programs'] + p['database_bindings'] + [reg[k] for k in ('protocol', 'preflight', 'predecessors')]
        source_bindings += [done['registration'], done['assessment'], p['signal_map']]
        for item in reg['profiles'].values():
            source_bindings += [item[k] for k in ('manifest', 'plan', 'policy')]
            manifest = read(item['manifest']['path']); identity = manifest.pop('manifest_hash')
            require(stable_signature(manifest) == identity, 'Source manifest signature mismatch')
            code = Path(item['manifest']['path']).parent / 'files'
            source_bindings += [dict(path=str(code / rel), sha256=sha) for rel, sha in manifest['files'].items()]
            source_bindings += [dict(path=f, sha256=sha) for f, sha in read(item['plan']['path'])['input_sha256'].items()]
        signals = read(p['signal_map']['path']); check_signal_map(signals)
        horizon = Path(p['signal_map']['path']).parent
        hp = read(horizon / 'protocol.json')
        source_bindings += hp['inputs'] + [bound(horizon / f) for f in ('protocol.json', 'prediction_bridge.json', 'map_coverage.json')]
        require(read(horizon / 'prediction_bridge.json')['passed'], 'Prediction reconstruction failed')
        authenticated = authenticate_files(source_bindings, output / 'status.json')
        write(output / 'source_authentication.json', dict(passed=True, at_utc=datetime.now(timezone.utc).isoformat(),
            checked_paths=len(authenticated), bindings=authenticated,
            scope='Fresh file authentication of frozen code/data/model identities; inherited full source-engine audit receipts, not a new replay.'))
        reviews = {}; bindings = [bound(source / f) for f in ('protocol.json', 'registration.json', 'completion.json', 'assessment.json', 'completed_cases.json', 'runtime_preflight.json')]
        profiles = {}
        for row in ledger:
            c = row['case']; folder = source / 'cases' / c['id']; item = reg['profiles'][c['period'] + '_cap' + str(c['cap'])]
            reviews[c['id']] = validate_selected_account(source, c, item, signals)
            if c['kind'] == 'bridge':
                reviews[c['id']]['bridge'] = verify_identity_bridge(folder, c['bridge'])
                bindings += [bound(Path(c['bridge']) / f) for f in ('daily.json', 'equity.json', 'fills.json')]
            else:
                profiles[c['period'] + '/' + c['scenario']] = dict(case=c, item=item, profile=profile_for(source, c, item))
            bindings += [bound(f) for f in folder.iterdir() if f.is_file()]
            print('Fresh account/action/funding acceptance:', c['id'], flush=True)
            write(output / 'status.json', dict(stage='account_acceptance', completed=len(reviews), total=12))
        require(set(profiles) == {p + '/' + s for p in ('is', 'oos') for s in SCENARIOS}, 'Incomplete final report matrix')
        write(output / 'account_acceptance.json', dict(passed=True, cases=reviews))
        for item in reg['profiles'].values():
            bindings += [item[k] for k in ('manifest', 'plan', 'policy')]
            bindings += [bound(read(item['plan']['path'])['config'])]
        bindings += [p['signal_map'], bound(horizon / 'prediction_bridge.json')]
        repo = Path(__file__).resolve().parents[3]
        report_programs = [bound(repo / f) for f in (
            'backtests/strategies/kalcb/veto_cap_round.py', 'backtests/strategies/kalcb/full_diagnostics.py',
            'backtests/strategies/kalcb/qualified_artifacts.py', 'backtests/strategies/kalcb/qualified_round.py',
            'backtests/strategies/kalcb/late_artifact_validation_v2.py', 'backtests/strategies/kalcb/funded_horizon_v1.py',
            'backtests/strategies/kalcb/fixed_allocation_funding_v1.py', 'backtests/strategies/kalcb/fixed_allocation_reference_v1.py',
            'backtests/auto/shared/phase_runner.py', 'backtests/auto/shared/round_manager.py',
            'tests/backtests/strategies/test_kalcb_veto_cap_round.py')]
        test_receipt = read(tests)
        for b in test_receipt['programs']:
            require(digest(b['path']) == b['sha256'], 'Tested code changed')
        bindings += report_programs + [bound(tests)] + [bound(output / f) for f in ('source_authentication.json', 'account_acceptance.json')]
        payload = dict(version=VERSION, source=str(source), profiles=profiles, bindings=bindings,
            tests=bound(tests), programs=report_programs, source_authentication=bound(output / 'source_authentication.json'),
            account_acceptance=bound(output / 'account_acceptance.json'), signal_map=p['signal_map'],
            policy='C12-B_VETO_plus_cap', production_enabled=False, joint_economic_objective_achieved=False,
            limitations=LIMITS, native_reruns=0, new_fits=0)
        write(output / 'acceptance_manifest.json', {**payload, 'manifest_hash': stable_signature(payload)})
        write(output / 'status.json', dict(stage='acceptance_complete', accounts=12, reports=8))
        return output / 'acceptance_manifest.json'
    except Exception as error:
        write(output / 'failure.json', dict(error=repr(error)))
        raise


def load_acceptance(path):
    capsule = read(path); sig = capsule.pop('manifest_hash')
    require(stable_signature(capsule) == sig and capsule['version'] == VERSION, 'Changed acceptance capsule')
    require(capsule['production_enabled'] is False and capsule['policy'] == 'C12-B_VETO_plus_cap', 'Wrong release policy')
    require(set(capsule['profiles']) == {p + '/' + s for p in ('is', 'oos') for s in SCENARIOS}, 'Missing report accounts')
    authenticate_files(capsule['bindings'])
    require(read(capsule['source_authentication']['path'])['passed'] and read(capsule['account_acceptance']['path'])['passed'],
            'Incomplete source/account acceptance')
    return {**capsule, 'manifest_hash': sig}


def load_bundle(capsule, path, period, scenario):
    entry = capsule['profiles'][period + '/' + scenario]; profile = entry['profile']
    # The policy loader reconstructs existing snapshots; no fitting or selection.
    from scripts.kalcb_executed_value_replay import verify_policy
    policy_dir = Path(profile['policy']['directory'])
    policy, opening = verify_policy(policy_dir, 'interaction_cost_margin')
    updates = cumulative_timed_policy(policy_dir, policy, opening)
    case = validate_case(Path(profile['result_paths'][scenario]).parent, profile, scenario)
    return dict(profile=profile, cases={scenario: case}, opening=opening, updates=updates,
        policy={**policy, 'prepared_sha256': capsule['manifest_hash']},
        manifest=str(path), manifest_hash=capsule['manifest_hash'])


class VetoCapRoundPlugin(KALCBQualifiedRoundPlugin):
    """Exact selected policy, one shared phase, eight standard full reports."""
    def __init__(self, manifest):
        self.acceptance_manifest = Path(manifest).resolve(); self.capsule = load_acceptance(manifest)
        self.bundle = load_bundle(self.capsule, manifest, 'is', 'baseline'); self.profile = self.bundle['profile']
        self.source_fingerprint = self.profile['execution_source']['source_generation_hash']
        # Preserve executed settings exactly; current dataclass defaults must not
        # silently become settings attributed to a historical frozen engine.
        self.initial_mutations = read(self.profile['configuration'])
        KALCBConfig.from_mapping(self.initial_mutations)  # Current report interface compatibility only.
        self.initial_mutations_override = dict(self.initial_mutations)
        self.config = dict(initial_equity=100_000_000., date_range=self.profile['date_range'], promotion_status=STATUS,
            artifact_promotion_policy='user_authorized_numbered_research_round_only_no_live_activation',
            selected_candidate='C12-B_VETO_plus_cap', initial_mutations=self.initial_mutations,
            initial_mutations_path=self.profile['configuration'], risk_basis='native_bar_level_mark_to_market',
            live_parity_fill_timing='next_5m_open', auction_mode='non_auction_continuous', capability_level='real_replay',
            acceptance_manifest=str(self.acceptance_manifest), acceptance_manifest_sha256=digest(manifest),
            required_candidate_policy=dict(parent=self.profile['policy'], veto_map=self.capsule['signal_map'], mode='veto',
                opening_pending_cap=2., frozen_profiles=self.capsule['profiles'], acceptance_manifest=str(self.acceptance_manifest)),
            economic_objective_achieved=False, qualification_limits=LIMITS)
        self.execution_context = {**self.config, 'source_fingerprint': self.source_fingerprint,
            'feature_manifest_hash': self.capsule['manifest_hash'], 'candidate_snapshot_hash': stable_signature(self.config['required_candidate_policy']),
            'replay_mode': 'authenticated_C12B_native_full_accounts', 'expected_universe_size': 103,
            'train_start': self.profile['date_range']['start'], 'train_end': self.profile['date_range']['end'],
            'production_enabled': False, 'cost_policy': self.bundle['cases']['baseline']['audit']['audit_receipt']['qualification']['capacity']['policy']}
        self.result = diagnostic_result(self.bundle)
        self.result.metrics['invalid_order_rejection_count'] = self.bundle['cases']['baseline']['audit']['invalid_order_rejection_count']
        from .full_diagnostics import _final_metrics
        metrics = _final_metrics(self.result.metrics, self.result.trades, self.config, self.result.replay_result.equity_curve,
                                 timestamps=self.result.replay_result.timestamps)
        self.metrics = attach_official_metric_contract(metrics, requires_audit_pass=True, audit_pass=True,
            execution_contract={**self.execution_context, 'metric_units': 'Return/DD fractions; currency KRW'})
        self.metrics['archived_native_evidence'] = True

    def validate_archived_execution(self, metrics):
        """Exact v16 frozen native evidence, not the current checkout's v18 replay."""
        capsule = load_acceptance(self.acceptance_manifest)
        require(metrics.get('audit_pass') is True and metrics.get('archived_native_evidence') is True,
                'Explicit source audit failure')
        scored_source = metrics.get('source_fingerprint') or metrics['metric_contract']['execution_contract']['source_fingerprint']
        require(scored_source == self.source_fingerprint, 'Wrong scored source')
        source = Path(capsule['source']); p = read(source / 'protocol.json'); reg = read(source / 'registration.json')
        signals = read(capsule['signal_map']['path']); check_signal_map(signals)
        for case in p['cases']:
            item = reg['profiles'][case['period'] + '_cap' + str(case['cap'])]
            validate_selected_account(source, case, item, signals)
            folder = source / 'cases' / case['id']; audit = read(folder / 'execution_audit.json')
            receipt = audit['audit_receipt']
            require((receipt['audit_version'], receipt['metric_version'], receipt['execution_version']) ==
                    ('execution-ledger-v16', 'session-equity-trade-groups-v5', 'timestamp-phases-v6'), 'Wrong historical audit versions')
            require(all(v in ('pass', 'not_applicable') for v in receipt['coverage'].values()) and
                    all(receipt['coverage'].get(k) == 'pass' for k in ('ledger', 'clock', 'calendar', 'dated_references', 'replay_completeness', 'capacity')),
                    'Incomplete historical audit coverage')
            require(audit['invalid_order_rejection_count'] == 0, 'Invalid native order')
            if case['kind'] == 'bridge': verify_identity_bridge(folder, case['bridge'])
        native = self.bundle['cases']['baseline']['result']
        for key, value in [('official_mtm_net_return_pct', native['return_pct'] / 100),
                           ('max_drawdown_pct', native['drawdown_pct'] / 100), ('total_trades', native['trades'])]:
            close(metrics[key], value, 'Scored native metric mismatch: ' + key)
        return True

    def run_phase_diagnostics(self, phase, state, metrics, greedy_result):
        return json.dumps(dict(locked_policy='C12-B_VETO_plus_cap', source_account_acceptance=True, accounts=12,
            native_reruns=0, new_fits=0, research_registration_only=True, joint_economic_objective_achieved=False,
            limitations=LIMITS), indent=2)

    run_enhanced_diagnostics = run_phase_diagnostics

    def build_end_of_round_artifacts(self, state):
        require(bool(state.phase_gate_results) and all(g.get('passed') is True for g in state.phase_gate_results.values()),
                'Failed phase gate blocks locked round registration')
        self.canonicalize_mutations(state.cumulative_mutations); load_acceptance(self.acceptance_manifest)
        report = self.run_phase_diagnostics(1, state, self.metrics, None)
        return EndOfRoundArtifacts(report, {'locked_acceptance': report},
            'User-selected VETO + cap research round; joint participation objective unmet; no live activation')

    def write_full_diagnostics(self, state, output_dir, *, round_num=2, round_name=''):
        output_dir = Path(output_dir); reports = {}; summary = None
        for period in ('is', 'oos'):
            for scenario in SCENARIOS:
                primary = period == 'is' and scenario == 'baseline'
                bundle = self.bundle if primary else load_bundle(self.capsule, self.acceptance_manifest, period, scenario)
                result = self.result if primary else diagnostic_result(bundle, scenario)
                dest = output_dir if primary else output_dir / 'scenarios' / period / scenario
                current = write_report(bundle, result, self.capsule, self.config, state.cumulative_mutations,
                                       dest, period, scenario, round_num, round_name)
                if primary: summary = current
                reports[period + '/' + scenario] = bound(dest / 'round_final_diagnostics.txt')
                print('Saved full diagnostics:', period, scenario, flush=True)
                if not primary:
                    del bundle, result, current
                    gc.collect()
        load_acceptance(self.acceptance_manifest)
        write(output_dir / 'scenario_reports.json', dict(complete=True, reports=reports, report_count=8,
            primary='is/baseline', scope='Historical reused data; scenarios are not independent market episodes'))
        shutil.copy2(self.acceptance_manifest, output_dir / 'acceptance_manifest.json')
        write(output_dir / 'diagnostic_integrity.json', dict(passed=True, native_evidence_unchanged=True, reports=reports,
            source_authentication=self.capsule['source_authentication'], account_acceptance=self.capsule['account_acceptance'],
            production_enabled=False, joint_economic_objective_achieved=False))
        return summary


def write_report(bundle, result, capsule, config, mutations, dest, period, scenario, round_num, round_name):
    from .full_diagnostics import (analyze_kalcb_result, render_kalcb_diagnostics_report, _atomic_write_text,
        _atomic_write_json, _atomic_write_jsonl, _artifact_paths, _index_payload)
    dest = Path(dest); dest.mkdir(parents=True, exist_ok=True)
    cfg = {**config, 'date_range': bundle['profile']['date_range'], 'scenario': scenario, 'period': period}
    native = bundle['cases'][scenario]
    result.metrics['invalid_order_rejection_count'] = native['audit']['invalid_order_rejection_count']
    analysis = analyze_kalcb_result(result, config=cfg, mutations=mutations)
    for key, expected in [('net_return_pct', native['result']['return_pct'] / 100),
                          ('max_drawdown_pct', native['result']['drawdown_pct'] / 100),
                          ('profit_factor', native['validation']['net_profit_factor']), ('total_trades', native['result']['trades'])]:
        close(analysis['final_metrics'][key], expected, 'Full diagnostic metric drift: ' + key)
    analysis.update(round=round_num, round_name=round_name, promotion_status=STATUS)
    analysis['diagnostics_summary']['promotion_status'] = STATUS
    analysis['gate_result'] = dict(status='passed_source_account_integrity_user_selected_research_round',
        production_ready=False, joint_economic_objective_achieved=False, economic_frequency_qualified=False)
    analysis['live_parity_audit'].update(status='historical_frozen_shared_core_only_live_controller_unqualified',
        observation_watch_count=103, observation_window_minutes=bundle['policy'].get('observation_minutes', 45),
        active_symbol_count_basis='Admitted snapshot count, not verified live subscription occupancy', paper_promotion_blockers=LIMITS)
    analysis['diagnostics_summary']['live_parity_audit'] = analysis['live_parity_audit']
    assessment = read(Path(capsule['source']) / 'assessment.json')
    context = dict(policy='C12-B_VETO_plus_cap', period=period, scenario=scenario, profile=bundle['profile'],
        acceptance_manifest=bundle['manifest'], acceptance_manifest_hash=bundle['manifest_hash'],
        source_account_checks=native['validation'], source_authentication=capsule['source_authentication'],
        actual_behaviour=read(capsule['account_acceptance']['path'])['cases']['combined_' + period + '_' + scenario],
        funded_comparisons=assessment['cells'][period + '/' + scenario], limitations=LIMITS,
        metric_units='Canonical return/DD fractions; scenario/source return_pct/drawdown_pct use percent; KRW currency',
        training='Inherited causal expanding/refit/removal policy and frozen opening-only VETO models; no new fitting.',
        validation_scope='Reused development IS156/OOS69; no fresh OOS or live claim',
        safeguards=dict(completed_bar_availability=True, source_fingerprinted_native_broker=True,
            original_full_source_audits_authenticated=True, fresh_independent_account_cost_funding_review=True,
            fresh_action_order_reconciliation=True, full_bar_mtm=True, four_component_identity_bridges=True,
            native_replay_rerun=False, mutable_current_core_substitution=False),
        production_enabled=False, joint_economic_objective_achieved=False, new_search_trials=0)
    analysis['diagnostics_summary']['veto_cap_round_context'] = context
    report = render_kalcb_diagnostics_report(analysis).replace('Full-universe alpha', 'Non-selected outcome evidence').replace('Intraday Alpha Curve', 'Intraday realized PnL attribution')
    report += '\n\nVETO + CAP: LOCKED RESEARCH ROUND / EXISTING SAFEGUARDS AND LIMITS\n' + json.dumps(context, indent=2, default=str)
    report += '\n\nHistorical source audits authenticated; account/action checks refreshed. No new financial replay or independent validation.\n'
    _atomic_write_text(dest / 'round_final_diagnostics.txt', report)
    _atomic_write_text(dest / 'round_final_full_diagnostics.txt', report)
    for filename, data in [('diagnostics_summary.json', analysis['diagnostics_summary']), ('candidate_frontier.json', analysis['candidate_frontier']),
        ('live_parity_audit.json', analysis['live_parity_audit']), ('round_final_full_diagnostics.json', {k: v for k, v in analysis.items() if k not in ('trades', 'decisions')}),
        ('qualified_replay_diagnostics.json', context), ('selected_research_profile.json', bundle['profile'])]:
        _atomic_write_json(data, dest / filename)
    _atomic_write_jsonl(dest / 'trade_events.jsonl', analysis['trade_events'])
    _atomic_write_jsonl(dest / 'decision_events.jsonl', [d.to_json_dict() for d in result.decisions])
    shutil.copy2(bundle['profile']['configuration'], dest / 'strategy_config.json')
    write(dest / 'required_policy.json', config['required_candidate_policy'])
    paths = _artifact_paths(dest)
    for name in ('round_final_full_diagnostics.json', 'qualified_replay_diagnostics.json', 'required_policy.json', 'selected_research_profile.json', 'strategy_config.json', 'decision_events.jsonl'):
        paths[name] = str(dest / name)
    _atomic_write_json(_index_payload(analysis, paths, {'acceptance_manifest': bundle['manifest']}), dest / 'full_diagnostics_index.json')
    return analysis['diagnostics_summary']
