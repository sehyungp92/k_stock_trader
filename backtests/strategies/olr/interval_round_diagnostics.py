"""Read-only N0 evidence adapter for the existing final-round lifecycle.

Validation runs under each case's frozen engine, never a mixed IS/OOS import
environment. No simulator, selection loop or trading transition is implemented
here. The native audit remains mandatory; reconstructed exports supplement it.
"""
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from datetime import datetime, timedelta
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys


SCHEMA = 'olr-i0-n0-final-diagnostics-v1'
STATUS = 'user_selected_retained_research_policy_independent_validation_pending'
CASE_NAMES = {f'{w}_{s}' for w in ('train', 'oos') for s in ('nominal', 'stress', 'capped')}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def close(a, b, message):
    require(math.isfinite(float(a)) and math.isfinite(float(b))
            and math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-5), message)


def normalized(value):
    return json.loads(json.dumps(value, default=str))


def effective_policy(study, request):
    from strategy_olr.config import OLRConfig
    source = Path(study['code'])
    base = read(source / study['olr_request'])
    recipe = study['candidates'][request['arguments']['recipe']]
    cfg = replace(OLRConfig.from_mapping(base['config'], base['mutations']),
                  slippage_bps=5, tax_bps_on_sell=20, **recipe['olr_capacity'])
    settings = {key: read(source / study[key]) for key in
                ('structural_settings', 'funded_probe_settings',
                 'funded_population_settings', 'relative_context_settings')}
    require(settings['structural_settings'] == {'mode': 'control', 'interval_service': True},
            'N0 requires original control with interval service')
    require(settings['funded_probe_settings']['action'] == 'hold'
            and settings['funded_population_settings']['cell'] is None
            and settings['relative_context_settings']['fraction'] == 0,
            'An experimental intervention is enabled')
    require(not any(study.get(k) for k in ('economic_policy_settings', 'drawdown_reserve_contract',
                                          'expired_selection_settings')), 'Unexpected policy overlay')
    return normalized({'olr_config': asdict(cfg), 'recipe': recipe, 'settings': settings})


def require_native_receipt(audit, request, study, manifest_hash):
    """Exact interval identity, including the native runner's clock extensions.

    The older capacity validator predates these two required configuration
    fields. Never strip them from the signed receipt or weaken its other gates.
    """
    from backtests.auto.shared.cache_keys import stable_signature
    from backtests.engine.contracts import AUDIT_VERSION, METRIC_VERSION, EXECUTION_VERSION
    from backtests.engine.integrity import REQUIRED_COVERAGE
    args = request['arguments']
    configuration = dict(study['configuration'])
    configuration['source_dependency_scopes'] = {'OLR': configuration['source_dependency_scopes']['OLR']}
    configuration['minute_clock'] = dict(version='olr-minute-clock-v5',
        source_definition=study['minute_source_definition'], fast_exit=False)
    configuration['structural_decision_v1'] = {'settings_path': study['structural_settings']}
    expected = dict(study_hash=request['study_hash'], recipe=request['recipe'],
        scenario=args['scenario'], standalone='OLR', absolute_caps=False,
        no_upward_compounding=args['no_upward_compounding'], configuration=configuration)
    receipt = dict(audit.get('audit_receipt') or {})
    signature = receipt.pop('receipt_hash', None)
    identity = receipt.get('identity', {})
    require(audit.get('audit_pass') is True and audit.get('execution_validity_pass') is True
            and audit.get('audit_status') == 'passed' and args['full_audit'] is True,
            'Native audit did not pass')
    require(signature and signature == stable_signature(receipt), 'Native receipt signature mismatch')
    require(all(identity.get(k) == v for k, v in expected.items()), 'Native interval audit identity mismatch')
    require(identity.get('input_manifest', {}).get('manifest_hash') == manifest_hash
            and identity['input_manifest'].get('source_generation_hash') == study['source_generation_hash']
            and bool(identity.get('qualification_snapshot_hash')), 'Native audit source/input binding mismatch')
    require([receipt.get(k) for k in ('audit_version', 'metric_version', 'execution_version')]
            == [AUDIT_VERSION, METRIC_VERSION, EXECUTION_VERSION], 'Native audit version mismatch')
    require(receipt.get('source_generation_hash') == study['source_generation_hash']
            and not receipt.get('errors') and set(receipt.get('coverage', {})) == REQUIRED_COVERAGE
            and all(v in ('pass', 'not_applicable') for v in receipt['coverage'].values()),
            'Incomplete or failed native audit coverage')


def reconstruct(case, capital, window):
    """Stream causal marks/fills, independently reconstruct cash and full NAV."""
    case = Path(case)
    cash = peak = float(capital)
    qty, marks, buy_by_order = defaultdict(int), {}, defaultdict(float)
    counts, daily, groups = Counter(), {}, defaultdict(list)
    dd = money_dd = fees = buy_notional = sell_notional = maximum_name = 0.
    last = first = underwater = peak_at = None
    weighted = seconds = intraday_weighted = intraday_seconds = 0.
    previous_nav, previous_gross = capital, 0.
    longest = 0.
    points = 0
    with gzip.open(case / 'minute_events.jsonl.gz', 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            kind = row['event']
            if kind in ('opening_marks', 'interval_marks', 'closing_marks'):
                at = datetime.fromisoformat(row['timestamp']) + timedelta(minutes=kind != 'opening_marks')
            elif kind == 'fill':
                at = datetime.fromisoformat(row['fill']['timestamp'])
            else:
                continue
            require(at.tzinfo is not None and window['start'] <= at.date().isoformat() <= window['end'],
                    'Event outside declared aware clock/window')
            if last is not None:
                delta = (at - last).total_seconds()
                require(delta >= 0, 'Reversed event availability')
                weighted += delta * previous_gross / previous_nav
                seconds += delta
                if at.date() == last.date():
                    intraday_weighted += delta * previous_gross / previous_nav
                    intraday_seconds += delta
            if kind == 'fill':
                f = row['fill']
                require(f['strategy_id'] == 'OLR' and f['side'] in ('BUY', 'SELL'), 'Wrong fill owner/side')
                n, price = f['qty'], f['price']
                require(n > 0 and n == int(n) and price > 0, 'Invalid fill quantity/price')
                buy = f['side'] == 'BUY'
                fee = n * price * (2 if buy else 22) / 10000
                close(fee, f['metadata']['fee'], 'Wrong commission/sell tax')
                close(cash, row['cash_before'], 'Discontinuous fill cash')
                cash += (-1 if buy else 1) * n * price - fee
                close(cash, row['cash_after'], 'Incorrect fill accounting')
                qty[f['symbol']] += n if buy else -n
                require(cash >= -1e-5 and qty[f['symbol']] >= 0, 'Unfunded/oversold ledger')
                fees += fee
                if buy:
                    buy_notional += n * price
                    buy_by_order[f['order_id']] += n * price
                else:
                    sell_notional += n * price
            else:
                marks.update(row['prices'])
            gross = sum(n * marks[s] for s, n in qty.items() if n)
            nav = cash + gross
            require(nav > 0, 'Nonpositive account NAV')
            maximum_name = max(maximum_name, max((n * marks[s] / nav for s, n in qty.items() if n), default=0))
            if nav >= peak:
                if underwater is not None:
                    longest = max(longest, (at - underwater).total_seconds() / 86400)
                peak, peak_at, underwater = nav, at, None
            elif underwater is None:
                underwater = peak_at or at
            dd = max(dd, 1 - nav / peak)
            money_dd = max(money_dd, peak - nav)
            counts[kind] += 1
            points += 1
            first = first or at
            last, previous_nav, previous_gross = at, nav, gross
            daily[at.date().isoformat()] = (nav, cash, {str(('OLR', s)): n for s, n in qty.items() if n})
    require(last is not None and not any(qty.values()), 'Missing events or unresolved inventory')
    if underwater is not None:
        longest = max(longest, (last - underwater).total_seconds() / 86400)
    evidence = read(case / 'minute_nav_evidence.json')
    close(nav, evidence['final_equity'], 'Full-event terminal NAV mismatch')
    close(100 * dd, evidence['all_event_drawdown_pct'], 'Full-event drawdown mismatch')
    require(points == evidence['points'] and dict(counts) == evidence['events'], 'Omitted fine events')
    saved_daily, trades = read(case / 'daily.json'), read(case / 'trades.json')
    require([d['day'] for d in saved_daily] == sorted(daily), 'Missing/duplicate daily observations')
    returns, previous = [], capital
    for row in saved_daily:
        value, day_cash, inventory = daily[row['day']]
        close(value, row['equity'], 'Daily equity mismatch')
        close(day_cash, row['cash'], 'Daily cash mismatch')
        require(inventory == row['positions'], 'Daily positions mismatch')
        returns.append(value / previous - 1)
        previous = value
    pnl = [t['net_pnl'] for t in trades]
    close(sum(pnl), nav - capital, 'Trade net PNL mismatch')
    close(sum(t['gross_pnl'] - t['net_pnl'] for t in trades), fees, 'Trade/fill fee mismatch')
    identities = set()
    for t in trades:
        key = (t['symbol'], t['entry_decision_time'][:10])
        require(key not in identities, 'Duplicate independent opportunity')
        identities.add(key)
        require(t['entry_fill_time'] <= t['exit_fill_time'], 'Negative trade duration')
        for dim, label in [('entry_month', t['entry_decision_time'][:7]), ('symbol', t['symbol']),
                           ('exit_reason', t['exit_reason']), ('rank', t['route_metadata'].get('candidate_rank', 'unknown'))]:
            groups[(dim, str(label))].append(t)
    winners = sorted((p for p in pnl if p > 0), reverse=True)
    losses = -sum(p for p in pnl if p < 0)
    sd = statistics.stdev(returns) if len(returns) > 1 else 0
    metrics = dict(official_mtm_net_return_pct=nav / capital - 1, net_return_pct=nav / capital - 1,
        net_profit=nav - capital, max_drawdown_pct=dd, maximum_currency_decline=money_dd,
        longest_underwater_calendar_days=longest, unrecovered_at_end=underwater is not None,
        total_trades=len(trades), entry_lifecycle_count=len(identities),
        meaningful_acquisitions=sum(t['qty'] * t['entry_price'] >= .01 * capital for t in trades),
        active_entry_sessions=len({d for _, d in identities}), sessions=len(saved_daily),
        trades_per_21_sessions=21 * len(identities) / len(saved_daily),
        winning_wealth=sum(winners), losing_wealth=losses, profit_factor=sum(winners) / losses if losses else 0.,
        win_rate=len(winners) / len(trades), expectancy=sum(pnl) / len(trades),
        sharpe=math.sqrt(252) * statistics.mean(returns) / sd if sd else 0.,
        top5_share_of_positive_pnl=sum(winners[:5]) / sum(winners) if winners else 0.,
        max_single_name_nav_share=maximum_name, buy_notional=buy_notional, sell_notional=sell_notional,
        explicit_fees=fees, calendar_time_weighted_gross_nav=weighted / seconds,
        observed_same_day_time_weighted_gross_nav=intraday_weighted / intraday_seconds,
        end_open_position_count=0, full_event_points=points,
        official_metric_basis='native_cash_plus_inventory_full_open_interval_close_fill_MTM',
        net_return_pct_basis='net_cash_and_inventory_change_divided_by_initial_equity',
        risk_metric_basis='full_event_drawdown; daily_close_sharpe_zero_risk_free_sample_sd_sqrt252')
    cohorts = [dict(dimension=d, label=label, closed_entry_count=len(rows),
        wins=sum(t['net_pnl'] > 0 for t in rows), net_pnl_krw=sum(t['net_pnl'] for t in rows),
        explicit_fees_krw=sum(t['gross_pnl'] - t['net_pnl'] for t in rows)) for (d, label), rows in sorted(groups.items())]
    return dict(metrics=metrics, cohorts=cohorts, account_reconciliation=dict(passed=True,
        terminal_cash=cash, event_counts=dict(counts), buy_order_count=len(buy_by_order),
        scope='Independent full-event cash/inventory/fees/marks; source execution validated separately by native audit'))


def validate_frozen(plan, name):
    """Called only in an isolated subprocess rooted at the accepted engine."""
    from backtests.auto.shared.cache_keys import source_generation_hash, stable_signature
    from backtests.strategies.olr.economic_evidence import completed_case
    from strategy_olr.config import OLRConfig
    expected = plan['cases'][name]
    case, root = Path(expected['path']), Path(expected['execution_root'])
    require(Path(sys.modules[OLRConfig.__module__].__file__).resolve().is_relative_to(root.resolve()), 'Mixed engine imports')
    require(source_generation_hash(root=root) == expected['source_generation_hash'], 'Changed execution source')
    require(digest(case / 'completion.json') == expected['completion_sha256'], 'Changed completion')
    completion = read(case / 'completion.json')
    before = dict(completion['artifacts'])
    valid = completed_case(case)
    require(valid['valid'], 'Failed or incomplete native economic evidence')
    request = read(case / 'request.json')
    study_path = Path(request['arguments']['study'])
    require(str(study_path) == expected['study'] and digest(study_path) == expected['study_sha256'], 'Changed study')
    study = read(study_path)
    require(Path(study['code']).resolve() == root.resolve() and study['initial_equity'] == 100_000_000,
            'Wrong engine/account')
    require(request['study_hash'] == expected['study_sha256']
            and valid['source'] == study['source_generation_hash'] == expected['source_generation_hash'],
            'Audit request and executed source are not bound to the registered study')
    args = request['arguments']
    require(args['recipe'] == 'candidate' and args['standalone'] == 'OLR' and args['full_audit']
            and not args['canonical_costs'] and not args['absolute_caps']
            and args['scenario'] == ('combined' if name.endswith('stress') else 'baseline')
            and args['no_upward_compounding'] == name.endswith('capped'), 'Wrong scenario/recipe')
    require(study['configuration']['date_range'] == plan['windows'][name.split('_')[0]], 'Wrong window')
    require(effective_policy(study, request) == plan['effective_policy'], 'Selected policy differs from accepted N0')
    require(normalized(asdict(OLRConfig.from_mapping(plan['saved_config'], plan['saved_mutations'])))
            == plan['effective_policy']['olr_config'], 'Saved round config cannot reproduce N0')
    manifest = read(study['manifest'])
    manifest_hash = manifest.pop('manifest_hash')
    require(stable_signature(manifest) == manifest_hash, 'Changed input manifest')
    # source_root records the original workspace. InputManifest executes from
    # the immutable sibling files directory, not that historical origin.
    require((Path(study['manifest']).parent / 'files').resolve() == root.resolve(), 'Wrong frozen input root')
    for number, (relative, sha) in enumerate(manifest['files'].items()):
        path = (root / relative).resolve()
        require(path.is_relative_to(root.resolve()) and digest(path) == sha, 'Changed input bytes: ' + relative)
        if number % 2000 == 0:
            print(f'{name}: authenticated {number}/{len(manifest["files"])} inputs', flush=True)
    result, audit = read(case / 'result.json'), read(case / 'execution_audit.json')
    require_native_receipt(audit, request, study, manifest_hash)
    require(read(case / 'funded_checkpoint_receipt.json')['no_action_continuation_equal'], 'Checkpoint continuation mismatch')
    require(read(case / 'funded_population_receipt.json')['audit']['pass_'], 'Population receipt failed')
    restart = None
    if name.startswith('oos_'):
        parent = Path(plan['cases']['train_' + name.split('_')[1]]['path'])
        restart = read(case / 'capacity_restart_receipt.json')
        require(restart['parent_completion_sha256'] == digest(parent / 'completion.json')
                and restart['parent_checkpoint_sha256'] == digest(parent / 'capacity_terminal_state.json')
                and restart['same_configuration_and_scenario']
                and restart['all_samples_verified_against_oos_retained_ohlcv'], 'Wrong OOS market-history parent')
        require(restart['restored_state'] == read(parent / 'capacity_terminal_state.json')['capacity_history_v1'],
                'OOS restored history differs from own IS')
    report = reconstruct(case, study['initial_equity'], study['configuration']['date_range'])
    close(100 * report['metrics']['official_mtm_net_return_pct'], result['return_pct'], 'Inflated return')
    require(report['metrics']['total_trades'] == result['trades'], 'Trade denominator mismatch')
    coverage = audit['audit_receipt']['daily_equity_coverage']
    require(coverage['status'] == 'complete' and coverage['session_count'] == report['metrics']['sessions']
            and coverage['declared_window'] == study['configuration']['date_range'], 'Incomplete daily scope')
    report['metrics'].update(audit_pass=True, audit_status='passed', execution_validity_pass=True,
        same_bar_fill_count=audit['same_bar_fill_count'], forced_replay_close_count=audit['forced_replay_close_count'],
        rejected_order_count=audit['rejected_order_count'], daily_equity_coverage=coverage,
        source_fingerprint=study['source_generation_hash'],
        feature_manifest_hash=manifest_hash,
        candidate_snapshot_hash=audit['audit_receipt']['identity']['qualification_snapshot_hash'])
    report.update(path=str(case), native_reported_metrics=result['metrics'], native_audit=audit,
        capacity_restart={k: v for k, v in (restart or {}).items() if k != 'restored_state'},
        source_binding=dict(execution_root=str(root), source_generation_hash=study['source_generation_hash'],
            manifest=study['manifest'], manifest_sha256=digest(study['manifest']),
            authenticated_input_files=len(manifest['files']), completion_sha256=expected['completion_sha256'],
            artifacts=before, study_sha256=expected['study_sha256']))
    require(all(digest(case / f) == sha for f, sha in before.items()), 'Reporting altered native evidence')
    require(digest(case / 'completion.json') == expected['completion_sha256'], 'Completion changed during reporting')
    return report


class IntervalDiagnosticsPlugin:
    name = 'olr'
    num_phases = 0
    requires_full_diagnostics = True
    capability_level = 'real_replay'

    def __init__(self, config, plan_path, *, output_dir):
        self.config = self.training_config = dict(config)
        self.plan_path, self.output_dir = Path(plan_path), Path(output_dir)
        self.plan = read(plan_path)
        require(self.plan['schema'] == SCHEMA and set(self.plan['cases']) == CASE_NAMES, 'Incomplete N0 diagnostic matrix')
        require(config['native_case_plan_sha256'] == digest(plan_path), 'Changed N0 diagnostic plan')
        self.initial_mutations = dict(config['initial_mutations'])
        require(self.initial_mutations == self.plan['saved_mutations'], 'Changed saved policy')
        require({k: config.get(k) for k in self.plan['saved_config']} == self.plan['saved_config'], 'Changed base config')
        require(config['promotion_status'] == STATUS and config['live_promotion'] is False
                and config['independent_oos_pass'] is False, 'Unsupported qualification claim')
        self.source_fingerprint = self.plan['cases']['train_nominal']['source_generation_hash']
        self.execution_context = {**config, 'source_fingerprint': self.source_fingerprint,
            'date_window': config['date_range'], 'shared_decision_core': 'strategy_olr.core in bound frozen source',
            'replay_mode': 'native_N0_completed_interval_service',
            'risk_basis': 'full_event_mark_to_market',
            'live_parity_fill_timing': 'completed_5m_decision_then_later_completed_1m_service; actual_closing_auction',
            'auction_mode': 'resting_next_close_auction_with_native_nonfill_fallback',
            'cost_policy': {'commission_bps': 2, 'tax_bps_on_sell': 20, 'slippage_bps': 5, 'participation': .01}}
        self.diagnostic_scope = config['diagnostic_scope']
        self.diagnostic_limitations = config['diagnostic_limitations']
        self.exported_cases = {}
        self.analytical_plan = self.plan
        self.reused_validation = None

    def reuse_verified_report(self, directory, expected_sha256):
        """Reuse an explicitly sealed completed validation, never a loose cache.

        Economic evidence is rehashed; prior exhaustive market-input validation
        is reused, not falsely claimed to have run again. New analysis inputs
        and frozen Python bytes are independently authenticated here/at use.
        """
        directory = Path(directory)
        report_path = directory / 'round_final_full_diagnostics.json'
        require(digest(report_path) == expected_sha256, 'Changed parent report bytes')
        receipt = read(directory / 'integrity_verification.json')
        require(receipt['passed'] is True and receipt['case_plan_sha256'] == digest(self.plan_path),
                'Unverified parent report or changed plan')
        require(receipt['canonical_artifacts']['round_final_full_diagnostics.json'] == expected_sha256,
                'Parent integrity receipt binds a different report')
        saved = read(report_path)
        require(set(saved['cases']) == CASE_NAMES and saved['mutations'] == self.initial_mutations,
                'Wrong cached policy/matrix')
        require(saved['final_diagnostics_text_sha256'] == digest(directory / 'round_final_diagnostics.txt'),
                'Changed parent text')
        authenticated_sources = set()
        artifact_count = 0
        for name, spec in self.plan['cases'].items():
            case = saved['cases'][name]
            bound = case['source_binding']
            require(Path(case['path']).resolve() == Path(spec['path']).resolve()
                    and bound['source_generation_hash'] == spec['source_generation_hash']
                    and bound['completion_sha256'] == spec['completion_sha256']
                    and bound['study_sha256'] == spec['study_sha256'], 'Cached source/case mismatch')
            require(digest(Path(spec['path']) / 'completion.json') == spec['completion_sha256']
                    and digest(spec['study']) == spec['study_sha256'], 'Changed native completion/study')
            completion = read(Path(spec['path']) / 'completion.json')
            require(completion['artifacts'] == bound['artifacts'], 'Changed completion artifact set')
            for filename, expected in bound['artifacts'].items():
                require(Path(filename).name == filename, 'Invalid completion artifact path')
                require(digest(Path(spec['path']) / filename) == expected, 'Changed accepted artifact: ' + filename)
                artifact_count += 1
            require(case['account_reconciliation']['passed'] and case['metrics']['audit_pass']
                    and case['metrics']['execution_validity_pass'], 'Cached native validity failed')
            require(all(case['metrics'][k] == v for k, v in receipt['accepted_cases'][name].items()),
                    'Cached metrics differ from independently verified publication')
            require(digest(bound['manifest']) == bound['manifest_sha256'], 'Changed frozen manifest')
            if bound['manifest'] not in authenticated_sources:
                manifest = read(bound['manifest'])
                root = Path(spec['execution_root'])
                for relative, expected in manifest['files'].items():
                    if relative.endswith(('.py', '.yaml', '.yml')):
                        require((root / relative).resolve().is_relative_to(root.resolve()), 'Invalid frozen path')
                        require(digest(root / relative) == expected, 'Changed frozen Python: ' + relative)
                authenticated_sources.add(bound['manifest'])
            target = self.output_dir / 'diagnostics' / name / 'summary.json'
            require(not target.exists(), 'Use a fresh analysis generation')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(case, indent=2), encoding='utf-8')
            self.exported_cases[name] = case
        self.reused_validation = dict(parent=str(report_path), sha256=expected_sha256,
            completion_artifacts_rehashed=artifact_count,
            scope='Prior exhaustive native/source validation reused; all completion artifacts, frozen Python, study/manifest and newly analysed source files reauthenticated.')

    def canonicalize_mutations(self, mutations):
        require(mutations == self.initial_mutations, 'Diagnostics cannot change retained policy')
        return dict(mutations)

    def require_bound_cases(self, mutations):
        self.canonicalize_mutations(mutations)
        if self.exported_cases:
            return
        for name in sorted(CASE_NAMES, key=lambda n: (n.startswith('oos'), n)):
            expected = self.plan['cases'][name]
            directory = self.output_dir / 'diagnostics' / name
            directory.mkdir(parents=True, exist_ok=True)
            target = directory / 'summary.json'
            require(not target.exists(), 'Preserve prior report; use a fresh reporting generation')
            command = [sys.executable, '-B', str(Path(__file__).resolve()), '--verify-case',
                       str(self.plan_path.resolve()), name, str(target.resolve())]
            print('Validating N0 case ' + name, flush=True)
            with (directory / 'validation.log').open('x', encoding='utf-8') as log:
                done = subprocess.run(command, cwd=expected['execution_root'], stdout=log, stderr=subprocess.STDOUT,
                    env={**os.environ, 'PYTHONPATH': expected['execution_root'], 'PYTHONDONTWRITEBYTECODE': '1',
                         'PYTHONIOENCODING': 'utf-8', 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'})
            require(done.returncode == 0 and target.exists(), 'Native validation failed: ' + str(directory / 'validation.log'))
            self.exported_cases[name] = read(target)

    def compute_final_metrics(self, mutations):
        from backtests.strategies.common.plugin_base import attach_official_metric_contract, build_execution_contract
        self.require_bound_cases(mutations)
        metrics = dict(self.exported_cases['train_nominal']['metrics'])
        metrics.update(promotion_status=STATUS, native_receipt_validation_required=True,
                       native_receipt_validation_pass=True, independent_oos_pass=False, live_promotion=False)
        return attach_official_metric_contract(metrics, requires_audit_pass=True,
                                               execution_contract=build_execution_contract(self, metrics))

    def build_end_of_round_artifacts(self, state):
        from backtests.auto.shared.types import EndOfRoundArtifacts
        self.require_bound_cases(state.cumulative_mutations)
        text = ('Explicit user adoption of retained I0 / N0 as research round_2. No new optimisation phases. '
                'All six full native audits, frozen input/source identities and independent event/account checks pass. '
                'No new alpha, untouched OOS or live qualification is claimed.')
        return EndOfRoundArtifacts(text, {'selection_and_integrity': text}, text)

    def write_full_diagnostics(self, state, output_dir, *, round_num, round_name):
        from backtests.strategies.olr.final_diagnostics import render_full_diagnostics
        self.require_bound_cases(state.cumulative_mutations)
        self.reporting_source = dict(adapter=str(Path(__file__).resolve()), adapter_sha256=digest(__file__),
            case_plan_sha256=digest(self.plan_path), mode='frozen_native_N0_six_case_revalidation')
        if self.reused_validation:
            self.reporting_source.update(mode='authenticated_prior_native_validation_new_analysis',
                                         reused_validation=self.reused_validation)
        lines = ['Retained I0 policy / N0 interval control. Full-event risk is the headline basis.',
                 'Economic thesis: daily leader, flow and sector context followed by afternoon acceptance should capture overnight and next-session continuation.',
                 'Actual entry: original snapshot six-slot selection, then first of six completed five-minute confirmation bars; return >= -0.1%, upper-half close, at/above VWAP and <= 2.5% VWAP extension. Special sessions retain calendar-adjusted cutoffs. A fresh breakout/reclaim is not required.',
                 'Protection/management: original R = max(daily ATR, 0.5% signal close, KRW1); no active resting hard stop. Native failed followthrough (-1R while MFE <4R) and MFE fade (activation 1R, giveback 1.25R) use their completed-bar evidence and later executable service.',
                 'Exit/access: planned next-session closing auction; cancellation acknowledgement and actual fills govern release. Unfilled auction remainder may extend holding through native fallback. New opportunity allocation is endogenous, not a guaranteed nominated replacement.',
                 'Allocation: rank decay 1.5; 65% name cap; native cash/gross, capacity and 4/6/8% position/sector/account scenario-loss budgets. Scenario charges are not guaranteed realised-loss ceilings. Original quantity-dependent management and legacy-feed capacity history remain active.',
                 'IS and development OOS are separate KRW100m accounts; returns are not stitched or compounded across them.',
                 'Every OOS scenario restores its own IS market history. No parameter search or financial rerun.', '']
        for name, case in self.exported_cases.items():
            m = case['metrics']
            lines.append(f"{name}: meaningful {m['meaningful_acquisitions']}; entry days {m['active_entry_sessions']}; "
                f"currency DD KRW {m['maximum_currency_decline']:,.0f}; longest underwater {m['longest_underwater_calendar_days']:.3f} days; "
                f"unrecovered {m['unrecovered_at_end']}; winning/losing wealth {m['winning_wealth']:,.0f}/{m['losing_wealth']:,.0f}; "
                f"calendar/intraday exposure {m['calendar_time_weighted_gross_nav']:.4f}/{m['observed_same_day_time_weighted_gross_nav']:.4f}; "
                f"top-five winner share {m['top5_share_of_positive_pnl']:.4f}; max name/NAV {m['max_single_name_nav_share']:.4f}.")
        payload = render_full_diagnostics(self, state, Path(output_dir), round_num=round_num, round_name=round_name,
            cases=self.exported_cases, training_text='\n'.join(lines))
        from backtests.auto.shared.phase_state import _atomic_write_json
        _atomic_write_json(dict(complete=True, all_native_audits_pass=True,
            plan_sha256=digest(self.plan_path), cases={k: v['source_binding'] for k, v in self.exported_cases.items()},
            reporting_source=self.reporting_source), Path(output_dir) / 'native_diagnostic_evidence.json')
        return payload


if __name__ == '__main__':
    require(len(sys.argv) == 5 and sys.argv[1] == '--verify-case', 'Use the existing final-round script')
    # The source root is supplied by the sealed plan and becomes the only engine
    # import root. The reporting file itself remains outside that frozen tree.
    plan = read(sys.argv[2])
    sys.path.insert(0, plan['cases'][sys.argv[3]]['execution_root'])
    result = validate_frozen(plan, sys.argv[3])
    with Path(sys.argv[4]).open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, default=str)
