"""Final round reporting for a user-assigned, sealed native portfolio candidate.

Observational adapter for the existing native replay and attribution scripts.
It does not fit, rank or trade, and cannot convert a failed OOS gate into a pass.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import math
from pathlib import Path
import json

from backtests.auto.shared.phase_state import _atomic_write_json
from backtests.auto.shared.round_manager import RoundManager
from backtests.analysis.metrics import max_drawdown_pct
from scripts.portfolio_native_budget_study import checked, read, verified_spec
from scripts.replay_selected_portfolio import binding, bound, digest

VERSION = 'portfolio-user-assigned-native-round-v1'
STATUS = 'user_selected_research_round_oos_failed'
WINDOW = {'start': '2025-08-07', 'end': '2026-03-31'}
CAPITAL = 25_000_000
ECONOMIC_FILES = ('trades.json', 'daily.json', 'equity.json', 'fills.json', 'marginal_quantity_evidence.json')


def battery_case(label, recipe):
    if label in ('baseline', 'cost', 'participation', 'combined'):
        return recipe, label
    for prefix, suffix in (('auction_price_', '_price'), ('auction_nonfill_', '_nonfill')):
        if label.startswith(prefix) and label[len(prefix):] in ('baseline', 'combined'):
            return recipe + suffix, label[len(prefix):]
    raise ValueError('Unregistered battery scenario: ' + label)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b, message):
    require(math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-5), message)


def verify_window(daily, equity, trades, window):
    require(bool(daily) and bool(equity), 'Empty native replay')
    days = [d['day'] for d in daily]
    require(days == sorted(set(days)), 'Duplicate or unsorted session dates')
    require(days[0] == window['start'] and days[-1] == window['end'], 'IS boundary mismatch')
    require(all(window['start'] <= str(t)[:10] <= window['end'] for t, _ in equity),
            'Equity observations outside IS')
    require(all(window['start'] <= t[k][:10] <= window['end'] for t in trades
                for k in ('entry_decision_time', 'entry_fill_time', 'exit_fill_time')), 'Trade outside IS')


def case_evidence(path, study, recipe, scenario):
    path = Path(path)
    result = checked(path, True)
    request = read(path / 'request.json')
    arguments = request['arguments']
    require(arguments.get('full_audit') is True
            and not any(arguments.get(k) for k in ('standalone', 'canonical_costs', 'absolute_caps', 'no_upward_compounding')),
            'Unexpected diagnostic replay flags')
    require(request['recipe'] == study['candidates'][recipe]
            and result['recipe'] == recipe and result['scenario'] == scenario
            and result['source_generation_hash'] == study['source_generation_hash'], 'Case identity mismatch')
    actual_study = read(request['arguments']['study'])
    require(actual_study['configuration']['date_range'] == WINDOW
            and actual_study['initial_equity'] == CAPITAL, 'Wrong replay window or account capital')
    audit = read(path / 'execution_audit.json')
    require(audit['audit_pass'] is True and audit['native_budget_audit']['pass'] is True,
            'Full native audit must pass')
    daily, equity, trades = [read(path / (n + '.json')) for n in ('daily', 'equity', 'trades')]
    verify_window(daily, equity, trades, WINDOW)
    require(len(daily) == result['sessions'] and len(trades) == result['trades'], 'Counts do not reconcile')
    close(equity[0][1], CAPITAL, 'Incorrect starting NAV')
    close(daily[-1]['equity'], equity[-1][1], 'Final daily/bar NAV differs')
    close(sum(t['net_pnl'] for t in trades), equity[-1][1] - CAPITAL, 'Trade/NAV reconciliation failed')
    close(result['return_pct'], (equity[-1][1] / CAPITAL - 1) * 100, 'Return drift')
    close(result['drawdown_pct'], max_drawdown_pct([CAPITAL] + [v for _, v in equity]) * 100, 'MTM drawdown drift')
    require(not result.get('terminal_deferred_intents') and not result.get('terminal_rotation_intents'),
            'Unresolved strategy intents')
    return {'output': str(path.resolve()), 'recipe': recipe, 'scenario': scenario,
        'metrics': {k: result[k] for k in ('return_pct', 'drawdown_pct', 'trades', 'sessions',
                    'active_days', 'trades_per_21_sessions', 'strategy_counts', 'audit_status')},
        'evidence': {n: binding(path / n) for n in ('completion.json', 'request.json', 'result.json', 'execution_audit.json')},
        'reconciliation_pass': True}


def make_profile(round_dir):
    round_dir = Path(round_dir).resolve()
    context = read(round_dir / 'round_registration.json')
    require(context['date_range'] == WINDOW and context['initial_equity_krw'] == CAPITAL
            and context['live_deployment_enabled'] is False and context['oos_acceptance_pass'] is False,
            'Invalid round registration')
    study, seal, oos = [bound(context[k]) for k in ('study', 'selection', 'oos_acceptance')]
    for key in ('manifest', 'protocol', 'archive_receipt', 'archived_manifest', 'previous_pointer'):
        bound(context[key])
    verified_spec(study)
    from scripts.portfolio_allocation_profile import verify_frozen_bytes
    verify_frozen_bytes(study, {})
    recipe = context['recipe']
    require(recipe == seal['finalist'] == oos['finalist'] == 'k_marginal'
            and context['settings'] == seal['recipe'] == study['candidates'][recipe], 'Changed sealed candidate')
    require(study['configuration']['date_range'] == WINDOW and study['initial_equity'] == CAPITAL,
            'Wrong source study boundary/capital')
    require(oos['acceptance']['pass'] is False and not oos['promotion_eligible']
            and oos['source_and_execution_pass'] is True, 'OOS result changed')
    cases, parity = {}, {}
    for label, item in seal['candidate_batteries'][recipe].items():
        bound(item['completion'])
        original = Path(item['completion']['path']).parent
        case = round_dir / 'diagnostics' / label if label in context['replay_scenarios'] else original
        actual_recipe, actual_scenario = battery_case(label, recipe)
        cases[label] = case_evidence(case, study, actual_recipe, actual_scenario)
        if label in context['replay_scenarios']:
            hashes = {name: {'sealed': digest(original / name), 'rerun': digest(case / name)} for name in ECONOMIC_FILES}
            require(all(x['sealed'] == x['rerun'] for x in hashes.values()), 'Fresh replay changed sealed economic paths')
            parity[label] = {'pass': True, 'files': hashes, 'sealed_completion': item['completion']}
    require(set(cases) == {'baseline', 'cost', 'participation', 'combined', 'auction_price_baseline',
        'auction_price_combined', 'auction_nonfill_baseline', 'auction_nonfill_combined'}
        and set(parity) == {'baseline', 'combined'}, 'Incomplete final battery')
    return {'version': VERSION, 'status': STATUS, 'round': 1, 'round_directory': str(round_dir),
        'recipe': recipe, 'settings': context['settings'], 'initial_equity_krw': CAPITAL,
        'evaluation_window': WINDOW, 'study': context['study'], 'selection': context['selection'],
        'registration': binding(round_dir / 'round_registration.json'),
        'source_generation_hash': study['source_generation_hash'],
        'qualified_cases': cases, 'fresh_replay_parity': parity,
        'oos_acceptance': context['oos_acceptance'], 'oos_acceptance_pass': False,
        'selection_basis': context['assignment'], 'new_search_trials': 0,
        'independent_performance_established': False, 'live_deployment_enabled': False,
        'operational_scope': 'User-assigned research round. Native audits passed; reserved OOS acceptance failed. No live route.'}


def verify_user_assigned_profile(profile):
    require(profile.get('version') == VERSION and profile.get('status') == STATUS,
            'Unsupported user-assigned profile')
    bound(profile['registration'])
    require(profile == make_profile(profile['round_directory']), 'Profile differs from bound round evidence')
    return profile


def extra_diagnostics(path):
    """Compute descriptive dimensions without inventing shadow trade outcomes."""
    import numpy as np
    from scripts.analyze_native_portfolio import describe, distribution
    from scripts.portfolio_targeted_validation import direct_support
    path = Path(path)
    attribution = describe(path)
    daily, equity, trades = [read(path / (n + '.json')) for n in ('daily', 'equity', 'trades')]
    nav = np.asarray([CAPITAL] + [d['equity'] for d in daily], dtype=float)
    returns = nav[1:] / nav[:-1] - 1
    groups = {}
    for dimension, fn in (
        ('strategy', lambda t: t['strategy_id']), ('symbol', lambda t: t['symbol']),
        ('exit_reason', lambda t: t['exit_reason']),
        ('entry_route', lambda t: t['strategy_id'] + ':' + str(t.get('route_metadata', {}).get('entry_route', 'native')))):
        buckets = defaultdict(list)
        for trade in trades:
            buckets[fn(trade)].append(trade)
        groups[dimension] = {k: distribution(v) for k, v in sorted(buckets.items())}
    months = {}
    previous = float(CAPITAL)
    for i, day in enumerate(daily):
        month = day['day'][:7]
        row = months.setdefault(month, {'start_equity': previous, 'sessions': 0})
        row['sessions'] += 1
        row['end_equity'] = day['equity']
        row['return_pct'] = 100 * (row['end_equity'] / row['start_equity'] - 1)
        previous = day['equity']
    downside = math.sqrt(float(np.mean(np.minimum(returns, 0) ** 2)))
    annualized = (nav[-1] / CAPITAL) ** (252 / len(daily)) - 1
    dd = max_drawdown_pct([v for _, v in equity])
    evidence = read(path / 'marginal_quantity_evidence.json')
    observations = evidence['decisions']
    return {'attribution': attribution, 'trade_dimensions': groups, 'monthly_mtm': months,
        'session_returns': [{'day': d['day'], 'return_pct': float(r * 100)} for d, r in zip(daily, returns)],
        'worst_sessions': sorted([{'day': d['day'], 'return_pct': float(r * 100),
                                  'pnl_krw': float(nav[i + 1] - nav[i])}
                                 for i, (d, r) in enumerate(zip(daily, returns))], key=lambda r: r['return_pct'])[:10],
        'risk_statistics': {'sortino': float(np.mean(returns) / downside * math.sqrt(252)) if downside else None,
            'calmar': annualized / dd if dd else None, 'annualized_return_pct': annualized * 100,
            'basis': '252 exchange sessions/year; daily close returns including cash days; zero downside target; bar MTM DD.'},
        'marginal_quantity': {'summary': {k: v for k, v in evidence.items() if k != 'decisions'},
            'decision_count': len(observations), 'reason_counts': dict(Counter(r.get('reason', 'unspecified') for r in observations)),
            'direct_support': direct_support(path, 'k_marginal')},
        'limits': ['Blocked trade outcome superiority is unmeasured without causal, matched shadow execution.',
                   'Month/strategy/trade dimensions are attribution of this account, not independently funded portfolios.',
                   'Higher historical return and frequency do not establish maximum expected future returns.']}


def render_report(payload):
    p = payload['profile']
    lines = ['PORTFOLIO SYNERGY — ROUND 1 FINAL NATIVE DIAGNOSTICS',
        f"Generated UTC: {payload['generated_at_utc']}",
        'IS: 2025-08-07 through 2026-03-31 inclusive; 156 exchange sessions.',
        'Starting equity: KRW25,000,000. Candidate: k_marginal.',
        'Status: user-assigned research round; full native IS audits passed; OOS acceptance FAILED.',
        'No new search, refit or changed strategy parameters. June–July is excluded from IS metrics.',
        'KALCB sizing 0.75; OLR allocation 0.625; aggregate KALCB planned-risk ceiling 2% NAV.',
        'KALCB supported marginal lower-tier quantity rule; native OLR capacity redistribution enabled.', '',
        'IS SCENARIO BATTERY (returns/DD in percent; entries are closed economic lifecycles)',
        'Scenario | net return | max MTM DD | entries | trades/21 sessions | full audit']
    for name, row in p['qualified_cases'].items():
        m = row['metrics']
        lines.append(f"{name} | {m['return_pct']:.6f}% | {m['drawdown_pct']:.6f}% | {m['trades']} | {m['trades_per_21_sessions']:.4f} | {m['audit_status']}")
    lines += ['', 'BASELINE AND COMBINED STRESS: FRESH REPLAY / SEALED CANDIDATE PARITY',
        'Trades, fills, daily equity, bar equity and marginal policy decisions reproduce the sealed candidate exactly.',
        'Other six stress cases are authenticated saved full audits, not represented as fresh runs.', '',
        'CURRENT CONTROL COMPARISON (same IS and capital; differences are candidate minus control)']
    for name, row in payload['control_comparison'].items():
        lines.append(f"{name}: return {row['delta_return_pp']:+.6f}pp; DD {row['delta_drawdown_pp']:+.6f}pp; entries {row['delta_trades']:+d}")
    for name, d in payload['diagnostics'].items():
        r = payload['profile']['qualified_cases'][name]['metrics']
        lines += ['', name.upper() + ' FULL DIAGNOSTICS',
            f"Sessions {r['sessions']}; active entry days {r['active_days']}; closed entries by sleeve {r['strategy_counts']}.",
            'Canonical metrics: ' + json.dumps(payload['native_metrics'][name], sort_keys=True),
            'Risk statistics: ' + json.dumps(d['risk_statistics']),
            'Reconciled net PNL by sleeve: ' + json.dumps(d['attribution']['pnl_by_strategy']),
            'Overnight exposure: ' + json.dumps(d['attribution']['overnight_exposure']),
            'Daily marked sleeve contributions: ' + json.dumps(d['attribution']['daily_marked_sleeve_contributions']),
            'Funding exposure: ' + json.dumps(d['attribution']['observed_funding_exposure']),
            'Admission diagnostics: ' + json.dumps(d['attribution']['admission_diagnostics']),
            'OLR liquidation delay: ' + json.dumps(d['attribution']['olr_liquidation_delay']),
            'Marginal quantity policy: ' + json.dumps(d['marginal_quantity'], default=str),
            'Monthly account MTM: ' + json.dumps(d['monthly_mtm']),
            'Worst sessions: ' + json.dumps(d['worst_sessions']),
            'Exit reasons: ' + json.dumps(d['trade_dimensions']['exit_reason']),
            'Entry routes: ' + json.dumps(d['trade_dimensions']['entry_route']),
            'Symbol attribution: ' + json.dumps(d['trade_dimensions']['symbol']),
            'Recycling interventions: ' + json.dumps(d['attribution']['recycling_interventions']),
            'Opposite strategy flows: ' + json.dumps(d['attribution']['same_phase_opposite_strategy_flows']),
            'Native audit summary: ' + json.dumps(payload['audit_summaries'][name])]
    lines += ['', 'SEPARATE OOS VALIDATION — NOT INCLUDED IN THE ABOVE IS METRICS',
        'Reserved test: 2026-06-01–2026-07-10, now consumed. Full source/execution audits passed; acceptance failed.',
        json.dumps(payload['oos_summary'], indent=2), '', 'INTERPRETATION AND LIMITS',
        'The candidate improves reused development results modestly. Its lower-tier action did not activate in the reserved OOS; current and candidate OOS paths were identical.',
        'Round assignment records the user-selected research baseline. It does not override failed OOS gates, prove maximum synergy, or authorize live deployment.',
        'The sealed IS source is retained because the subsequent repaired data dates are outside this development window. No later observations are appended.',
        'Trade rejections and quantity reductions are collapsed by order ID. Missing entries are not automatically funding blocks; hindsight PNL is never used as an entry ranking input.',
        'Native buying power can produce negative intraday cash. Financing and real broker parity remain unqualified; marked exposure can change after admission.',
        'Baseline/stress cost and participation assumptions are preserved in the saved requests and frozen study. Planned stop risk is not a guaranteed realized-loss ceiling.',
        'Full JSON includes session returns, all attribution dimensions, evidence hashes, selected settings and provenance.',
        'Metric units: canonical manifest return/DD fields are fractions; scenario tables explicitly use percent.', '',
        'PROVENANCE', json.dumps(payload['provenance'], indent=2)]
    return '\n'.join(lines) + '\n'


def plot_round(output, profile):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, layout='constrained')
    for name, case in profile['qualified_cases'].items():
        if name not in ('baseline', 'combined'):
            continue
        data = read(Path(case['output']) / 'equity.json')
        dates = [datetime.fromisoformat(t) for t, _ in data]
        values = np.asarray([v for _, v in data])
        axes[0].plot(dates, values / 1e6, label=name)
        axes[1].plot(dates, 100 * (values / np.maximum.accumulate(values) - 1), label=name)
    axes[0].set_ylabel('Account equity (KRW million)')
    axes[1].set_ylabel('Bar MTM drawdown (%)')
    axes[0].set_title('Portfolio synergy round 1 | k_marginal | IS through 31 March 2026 | KRW25m')
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend()
    fig.savefig(output / 'round_final_equity_drawdown.png', dpi=160)
    plt.close(fig)


def write_round_diagnostics(round_dir):
    from scripts.portfolio_allocation_challenge import risk_metrics
    output = Path(round_dir).resolve()
    profile = make_profile(output)
    context = read(output / 'round_registration.json')
    seal = bound(context['selection'])
    study = bound(context['study'])
    oos = bound(context['oos_acceptance'])
    diagnostics, audit_summaries, native_metrics, comparison = {}, {}, {}, {}
    for scenario in ('baseline', 'combined'):
        case = Path(profile['qualified_cases'][scenario]['output'])
        diagnostics[scenario] = extra_diagnostics(case)
        result = read(case / 'result.json')
        native_metrics[scenario] = result['metrics']
        audit = read(case / 'execution_audit.json')
        audit_summaries[scenario] = {k: audit[k] for k in ('audit_pass', 'audit_status', 'execution_validity_pass',
            'same_bar_fill_count', 'forced_replay_close_count', 'rejected_order_count', 'invalid_order_rejection_count',
            'intraday_residual_session_count', 'native_budget_audit')}
    for scenario, item in seal['control_batteries']['control'].items():
        bound(item['completion'])
        path = Path(item['completion']['path']).parent
        control = checked(path, True)
        control_study = read(read(path / 'request.json')['arguments']['study'])
        require(control_study['initial_equity'] == CAPITAL and control_study['configuration']['date_range'] == WINDOW,
                'Control does not share IS window/capital')
        candidate = profile['qualified_cases'][scenario]['metrics']
        comparison[scenario] = {'control_completion': item['completion'],
            'control': {k: control[k] for k in ('return_pct', 'drawdown_pct', 'trades')},
            'candidate': candidate, 'delta_return_pp': candidate['return_pct'] - control['return_pct'],
            'delta_drawdown_pp': candidate['drawdown_pct'] - control['drawdown_pct'],
            'delta_trades': candidate['trades'] - control['trades']}
    oos_summary = {'window': {'start': '2026-06-01', 'end': '2026-07-10'},
        'acceptance': oos['acceptance'], 'candidate': {}, 'current_control': {},
        'source_and_execution_pass': oos['source_and_execution_pass'], 'validation_window_now_consumed': True}
    for name, key in (('k_marginal', 'candidate'), ('control', 'current_control')):
        for scenario, case in oos['qualified_cases'][name].items():
            for evidence in case['evidence'].values():
                bound(evidence)
            m = checked(Path(case['output']), True)
            oos_summary[key][scenario] = {k: m[k] for k in ('return_pct', 'drawdown_pct', 'trades')}
    payload = {'strategy': 'portfolio_synergy', 'round': 1, 'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'diagnostics_version': 'native-portfolio-full-eor-v1', 'profile': profile,
        'promotion_status': STATUS, 'native_metrics': native_metrics, 'audit_summaries': audit_summaries,
        'control_comparison': comparison, 'diagnostics': diagnostics, 'oos_summary': oos_summary,
        'risk_battery': {s: risk_metrics(Path(c['output'])) for s, c in profile['qualified_cases'].items()},
        'provenance': {k: context[k] for k in ('selection', 'study', 'manifest', 'protocol', 'oos_acceptance', 'archive_receipt')},
        'live_deployment_enabled': False}
    for path in sorted((output / 'source_preflight').glob('*.json')):
        payload['provenance'][path.stem] = binding(path)
    if (output / 'verification_tests.json').exists():
        payload['provenance']['verification_tests'] = binding(output / 'verification_tests.json')
    _atomic_write_json(profile, output / 'selected_research_profile.json')
    report = render_report(payload)
    (output / 'round_final_diagnostics.txt').write_text(report, encoding='utf-8')
    plot_round(output, profile)
    base = read(output / 'diagnostics/baseline/result.json')
    audit = read(output / 'diagnostics/baseline/execution_audit.json')
    contract = {'strategy': 'portfolio_synergy', 'date_window': WINDOW, 'initial_equity': CAPITAL,
        'source_fingerprint': profile['source_generation_hash'],
        'feature_manifest_hash': read(study['manifest'])['manifest_hash'],
        'candidate_snapshot_hash': audit['audit_receipt']['identity']['qualification_snapshot_hash'],
        'capability_level': 'real_replay', 'replay_mode': 'native_shared_account_sim_broker',
        'fill_timing': 'next_completed_5m_open_and_native_auction_contract',
        'auction_mode': 'native_olr_close_auction_with_liquidity_fallback',
        'primary_promotion_basis': 'SimBroker.equity_curve_bar_level_mtm',
        'cost_policy': {'source': context['study'], 'scenario': 'baseline',
                        'basis': 'Frozen native strategy commissions, sell tax, slippage and participation'}}
    metrics = {**base['metrics'], 'official_mtm_net_return_pct': base['return_pct'] / 100,
        'net_return_pct_basis': 'native_marked_account_equity_over_initial_equity',
        'official_metric_basis': 'SimBroker.equity_curve_bar_level_mtm',
        'primary_promotion_metric': 'official_mtm_net_return_pct',
        'primary_promotion_value': base['return_pct'] / 100,
        'primary_promotion_basis': 'SimBroker.equity_curve_bar_level_mtm',
        'official_replay_pass': True, 'audit_pass': True, 'audit_status': 'passed',
        'promotion_status': STATUS, 'promotion_requires_audit_pass': True,
        'source_fingerprint': profile['source_generation_hash'],
        'feature_manifest_hash': contract['feature_manifest_hash'], 'candidate_snapshot_hash': contract['candidate_snapshot_hash'],
        'end_open_position_count': 0, 'execution_contract': contract,
        **{k: audit[k] for k in ('same_bar_fill_count', 'forced_replay_close_count', 'rejected_order_count')}}
    metadata = {'promotion_status': STATUS,
        'artifact_promotion_policy': 'user_assigned_research_round_full_IS_audits_passed_OOS_failed_no_live',
        'execution_contract': contract, 'initial_equity_krw': CAPITAL, 'date_range': WINDOW,
        'selected_profile': binding(output / 'selected_research_profile.json'),
        'oos_acceptance_pass': False, 'live_deployment_enabled': False,
        'metric_contract': {'primary_promotion_metric': 'official_mtm_net_return_pct',
            'primary_promotion_value': base['return_pct'] / 100,
            'primary_promotion_basis': 'SimBroker.equity_curve_bar_level_mtm',
            'metric_units': 'Canonical return/drawdown fractions; scenario tables percent.',
            'execution_contract': contract, 'oos_acceptance_pass': False}}
    payload.update(headline_metrics=metrics, metric_contract=metadata['metric_contract'], execution_contract=contract)
    _atomic_write_json(payload, output / 'round_final_full_diagnostics.json')
    _atomic_write_json({'status': 'pending_paper_live_execution',
        'scope': 'Frozen native research execution only', 'native_research_audit_pass': True,
        'paper_live_parity_established': False, 'oos_acceptance_pass': False,
        'live_deployment_enabled': False,
        'outstanding': ['Broker buying-power and financing equivalence',
            'Live observation/model delivery timing', 'Order, partial-fill, cancellation and restart reconciliation',
            'Independent performance validation after failed June-July acceptance']},
        output / 'live_backtest_parity_alignment.json')
    manager = RoundManager('stock', 'portfolio_synergy', base_dir=output.parent.parent)
    mutations = {'recipe': profile['recipe']}
    manager.write_run_spec(output, 1, 'portfolio_synergy',
        description='User-assigned sealed candidate; fresh final native replay through 2026-03-31.',
        baseline_mutations=mutations, baseline_source=context['selection']['path'], execution_context=contract, overwrite=True)
    manager.write_optimized_config(output, mutations, artifact_metadata={**metadata, 'settings': profile['settings']})
    manager.write_run_summary(output, mutations, metrics, [], round_num=1, artifact_metadata=metadata)
    (output / 'round_evaluation.txt').write_text(
        'User-assigned research round_1: k_marginal. No optimization phases rerun.\n'
        'IS ends 2026-03-31; KRW25m; native full audits and fresh replay parity passed.\n'
        'Reserved June–July OOS failed; independent performance and live readiness are not established.\n', encoding='utf-8')
    manager.append_to_manifest(1, mutations, metrics, artifact_metadata=metadata)
    manifest = manager.load_manifest()
    manifest.pop('pending_round', None)
    entry = next(r for r in manifest['rounds'] if r['round'] == 1)
    entry.update(date_range=WINDOW, initial_equity_krw=CAPITAL,
        selected_profile=metadata['selected_profile'], oos_acceptance_pass=False, live_deployment_enabled=False,
        final_diagnostics=binding(output / 'round_final_full_diagnostics.json'))
    _atomic_write_json(manifest, manager.manifest_path)
    pointer_path = Path(context['selected_pointer_path'])
    replacement = {'version': 'portfolio-research-pointer-v1',
        'profile': binding(output / 'selected_research_profile.json'), 'live_deployment_enabled': False}
    require(digest(pointer_path) == context['previous_pointer']['sha256'] or read(pointer_path) == replacement,
            'Selected research pointer changed concurrently')
    _atomic_write_json(replacement, pointer_path)
    (output / 'round_readme.txt').write_text(
        'Canonical portfolio_synergy round_1: k_marginal, KRW25m.\n'
        'In sample: 2025-08-07 through 2026-03-31 inclusive.\n'
        'User-assigned research round; native IS audits passed; June-July OOS acceptance failed.\n\n'
        'Existing report entry point (regenerates diagnostics from bound native replays):\n'
        'python -B scripts/report_native_portfolio.py --round-dir data/backtests/output/portfolio_synergy/round_1\n\n'
        'Verify canonical research profile:\n'
        'python -B scripts/replay_selected_portfolio.py verify\n\n'
        'Replay a declared IS scenario into a NEW folder:\n'
        'python -B scripts/replay_selected_portfolio.py replay --scenario baseline --output <new-folder>\n\n'
        'The legacy synthetic optimization YAML does not implement this native candidate.\n'
        'Fresh final baseline/combined logs and command receipts: logs/.\n'
        'Native replay exports and full audits: diagnostics/baseline and diagnostics/combined.\n'
        'Six remaining IS stress cases: authenticated frozen evidence paths in selected_research_profile.json.\n'
        'Archive location and original checksums: round_registration.json -> archive_receipt.\n'
        'Canonical manifest return/DD values are fractions; human scenario tables are percent.\n', encoding='utf-8')
    # Authenticate output bytes again; report code never changes native execution.
    for case in profile['qualified_cases'].values():
        checked(Path(case['output']), True)
    _atomic_write_json({'strategy': 'portfolio_synergy', 'round': 1, 'date_range': WINDOW,
        'initial_equity_krw': CAPITAL, 'final_metrics': metrics,
        'scenarios': {s: c['metrics'] for s, c in profile['qualified_cases'].items()},
        'promotion_status': STATUS, 'oos_acceptance_pass': False,
        'native_audits_pass': True, 'live_deployment_enabled': False}, output / 'diagnostics_summary.json')
    names = ('round_final_full_diagnostics.json', 'round_final_diagnostics.txt', 'round_final_equity_drawdown.png',
             'round_evaluation.txt', 'run_spec.json', 'optimized_config.json', 'run_summary.json',
             'selected_research_profile.json', 'diagnostics_summary.json', 'round_readme.txt',
             'live_backtest_parity_alignment.json')
    _atomic_write_json({'strategy': 'portfolio_synergy', 'round': 1,
        'artifacts': {n: binding(output / n) for n in names},
        'native_replays': profile['qualified_cases'], 'source_lineage': payload['provenance']},
        output / 'full_diagnostics_index.json')
    _atomic_write_json({'strategy': 'portfolio_synergy', 'round': 1, 'complete': True,
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'mode': 'existing_native_replay_and_report_script', 'date_range': WINDOW, 'initial_equity_krw': CAPITAL,
        'full_native_audits_pass': True, 'fresh_baseline_combined_parity_pass': True,
        'fresh_replay_scenarios': ['baseline', 'combined'], 'authenticated_scenario_count': 8,
        'oos_acceptance_pass': False, 'live_deployment_enabled': False,
        'round_final_diagnostics_exists': True, 'round_evaluation_exists': True,
        'round_final_diagnostics_path': str(output / 'round_final_diagnostics.txt'),
        'round_final_diagnostics_bytes': (output / 'round_final_diagnostics.txt').stat().st_size,
        'round_final_diagnostics_lines': len(report.splitlines()),
        'artifacts': {n: binding(output / n) for n in (*names, 'full_diagnostics_index.json')}},
        output / 'round_final_diagnostics_status.json')
    return payload
