"""Read-only critical diagnostics for the frozen KRW25m portfolio selection.

Creates separate review evidence; never retunes, publishes, or runs an allocator.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import numpy as np
from scripts.analyze_native_portfolio import describe, rows, key
from scripts.portfolio_native_budget_study import read, checked, write_json, digest, SCENARIOS
from scripts.replay_selected_portfolio import binding, verify_profile


def ordinal_conflicts(case):
    def stamp(value):
        return datetime.fromisoformat(value).isoformat()
    fills = read(case / 'fills.json')
    filled = Counter()
    fill_times = defaultdict(list)
    for fill in fills:
        if fill['side'] == 'BUY':
            filled[fill['order_id']] += fill['qty']
            fill_times[fill['order_id']].append(stamp(fill['timestamp']))
    blocked = defaultdict(set)
    for row in rows(case / 'admissions.jsonl.gz'):
        if row['admitted_qty'] == 0:
            blocked[row['order']['order_id']].add(stamp(row['timestamp']))
    for row in rows(case / 'native_budget_admissions.jsonl.gz'):
        if row['approved_qty'] == 0:
            blocked[row['order']['order_id']].add(stamp(row['timestamp']))
    groups = defaultdict(list)
    for event in rows(case / 'allocation_events.jsonl.gz'):
        if event['event'] != 'native_submission' or not event['submitted_order_id']:
            continue
        oid = event['submitted_order_id']
        groups[(event['strategy_id'], event['timestamp'])].append({
            'order_id': oid, 'symbol': event['symbol'],
            'candidate_rank': event['native_metadata'].get('candidate_rank'),
            'filled_qty': filled[oid], 'requested_qty': event['native_qty'],
            'fill_timestamps': fill_times[oid], 'zero_admission_timestamps': sorted(blocked[oid])})
    found = []
    for (sid, timestamp), group in groups.items():
        for rejected in group:
            rank = rejected['candidate_rank']
            if rejected['filled_qty'] or not rejected['zero_admission_timestamps'] or not isinstance(rank, (int, float)):
                continue
            peers = [p for p in group if p['filled_qty'] and isinstance(p['candidate_rank'], (int, float))
                     and p['candidate_rank'] > rank]
            if peers:
                found.append({'strategy': sid, 'submission_timestamp': timestamp,
                    'zero_filled_order': rejected, 'filled_worse_ordinal_rank_peers': peers,
                    'same_execution_timestamp_peer_count': sum(bool(set(p['fill_timestamps']) & blocked[rejected['order_id']]) for p in peers)})
    return {'cases': found, 'distinct_sessions': sorted({r['submission_timestamp'][:10] for r in found}),
            'interpretation': 'Ordinal rank inversions among simultaneous same-strategy requests. Rank is not calibrated expected profitability. Observed admission/fill outcomes do not establish the profit of changing execution order.'}


def path_statistics(case):
    trades, daily, bars = (read(case / name) for name in ('trades.json', 'daily.json', 'equity.json'))
    initial = 25_000_000.
    nav = np.asarray([initial] + [d['equity'] for d in daily])
    ret = nav[1:] / nav[:-1] - 1
    k_daily = defaultdict(float)
    for t in trades:
        if t['strategy_id'] == 'KALCB':
            k_daily[t['exit_fill_time'][:10]] += t['net_pnl']
    k = np.asarray([k_daily[d['day']] for d in daily]) / nav[:-1]
    o = ret - k
    threshold = np.quantile(ret, .1)
    tail = ret <= threshold
    close_months = defaultdict(list)
    for i, d in enumerate(daily):
        close_months[d['day'][:7]].append(i)
    months = {m: 100 * (nav[ix[-1] + 1] / nav[ix[0]] - 1) for m, ix in close_months.items()}
    values = np.asarray([v for _, v in bars])
    drawdowns = 1 - values / np.maximum.accumulate(values)
    trough = int(np.argmax(drawdowns)); peak = int(np.argmax(values[:trough + 1]))
    by_symbol = defaultdict(float)
    for t in trades:
        by_symbol[t['symbol']] += t['net_pnl']
    total = sum(t['net_pnl'] for t in trades)
    assert abs(total - (nav[-1] - initial)) < .01
    positive = sorted((t['net_pnl'] for t in trades if t['net_pnl'] > 0), reverse=True)
    return {'monthly_marked_returns_pct': months,
        'before_2026_marked_return_pct': 100 * (next(d['equity'] for d in reversed(daily) if d['day'] < '2026-01-01') / initial - 1),
        'daily_sample_std_pct': float(np.std(ret, ddof=1) * 100),
        'daily_worst_decile_mean_return_pct': float(np.mean(ret[tail]) * 100),
        'daily_worst_decile_both_sleeves_negative': int(sum((k < 0) & (o < 0) & tail)),
        'daily_worst_decile_count': int(sum(tail)),
        'volatility_reduction_vs_perfect_positive_correlation_same_realized_sleeves_pct': float(100 * (1 - np.std(ret) / (np.std(k) + np.std(o)))),
        'max_bar_drawdown': {'peak_time': bars[peak][0], 'trough_time': bars[trough][0], 'pct': float(drawdowns[trough] * 100)},
        'top5_profitable_trades_share_of_net_pnl': sum(positive[:5]) / total,
        'top10_profitable_trades_share_of_net_pnl': sum(positive[:10]) / total,
        'top5_symbols_share_of_net_pnl': sum(sorted(by_symbol.values(), reverse=True)[:5]) / total,
        'interpretation': 'Descriptive diagnostics on the reused path. Perfect-correlation comparator and subperiod returns are not independently tradable counterfactuals or new holdouts.'}


def identity_decomposition(shared_case, standalone_case, sid):
    shared_rows = [t for t in read(shared_case / 'trades.json') if t['strategy_id'] == sid]
    ref_rows = [t for t in read(standalone_case / 'trades.json') if t['strategy_id'] == sid]
    shared = {key(t): t for t in shared_rows}
    ref = {key(t): t for t in ref_rows}
    assert len(shared) == len(shared_rows) and len(ref) == len(ref_rows)
    matched = shared.keys() & ref.keys()
    matched_change = sum(shared[k]['net_pnl'] - ref[k]['net_pnl'] for k in matched)
    new = sum(shared[k]['net_pnl'] for k in shared.keys() - ref.keys())
    missing = sum(ref[k]['net_pnl'] for k in ref.keys() - shared.keys())
    total_change = sum(t['net_pnl'] for t in shared.values()) - sum(t['net_pnl'] for t in ref.values())
    assert abs(total_change - (matched_change + new - missing)) < .01
    return {'matched_entries': len(matched), 'total_sleeve_pnl_change_krw': total_change,
        'matched_entry_pnl_change_krw': matched_change, 'new_entry_pnl_krw': new,
        'missing_reference_pnl_krw': missing, 'new_minus_missing_pnl_krw': new - missing,
        'interpretation': 'Accounting identity, not a causal alpha decomposition. Matching decisions still differ in quantities, execution, exits, account compounding and risk. Separately funded standalone PNL must not be added into an investable KRW25m portfolio.'}


def review(profile_path, report_path, output):
    profile = verify_profile(profile_path)
    base_report = read(report_path)
    assert base_report['profile'] == binding(profile_path)
    study_root = Path(profile['study']['path']).parent
    controls = Path(profile['native_comparison_evidence']['reference_study']['path']).parent
    candidates = ('k075_o075_r1', 'k050_o100_r1', 'k050_o075_r1', 'k050_o050_r1')
    diagnostics = {}
    for name in candidates:
        diagnostics[name] = {}
        for scenario in SCENARIOS:
            case = study_root / 'qualified' / (profile['recipe'] + '_' + scenario) if name == candidates[0] else controls / 'cases' / (name + '_' + scenario)
            summary = describe(case)
            diagnostics[name][scenario] = {k: summary[k] for k in (
                'result', 'frequency_detail', 'pnl_by_strategy', 'olr_liquidation_delay',
                'overnight_exposure', 'admission_diagnostics', 'observed_funding_exposure',
                'worst_session_pct', 'recycling_support')}
            diagnostics[name][scenario]['path_statistics'] = path_statistics(case)
            diagnostics[name][scenario]['case_completion'] = binding(case / 'completion.json')
    conflicts, decomposition = {}, {}
    for scenario in SCENARIOS:
        case = study_root / 'qualified' / (profile['recipe'] + '_' + scenario)
        conflicts[scenario] = ordinal_conflicts(case)
        decomposition[scenario] = {sid: identity_decomposition(case, controls / 'cases' / (name + '_' + scenario), sid)
            for sid, name in (('KALCB', 'control_k075'), ('OLR', 'control_o075_r1'))}
    grid = base_report['base_grid_results']
    out = {'created_utc': datetime.now(timezone.utc).isoformat(), 'profile': binding(profile_path),
        'review_generator': binding(Path(__file__)),
        'original_report': binding(report_path), 'canonical_pointer': binding(REPO / 'config/research/portfolio_synergy_selected.json'),
        'scope': 'Post-selection descriptive review only; no new performance validation or policy promotion.',
        'candidate_diagnostics': diagnostics, 'same_strategy_ordinal_conflicts': conflicts,
        'pnl_identity_decomposition': decomposition,
        'close_alternative_tradeoff': {s: {k: grid['k075_o075_r1'][i][k] - grid['k050_o100_r1'][i][k]
            for k in ('return_pct', 'drawdown_pct', 'trades')} for i, s in enumerate(SCENARIOS)},
        'source_files': {p: binding(REPO / p) for p in (
            'deployment/olr_kalcb/native_budget.py', 'backtests/strategies/portfolio_synergy/native_policy.py',
            'scripts/portfolio_native_budget_study.py')},
        'no_default_or_trading_runtime_changed': True}
    write_json(output, out)
    print(output)


def plot_review(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data = read(output); original = read(data['original_report']['path'])
    colors = {'k075_o075_r1': '#b45309', 'k050_o100_r1': '#2563eb',
              'k050_o075_r1': '#0f766e', 'k050_o050_r1': '#64748b'}
    labels = {'k075_o075_r1': 'Selected 0.75 / 0.75', 'k050_o100_r1': '0.50 / 1.00',
              'k050_o075_r1': '0.50 / 0.75', 'k050_o050_r1': '0.50 / 0.50'}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    left, right = axes
    for name, values in original['base_grid_results'].items():
        if name.startswith('k') and name.endswith('_r1'):
            row = values[1]
            left.scatter(row['drawdown_pct'], row['return_pct'], s=45, color='#cbd5e1', zorder=2)
    left.axvspan(12.5, 17, color='#fee2e2', alpha=.45)
    left.axvline(12.5, color='#b91c1c', linestyle='--', linewidth=1)
    offsets = {'k075_o075_r1': (-108, 18), 'k050_o100_r1': (-80, -22),
               'k050_o075_r1': (-65, 13), 'k050_o050_r1': (-33, -22)}
    for name, values in data['candidate_diagnostics'].items():
        row = values['combined']['result']
        left.scatter(row['drawdown_pct'], row['return_pct'], s=140 if name == 'k075_o075_r1' else 75,
                     marker='*' if name == 'k075_o075_r1' else 'o', color=colors[name], zorder=4)
        left.annotate(labels[name], (row['drawdown_pct'], row['return_pct']),
                      xytext=offsets[name], textcoords='offset points', color=colors[name], fontsize=9)
        exposure = values['combined']['overnight_exposure']['max_nav_fraction'] * 100
        right.scatter(exposure, row['drawdown_pct'], s=150 if name == 'k075_o075_r1' else 85,
                      marker='*' if name == 'k075_o075_r1' else 'o', color=colors[name])
        right.annotate(labels[name], (exposure, row['drawdown_pct']), xytext=(0, 13),
                       textcoords='offset points', ha='center', color=colors[name], fontsize=9)
    adverse = original['robustness']['auction/auction_nonfill50_combined']['metrics']
    left.scatter(adverse['drawdown_pct'], adverse['return_pct'], color='#b91c1c', marker='x', s=85, linewidth=2, zorder=5)
    left.annotate('Selected + 50%\nauction nonfills', (adverse['drawdown_pct'], adverse['return_pct']),
                  xytext=(14, 7), textcoords='offset points', fontsize=9, color='#b91c1c')
    left.set(xlim=(5, 17), ylim=(40, 132), xlabel='Combined-stress maximum drawdown (%)',
             ylabel='Combined-stress cumulative net return (%)', title='Best tested under one set of constraints')
    right.set(xlim=(43, 108), ylim=(6.5, 13), xlabel='Maximum overnight exposure (% of NAV)',
              ylabel='Combined-stress maximum drawdown (%)', title='Lower historical drawdown can conceal more overnight risk')
    for ax in axes:
        ax.grid(alpha=.2); ax.set_axisbelow(True); ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('KRW25m portfolio: evidence of benefit, no proof of maximum synergy', fontsize=15, y=.98)
    fig.text(.06, .05, 'Budget labels: KALCB native sizing factor / OLR sleeve budget. They are not fixed weights summing to one.\n'
             'Reused development window: 7 Aug 2025–31 Mar 2026. Gray points: other recycling-on grid cases. Dashed line: original 12.5% stress gate.', fontsize=9)
    fig.tight_layout(rect=(0, .12, 1, .94))
    path = output.with_suffix('.png'); fig.savefig(path, dpi=170); plt.close(fig)
    print(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    review(args.profile.resolve(), args.report.resolve(), args.output.resolve())
    plot_review(args.output.resolve())
