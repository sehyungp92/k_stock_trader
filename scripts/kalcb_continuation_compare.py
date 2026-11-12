"""Paired account and added-trade diagnostics for continuation experiments."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backtests.strategies.kalcb import executed_value as ev


def load(directory):
    result = json.loads((directory / 'result.json').read_text())
    days = json.loads((directory / 'daily.json').read_text())
    trades = json.loads((directory / 'trades.json').read_text())
    if not json.loads((directory / 'completion.json').read_text())['complete']:
        raise ValueError('Incomplete replay')
    return result, days, trades


def day_pnl(days):
    previous = ev.REFERENCE_EQUITY
    result = {}
    for row in days:
        result[row['day']] = row['equity'] - previous
        previous = row['equity']
    return result


def episodes(trades):
    result = defaultdict(float)
    for t in trades:
        result[(t['entry_fill_time'][:10], t['symbol'])] += t['net_pnl']
    return dict(result)


def compare(control, candidate):
    import numpy as np
    cr, cd, ct = load(control)
    nr, nd, nt = load(candidate)
    cp, npnl = day_pnl(cd), day_pnl(nd)
    if list(cp) != list(npnl):
        raise ValueError('Different evaluation calendar')
    if cr['scenario'] != nr['scenario']:
        raise ValueError('Different execution scenario')
    old, new = episodes(ct), episodes(nt)
    added, removed, shared = new.keys() - old.keys(), old.keys() - new.keys(), new.keys() & old.keys()
    attribution = {
        'added_stock_days': len(added), 'added_net_pnl': sum(new[k] for k in added),
        'removed_stock_days': len(removed), 'removed_control_net_pnl': sum(old[k] for k in removed),
        'shared_stock_days': len(shared), 'shared_net_pnl_change': sum(new[k] - old[k] for k in shared),
    }
    change = sum(npnl.values()) - sum(cp.values())
    reconciled = attribution['added_net_pnl'] - attribution['removed_control_net_pnl'] + attribution['shared_net_pnl_change']
    if abs(change - reconciled) > .1:
        raise ValueError('Trade/account attribution does not reconcile; investigate inventory')
    by_month = defaultdict(lambda: {'control_net_pnl': 0., 'candidate_net_pnl': 0., 'incremental_net_pnl': 0.})
    for d in cp:
        m = by_month[d[:7]]
        m['control_net_pnl'] += cp[d]
        m['candidate_net_pnl'] += npnl[d]
        m['incremental_net_pnl'] += npnl[d] - cp[d]
    delta = np.asarray([npnl[d] - cp[d] for d in cp]) / ev.REFERENCE_EQUITY
    rng = np.random.default_rng(20261006)
    bootstrap = []
    # Five-session moving blocks preserve short-range serial dependence. This
    # is descriptive uncertainty on reused history, not selection-adjusted OOS.
    for _ in range(5000):
        starts = rng.integers(0, max(1, len(delta) - 4), size=(len(delta) + 4) // 5)
        sample = np.concatenate([delta[s:s + 5] for s in starts])[:len(delta)]
        bootstrap.append(sample.sum() * 100)
    gross = sum(t['gross_pnl'] for t in nt)
    loss = -sum(min(t['net_pnl'], 0) for t in nt)
    wins = sum(max(t['net_pnl'], 0) for t in nt)
    required = ('return_pct', 'drawdown_pct', 'trades', 'active_days', 'execution_validity_pass', 'audit_status', 'residual_sessions', 'terminal_inventory')
    return {
        'control': {k: cr[k] for k in required}, 'candidate': {k: nr[k] for k in required},
        'return_change_percentage_points': nr['return_pct'] - cr['return_pct'],
        'trade_count_change': nr['trades'] - cr['trades'], 'attribution': attribution,
        'attribution_reconciliation_error_krw': change - reconciled,
        'monthly': dict(by_month),
        'candidate_profit_factor': wins / loss if loss else None,
        'candidate_fee_share_of_gross_profit': sum(t['commission'] for t in nt) / gross if gross else None,
        'candidate_top10_winner_net_pnl': sum(sorted((t['net_pnl'] for t in nt), reverse=True)[:10]),
        'incremental_return_excluding_five_best_incremental_days_pp': float((delta.sum() - np.sort(delta)[-5:].sum()) * 100),
        'paired_five_session_block_bootstrap_incremental_return_pp_95pct': list(map(float, np.quantile(bootstrap, [.025, .975]))),
        'bootstrap_limitation': 'Descriptive, conditional on already selected policies and reused development observations; not a fresh OOS confidence interval.',
        'attribution_limitation': 'Realised path reconciliation, not a causal insert/remove experiment; shared account sizing and capital differ.',
        'inputs': {str(p): ev.digest(p) for d in (control, candidate) for p in (d / 'result.json', d / 'daily.json', d / 'trades.json')},
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--control', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    ev.write_json(a.output, compare(a.control, a.candidate))


if __name__ == '__main__':
    main()
