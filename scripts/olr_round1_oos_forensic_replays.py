"""Small, source-bound loss-control follow-up; never promotes a strategy.

Uses the existing audited study's native evaluator and frozen market sources.
Writes a separate registration and result tree. No threshold selection is made
between windows: all three predeclared guard values run on both windows.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import olr_round1_mechanism_study as mechanism

native = mechanism.native
PARENT = Path('E:/CodexResearch/k_stock_trader/olr_round1_comprehensive_20261009/full_oos_20260401_20260710/continuous_history_v10')
ROOT = REPO / 'data/backtests/analysis/olr_round1_oos_review_20261010/replays'
IDS = ('c609a66b6ce80f748', 'cbd14aff8c9557723', 'c272fe0390df855bd')


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    plan = native.read(PARENT/'mechanism_plan.json')
    catalog = {c['id']: c for c in plan['candidates']}
    protocol = native.read(PARENT/'protocol.json')
    protocol['generation_directories'] = {
        w: str(native.generation_folder(PARENT, w).parent) for w in ('train', 'oos')}
    protocol['review_scope'] = 'Three predeclared completed-bar loss guards, paired IS and full OOS; no promotion.'
    registration = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'parent': str(PARENT), 'parent_plan_sha256': native.sha(PARENT/'mechanism_plan.json'),
        'windows': protocol['windows'], 'reference_capital': 100_000_000,
        'candidate_ids': list(IDS),
        'hypothesis': 'Large post-entry losses, with disabled hard stops, impair OOS payoff. Test loss guards at 0.4/0.8/1.2 original ATR risk units, active after one completed bar while MFE is below 4 risk units; retain existing MFE-fade exit.',
        'causal_execution': 'Completed-bar decision followed by native next-open market execution; no assumed threshold-price fills.',
        'selection': protocol['selection'], 'adaptive_development': True,
        'independent_validation': False, 'live_or_canonical_changes': False,
        'harness_sha256': native.sha(native.__file__),
        'mechanism_sha256': native.sha(mechanism.__file__),
    }
    if not (ROOT/'registration.json').exists():
        native.write(ROOT/'registration.json', registration)
        native.write(ROOT/'protocol.json', protocol)
        native.write(ROOT/'candidate_plan.json', {
            'baseline_id': plan['baseline_id'], 'baseline': plan['baseline'],
            'candidates': [catalog[plan['baseline_id']], *[catalog[cid] for cid in IDS]]})
        native.write(ROOT/'mechanism_plan.json', plan)
    else:
        prior = native.read(ROOT/'registration.json')
        assert prior['parent_plan_sha256'] == registration['parent_plan_sha256']
        assert prior['harness_sha256'] == registration['harness_sha256']
        assert prior['candidate_ids'] == list(IDS)
    results = []
    for cid in IDS:
        candidate = catalog[cid]
        for window in ('train', 'oos'):
            print(json.dumps({'starting': candidate['label'], 'window': window}), flush=True)
            row = mechanism.evaluate(ROOT, candidate, window)
            results.append(row)
            native.write(ROOT/'results.json', results)
            print(json.dumps({k: row.get(k) for k in ('candidate_id', 'window', 'qualified', 'return_pct', 'drawdown_pct', 'trades', 'status')}), flush=True)
            if not row.get('qualified'):
                raise RuntimeError('Unqualified replay; retained for diagnosis: '+str(row))
    native.write(ROOT/'completion.json', {
        'completed_utc': datetime.now(timezone.utc).isoformat(),
        'paired_trials': 3, 'full_native_audits': 6,
        'independent_validation': False, 'promoted': False})


if __name__ == '__main__':
    main()
