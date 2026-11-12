"""Finite read-only execution materiality, with no policy, fit, or replay."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASE = Path('E:/CodexResearch/k_stock_trader/kalcb_late_entry_20261011')
SOURCE = BASE / 'v49_staged_support'
HELPER = REPO / 'backtests/strategies/kalcb/entry_reference_drift_v1.py'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def bound(path):
    path = Path(path).resolve()
    with path.open('rb') as f:
        return {'path': str(path), 'sha256': hashlib.file_digest(f, 'sha256').hexdigest()}


def verify(bindings):
    for b in bindings:
        if bound(b['path'])['sha256'] != b['sha256']:
            raise ValueError('Changed binding ' + b['path'])


def write(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(data, f, indent=2, allow_nan=False)


def memory_headroom():
    import ctypes
    from ctypes import wintypes
    class PerformanceInfo(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD)] + [(name, ctypes.c_size_t) for name in
            ['CommitTotal', 'CommitLimit', 'CommitPeak', 'PhysicalTotal', 'PhysicalAvailable',
             'SystemCache', 'KernelTotal', 'KernelPaged', 'KernelNonpaged', 'PageSize']] + [
             (name, wintypes.DWORD) for name in ['HandleCount', 'ProcessCount', 'ThreadCount']]
    info = PerformanceInfo(); info.cb = ctypes.sizeof(info)
    if not ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(info), info.cb):
        raise OSError('Cannot measure Windows commit headroom')
    physical = info.PhysicalAvailable * info.PageSize
    commit = (info.CommitLimit - info.CommitTotal) * info.PageSize
    if physical < 384 * 2**20 or commit < 1280 * 2**20:
        raise RuntimeError('Insufficient declared physical/commit headroom')
    return {'physical_available_bytes': physical, 'commit_available_bytes': commit}


def register(root, tests):
    t = read(tests)
    if not t['passed'] or t['tests'] != 10:
        raise ValueError('Missing focused verification')
    verify(t['programs'] + [t['log']])
    original = read(SOURCE / 'protocol.json'); done = read(SOURCE / 'completion.json')
    if not read(SOURCE / 'review_v1/completion.json')['passed']:
        raise ValueError('Unreviewed source')
    verify(original['programs'] + original['bindings'] + done['outputs'] + [done['assessment']])
    cases = original['cases']
    paths = [SOURCE / name for name in ['protocol.json', 'completion.json', 'review_v1/completion.json',
             'review_v1/review.json', 'post_completion_health.json']]
    paths += [Path(tests), BASE / 'v53_normalized_decision_regret/economic_decision.json',
              REPO / 'strategy_kalcb/core/logic.py', REPO / 'docs/strategy-implementation-lessons.md']
    origins = {}
    for c in cases:
        folder = Path(c['folder'])
        paths += [SOURCE / 'accounts' / Path(c['key']).with_suffix('.json')]
        paths += [folder / name for name in ['fills.json', 'completion.json', 'execution_audit.json',
                  'independent_account_verification.json', 'independent_exposure_mtm.json', 'identity_bridge.json']]
        if not read(folder / 'execution_audit.json')['execution_validity_pass']:
            raise ValueError('Original execution failed')
        for name in ['independent_account_verification.json', 'independent_exposure_mtm.json', 'identity_bridge.json']:
            if not read(folder / name)['passed']:
                raise ValueError('Original review failed ' + name)
        completion = read(folder / 'completion.json')
        for key in ['origins_before', 'origins_after']:
            for b in completion[key]:
                if b['path'] in origins and origins[b['path']] != b:
                    raise ValueError('Conflicting frozen origin')
                origins[b['path']] = b
    verify(list(origins.values()))
    root.mkdir(exist_ok=False)
    programs = [REPO / x for x in ['scripts/kalcb_entry_reference_drift_v1.py',
                'scripts/kalcb_entry_reference_drift_review_v1.py',
                'backtests/strategies/kalcb/entry_reference_drift_v1.py',
                'tests/test_kalcb_entry_reference_drift_v1.py']]
    write(root / 'protocol.json', {
        'registered_at_utc': datetime.now(timezone.utc).isoformat(), 'cases': cases, 'source': str(SOURCE),
        'kind': 'read_only_original_opening_initial_execution_reference_drift',
        'question': 'Do initial MARKET fills drift materially from the completed price used for selection/sizing, separate from modeled slippage and later partial fills?',
        'population': 'All2620 original opening intents/12 unchanged accounts, including unfilled orders. Both priors baseline/combined and prior100 participation/cost. No outcome-selected subset.',
        'decomposition': 'Every filled share: qty*(first-or-later base minus decision reference) plus qty*(execution price minus base). First means all physical fills at the earliest timestamp of the original order. Entry fees separate. Validate original open price and configured3/13bps slippage; historical independent raw-source/account/MTM audits authenticated/reused.',
        'groups': 'All, unfilled, adverse/flat/favorable initial market gap at zero. Diagnostic future realized execution groups, NOT available at original decision and NOT policy filters. All months and issuers retained.',
        'limits': 'Reference price is NOT an attainable substitute fill. Drift is NOT saved PnL. Whole lifetime PnL attaches once and is NOT slice PnL. No limit-order queue/alternative fill, cancellation, quantity or funded portfolio result. Scenarios/priors are dependent. No new entry clock or training/model.',
        'next_gate': 'No automatic cap/limit grid. Only a material controllable execution discrepancy with defensible winner opportunity cost could justify separately frozen native order semantics and exact-account pilot. Small or mixed drift closes this hypothesis.',
        'estimate': '12existing accounts,1-5minutes plus independent review; zero native cases/fits.',
        'programs': [bound(p) for p in programs], 'bindings': [bound(p) for p in dict.fromkeys(paths)],
        'origins': list(origins.values()), 'study_complete': False})


def run(root):
    import psutil
    p = read(root / 'protocol.json'); verify(p['programs'] + p['bindings'] + p['origins'])
    me = psutil.Process()
    write(root / 'supervisor_start.json', {'pid': me.pid, 'create_time': me.create_time(), 'cmdline': me.cmdline()})
    spec = importlib.util.spec_from_file_location('bound_drift', HELPER)
    h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
    outputs = []; accounts = {}
    for c in p['cases']:
        memory_headroom()
        previous = read(Path(p['source']) / 'accounts' / Path(c['key']).with_suffix('.json'))
        fills = read(Path(c['folder']) / 'fills.json'); by_order = defaultdict(list)
        execution_ids = [f['metadata']['execution_id'] for f in fills]
        if len(execution_ids) != len(set(execution_ids)):
            raise ValueError('Duplicate physical execution ID')
        for f in fills:
            if f['side'] == 'BUY': by_order[f['order_id']].append(f)
        slip = 13 if c['key'].split('/')[-1] in ['combined', 'cost'] else 3
        rows = [h.measure(r, by_order[r['order_id']], slip) for r in previous['rows']]
        groups = {'all': rows}
        groups.update({k: [r for r in rows if r['cohort'] == k] for k in
                       ['unfilled', 'adverse_initial_gap', 'flat_initial_gap', 'favorable_initial_gap']})
        account = {'rows': rows, 'groups': {k: h.summary(rs) for k, rs in groups.items()},
                   'months': {m: {k: h.summary([r for r in rs if r['day'][:7] == m]) for k, rs in groups.items()}
                              for m in sorted({r['day'][:7] for r in rows})},
                   'issuers': {s: {k: h.summary([r for r in rs if r['symbol'] == s]) for k, rs in groups.items()}
                               for s in sorted({r['symbol'] for r in rows})}}
        out = root / 'accounts' / Path(c['key']).with_suffix('.json'); write(out, account)
        outputs.append(bound(out)); accounts[c['key']] = account['groups']
        (root / 'progress.json').write_text(json.dumps({'complete_accounts': len(outputs), 'total': len(p['cases'])}), encoding='utf-8')
        print(json.dumps({'account': c['key'], 'orders': len(rows)}), flush=True)
        del fills, by_order, rows, previous, account
    verify(p['programs'] + p['bindings'] + p['origins'])
    write(root / 'assessment.json', {'accounts': accounts, 'native_cases': 0, 'fits': 0, 'study_complete': False})
    write(root / 'completion.json', {'complete': True, 'at_utc': datetime.now(timezone.utc).isoformat(),
          'outputs': outputs + [bound(root / 'assessment.json')], 'study_complete': False})


if __name__ == '__main__':
    a = argparse.ArgumentParser(); a.add_argument('mode', choices=['register', 'run'])
    a.add_argument('--root', type=Path, required=True); a.add_argument('--tests', type=Path); args = a.parse_args()
    try:
        register(args.root, args.tests) if args.mode == 'register' else run(args.root)
    except Exception as exc:
        if args.root.exists() and not (args.root / 'failure.json').exists():
            write(args.root / 'failure.json', {'error': repr(exc), 'preserve_evidence': True})
        raise
