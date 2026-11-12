"""Fixed C8-D1 diagnostic of existing funded suffixes; no policy or fit."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import time

ROOT = Path('E:/CodexResearch/k_stock_trader/holding_account_20261013')
CODE = Path(__file__).resolve().parents[1]
SCENARIOS = ('baseline', 'participation', 'cost', 'combined')
FIELDS = ('value_bps', 'dd_change_bps', 'own_initial_bps', 'other_initial_bps',
          'preexisting_pending_bps', 'new_descendant_bps', 'fee_change_bps',
          'material_entry_change', 'mean_cash_change_krw')


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def write(p, obj):
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, allow_nan=False)


def bound(p):
    p = Path(p)
    with p.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    return {'path': str(p), 'sha256': digest}


def verify(b):
    if bound(b['path'])['sha256'] != b['sha256']:
        raise ValueError('Changed bound input: ' + b['path'])


def key(o):
    return o['day'], o['available_at'], o['entry_order_id']


def conditions(o, atr):
    risk = o['management_risk_per_share']
    if risk <= 0 or atr <= 0 or o['reason'] != 'eligible':
        raise ValueError('Unsupported native observation')
    current = (o['current_price'] - o['entry_price']) / risk
    f = o['features']
    if not math.isclose(current * risk / atr, f['return_atr'], rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError('ATR/price semantic mismatch')
    mfe = f['mfe_atr'] * atr / risk
    return {'F': o['held_bars'] > 6 and mfe <= .75 and current <= -.25,
            'Q': o['held_bars'] > 10 and current < -.5}


def select(rows, entry_atr):
    selected = {name: [] for name in ('F', 'Q', 'F_or_Q')}
    seen = {name: set() for name in selected}
    for row in sorted(rows, key=lambda r: key(r['observation'])):
        o = row['observation']; predicates = conditions(o, entry_atr[o['entry_order_id']])
        predicates['F_or_Q'] = predicates['F'] or predicates['Q']
        lifetime = o['day'], o['entry_order_id']
        for name, eligible in predicates.items():
            if eligible and lifetime not in seen[name]:
                seen[name].add(lifetime); selected[name].append(row)
    return selected


def summary(rows):
    days = defaultdict(list)
    for r in rows:
        days[r['day']].append(r)
    means = {k: sum(sum(r[k] for r in v) / len(v) for v in days.values()) / len(days)
             if days else None for k in FIELDS}
    return {'actions': len(rows), 'days': len(days), 'mean_by_session': means,
            'positive_actions': sum(r['value_bps'] > 0 for r in rows),
            'positive_continuation_cut': sum(r['hold_own_remaining_pnl_krw'] > 0 for r in rows),
            'top_decile_winners_cut': sum(r['parent_top_decile_winner'] for r in rows),
            'native_buying_power_at_least_equity': sum(r['features']['buying_power_fraction'] >= 1 for r in rows),
            'changed_material_entry_count': sum(r['material_entry_change'] != 0 for r in rows),
            'at_least_100k_absolute_effect': sum(abs(r['whole_account_delta_krw']) >= 100_000 for r in rows),
            'intended_quantity_mismatch': sum(r['intended_qty'] != r['actual_action_qty'] for r in rows),
            'worst_value_bps': min((r['value_bps'] for r in rows), default=None),
            'best_value_bps': max((r['value_bps'] for r in rows), default=None),
            'total_positive_branch_krw_nonadditive': sum(max(0, r['whole_account_delta_krw']) for r in rows),
            'total_negative_branch_krw_nonadditive': sum(min(0, r['whole_account_delta_krw']) for r in rows)}


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    receipt = read(ROOT / 'completion_assessment_v1/receipt.json')
    known = {str(Path(b['path']).resolve()): b for b in receipt['inputs'] + receipt['outputs']}
    inputs = [bound(ROOT / 'completion_assessment_v1/receipt.json')]
    cases = []
    for period in ('is', 'oos'):
        for scenario in SCENARIOS:
            case = ROOT / 'study_v2/cases/control' / period / scenario
            attrs = ROOT / 'completion_assessment_v1' / (period + '_' + scenario + '_actions.jsonl.gz')
            for p in (case / 'labels.jsonl', case / 'fills.json', attrs):
                b = known[str(p.resolve())]; verify(b); inputs.append(b)
            profile = read(case / 'input_profile.json')
            for name in ('manifest', 'plan', 'policy'):
                b = profile[name]; verify(b); inputs.append(b)
            if not read(case / 'identity_bridge.json')['passed']:
                raise ValueError('Invalid control bridge')
            inputs += [bound(case / name) for name in ('input_profile.json', 'identity_bridge.json', 'hold_identity.json')]
            cases.append((period, scenario, case, attrs))
    registration = {'at_utc': datetime.now(timezone.utc).isoformat(), 'experiment': 'kalcb-exp-0032',
                    'criteria': 'C8-D1', 'inputs': inputs,
                    'program': bound(__file__),
                    'design': bound(CODE / 'docs/optimisation/kalcb/experiments/kalcb-exp-0032.md'),
                    'native_replays': 0, 'fits': 0, 'conditions': ['F', 'Q', 'F_or_Q'],
                    'comparison_actions': ['exit', 'half'], 'target_time': 'first qualifying saved state per lifetime'}
    write(out / 'registration.json', registration)
    reports = {}; checked = {}
    for period, scenario, case, attrs in cases:
        entry_atr = {}
        for f in read(case / 'fills.json'):
            if f['side'] == 'BUY':
                atr = float(f['metadata']['daily_atr'])
                previous = entry_atr.setdefault(f['order_id'], atr)
                if previous != atr:
                    raise ValueError('ATR changed within entry order')
        rows = [json.loads(line) for line in (case / 'labels.jsonl').open(encoding='utf-8')]
        selected = select(rows, entry_atr)
        identities = {r['day']: r['exact_hold_eod_state'] for r in read(case / 'hold_identity.json')}
        with gzip.open(attrs, 'rt', encoding='utf-8') as f:
            details = {(r['day'], r['clock'], r['entry_order_id'], r['arm']): r for r in map(json.loads, f)}
        report = {'label_rows': len(rows), 'cohorts': {}, 'selected_observations': {}}
        for name, members in selected.items():
            report['cohorts'][name] = {}
            report['selected_observations'][name] = [r['observation'] for r in members]
            for arm in ('exit', 'half'):
                actual = []
                for r in members:
                    o = r['observation']
                    if identities[o['day']] != r['hold_eod_state']:
                        raise ValueError('Label hold identity mismatch')
                    a = details[(*key(o), arm)]
                    for field in ('value_bps', 'dd_change_bps', 'material_entry_change', 'fee_change_krw'):
                        if a[field] != r['contrasts'][arm]['target'][field]:
                            raise ValueError('Attribution target mismatch')
                    if abs(sum(a['components_krw'].values()) - a['whole_account_delta_krw']) > .011:
                        raise ValueError('Funded attribution mismatch')
                    for b in (a['hold_binding'], a['action_binding']):
                        if b['path'] not in checked:
                            verify(b); checked[b['path']] = b
                    actual.append(a)
                months = sorted({a['day'][:7] for a in actual})
                tails = sorted(actual, key=lambda a: a['value_bps'])
                result = summary(actual)
                result['months'] = {m: summary([a for a in actual if a['day'].startswith(m)]) for m in months}
                result['worst_three'] = [{k: a[k] for k in ('day', 'clock', 'symbol', 'value_bps',
                    'whole_account_delta_krw', 'hold_own_remaining_pnl_krw', 'parent_lifetime_pnl_krw',
                    'components_krw', 'material_entry_change')} for a in tails[:3]]
                report['cohorts'][name][arm] = result
                write(out / (period + '_' + scenario + '_' + name + '_' + arm + '_actions.json'), actual)
        reports[period + '/' + scenario] = report
        write(out / (period + '_' + scenario + '.json'), report)
        print(json.dumps({'case': period + '/' + scenario, 'elapsed_seconds': time.monotonic() - started,
                          'exit_means': {k: v['exit']['mean_by_session']['value_bps'] for k, v in report['cohorts'].items()}}), flush=True)
    for b in inputs + [registration['program'], registration['design']]:
        verify(b)
    write(out / 'selected_branch_bindings.json', list(checked.values()))
    write(out / 'assessment.json', {'reports': reports, 'verified_branch_bindings': len(checked),
          'scope': 'Sampled native decision-state contrasts; mutually exclusive alternatives, not policy returns. Reused development history; no fresh predictive confirmation.',
          'economic_objective_achieved': False})
    write(out / 'completion.json', {'completed': True, 'at_utc': datetime.now(timezone.utc).isoformat(),
          'elapsed_seconds': time.monotonic() - started, 'economic_qualification': False,
          'registration': bound(out / 'registration.json'), 'assessment': bound(out / 'assessment.json')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.output)
    except Exception as exc:
        if args.output.exists() and not (args.output / 'failure.json').exists():
            write(args.output / 'failure.json', {'at_utc': datetime.now(timezone.utc).isoformat(), 'error': repr(exc)})
        raise
