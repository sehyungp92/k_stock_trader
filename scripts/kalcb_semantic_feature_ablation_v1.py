"""Registered fixed-feature IS ablation; no live hook or market-data access.

Commands: --phase audit, then --phase fit, with the same fresh --output.
Preserves original targets, models and selection failures. Never calls native
policy qualification or treats overlapping contrasts as account returns.
"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import sys
import time

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
from backtests.strategies.kalcb.holding_account_observation_v2 import from_saved_prefix
from backtests.strategies.kalcb import holding_account_model_v1 as model_api
from backtests.strategies.kalcb.retention_research_io_v1 import read, write, bound, verify
from backtests.strategies.kalcb.pending_entry_account_pilot_v1 import canonical
import numpy as np

ROOT = Path('E:/CodexResearch/k_stock_trader/holding_account_20261013/study_v2')
OLD_AUDIT = ROOT.parent / 'completion_assessment_v1/feature_contract_audit_v1.json'


def key(row):
    return (row['day'], row['available_at'], row['entry_order_id'])


def log(out, phase, **extra):
    data = dict(at_utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid(), phase=phase, **extra)
    write(out / 'progress.json', data)
    print(json.dumps(data), flush=True)


def checked_prefix(binding):
    verify([binding])
    with gzip.open(binding['path'], 'rt', encoding='utf-8') as f:
        prefix = json.load(f)
    if hashlib.sha256(canonical(prefix['state']).encode()).hexdigest() != prefix['sha256']:
        raise ValueError('Saved native state receipt mismatch')
    return prefix


def close(a, b, name):
    if not math.isclose(a, b, abs_tol=1e-10, rel_tol=1e-10):
        raise ValueError(f'{name}: {a} != {b}')


def audit(out):
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    inputs = [bound(OLD_AUDIT), bound(ROOT / 'models/manifest.json')]
    previous = read(OLD_AUDIT)
    # Every previously authenticated positive/negative extreme is retained,
    # including OOS semantics only. No OOS action outcomes are loaded here.
    for case in previous['cases'].values():
        for x in case['authenticated_counterexamples']:
            p = checked_prefix(x['prefix'])
            meta = dict(p['state']['adapter']['state']['meta'])
            close(meta['portfolio_session_start_equity'], x['prefix_session_start_equity'], 'counterexample baseline')
            close(x['observed_equity'] / meta['portfolio_session_start_equity'] - 1,
                  x['true_prefix_session_return'], 'counterexample fraction')
            inputs.append(x['prefix'])
    sources = [CODE / 'scripts/kalcb_semantic_feature_ablation_v1.py',
               CODE / 'backtests/strategies/kalcb/holding_account_observation_v2.py',
               CODE / 'tests/test_kalcb_account_observation_v2.py',
               CODE / 'docs/optimisation/kalcb/experiments/kalcb-exp-0028.md']
    for filename in ('holding_action_v1.py', 'holding_account_model_v1.py'):
        working = CODE / 'backtests/strategies/kalcb' / filename
        frozen = ROOT / 'overlays/backtests/strategies/kalcb' / filename
        if bound(working)['sha256'] != bound(frozen)['sha256']:
            raise ValueError('Fixed source differs from frozen predecessor')
        sources.extend((working, frozen))
    all_results = {}
    for scenario in model_api.SCENARIOS:
        case = ROOT / 'cases/control/is' / scenario
        lb = bound(case / 'labels.jsonl')
        if lb['sha256'] != previous['cases']['is/' + scenario]['labels']['sha256']:
            raise ValueError('Changed previously audited labels')
        inputs.extend(bound(case / f) for f in ('labels.jsonl', 'equity.json', 'daily.json',
                                                'hold_identity.json', 'identity_bridge.json', 'input_profile.json'))
        profile = read(case / 'input_profile.json')
        for name in ('manifest', 'plan', 'policy'):
            binding = profile[name]
            if 'hash' in binding:
                binding = dict(path=binding['path'], sha256=binding['hash'])
            verify([binding]); inputs.append(binding)
        if not read(case / 'identity_bridge.json')['passed']:
            raise ValueError('Invalid native control bridge')
        identities = {r['day']:r['exact_hold_eod_state'] for r in read(case / 'hold_identity.json')}
        rows = [json.loads(s) for s in (case / 'labels.jsonl').open(encoding='utf-8')]
        equity = read(case / 'equity.json'); peaks = {}; running = 100_000_000.
        for stamp, value in equity:
            running = max(running, value); peaks[str(stamp).replace(' ', 'T')] = (value, running)
        grouped = defaultdict(list)
        for r in rows:
            grouped[r['prefix']['path']].append(r)
        corrected = {}; counts = Counter(); baselines = {}; bindings = []; max_delta = 0.
        for n, members in enumerate(grouped.values()):
            first = members[0]; prefix = checked_prefix(first['prefix']); bindings.append(first['prefix'])
            if prefix['sha256'] != first['boundary']['state_sha256']:
                raise ValueError('Boundary does not bind prefix')
            broker = prefix['state']['broker']; positions = {k[1]:p for k,p in broker['positions']}
            orders = broker['orders']
            for r in members:
                o = r['observation']; f = o['features']; account = from_saved_prefix(prefix, o)
                if prefix['sha256'] != r['boundary']['state_sha256'] or identities[o['day']] != r['hold_eod_state']:
                    raise ValueError('Unbound prefix or hold target')
                if o['features']['session_return'] != 0:
                    raise ValueError('Control is not the registered constant-feature generation')
                mark, peak = peaks[o['signal_bar']]
                close(mark, o['equity'], 'native equity mark'); close(peak, o['running_peak'], 'native running peak')
                close(positions[o['symbol']]['qty'], o['qty'], 'native holding quantity')
                close(o['qty'] * o['current_price'] / o['equity'], f['position_weight'], 'position weight')
                close(sum(x['qty'] for x in orders if x['side'] == 'BUY' and x['symbol'] == o['symbol']) / o['qty'],
                      f['pending_to_held'], 'pending remaining shares')
                for name, value in account.features().items():
                    if name != 'session_return':
                        close(f[name], value, name)
                old = baselines.setdefault(o['day'], account.session_start_equity_krw)
                close(old, account.session_start_equity_krw, 'stable same-session baseline')
                corrected['|'.join(key(o))] = account.session_return
                counts['positive' if account.session_return > 0 else 'negative' if account.session_return < 0 else 'flat'] += 1
                counts['pending_buy_rows'] += int(any(x['side'] == 'BUY' for x in orders))
                counts['partial_action_fill_rows'] += int(any(c['first_fill'] != c['last_fill'] for c in r['contrasts'].values()))
                max_delta = max(max_delta, abs(account.session_return))
            if n % 50 == 0:
                log(out, 'semantic_audit', scenario=scenario, prefixes_done=n+1, prefixes_total=len(grouped), elapsed=time.monotonic()-started)
        dest = out / (scenario + '_corrected_feature.json')
        write(dest, corrected)
        write(out / (scenario + '_prefix_bindings.json'), bindings)
        inputs.extend((bound(dest), bound(out / (scenario + '_prefix_bindings.json'))))
        all_results[scenario] = dict(rows=len(rows), prefixes=len(grouped), lifetimes=len({(r['observation']['day'],r['observation']['entry_order_id']) for r in rows}),
                                    sessions=len(baselines), counts=dict(counts), max_abs_session_return=max_delta,
                                    targets_unchanged=True)
        inputs.append(bound(ROOT / 'models' / (scenario + '_out_of_time.json')))
    manifest = read(ROOT / 'models/manifest.json')
    inputs.extend(s['model'] for specs in manifest['snapshots'].values() for s in specs)
    verify(inputs)
    report = dict(passed=True, cases=all_results, prior_counterexamples=16, elapsed_seconds=time.monotonic()-started,
                  target_reuse='Only observation.features.session_return changes; immutable arm, quantity, timing, management, source and target records retained. Native branch regression is separately tested.',
                  semantics='Fractional session return from session-start native equity. Buying power is native pre-reservation capacity; pending-to-held is separate.')
    write(out / 'semantic_audit.json', report)
    import sklearn
    registration = dict(at_utc=datetime.now(timezone.utc).isoformat(), experiment='kalcb-exp-0028', revision='R1',
                        objective='O3', criteria='C4', inputs=inputs, code=[bound(p) for p in sources],
                        semantic_audit=bound(out / 'semantic_audit.json'),
                        python=sys.version, numpy=np.__version__, sklearn=sklearn.__version__,
                        treatment='session_return only', specifications=2, treatment_fits=20,
                        fit_folds='five existing chronological evaluation folds per IS scenario; final OOS fit deferred',
                        command=f'{sys.executable} -B {Path(__file__).resolve()} --phase fit --output {out}',
                        untouched_validation=False, native_candidate_accounts=0)
    with (out / 'registration.json').open('x', encoding='utf-8') as f:
        json.dump(registration, f, indent=2)
    log(out, 'registered', elapsed_seconds=time.monotonic()-started, cases=all_results)


def mean_by_day(selected, arm=None):
    grouped = defaultdict(list)
    for r in selected:
        target = r[arm] if arm else r['target']
        grouped[r['day']].append((target['value_bps'], target['dd_change_bps']))
    if not grouped:
        return dict(wealth_bps=None, dd_bps=None, days=0, actions=0)
    arr = np.array([np.mean(v, axis=0) for _,v in sorted(grouped.items())])
    return dict(wealth_bps=float(arr[:,0].mean()), dd_bps=float(arr[:,1].mean()), days=len(grouped), actions=len(selected))


def describe(selected, rowmap, specs):
    result = mean_by_day(selected)
    result['historical_C3'] = model_api.readiness(selected)
    result['same_selected_population_comparators'] = {a:mean_by_day(selected, a) for a in ('half','exit')}
    result['hold_comparator_wealth_bps'] = 0.
    result['intervals'] = []
    for spec in specs:
        part = [s for s in selected if spec['effective_from'] <= s['day'] <= spec['through']]
        result['intervals'].append(dict(effective_from=spec['effective_from'], through=spec['through'], **mean_by_day(part)))
    result['positive_intervals'] = sum((p['wealth_bps'] or 0) > 0 for p in result['intervals'])
    if selected:
        values = [s['target']['value_bps'] for s in selected]
        result['positive_negative_zero'] = [sum(v>0 for v in values),sum(v<0 for v in values),sum(v==0 for v in values)]
        result['median_action_bps'] = float(np.median(values))
        result['material_entry_change_counts'] = dict(Counter(s['target']['material_entry_change'] for s in selected))
        tail = sorted(selected, key=lambda s:s['target']['value_bps'])
        result['worst_actions'] = tail[:5]; result['best_actions'] = tail[-5:]
        result['local_net_krw_extrema'] = [min(s['target']['value_bps']*rowmap[key(s)]['observation']['equity']/10000 for s in selected),
                                          max(s['target']['value_bps']*rowmap[key(s)]['observation']['equity']/10000 for s in selected)]
    return result


def fit_all(out):
    started = time.monotonic(); registration = read(out / 'registration.json')
    verify(registration['inputs'] + registration['code'] + [registration['semantic_audit']])
    if not read(out / 'semantic_audit.json')['passed']:
        raise ValueError('Failed semantic audit')
    results_dir = out / 'fit'; results_dir.mkdir(exist_ok=False)
    manifest = read(ROOT / 'models/manifest.json'); reports = {}
    for scenario in model_api.SCENARIOS:
        rows = [json.loads(s) for s in (ROOT / 'cases/control/is' / scenario / 'labels.jsonl').open(encoding='utf-8')]
        rowmap = {key(r['observation']):r for r in rows}
        corrected = read(out / (scenario + '_corrected_feature.json'))
        treatment = deepcopy(rows)
        for r in treatment:
            r['observation']['features']['session_return'] = corrected['|'.join(key(r['observation']))]
        previous = read(ROOT / 'models' / (scenario + '_out_of_time.json'))
        expected = {key(r['observation']):r['decision'] for r in previous['predictions']}
        validation = []; controls = []; decisions = []; timings = []
        specs = manifest['snapshots'][scenario][:-1]
        for i,spec in enumerate(specs):
            fold_start = time.monotonic()
            with Path(spec['model']['path']).open('rb') as f:
                old_model = pickle.load(f)
            held = [r for r in rows if spec['effective_from'] <= r['observation']['day'] <= spec['through']]
            control = [model_api.choose(old_model,r['observation']) for r in held]
            for r,d in zip(held,control):
                if d != expected[key(r['observation'])]:
                    raise ValueError('A0 exact prediction reproduction failed')
            train_days = old_model['training_days']
            train = [r for r in treatment if r['observation']['day'] in train_days]
            if len(train) != spec['training_rows']:
                raise ValueError('Changed chronological training population')
            model = model_api.fit(train, train_days)
            with (results_dir / f'{scenario}_{i}.pkl').open('xb') as f:
                pickle.dump(model, f, protocol=5)
            corrected_held = [r for r in treatment if spec['effective_from'] <= r['observation']['day'] <= spec['through']]
            new_decisions = [model_api.choose(model,r['observation']) for r in corrected_held]
            validation.extend(held); controls.extend(control); decisions.extend(new_decisions)
            timings.append(time.monotonic()-fold_start)
            log(out, 'fit_fold_complete', scenario=scenario, fold=i, rows=len(train), held=len(held),
                fold_seconds=timings[-1], elapsed_seconds=time.monotonic()-started)
        a0 = model_api.selected_contrasts(validation,controls)
        if a0 != previous['selected']:
            raise ValueError('A0 exact selected-action reproduction failed')
        a1 = model_api.selected_contrasts(validation,decisions)
        write(results_dir / (scenario + '_predictions.json'),
              dict(predictions=[dict(observation=t['observation'],
                                     treatment_session_return=corrected['|'.join(key(t['observation']))],
                                     control=a,treatment=b) for t,a,b in zip(validation,controls,decisions)],
                   selected_A0=a0, selected_A1=a1))
        s0 = {key(s):s for s in a0}; s1 = {key(s):s for s in a1}
        union = []
        for k in sorted(s0.keys() | s1.keys()):
            left = s0[k]['target'] if k in s0 else dict(value_bps=0.,dd_change_bps=0.)
            right = s1[k]['target'] if k in s1 else dict(value_bps=0.,dd_change_bps=0.)
            union.append(dict(day=k[0],target={f:right[f]-left[f] for f in ('value_bps','dd_change_bps')}))
        reports[scenario] = dict(A0=describe(a0,rowmap,specs), A1=describe(a1,rowmap,specs),
            union_paired_change=mean_by_day(union),
            action_overlap=dict(shared_states=len(s0.keys()&s1.keys()),removed=len(s0.keys()-s1.keys()),added=len(s1.keys()-s0.keys()),
                                changed_arm=sum(s0[k]['arm']!=s1[k]['arm'] for k in s0.keys()&s1.keys())),
            reasons_A0=dict(Counter(d['reason'] for d in controls)),reasons_A1=dict(Counter(d['reason'] for d in decisions)),
            unchanged_raw_decisions=sum(a['arm']==b['arm'] for a,b in zip(controls,decisions)),
            eligible_validation_rows=len(validation), fold_seconds=timings,
            unconditional_mature_population=previous['unconditional_local_means'])
        write(results_dir / (scenario + '_report.json'), reports[scenario])
    positive = sum((v['A1']['wealth_bps'] or 0)>0 for v in reports.values())
    primary = [reports[s] for s in ('baseline','combined')]
    admitted = positive >= 3 and all((v['A1']['wealth_bps'] or 0)>0 and
               (v['union_paired_change']['wealth_bps'] or 0) >= 2 and
               v['A1']['actions'] >= 20 and v['A1']['days'] >= 10 and v['A1']['positive_intervals'] >= 2 for v in primary)
    verify(registration['inputs'] + registration['code'])
    output = dict(at_utc=datetime.now(timezone.utc).isoformat(), completed=True,
                  registration=bound(out / 'registration.json'), reports=reports, C4_account_admission=admitted,
                  elapsed_seconds=time.monotonic()-started, native_candidate_accounts=0,
                  economic_objective_achieved=False, interpretation='Local overlapping suffix decisions on reused IS; not funded policy returns.')
    write(out / 'result.json', output)
    log(out, 'complete', C4_account_admission=admitted, elapsed_seconds=output['elapsed_seconds'],
        means={s:{a:reports[s][a]['wealth_bps'] for a in ('A0','A1')} for s in reports})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('audit','fit'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        (audit if args.phase == 'audit' else fit_all)(args.output)
    except Exception as exc:
        if args.output.exists():
            write(args.output / ('failure_' + args.phase + '.json'),
                  dict(at_utc=datetime.now(timezone.utc).isoformat(), error=repr(exc), economic_objective_achieved=False))
        raise
