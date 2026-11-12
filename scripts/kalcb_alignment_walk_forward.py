"""Registered target-alignment contrasts with one continuous, causal fit clock.

This is research orchestration. Native economics remain in the frozen shared
core; this program never promotes a configuration or enables live trading.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.kalcb_full_oos_recovery import ROOT, POLICY, read, write, rows, jsonl, bound, digest, env

ALIGNMENT = Path('E:/CodexResearch/k_stock_trader/kalcb_selected_target_alignment_20261010/v2')
DEST = Path('E:/CodexResearch/k_stock_trader/kalcb_aligned_walk_forward_20261010/v1')
RECIPES = {
    'original_target_full_history': {'value_target': 'original', 'risk_target': 'original'},
    'selected_target_value_only': {'value_target': 'selected', 'risk_target': 'original'},
    'selected_target_all': {'value_target': 'selected', 'risk_target': 'selected'},
}
PERIODS = {'is': ('2025-08-07', '2026-03-31', 156), 'oos': ('2026-04-01', '2026-07-10', 69)}
ROUND = 'aligned_target_round_v1'


def verify_bindings(items):
    for item in items:
        if digest(item['path']) != item['sha256']:
            raise ValueError('Changed registered input: ' + item['path'])


def register(out, alignment):
    if out.exists():
        raise ValueError('Preserve existing registration ' + str(out))
    protocol = read(alignment/'protocol.json')
    verify_bindings(protocol['bound_inputs'])
    inputs = [bound(alignment/'protocol.json'), bound(ROOT/'ablation_v3/study.json'),
              bound(ROOT/'boundary_continuity.json'), bound(ROOT/'history_boundary_audit.json')]
    history = Path('E:/CodexResearch/k_stock_trader')
    inputs += [bound(history/'kalcb_continuation_20261006/policies_v2/forecasts.jsonl'),
               bound(history/'kalcb_intraday_advantage_20261007/checkpoint/policies/forecasts.jsonl')]
    for cell in protocol['cells']:
        for field in ('plan', 'parent_plan'):
            inputs.append(bound(cell[field]))
        inputs.append(bound(Path(cell['prepared'])/'prepared.json'))
    # Bind the fitting/admission implementation, independently of the frozen
    # economic generation used by the native replay subprocesses.
    programs = set(REPO.glob('scripts/kalcb_*.py'))
    programs.update(REPO.glob('backtests/strategies/kalcb/*.py'))
    programs.update(REPO.glob('strategy_kalcb/**/*.py'))
    programs.update(REPO.glob('backtests/auto/shared/*.py'))
    programs.add(REPO/'scripts/portfolio_frozen_inference.py')
    out.mkdir(parents=True)
    write(out/'protocol.json', {
        'registered_at_utc': datetime.now(timezone.utc).isoformat(),
        'alignment_root': str(alignment), 'recovery_root': str(ROOT),
        'recipes': RECIPES, 'periods': PERIODS, 'round_name': ROUND,
        'source_programs': [bound(p) for p in sorted(programs)], 'bound_inputs': inputs,
        'fit_clock': '60 historical sessions, one-session embargo, expanding fits every20 sessions from May12; no reporting-boundary reset',
        'calibration': 'Empty OOF pool at first fold; only predictions from completed earlier folds and mature matched-target outcomes. Existing60-session/100-observation shrinkage.',
        'is_interpretation': 'Every learner gets new156-session native IS and69-session native OOS evaluations. Historical selected control is separately labelled.',
        'constants': 'Same features, interaction architecture, quantities, admission budgets, exits, thresholds and native execution. No ATR overlay in main contrast.',
        'control_bridge': 'Original-target IS economic predictions must match recorded historical OOF predictions within1e-7 bps. Physical models are uniformly refitted on paired original outcomes; risk differences are reported, never hidden.',
        'failure_policy': 'Keep all failed and unresolved evidence. Invalid economic labels excluded from value fitting, retained as mature physical-risk targets.',
        'independence': 'All April1-July10 outcomes are reused development, never independent validation.',
        'qualification': 'No automatic canonical replacement. Four stress scenarios and complete source/model/label/account audits still required for finalists.',
    })
    print(json.dumps({'registered': str(out/'protocol.json'), 'recipes': list(RECIPES)}), flush=True)


def protocol_for(out):
    p = read(out/'protocol.json')
    verify_bindings(p['bound_inputs'] + p['source_programs'])
    if p['recipes'] != RECIPES or p['periods'] != {k: list(v) for k, v in PERIODS.items()}:
        raise ValueError('Changed experimental definition')
    return p


def paired_risk(grid):
    """An invalid cash outcome can still be an observed physical failure."""
    from scripts.kalcb_management_refit import failure_outcomes
    return failure_outcomes(grid)


def aligned_targets(alignment, data, expected_source):
    from scripts.kalcb_continuation_study import compact_labels
    protocol = read(alignment/'protocol.json'); done = read(alignment/'completion.json')
    if not done.get('complete') or done['protocol'] != bound(alignment/'protocol.json'):
        raise ValueError('Incomplete or changed selected-target generation')
    verify_bindings(protocol['bound_inputs'] + done['grids'] + [done['preview']])
    if not read(alignment/'preview_bridge.json').get('passed'):
        raise ValueError('Entry population did not bridge')
    economic = {k: {s: {} for s in ('baseline', 'combined')} for k in data}
    physical = {k: {} for k in data}; evidence = {}; summary = []
    for cell in protocol['cells']:
        period, kind = cell['period'], cell['kind']
        jobs = [j for j in protocol['jobs'] if j['period'] == period and j['kind'] == kind]
        folders = [Path(j['output']) for j in jobs]
        grid, proof = compact_labels(folders, digest(Path(cell['prepared'])/'prepared.json'), grid=True)
        expected = {r['opportunity_id'] for r in data[kind] if r['trade_date'] in cell['sessions']}
        if any(set(grid[s]) != expected for s in ('baseline', 'combined')):
            raise ValueError('Incomplete selected-target population')
        risk_grid = {s: {} for s in ('baseline', 'combined')}
        for job in jobs:
            folder = Path(job['output']); request = read(folder/'request.json'); manifest = read(folder/'grid_manifest.json')
            completion = read(folder/'completion.json'); args = request['arguments']
            if {request['source_generation_hash'], completion['source_generation_hash']} != {expected_source}:
                raise ValueError('Wrong native target source generation')
            if Path(args['plan']).resolve() != Path(job['plan']).resolve() or Path(args['prepared']).resolve() != Path(job['prepared']).resolve():
                raise ValueError('Wrong target plan or preparation')
            if args['scenario'] != job['scenario'] or args['scales'] != [.25, .5, 1.] or args['start_index'] != job['start_index']:
                raise ValueError('Wrong target replay arguments')
            if manifest['sessions'] != cell['sessions']:
                raise ValueError('Wrong target session coverage')
            statuses = Counter(); scenario = job['scenario']
            for name in manifest['files']:
                if not name.endswith('.jsonl'):
                    continue
                for r in rows(folder/name):
                    key = r['opportunity_id']
                    if key in risk_grid[scenario]:
                        raise ValueError('Duplicate selected physical outcome')
                    risk_grid[scenario][key] = {k: r[k] for k in ('status', 'risk_available_at', 'physical_fill_failures', 'deadline_violation', 'unresolved')}
                    if (r['physical_fill_failures'] or r['deadline_violation'] or r['unresolved']) and r['label_valid']:
                        raise ValueError('Unsafe episode incorrectly has a valid economic label')
                    statuses[r['status']] += 1
            summary.append({'period': period, 'kind': kind, 'scenario': scenario, 'statuses': dict(statuses), 'actions': len(risk_grid[scenario])})
            evidence.update({str(folder/f): digest(folder/f) for f in ('request.json', 'completion.json')})
        risk = paired_risk(risk_grid)
        if set(physical[kind]) & set(risk):
            raise ValueError('Overlapping physical target periods')
        physical[kind].update(risk)
        for s in ('baseline', 'combined'):
            if set(economic[kind][s]) & set(grid[s]):
                raise ValueError('Overlapping economic target periods')
            economic[kind][s].update(grid[s])
        evidence.update(proof)
    return economic, physical, evidence, summary


def calibrate(history, labels, sessions, first, last):
    from scripts.kalcb_quantity_calibration import corrections
    from backtests.strategies.kalcb.value_calibration import fit_support_offsets
    if any(r['trade_date'] >= first for r in history):
        raise ValueError('Calibration pool contains current/future fold forecasts')
    out = {}
    for origin, mode, method in [('frozen_core', 'offset_all', corrections), ('new_forecast', 'support_offset', fit_support_offsets)]:
        _, proof = method(history, labels, sessions, first)
        proof.update(mode=mode, valid_through=last)
        out[origin] = proof
    return out


def split_dates(sessions, first, last):
    return {p: [d for d in sessions if max(first, a) <= d <= min(last, b)] for p, (a, b, _) in PERIODS.items()}


def save_period(out, parent, period, result, folds, calibrations, evidence, recipe, registration):
    from scripts.kalcb_executed_value_replay import verify_policy, timed_policy
    first, last, count = PERIODS[period]
    dates = [str(d) for d in result['snapshots']]
    if len(dates) != count or dates[0] != first or dates[-1] != last:
        raise ValueError('Wrong scored policy coverage')
    out.mkdir(parents=True, exist_ok=False)
    write(out/(POLICY+'.snapshots.json'), [s.to_json_dict() for s in result['snapshots'].values()])
    jsonl(out/(POLICY+'.admission.jsonl'), result['admissions'])
    jsonl(out/'timed.snapshots.jsonl', (s.to_json_dict() for items in result['updates'].values() for s in items))
    jsonl(out/'timed.admission.jsonl', result['timed_admissions'])
    relevant = lambda x: x['valid_from'] <= last and x['valid_through'] >= first
    pf = [p for p in folds if relevant(p)]
    write(out/'chronology.json', {'folds': pf, 'calibration': [p for p in calibrations if relevant(p)], 'scope': 'Full-history causal development; reporting split does not refit or reset calibration.'})
    write(out/'lineage.json', {'label_manifests_and_receipts': evidence, 'recipe': recipe, 'registration': bound(registration),
                              'risk_models': 'Expanding paired physical-feasibility models, not loss classifiers.',
                              'models': [{'path': p['model_path'], 'sha256': p['model_sha256']} for p in pf]})
    old = read(parent/'policies_manifest.json')
    manifest = {k: deepcopy(old[k]) for k in ('contract', 'timed_snapshot_file', 'timed_admission_file', 'observation_symbols', 'observation_minutes')}
    manifest.update(version='kalcb-aligned-prequential-v1', policies=[POLICY], evaluation_from=first, evaluation_through=last,
                    files={p.name: digest(p) for p in out.iterdir()}, scope='Reused development; actual newly fitted IS and OOS policies.',
                    reference_policy_manifest=bound(parent/'policies_manifest.json'), recipe=recipe)
    write(out/'policies_manifest.json', manifest); write(out/'completion.json', {'complete': True})
    m, snapshots = verify_policy(out, POLICY); timed_policy(out, m, snapshots)


def control_preflight(out):
    """Reproduce a real historical fold before waiting for new target outcomes."""
    from threadpoolctl import threadpool_limits
    from scripts.kalcb_recovery_refit import datasets, HISTORY
    from backtests.strategies.kalcb import executed_value as ev, continuation_value as cv, paired_value as pv
    p = protocol_for(out)
    if (out/'control_preflight.json').exists():
        raise ValueError('Preserve previous control preflight')
    print('Authenticating original target evidence for historical forecast bridge.', flush=True)
    data, _, targets, sessions, _, evidence = datasets(ROOT, False)
    first, last = ev.outer_windows(sessions, min_train=60, block=20)[0]
    comparisons = []
    with threadpool_limits(limits=1):
        for kind, path, model_name in [('opening', HISTORY/'kalcb_continuation_20261006/policies_v2/forecasts.jsonl', 'cv_interaction'),
                                      ('timed', HISTORY/'kalcb_intraday_advantage_20261007/checkpoint/policies/forecasts.jsonl', 'interaction')]:
            historical = {r['opportunity_id']: r for r in rows(path) if r['model'] == model_name}
            for scenario in ('baseline', 'combined'):
                _, predictions, proof = cv.fit_forecast(data[kind], targets[kind][scenario], sessions, first, last,
                    columns=cv.COLUMNS + (pv.CHECKPOINT_COLUMNS if kind == 'timed' else ()))
                errors = [abs(v-historical[r['opportunity_id']][scenario+'_value_bps']) for r, v in predictions]
                row = {'kind': kind, 'scenario': scenario, 'comparisons': len(errors), 'max_abs_difference': max(errors), 'training_proof': proof}
                comparisons.append(row)
                write(out/'control_preflight_progress.json', comparisons)
                if row['max_abs_difference'] > 1e-7:
                    raise ValueError('Historical economic forecast drift: '+json.dumps(row))
    verify_bindings(p['bound_inputs'] + p['source_programs'])
    write(out/'control_preflight.json', {'complete': True, 'passed': True, 'comparisons': comparisons,
        'protocol': bound(out/'protocol.json'), 'original_target_evidence': evidence})


def fit(out):
    import joblib
    from threadpoolctl import threadpool_limits
    from scripts.kalcb_recovery_refit import datasets, HISTORY, assemble
    from scripts.kalcb_recovery_continuous import risk_labels
    from scripts.kalcb_action_quantity_ranker import capacity_ok
    from backtests.strategies.kalcb import executed_value as ev, continuation_value as cv, paired_value as pv, action_value as av
    p = protocol_for(out); alignment = Path(p['alignment_root'])
    preflight = read(out/'control_preflight.json')
    if not preflight.get('passed') or preflight['protocol'] != bound(out/'protocol.json'):
        raise ValueError('Missing historical forecast bridge')
    work = out/'fit'; work.mkdir(exist_ok=False)
    print('Authenticating original and selected native target generations.', flush=True)
    data, _, original, sessions, _, evidence = datasets(ROOT, True)
    physical = {}; original_proof = dict(evidence)
    for kind in data:
        physical[kind], ep = risk_labels(ROOT, kind, {r['opportunity_id'] for r in data[kind]}); original_proof.update(ep)
    selected, selected_physical, ep, target_summary = aligned_targets(alignment, data, read(ROOT/'preparation.json')['source_generation_hash'])
    write(work/'target_audit.json', {'selected': target_summary, 'selected_evidence': ep, 'original_evidence': original_proof})
    targets = {'original': original, 'selected': selected}; risks = {'original': physical, 'selected': selected_physical}
    evidence = {'original': original_proof, 'selected': ep}
    ap = read(alignment/'protocol.json'); cells = {(c['period'], c['kind']): c for c in ap['cells']}
    plans = {k: read(c['parent_plan']) for k, c in cells.items()}
    sources = {k: digest(Path(c['prepared'])/'opportunities.jsonl') for k, c in cells.items()}
    study = read(ROOT/'ablation_v3/study.json')
    windows = ev.outer_windows(sessions, min_train=60, block=20)
    if len(sessions) != 286 or sessions[0] != '2025-05-12' or windows[0][0] != PERIODS['is'][0]:
        raise ValueError('Wrong continuous training calendar')
    oos_windows = [w for w in windows if w[1] >= PERIODS['oos'][0]]
    if [a for a, _ in oos_windows] != ['2026-03-10', '2026-04-07', '2026-05-07', '2026-06-08', '2026-07-06']:
        raise ValueError('Training clock was reset')
    write(work/'calendar.json', {'sessions': sessions, 'windows': windows, 'periods': PERIODS})
    references = {}
    for kind, path, model in [('opening', HISTORY/'kalcb_continuation_20261006/policies_v2/forecasts.jsonl', 'cv_interaction'),
                              ('timed', HISTORY/'kalcb_intraday_advantage_20261007/checkpoint/policies/forecasts.jsonl', 'interaction')]:
        references[kind] = {r['opportunity_id']: r for r in rows(path) if r['model'] == model}
    results = {n: {period: {'snapshots': {}, 'updates': {}, 'admissions': [], 'timed_admissions': []} for period in PERIODS} for n in RECIPES}
    histories = {n: [] for n in RECIPES}; proofs = {n: [] for n in RECIPES}; calibrations = {n: [] for n in RECIPES}
    forecasts = {n: {k: [] for k in data} for n in RECIPES}; bridge = []
    (work/'models').mkdir()
    with threadpool_limits(limits=1):
        for first, last in windows:
            foldrows = {k: [r for r in rs if first <= r['trade_date'] <= last] for k, rs in data.items()}
            fitted = {}
            for family in ('original', 'selected'):
                for kind in data:
                    for scenario in ('baseline', 'combined', 'failure'):
                        if scenario == 'failure':
                            model, predictions, proof = av.fit_feasibility(data[kind], risks[family][kind], sessions, first, last)
                        else:
                            model, predictions, proof = cv.fit_forecast(data[kind], targets[family][kind][scenario], sessions, first, last,
                                columns=cv.COLUMNS + (pv.CHECKPOINT_COLUMNS if kind == 'timed' else ()))
                        path = work/'models'/f'{family}_{kind}_{scenario}_{first}.joblib'; joblib.dump(model, path)
                        proof.update(kind=kind, scenario=scenario, target_family=family, model_path=str(path.resolve()), model_sha256=digest(path))
                        values = {r['opportunity_id']: v for r, v in predictions}
                        if set(values) != {r['opportunity_id'] for r in foldrows[kind]}:
                            raise ValueError('Incomplete held-out predictions')
                        fitted[(family, kind, scenario)] = (values, proof)
                        if family == 'original':
                            field = 'failure_probability' if scenario == 'failure' else scenario+'_value_bps'
                            historical = [r for r in foldrows[kind] if r['trade_date'] <= PERIODS['is'][1]]
                            if any(r['opportunity_id'] not in references[kind] for r in historical):
                                raise ValueError('Missing historical control forecast')
                            errors = [abs(values[r['opportunity_id']]-references[kind][r['opportunity_id']][field]) for r in historical]
                            entry = {'fold': first, 'kind': kind, 'scenario': scenario, 'comparisons': len(errors), 'max_abs_difference': max(errors, default=0.)}
                            bridge.append(entry); write(work/'control_forecast_bridge.json', bridge)
                            if scenario != 'failure' and entry['max_abs_difference'] > 1e-7:
                                raise ValueError('Unexpected original economic forecast drift: '+json.dumps(entry))
            for name, recipe in RECIPES.items():
                current = {k: {} for k in data}
                for kind in data:
                    fields = {}
                    for scenario in ('baseline', 'combined', 'failure'):
                        family = recipe['risk_target' if scenario == 'failure' else 'value_target']
                        values, proof = fitted[(family, kind, scenario)]; proofs[name].append(proof)
                        fields['failure_probability' if scenario == 'failure' else scenario+'_value_bps'] = values
                    for r in foldrows[kind]:
                        key = r['opportunity_id']
                        current[kind][key] = {'opportunity_id': key, 'valid_from': first, 'valid_through': last,
                            'trade_date': r['trade_date'], 'quantity_scale': r['features']['action_quantity_scale'],
                            'capacity_ok': capacity_ok(r), **{field: values[key] for field, values in fields.items()}}
                cals = calibrate(histories[name], targets[recipe['value_target']]['opening'], sessions, first, last)
                calibrations[name].extend(cals.values())
                for period, dates in split_dates(sessions, first, last).items():
                    if not dates:
                        continue
                    chosen = assemble(*([r for r in foldrows[k] if r['trade_date'] in dates] for k in ('opening', 'timed')),
                        current['opening'], current['timed'], cals, plans[(period, 'opening')], plans[(period, 'timed')], dates,
                        sources[(period, 'opening')], sources[(period, 'timed')])
                    for key in ('snapshots', 'updates'): results[name][period][key].update(chosen[key])
                    for key in ('admissions', 'timed_admissions'): results[name][period][key].extend(chosen[key])
                # Only AFTER this entire fold is selected do its raw forecasts
                # enter the pool available to later calibration folds.
                histories[name].extend(current['opening'].values())
                for kind in data: forecasts[name][kind].extend(current[kind].values())
            print(json.dumps({'fit_from': first, 'through': last, 'recipes': list(RECIPES)}), flush=True)
    for name, recipe in RECIPES.items():
        label_proofs = dict(evidence[recipe['value_target']]); label_proofs.update(evidence[recipe['risk_target']])
        for period in PERIODS:
            save_period(work/'policies'/name/period, Path(study['periods'][period]['policies']), period, results[name][period],
                        proofs[name], calibrations[name], label_proofs, {'name': name, **recipe}, out/'protocol.json')
        folder = work/'forecasts'/name; folder.mkdir(parents=True)
        for kind in data: jsonl(folder/(kind+'.jsonl'), forecasts[name][kind])
    verify_bindings(p['bound_inputs'] + p['source_programs'])
    write(work/'completion.json', {'complete': True, 'protocol': bound(out/'protocol.json'), 'calendar': bound(work/'calendar.json'),
        'bridge': bound(work/'control_forecast_bridge.json'), 'target_audit': bound(work/'target_audit.json'),
        'policies': [bound(work/'policies'/n/period/'policies_manifest.json') for n in RECIPES for period in PERIODS],
        'models': [bound(path) for path in sorted((work/'models').glob('*.joblib'))]})


def replay(out):
    from scripts.kalcb_recovery_experiment_round import prepare, run
    p = protocol_for(out); completion = read(out/'fit/completion.json')
    if not completion.get('complete') or completion['protocol'] != bound(out/'protocol.json'):
        raise ValueError('Missing authenticated fitted policies')
    verify_bindings(completion['models'] + completion['policies'] + [completion['bridge'], completion['target_audit'], completion['calendar']])
    definitions = {name: {'config': {}, **{period+'_policy': str(out/'fit/policies'/name/period) for period in PERIODS}} for name in RECIPES}
    write(out/'replay_definitions.json', definitions)
    prepare(ROOT, ROUND, definitions, 1)
    # Registration is finalized before plugin instantiation or native replay.
    study_path = ROOT/ROUND/'study.json'; study = read(study_path)
    study['is_interpretation'] = p['is_interpretation']
    study['purpose'] = 'Controlled original vs selected economic/physical targets; complete historical OOF and native IS/OOS replay.'
    study['bound_inputs'] += [bound(out/'protocol.json'), bound(out/'fit/completion.json')] + completion['models']
    study['source_programs'] += [bound(__file__)]
    if study['cached_interaction_is']:
        raise ValueError('Fresh learner IS must never use a cached historical account')
    write(study_path, study)
    run(ROOT, ROUND)
    final = list((ROOT/ROUND/'framework').glob('*/round_1/round_final_full_diagnostics.json'))
    if len(final) != 1:
        raise ValueError('Missing existing-script full final diagnostics')
    write(out/'replay_completion.json', {'complete': True, 'study': bound(study_path), 'diagnostics': bound(final[0]),
        'next_step': 'Review target-alignment contrasts, component tradeoffs and full source qualification; no automatic selection.'})


def run_background(out):
    import msvcrt
    import psutil
    p = protocol_for(out); alignment = Path(p['alignment_root'])
    lock = (out/'run.lock').open('a+b'); lock.seek(0)
    if not lock.read(1): lock.write(b'0'); lock.flush()
    lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    state = {'pid': os.getpid(), 'process_created': psutil.Process().create_time(), 'child': None}
    def status(stage, **kw):
        state.update(stage=stage, updated_at_utc=datetime.now(timezone.utc).isoformat(), **kw)
        tmp = out/'status.tmp'; write(tmp, state); os.replace(tmp, out/'status.json')
    def wait_for(label, predicate):
        start = time.monotonic()
        while not predicate():
            status('waiting', waiting_for=label)
            if (out/'hold.json').exists(): raise ValueError('Explicit study hold')
            upstream = alignment/'status.json'
            if upstream.exists() and read(upstream).get('stage') == 'attention_required':
                raise ValueError('Upstream selected-target generation needs attention')
            if time.monotonic()-start > 48*3600: raise TimeoutError(label)
            time.sleep(30)
    try:
        for stage, receipt in [('control-preflight', out/'control_preflight.json'), ('fit', out/'fit/completion.json'), ('replay', out/'replay_completion.json')]:
            if stage == 'fit':
                wait_for('all8 authenticated selected-target grids', lambda: (alignment/'completion.json').exists())
            if receipt.exists(): raise ValueError('Preserve previous run; do not silently resume completed stages')
            wait_for('at least2GiB available RAM', lambda: psutil.virtual_memory().available >= 2*1024**3)
            protocol_for(out)
            command = [sys.executable, '-B', '-u', str(Path(__file__).resolve()), stage, '--root', str(out)]
            with (out/(stage+'.log')).open('x', encoding='utf-8') as stream:
                child = subprocess.Popen(command, cwd=REPO, env=env(REPO), stdout=stream, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
                status(stage, waiting_for=None, child={'pid': child.pid, 'process_created': psutil.Process(child.pid).create_time(), 'command': command})
                start = time.monotonic()
                while child.poll() is None:
                    if time.monotonic()-start > 48*3600: raise TimeoutError('Child preserved after48hours')
                    time.sleep(30)
                if child.returncode: raise RuntimeError('Failed '+stage+'; preserve '+str(out/(stage+'.log')))
            if not read(receipt).get('complete'): raise ValueError('Stage did not complete')
            status(stage+'_completed', child=None)
        status('ready_for_target_alignment_review', child=None)
    except Exception as exc:
        status('attention_required', error=repr(exc)); raise
    finally:
        lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1); lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['register', 'control-preflight', 'fit', 'replay', 'run'])
    parser.add_argument('--root', type=Path, default=DEST)
    parser.add_argument('--alignment', type=Path, default=ALIGNMENT)
    args = parser.parse_args()
    if args.stage == 'register': register(args.root, args.alignment)
    else: {'control-preflight': control_preflight, 'fit': fit, 'replay': replay, 'run': run_background}[args.stage](args.root)
