"""Authenticate and reconstruct stored finalist forecasts without refitting."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import os
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.kalcb_full_oos_recovery import ROOT, read, write, bound, digest, rows
from scripts import kalcb_alignment_walk_forward_v3 as aligned
from scripts import kalcb_recovery_mechanism_round as mechanism
from scripts import kalcb_recovery_blend_checkpoint_round as blend

DEST = Path('E:/CodexResearch/k_stock_trader/kalcb_forecast_qualification_20261010/v1')
Q = Path('E:/CodexResearch/k_stock_trader/kalcb_recovery_qualification_20261010/v4')
RECIPES = {'expanding': (aligned.DEST, 'original_target_full_history'),
           'decay60': (mechanism.DEST, 'original_decay60')}


def validate_training_proof(proof, selected, target, weights=None):
    """Reject future membership, wrong target population or lost recency weights."""
    from backtests.strategies.kalcb.executed_value import signature
    if not selected or max(r['trade_date'] for r in selected) >= proof['valid_from']:
        raise ValueError('Future or empty training membership')
    if signature([r['opportunity_id'] for r in selected]) != proof['training_keys_hash']:
        raise ValueError('Training membership differs from fitted model')
    if signature(target) != proof['training_target_hash']:
        raise ValueError('Training outcomes differ from fitted model')
    if len(selected) != proof['train_rows']:
        raise ValueError('Training denominator drift')
    if (min(r['trade_date'] for r in selected), max(r['trade_date'] for r in selected)) != (proof['train_from'], proof['train_through']):
        raise ValueError('Training date extent drift')
    if 'training_weights_hash' in proof and (weights is None or signature(weights) != proof['training_weights_hash']):
        raise ValueError('Recency weights drift')


def register(out):
    if out.exists(): raise ValueError('Preserve prior forecast qualification')
    from scripts.kalcb_recovery_qualification_round_v4 import check
    check(Q)
    paths = [Path(__file__), Q/'protocol.json', Q/'preparation.json',
             ROOT/'forecast_qualification_tests_20261010.log',
             REPO/'tests/backtests/strategies/test_kalcb_forecast_qualification.py',
             ROOT/'recovery_report_context_repair_20261010.json',
             REPO/'backtests/strategies/kalcb/action_value.py',
             REPO/'backtests/strategies/kalcb/continuation_value.py',
             REPO/'backtests/strategies/kalcb/value_calibration.py',
             REPO/'scripts/kalcb_quantity_calibration.py',
             REPO/'scripts/kalcb_recovery_refit.py',
             REPO/'scripts/kalcb_recovery_continuous.py',
             REPO/'scripts/kalcb_alignment_walk_forward_v3.py',
             REPO/'scripts/kalcb_recovery_blend_checkpoint_round.py',
             aligned.BOUNDARY/'completion.json', aligned.BOUNDARY/'overlays.json',
             blend.DEST/'build/control_bridges.json']
    for root, recipe in RECIPES.values():
        paths += [root/'protocol.json', root/'fit/completion.json']
        for kind in ('opening', 'timed'): paths.append(root/'fit/forecasts'/recipe/(kind+'.jsonl'))
        for period in ('is', 'oos'):
            paths += list((root/'fit/policies'/recipe/period).glob('*'))
    out.mkdir(parents=True)
    write(out/'protocol.json', {'registered_at_utc': datetime.now(timezone.utc).isoformat(),
        'bindings': [bound(p) for p in sorted(set(paths)) if p.is_file()],
        'recipes': list(RECIPES), 'scope': 'Load content-bound saved models; reconstruct every stored held-out economic and physical-risk prediction and opening calibration. Verify training membership, targets, weights, label maturity, original fold clock and first-OOS inherited fit. No refit, parameter search, market change or promotion.',
        'forecast_tolerance': 1e-8, 'calibration_contract': 'Exact saved proof equality after past-OOF reconstruction.'})


def run(out):
    import joblib
    import numpy as np
    import psutil
    from threadpoolctl import threadpool_limits
    from scripts.kalcb_recovery_continuous import risk_labels
    from backtests.strategies.kalcb import action_value as av, continuation_value as cv
    protocol = read(out/'protocol.json'); aligned.verify_bindings(protocol['bindings'])
    if (out/'completion.json').exists(): raise ValueError('Preserve completed verification')
    status = {'pid': os.getpid(), 'process_created': psutil.Process().create_time()}
    def update(stage, **extra):
        status.update(stage=stage, updated_at_utc=datetime.now(timezone.utc).isoformat(), **extra)
        write(out/'status.tmp', status); os.replace(out/'status.tmp', out/'status.json')
    try:
        update('authenticating_inputs')
        data, labels, sessions, evidence = blend.original_targets()
        calendar = read(aligned.DEST/'fit/calendar.json')
        if sessions != calendar['sessions']: raise ValueError('Changed historical calendar')
        physical = {}
        for kind in data:
            physical[kind], proof = risk_labels(ROOT, kind, {r['opportunity_id'] for r in data[kind]})
            evidence.update(proof)
        overlays = read(aligned.BOUNDARY/'overlays.json')
        count = 0
        for item in overlays:
            if item['family'] != 'original': continue
            label = item['label']; key = label['opportunity_id']; kind = item['kind']
            old = physical[kind][key]
            if not old['unsafe'] or not label['deadline_violation']: raise ValueError('Lost boundary physical failure')
            physical[kind][key] = {'unsafe': True, 'risk_available_at': max(old['risk_available_at'], label['risk_available_at'])}
            count += 1
        if count != 4: raise ValueError('Wrong original boundary population')
        all_checks = []; calibration_checks = []
        with threadpool_limits(limits=1):
            for recipe, (root, name) in RECIPES.items():
                forecasts = {k: {r['opportunity_id']: r for r in rows(root/'fit/forecasts'/name/(k+'.jsonl'))} for k in data}
                proofs = {}; calibrations = {}
                for period in ('is', 'oos'):
                    c = read(root/'fit/policies'/name/period/'chronology.json')
                    for proof in c['folds']:
                        key = (proof['kind'], proof['scenario'], proof['valid_from'])
                        if key in proofs and proofs[key] != proof: raise ValueError('IS/OOS model boundary mismatch')
                        proofs[key] = proof
                    for proof in c['calibration']:
                        key = (proof['valid_from'], proof['mode'])
                        if key in calibrations and calibrations[key] != proof: raise ValueError('IS/OOS calibration boundary mismatch')
                        calibrations[key] = proof
                expected_keys = {(k, s, first) for k in data for s in ('baseline', 'combined', 'failure') for first, _ in calendar['windows']}
                if set(proofs) != expected_keys: raise ValueError('Missing full-history model fold')
                for kind in data:
                    expected = {r['opportunity_id'] for r in data[kind] if r['trade_date'] >= calendar['windows'][0][0]}
                    if set(forecasts[kind]) != expected: raise ValueError('Stored forecasts omit eligible actions')
                for key, proof in sorted(proofs.items()):
                    kind, scenario, first = key
                    update('reconstructing_models', recipe=recipe, fold=first, kind=kind, scenario=scenario)
                    path = Path(proof['model_path'])
                    if digest(path) != proof['model_sha256']: raise ValueError('Changed fitted model')
                    if proof['label_available_through'] >= proof['fit_at'] or sessions.index(first)-sessions.index(proof['train_through']) < 2:
                        raise ValueError('Immature or unembargoed training outcomes')
                    valid = [r for r in data[kind] if first <= r['trade_date'] <= proof['valid_through']]
                    weights = None
                    if scenario == 'failure':
                        allowed = set(sessions[:max(0, sessions.index(first)-1)])
                        selected = [r for r in data[kind] if r['trade_date'] in allowed and r['opportunity_id'] in physical[kind]
                                    and physical[kind][r['opportunity_id']]['risk_available_at'] < proof['fit_at']]
                        target = [bool(physical[kind][r['opportunity_id']]['unsafe']) for r in selected]
                        available = max(physical[kind][r['opportunity_id']]['risk_available_at'] for r in selected)
                        field = 'failure_probability'; matrix = av.matrix(valid, tuple(proof['feature_columns']))
                    else:
                        target_labels = labels[kind][scenario]
                        indices, weight = av.training_indices(data[kind], target_labels, sessions, first, av.ModelSpec('interaction', window=proof.get('window', 'expanding')))
                        selected = [data[kind][i] for i in indices]
                        target = [target_labels[r['opportunity_id']]['net_value_equity_bps'] for r in selected]
                        available = max(target_labels[r['opportunity_id']]['label_available_at'] for r in selected)
                        weights = weight.tolist(); field = scenario+'_value_bps'; matrix = cv.matrix(valid, tuple(proof['feature_columns']))
                    validate_training_proof(proof, selected, target, weights)
                    if available != proof['label_available_through']: raise ValueError('Training availability extent drift')
                    model = joblib.load(path)
                    predicted = np.asarray(model.predict(matrix), dtype=float)
                    stored = np.asarray([forecasts[kind][r['opportunity_id']][field] for r in valid], dtype=float)
                    if predicted.shape != stored.shape or not np.isfinite(predicted).all(): raise ValueError('Prediction coverage/nonfinite failure')
                    error = float(np.max(np.abs(predicted-stored))) if len(valid) else 0.
                    if error > protocol['forecast_tolerance']: raise ValueError('Saved prediction reconstruction failed')
                    if any((forecasts[kind][r['opportunity_id']]['valid_from'],forecasts[kind][r['opportunity_id']]['valid_through']) != (first,proof['valid_through']) for r in valid):
                        raise ValueError('Forecast/model deployment clock differs')
                    all_checks.append({'recipe': recipe, 'kind': kind, 'scenario': scenario, 'first': first,
                        'model': bound(path), 'prediction_count': len(valid), 'max_abs_error': error,
                        'training_rows': len(selected), 'training_target_and_keys_passed': True,
                        'training_weights_checked': 'training_weights_hash' in proof})
                    print({k:v for k,v in all_checks[-1].items() if k!='model'},flush=True)
                history = []
                for first, last in calendar['windows']:
                    cals = aligned.calibrate(history, labels['opening'], sessions, first, last)
                    for cal in cals.values():
                        if cal != calibrations[(first,cal['mode'])]: raise ValueError('Past-only calibration reconstruction failed')
                    calibration_checks.append({'recipe':recipe,'first':first,'prior_oof_rows':len(history),'passed':True})
                    history.extend(r for r in forecasts['opening'].values() if first <= r['trade_date'] <= last)
        aligned.verify_bindings(protocol['bindings'])
        write(out/'forecast_reconstruction.json',{'passed':True,'models':all_checks,'calibrations':calibration_checks,
            'prediction_comparisons':sum(x['prediction_count'] for x in all_checks),
            'max_abs_error':max(x['max_abs_error'] for x in all_checks),
            'history_start':sessions[0],'first_oos_inherited_fit':'2026-03-10','first_oos_refit':'2026-04-07',
            'continuous_market_sessions':len(sessions),'inputs':evidence,'protocol':bound(out/'protocol.json'),
            'scope':'Authenticates the saved full-history models, training populations, weights and targets; reconstructs saved held-out forecasts and earlier-OOF calibration. Does not refit models, certify profitability or create independent observations.'})
        write(out/'completion.json',{'complete':True,'passed':True,'reconstruction':bound(out/'forecast_reconstruction.json'),'protocol':bound(out/'protocol.json')})
        update('complete')
    except Exception as error:
        update('attention_required', error=repr(error)); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['register','run'])
    args = parser.parse_args()
    {'register':register,'run':run}[args.stage](DEST)
