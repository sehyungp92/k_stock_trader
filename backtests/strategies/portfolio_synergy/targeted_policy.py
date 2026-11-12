"""Causal, lineage-matched marginal-quantity challenge of existing KALCB entries."""
from dataclasses import replace
from datetime import datetime
import json
import math


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def marginal_decision(current, smaller, calibration, day):
    """Unsupported estimates leave the incumbent action unchanged."""
    if current is None or smaller is None or calibration is None:
        return False, 'missing_paired_forecast_or_calibration'
    if current['opportunity_id'].rsplit('/q',1)[0] != smaller['opportunity_id'].rsplit('/q',1)[0]:
        raise ValueError('Marginal forecasts refer to different opportunities')
    if current['valid_from'] != smaller['valid_from'] or calibration['valid_from'] != current['valid_from']:
        raise ValueError('Marginal forecasts mix fit lineages')
    if current['valid_from'] > day or calibration.get('valid_through',day) < day:
        raise ValueError('Marginal forecast outside causal validity window')
    cutoff = datetime.fromisoformat(current['valid_from']+'T00:00:00+09:00')
    latest = calibration.get('latest_label_available_at')
    if latest and datetime.fromisoformat(latest) >= cutoff:
        raise ValueError('Unmatured marginal calibration labels')
    if calibration.get('observed_through') and calibration['observed_through'] >= current['valid_from']:
        raise ValueError('Calibration overlaps prediction period')
    values = [row[s+'_value_bps'] for row in (current,smaller) for s in ('baseline','combined')]
    if any(not math.isfinite(float(v)) for v in values):
        raise ValueError('Nonfinite marginal forecast')
    if any(not math.isfinite(float(r.get('failure_probability',1))) or r.get('failure_probability',1) >= .001
           for r in (current,smaller)):
        return False, 'unsupported_feasibility'
    if min(smaller[s+'_value_bps'] for s in ('baseline','combined')) <= 0:
        return False, 'smaller_action_not_positive_in_both_scenarios'
    for row in (current,smaller):
        q = float(row['opportunity_id'].rsplit('/q',1)[1])
        support = row.get('raw_score_support')
        if calibration.get('mode') == 'offset_all' and support != 'positive':
            return False, 'outside_core_calibration_population'
        if calibration.get('mode') == 'support_offset' and support not in ('positive','nonpositive'):
            return False, 'missing_raw_support_lineage'
        for scenario in ('baseline','combined'):
            key = f'{scenario}/q{q:g}' + (f'/{support}' if calibration.get('mode') == 'support_offset' else '')
            cohort = calibration.get('cohorts',{}).get(key,{})
            if cohort.get('stock_days',0) < 20 or cohort.get('sessions',0) < 8:
                return False, 'insufficient_mature_cohort_support'
    if any(current[s+'_value_bps'] <= smaller[s+'_value_bps'] for s in ('baseline','combined')):
        return True, 'nonpositive_increment_in_at_least_one_scenario'
    return False, 'positive_increment_in_both_scenarios'


def recover_core_support(directory, calibrations):
    """Recover missing session counts from the exact, frozen calibration inputs.

    The old offset-all fit used raw-positive observations for its offset. It
    saved counts but not session counts. Reconstruct those counts and verify
    the old offsets; never fit a new forecast or use current-fold labels.
    """
    from backtests.strategies.kalcb.executed_value import digest
    from scripts.kalcb_continuation_study import compact_labels
    from scripts.kalcb_action_quantity_labels import scaled_row
    from scripts.kalcb_action_quantity_ranker import capacity_ok
    from scripts.kalcb_quantity_calibration import corrections
    parent = directory.parent
    manifest = read(directory/'opening_core_policies_manifest.json')
    candidates = [p for p in parent.glob('*/prepared.json') if digest(p) == manifest['prepared_sha256']]
    if len(candidates) != 1: raise ValueError('Ambiguous frozen core prepared lineage')
    prepared = candidates[0]; meta = read(prepared)
    if digest(prepared.parent/'opportunities.jsonl') != meta['opportunities_sha256']:
        raise ValueError('Changed core opportunities')
    paths = {digest(p):p.parent for kind in ('labels_manifest.json','grid_manifest.json') for p in parent.glob('*/'+kind)}
    label_paths = [paths[sha] for sha in manifest['label_manifests'].values()]
    grid_paths = [paths[sha] for sha in manifest['grid_manifests'].values()]
    base,_ = compact_labels(label_paths, meta['parent_prepared_sha256'])
    grid,_ = compact_labels(grid_paths, meta['original_prepared_sha256'], grid=True)
    raw = {r['opportunity_id']:r for r in rows(directory/'opening_core_forecasts.jsonl') if r['model']=='cv_interaction'}
    observations = []
    for row in rows(prepared.parent/'opportunities.jsonl'):
        if not manifest['evaluation_from'] <= row['trade_date'] <= manifest['evaluation_through']: continue
        for q in (.25,.5,1.):
            scaled = scaled_row(row,q); key = scaled['opportunity_id']
            if q == 1.: grid['baseline'][key] = base['baseline'][scaled['base_opportunity_id']]
            observations.append({**raw[key], 'trade_date':row['trade_date'], 'quantity_scale':q,
                                 'capacity_ok':capacity_ok(scaled)})
    for calibration in calibrations:
        first = calibration['valid_from']; sessions = meta['sessions']
        _, proof = corrections(observations,grid,sessions,first)
        if proof['cohorts'] != calibration['cohorts']:
            raise ValueError('Reconstructed core calibration differs from frozen fit')
        stop = max(0,sessions.index(first)-1); allowed = set(sessions[max(0,stop-60):stop])
        cutoff = datetime.fromisoformat(first+'T00:00:00+09:00')
        for scenario in ('baseline','combined'):
            for q in (.25,.5,1.):
                population = [r for r in observations if r['trade_date'] in allowed and r['quantity_scale']==q
                    and r['baseline_value_bps']>0 and r['combined_value_bps']>0
                    and r['failure_probability']<.001 and r['capacity_ok']
                    and grid[scenario][r['opportunity_id']]['label_valid']
                    and datetime.fromisoformat(grid[scenario][r['opportunity_id']]['label_available_at'])<cutoff]
                cohort = calibration['cohorts'][f'{scenario}/q{q:g}']
                cohort['stock_days'] = len({r['opportunity_id'].rsplit('/q',1)[0] for r in population})
                cohort['sessions'] = len({r['trade_date'] for r in population})


def transform_marginal_policy(directory, name, snapshots, updates):
    admissions = {r['opportunity_id']:r for r in rows(directory/(name+'.admission.jsonl')) if r['reason']=='admitted'}
    tables, calibrations = {}, {}
    for origin, prefix in (('frozen_core','opening_core_'),('new_forecast','opening_')):
        forecasts = rows(directory/(prefix+'calibrated_forecasts.jsonl'))
        table = {r['opportunity_id']:r for r in forecasts}
        if len(table) != len(forecasts): raise ValueError('Ambiguous marginal forecast table')
        tables[origin] = table
        raw = {r['opportunity_id']:r for r in rows(directory/(prefix+'forecasts.jsonl')) if r['model']=='cv_interaction'}
        for key,value in table.items():
            source = raw[key]
            if source['valid_from'] != value['valid_from']: raise ValueError('Raw/calibrated fit mismatch')
            value['raw_score_support'] = 'positive' if min(source[s+'_value_bps'] for s in ('baseline','combined'))>0 else 'nonpositive'
        calibration = [r for r in read(directory/(prefix+'chronology.json')).get('calibration',[]) if not r.get('lineage')]
        if origin == 'frozen_core': recover_core_support(directory,calibration)
        calibrations[origin] = {r['valid_from']:r for r in calibration}
        if len(calibrations[origin]) != len(calibration): raise ValueError('Ambiguous marginal calibration')
    transformed, evidence = {}, []
    for day, snapshot in snapshots.items():
        candidates = []
        for candidate in snapshot.candidates:
            key = f'{day}/{candidate.symbol}'
            admission = admissions[key]
            q = float(candidate.metadata.get('action_quantity_scale',1))
            if q not in (.25,.5,1.): raise ValueError('Unsupported native quantity tier')
            row = {'identity':key, 'from_quantity_scale':q, 'to_quantity_scale':q, 'changed':False,
                   'forecast_origin':admission.get('forecast_origin'), 'source_snapshot_hash':snapshot.artifact_hash}
            if q == .25:
                row['reason'] = 'minimum_native_tier'
            else:
                origin = admission.get('forecast_origin')
                if origin not in tables: raise ValueError('Unknown incumbent forecast lineage')
                small = .25 if q == .5 else .5
                current = tables[origin].get(f'{key}/q{q:g}')
                smaller = tables[origin].get(f'{key}/q{small:g}')
                cal = calibrations[origin].get(current['valid_from']) if current else None
                changed, reason = marginal_decision(current, smaller, cal, str(day))
                row.update(reason=reason, changed=changed, current_forecast=current, smaller_forecast=smaller,
                           calibration_valid_from=cal['valid_from'] if cal else None)
                if changed:
                    row['to_quantity_scale'] = small
                    candidate = replace(candidate, metadata={**candidate.metadata,'action_quantity_scale':small,
                        'portfolio_marginal_quantity':{k:v for k,v in row.items() if k not in ('current_forecast','smaller_forecast')}})
            candidates.append(candidate); evidence.append(row)
        transformed[day] = replace(snapshot,candidates=tuple(candidates))
    timed = {}
    for day, daily in updates.items():
        replacements = transformed[day].by_symbol()
        timed[day] = [replace(snapshot,candidates=tuple(replacements.get(c.symbol,c) for c in snapshot.candidates)) for snapshot in daily]
    return transformed, timed, {'version':'supported-marginal-quantity-v1', 'decisions':evidence,
        'changed_issuer_days':sum(r['changed'] for r in evidence),
        'changed_sessions':len({r['identity'].split('/')[0] for r in evidence if r['changed']}),
        'scope':'Causal proxy quantity forecasts at their original reference capital; not calibrated proof of incremental KRW25m shared-account value. No new signals, no tier increase, no OOS outcomes.'}
