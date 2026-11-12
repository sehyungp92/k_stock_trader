"""Descriptive, matched-outcome review of the registered alignment contrasts.

No selection, training or portfolio accounting is performed here. Opportunity
outcomes are isolated KRW100m accounts and must never be added into a portfolio.
"""
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.kalcb_full_oos_recovery import ROOT, read, write, rows, bound, digest, POLICY
from scripts.kalcb_alignment_walk_forward_v3 import DEST, ALIGNMENT, BOUNDARY, RECIPES, verify_bindings
from scripts.portfolio_frozen_inference import calibrated


def stats(keys, labels, forecasts, scenario):
    import numpy as np
    from scipy.stats import spearmanr
    keys = list(keys)
    valid = [k for k in keys if labels[k]['label_valid']]
    result = {'actions': len(keys), 'valid': len(valid), 'invalid_retained': len(keys)-len(valid),
              'issuer_days': len({labels[k]['trade_date']+'/'+labels[k]['symbol'] for k in keys}),
              'sessions': len({labels[k]['trade_date'] for k in keys})}
    if not valid:
        return result
    y = np.array([labels[k]['net_value_equity_bps'] for k in valid])
    p = np.array([forecasts[k][scenario+'_value_bps'] for k in valid])
    result.update(mean_actual_bps=float(y.mean()), mean_prediction_bps=float(p.mean()),
                  mean_overprediction_bps=float((p-y).mean()), median_actual_bps=float(np.median(y)),
                  positive_pct=float(np.mean(y > 0)*100),
                  rank_correlation=float(spearmanr(p, y).statistic) if len(y)>2 and np.ptp(p)>0 and np.ptp(y)>0 else None)
    return result


def run():
    out = ROOT/'alignment_outcome_review_v1.json'
    if out.exists():
        raise ValueError('Preserve existing descriptive review')
    done = read(DEST/'fit/completion.json')
    verify_bindings(done['policies'] + done['models'] + [done['protocol'], done['target_audit'], done['calendar'], done['bridge']])
    protocol = read(ALIGNMENT/'protocol.json'); complete = read(ALIGNMENT/'completion.json')
    verify_bindings(complete['grids'] + [complete['protocol'], complete['preview']])
    labels = defaultdict(dict); bindings = []
    for job in protocol['jobs']:
        folder = Path(job['output']); manifest = read(folder/'grid_manifest.json')
        for name, checksum in manifest['files'].items():
            if not name.endswith('.jsonl'):
                continue
            if digest(folder/name) != checksum:
                raise ValueError('Changed outcome grid')
            for r in rows(folder/name):
                group = labels[(job['kind'], job['scenario'])]
                if r['opportunity_id'] in group:
                    raise ValueError('Duplicate native outcome')
                group[r['opportunity_id']] = r
        bindings.append(bound(folder/'grid_manifest.json'))
    overlay = read(BOUNDARY/'completion.json'); verify_bindings([overlay['overlays'], overlay['prefix_bridge']])
    for item in read(BOUNDARY/'overlays.json'):
        if item['family'] == 'selected':
            labels[(item['kind'], item['scenario'])][item['label']['opportunity_id']] = item['label']
    studies = {}
    periods = [('is', '2025-08-07', '2026-03-31'), ('oos', '2026-04-01', '2026-07-10'),
               ('Apr-May', '2026-04-01', '2026-05-31'), ('Jun-Jul', '2026-06-01', '2026-07-10')]
    for recipe in RECIPES:
        forecasts = {k: {r['opportunity_id']: r for r in rows(DEST/'fit/forecasts'/recipe/(k+'.jsonl'))}
                     for k in ('opening', 'timed')}
        cals = []
        admissions = {'opening': {}, 'timed': {}}
        for period in ('is', 'oos'):
            directory = DEST/'fit/policies'/recipe/period
            for c in read(directory/'chronology.json')['calibration']:
                if c not in cals:
                    cals.append(c)
            for kind, filename in [('opening', POLICY+'.admission.jsonl'), ('timed', 'timed.admission.jsonl')]:
                for r in rows(directory/filename):
                    if r['action_id'] in admissions[kind]:
                        raise ValueError('Duplicate admission')
                    admissions[kind][r['action_id']] = r
        adjusted = {'frozen_core': {}, 'new_forecast': {}}
        for c in cals:
            raw = {k: r for k, r in forecasts['opening'].items() if c['valid_from']<=r['trade_date']<=c['valid_through']}
            origin = 'frozen_core' if c['mode']=='offset_all' else 'new_forecast'
            new = calibrated(raw, c)
            if set(new) & set(adjusted[origin]):
                raise ValueError('Overlapping calibration folds')
            adjusted[origin].update(new)
        groups = {}
        for period, first, last in periods:
            groups[period] = {}
            for kind, forecast in forecasts.items():
                current = {k: r for k, r in forecast.items() if first<=r['trade_date']<=last}
                cells = {}
                for q in (.25, .5, 1.):
                    keys = [k for k, r in current.items() if r['quantity_scale']==q]
                    cells['all_q'+str(q)] = (keys, forecast)
                    feasible = [k for k in keys if current[k]['capacity_ok'] and current[k]['failure_probability'] < .001]
                    cells['feasible_q'+str(q)] = (feasible, forecast)
                    positive = [k for k in feasible if min(current[k][s+'_value_bps'] for s in ('baseline', 'combined'))>0]
                    cells['raw_positive_q'+str(q)] = (positive, forecast)
                origins = ('frozen_core', 'new_forecast') if kind=='opening' else ('checkpoint',)
                for origin in origins:
                    keys = [k for k, r in admissions[kind].items() if k in current and
                            (kind=='timed' or r['forecast_origin']==origin)]
                    scored = adjusted[origin] if kind=='opening' else forecast
                    # Existing selector maximizes baseline expected value,
                    # with strictly positive stress value as a constraint.
                    for k in keys:
                        if abs(admissions[kind][k]['score']-scored[k]['baseline_value_bps']) > 1e-7:
                            raise ValueError('Admission forecast attribution mismatch')
                    cells['admitted_'+origin] = (keys, scored)
                cells['not_admitted_q1'] = ([k for k, r in current.items() if r['quantity_scale']==1 and
                    k.rsplit('/q',1)[0] not in {a.rsplit('/q',1)[0] for a in admissions[kind]}], forecast)
                groups[period][kind] = {name: {s: stats(keys, labels[(kind,s)], values, s)
                    for s in ('baseline', 'combined')} for name, (keys, values) in cells.items()}
        studies[recipe] = groups
    payload = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'program': bound(__file__),
               'fit': bound(DEST/'fit/completion.json'), 'targets': bindings, 'boundary': bound(BOUNDARY/'completion.json'),
               'scope': 'Common selected-management native outcomes for every recipe; original-target predictions intentionally assessed against the deployed management outcome. Isolated fixed KRW100m action returns, not portfolio returns. No invalid labels imputed. Quantity/issuer/date dependence means counts are not independent samples. Reused development; diagnostic groups must not become symbol/date exclusion rules.',
               'groups': studies}
    write(out, payload)
    print(json.dumps({'saved': str(out), 'oos_admitted': {n: {k: {c: v for c,v in result['oos'][k].items()
        if c.startswith('admitted_')} for k in ('opening','timed')} for n,result in studies.items()}}, indent=2), flush=True)


if __name__ == '__main__':
    run()
