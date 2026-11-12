"""Causal research policy primitives; execution remains in the shared native core.

No outcomes enter descriptors or selection. Calibration consumes only mature
earlier OOF observations, with an embargo and issuer-day/session support gates.
"""
from collections import defaultdict
from datetime import datetime
import math

SCENARIOS = ('baseline', 'combined')
QUANTITIES = (.25, .5, 1.)


def finite(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('Nonfinite policy value')
    return value


def atr_budget(row):
    c = row['candidate']
    atr, close = finite(c['daily_atr']), finite(c['prior_day_close'])
    if min(atr, close) <= 0:
        raise ValueError('ATR budget requires positive prior-day values')
    return min(1., .05 * close / atr)


def supported_quantity(row):
    """Only actual labelled grid actions, never interpolated outcomes/quantities."""
    q = finite(row['features']['action_quantity_scale'])
    if q not in QUANTITIES:
        raise ValueError('Unlabelled quantity')
    return q <= atr_budget(row) + 1e-12


def descriptors(items, values, other):
    """Rank issuer-days on parent scores BEFORE any proposed correction."""
    best = {}
    for r in items:
        k, b = r['opportunity_id'], r['base_opportunity_id']
        v = finite(values[k]['baseline_value_bps'])
        best[b] = max(best.get(b, -math.inf), v)
    ranks = {b: i + 1 for i, (b, _) in enumerate(sorted(best.items(), key=lambda x: (-x[1], x[0])))}
    result = {}
    for r in items:
        k = r['opportunity_id']
        positive = lambda d: all(finite(d[k][s+'_value_bps']) > 0 for s in SCENARIOS)
        agreement = 'agree' if positive(values) == positive(other) else 'disagree'
        rank = ranks[r['base_opportunity_id']]
        result[k] = {'rank': rank, 'tier': 'top2' if rank <= 2 else 'rest', 'agreement': agreement,
                     'quantity_scale': r['features']['action_quantity_scale']}
    return result


def cohort(row):
    return f"q{row['quantity_scale']:g}/{row['tier']}/{row['agreement']}"


def residual_fit(observations, labels, sessions, first, *, lookback=60, prior=100., minimum=20, min_sessions=8):
    """Hierarchical intercept residuals in fixed reference-account equity bps.

    Parent-selected OOF action per issuer-day only. Broad quantity cohorts shrink
    toward zero; small conditional cohorts shrink toward that broad correction.
    Unsupported cohorts use identity. Labels are retained, not winsorized.
    """
    index = sessions.index(first)
    allowed = set(sessions[max(0, index-1-lookback):max(0, index-1)])
    cutoff = datetime.fromisoformat(first+'T00:00:00+09:00')
    groups, pools, seen = defaultdict(list), defaultdict(list), set()
    latest = None
    for r in observations:
        if r['trade_date'] not in allowed:
            continue
        if r['valid_from'] > r['trade_date']:
            raise ValueError('Forecast from the future')
        identity = (r['trade_date'], r['base_opportunity_id'])
        if identity in seen:
            raise ValueError('Dependent quantity alternatives in selected calibration cohort')
        seen.add(identity)
        for s in SCENARIOS:
            label = labels[s][r['opportunity_id']]
            if not label['label_valid']:
                continue
            at = datetime.fromisoformat(label['label_available_at'])
            if at.tzinfo is None:
                raise ValueError('Naive outcome availability')
            if at >= cutoff:
                continue
            error = finite(label['net_value_equity_bps']) - finite(r[s+'_value_bps'])
            item = (r['trade_date'], error)
            groups[(s, cohort(r))].append(item)
            pools[(s, r['quantity_scale'])].append(item)
            latest = max(latest or label['label_available_at'], label['label_available_at'])
    detail = {}
    for s in SCENARIOS:
        for q in QUANTITIES:
            pool = pools[(s, q)]
            n, days = len(pool), len({d for d, _ in pool})
            supported = n >= minimum and days >= min_sessions
            pooled = sum(e for _, e in pool)/(n+prior) if supported else 0.
            for tier in ('top2', 'rest'):
                for agreement in ('agree', 'disagree'):
                    key = f'q{q:g}/{tier}/{agreement}'
                    sample = groups[(s, key)]
                    local = len(sample) >= 8 and len({d for d, _ in sample}) >= 5
                    value = ((sum(e for _, e in sample)+prior*pooled)/(len(sample)+prior)
                             if supported and local else pooled)
                    detail[s+'/'+key] = {'offset_bps': value, 'supported': supported,
                        'pool_issuer_days': n, 'pool_sessions': days, 'cell_issuer_days': len(sample),
                        'cell_sessions': len({d for d, _ in sample})}
    return {'valid_from': first, 'latest_label_available_at': latest,
            'observed_through': max(allowed) if allowed else None, 'embargo_sessions': 1,
            'lookback_sessions': lookback, 'prior_count': prior, 'minimum_count': minimum,
            'minimum_sessions': min_sessions, 'cohorts': detail,
            'target': 'native selected-management mature cash outcomes; reference100m equity bps'}


def corrected(values, descriptions, proof):
    result = {}
    for k, v in values.items():
        if k not in descriptions:
            continue
        result[k] = dict(v)
        for s in SCENARIOS:
            result[k][s+'_value_bps'] = finite(v[s+'_value_bps']) + proof['cohorts'][s+'/'+cohort(descriptions[k])]['offset_bps']
    return result


def supported_correction(description, proof):
    return all(proof['cohorts'][s+'/'+cohort(description)]['supported'] for s in SCENARIOS)
