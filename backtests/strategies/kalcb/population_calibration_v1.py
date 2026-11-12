"""Research-only causal residual calibration within supported populations.

This is a policy primitive, not an optimization runner or a live strategy.
The caller must authenticate the original forecasts and isolated native labels.
Quantity actions and forecast origins remain separate dependent populations.
"""
from collections import Counter, defaultdict
from datetime import datetime
from itertools import product
import math

SCENARIOS = ('baseline', 'combined')
ORIGINS = ('frozen_core', 'new_forecast')
QUANTITIES = (.25, .5, 1.)


def finite(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('Nonfinite calibration value')
    return value


def aware(value):
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError('Naive information availability')
    return result


def descriptor(row, other, origin):
    """Only contemporaneous forecasts/rank/quantity; never labels or outcomes."""
    if origin not in ORIGINS:
        raise ValueError('Unknown forecast origin')
    if row['action_id'] != other['action_id'] or row['day'] != other['day']:
        raise ValueError('Mismatched counterpart action')
    decision_at = aware(row['available_at'])
    for forecast in (row, other):
        available_at = aware(forecast['available_at'])
        if (forecast['valid_from'] > row['day'] or available_at > decision_at or
                available_at.date().isoformat() != row['day']):
            raise ValueError('Future decision descriptor')
    quantity = finite(row['quantity'])
    tier = row['descriptor']['tier']
    if quantity not in QUANTITIES or tier not in ('top2', 'rest'):
        raise ValueError('Unlabelled quantity or rank cohort')
    field = 'calibrated_core' if origin == 'frozen_core' else 'calibrated_support'
    return {'origin': origin, 'quantity_scale': quantity, 'tier': tier,
            'self_positive': all(finite(row[field][s]) > 0 for s in SCENARIOS),
            'other_positive': all(finite(other[field][s]) > 0 for s in SCENARIOS)}


def historical_record(row, other, origin):
    field = 'calibrated_core' if origin == 'frozen_core' else 'calibrated_support'
    return {**descriptor(row, other, origin), 'family': row['family'],
            'trade_date': row['day'], 'action_id': row['action_id'],
            'base_opportunity_id': row['action_id'].rsplit('/', 1)[0],
            'feature_available_at': row['available_at'], 'valid_from': row['valid_from'],
            'failure_probability': row['failure_probability'], 'capacity_pass': row['capacity_pass'],
            'forecasts': dict(row[field]), 'labels': row['selected_management_labels']}


def pool_key(d):
    return (d['origin'], d['quantity_scale'], d['self_positive'], d['other_positive'])


def cohort_key(d):
    return (f"{d['origin']}/q{d['quantity_scale']:g}/"
            f"self{int(d['self_positive'])}_other{int(d['other_positive'])}/{d['tier']}")


def fit_population(records, family, first, sessions, *, lookback=60, prior=100.,
                   minimum=20, min_sessions=8, local_minimum=8, local_min_sessions=5):
    """Fit only earlier eligible actions with mature exact-quantity labels.

    No selected-trade conditioning, cross-origin borrowing, cross-sign borrowing,
    interpolation of quantity outcomes, or pooling of multiple quantities.
    Unsupported local cells are identity even when their broader pool is large.
    """
    if family not in ('expanding', 'decay60'):
        raise ValueError('Unknown forecast family')
    if sessions != sorted(set(sessions)) or first not in sessions:
        raise ValueError('Invalid continuous exchange calendar')
    if min(lookback, minimum, min_sessions, local_minimum, local_min_sessions) < 1 or prior <= 0:
        raise ValueError('Invalid support contract')
    prior = finite(prior)
    index = sessions.index(first)
    allowed = set(sessions[max(0, index - 1 - lookback):max(0, index - 1)])
    cutoff = aware(first + 'T00:00:00+09:00')
    pools, cells, seen = defaultdict(list), defaultdict(list), set()
    exclusions = Counter()
    latest = None
    for r in records:
        # Exclude unobserved dates before inspecting outcome values or validity.
        if r['family'] != family or r['trade_date'] not in allowed:
            continue
        if r['origin'] not in ORIGINS or r['quantity_scale'] not in QUANTITIES or r['tier'] not in ('top2', 'rest'):
            raise ValueError('Invalid historical cohort')
        if r['valid_from'] > r['trade_date']:
            raise ValueError('Future forecast')
        feature_at = aware(r['feature_available_at'])
        if feature_at.date().isoformat() != r['trade_date'] or feature_at >= cutoff:
            raise ValueError('Future feature availability')
        identity = (r['trade_date'], r['base_opportunity_id'], r['quantity_scale'], r['origin'])
        if identity in seen:
            raise ValueError('Duplicate dependent observation within population')
        seen.add(identity)
        failure_probability = finite(r['failure_probability'])
        if not 0 <= failure_probability <= 1:
            raise ValueError('Invalid failure probability')
        if failure_probability >= .001 or not r['capacity_pass']:
            exclusions['historically_ineligible_actions_by_origin'] += 1
            continue
        for scenario in SCENARIOS:
            label = r['labels'][scenario]
            available = aware(label['label_available_at'])
            if available >= cutoff:
                exclusions[scenario + '_immature'] += 1
                continue
            if not label['label_valid']:
                exclusions[scenario + '_invalid'] += 1
                continue
            if available < feature_at:
                raise ValueError('Outcome precedes its candidate decision')
            residual = finite(label['net_value_equity_bps']) - finite(r['forecasts'][scenario])
            item = (r['trade_date'], r['base_opportunity_id'], residual)
            pools[(scenario, pool_key(r))].append(item)
            cells[(scenario, cohort_key(r))].append(item)
            latest = max(latest, available) if latest is not None else available
    detail = {}
    for origin, quantity, own, other, tier, scenario in product(
            ORIGINS, QUANTITIES, (False, True), (False, True), ('top2', 'rest'), SCENARIOS):
        d = {'origin': origin, 'quantity_scale': quantity, 'self_positive': own,
             'other_positive': other, 'tier': tier}
        pool = pools[(scenario, pool_key(d))]
        cell = cells[(scenario, cohort_key(d))]
        pool_days, cell_days = len({x[0] for x in pool}), len({x[0] for x in cell})
        # Each quantity/origin pool must contain one observation per issuer-day.
        assert len({(x[0], x[1]) for x in pool}) == len(pool)
        supported = (len(pool) >= minimum and pool_days >= min_sessions and
                     len(cell) >= local_minimum and cell_days >= local_min_sessions)
        pooled = sum(x[2] for x in pool) / (len(pool) + prior) if supported else 0.
        offset = (sum(x[2] for x in cell) + prior * pooled) / (len(cell) + prior) if supported else 0.
        detail[scenario + '/' + cohort_key(d)] = {
            'offset_bps': finite(offset), 'supported': supported,
            'pool_issuer_days': len(pool), 'pool_sessions': pool_days,
            'cell_issuer_days': len(cell), 'cell_sessions': cell_days}
    return {'family': family, 'valid_from': first,
            'valid_through': sessions[min(index + 4, len(sessions) - 1)],
            'latest_label_available_at': latest.isoformat() if latest is not None else None,
            'observed_through': max(allowed) if allowed else None,
            'lookback_sessions': lookback, 'embargo_sessions': 1, 'prior_count': prior,
            'minimum_count': minimum, 'minimum_sessions': min_sessions,
            'local_minimum_count': local_minimum, 'local_minimum_sessions': local_min_sessions,
            'target': 'Authenticated original-management isolated exact-quantity native net equity bps',
            'population': 'Historically eligible candidates; origin/quantity/sign separated; no local fallback',
            'exclusions': dict(exclusions), 'cohorts': detail}


def corrected_values(row, other, origin, proof):
    """Apply a mature frozen proof using only the candidate's decision fields."""
    if row['family'] != proof['family'] or not proof['valid_from'] <= row['day'] <= proof['valid_through']:
        raise ValueError('Calibration family/clock mismatch')
    d = descriptor(row, other, origin)
    field = 'calibrated_core' if origin == 'frozen_core' else 'calibrated_support'
    return {s: finite(row[field][s]) + proof['cohorts'][s + '/' + cohort_key(d)]['offset_bps']
            for s in SCENARIOS}


def supported(d, proof):
    return all(proof['cohorts'][s + '/' + cohort_key(d)]['supported'] for s in SCENARIOS)
