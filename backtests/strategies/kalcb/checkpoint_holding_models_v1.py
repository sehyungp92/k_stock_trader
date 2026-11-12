"""Two frozen causal holding-value comparisons; no orders or portfolio PnL."""
from collections import Counter
from datetime import datetime
import hashlib
import json
import math

SCENARIOS = ('baseline', 'combined')
RECIPES = ('own_state', 'peer_context')
OWN = ('post_entry_close_location', 'post_entry_path_efficiency',
       'post_entry_range_atr', 'post_entry_return_atr',
       'post_entry_volume_ratio', 'vwap_extension_atr')
STATE = ('log_held_notional', 'log_held_to_last_bar_volume',
         'held_to_planned_quantity', 'pending_to_held_quantity',
         'marked_entry_return_atr')
CONTEXT = ('peer_market_vwap_breadth_change', 'peer_sector_vwap_breadth_change',
           'relative_post_entry_market_return_atr', 'relative_post_entry_sector_return_atr')
COLUMNS = OWN + STATE + CONTEXT


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def allowed_days(sessions, first):
    if sessions != sorted(set(sessions)) or first not in sessions:
        raise ValueError('Invalid original calendar')
    return set(sessions[:max(0, sessions.index(first)-1)])


def mature(day, available, allowed):
    if day not in allowed or not available:
        return False
    at = datetime.fromisoformat(available)
    if at.tzinfo is None:
        raise ValueError('Naive outcome clock')
    return at.date().isoformat() <= max(allowed)


def feature_vector(row):
    groups = (row['own_features'], row['state_features'], row['context_features'])
    if any(g is None for g in groups):
        return None
    merged = {k: v for g in groups for k, v in g.items()}
    if any(k not in merged for k in COLUMNS):
        raise ValueError('Missing registered feature')
    values = [merged[k] for k in COLUMNS]
    if any(v is None or not math.isfinite(float(v)) for v in values):
        return None
    return list(map(float, values))


def causal_eligible(row):
    return (row['eligibility'] == 'eligible' and row['held_quantity'] > 0
            and row['pending_buy_quantity'] == 0 and feature_vector(row) is not None)


def grouped_weights(rows):
    """One weight per issuer-day, equal session weight; q/scenarios aren't trials."""
    import numpy as np
    quantities = Counter((r['day'], r['symbol']) for r in rows)
    names = Counter(d for d, _ in quantities)
    values = np.array([1 / quantities[r['day'], r['symbol']] / names[r['day']] for r in rows])
    return values * len(quantities) / values.sum()


def fit(groups, sessions, first, through):
    import numpy as np
    allowed = allowed_days(sessions, first)
    train = [g for g in groups if all(causal_eligible(g[s]) and
             g[s]['current_holding_label_valid'] and
             mature(g[s]['day'], g[s]['label_available_at'], allowed) for s in SCENARIOS)]
    rows = [g['baseline'] for g in train]
    proof = {'valid_from': first, 'valid_through': through, 'status': 'unsupported',
             'mature_sessions': len({r['day'] for r in rows}),
             'issuer_days': len({(r['day'], r['symbol']) for r in rows}),
             'dependent_quantities': len(rows), 'embargo_cutoff': max(allowed, default=None),
             'training_keys_hash': signature([r['opportunity_id'] for r in rows]),
             'models': {}, 'quantity_support': {}}
    if proof['mature_sessions'] < 60:
        return proof
    weights = grouped_weights(rows)
    proof['training_weights_hash'] = signature(weights.tolist())
    for q in (.25, .5, 1.):
        subset = [r for r in rows if r['quantity_scale'] == q]
        n = len({(r['day'], r['symbol']) for r in subset}); d = len({r['day'] for r in subset})
        proof['quantity_support'][str(q)] = {'issuer_days': n, 'sessions': d, 'supported': n >= 20 and d >= 8}
    for s in SCENARIOS:
        x = np.array([feature_vector(g[s]) for g in train])
        y = np.array([g[s]['current_holding_advantage_bps'] for g in train])
        if not np.isfinite(y).all():
            raise ValueError('Invalid payoff target')
        center = np.average(x, axis=0, weights=weights)
        scale = np.sqrt(np.average((x-center)**2, axis=0, weights=weights))
        scale[scale < 1e-12] = 1.
        common = {'center': center.tolist(), 'scale': scale.tolist(),
                  'minimum': x.min(axis=0).tolist(), 'maximum': x.max(axis=0).tolist(),
                  'target_hash': signature(y.tolist()), 'coefficients': {}}
        z = (x-center)/scale
        for recipe in RECIPES:
            n = len(OWN+STATE) if recipe == 'own_state' else len(COLUMNS)
            design = np.column_stack((np.ones(len(x)), z[:, :n]))
            penalty = np.diag([0.] + [120.] * n)
            beta = np.linalg.solve(design.T @ (weights[:, None]*design) + penalty,
                                   design.T @ (weights*y))
            common['coefficients'][recipe] = beta.tolist()
        proof['models'][s] = common
    proof['latest_maturity'] = max(g[s]['label_available_at'] for g in train for s in SCENARIOS)
    proof['status'] = 'supported'
    return proof


def forecast(pair, proof, recipe):
    if recipe not in RECIPES:
        raise ValueError('Unregistered recipe')
    if set(pair) != set(SCENARIOS):
        return {'supported': False, 'reason': 'missing_scenario_holding'}
    if any(not proof['valid_from'] <= r['day'] <= proof['valid_through'] for r in pair.values()):
        raise ValueError('Stale or future model')
    if not all(causal_eligible(r) for r in pair.values()):
        return {'supported': False, 'reason': 'no_holding_pending_parent_exit_or_features'}
    if proof['status'] != 'supported':
        return {'supported': False, 'reason': 'mature_session_support'}
    values = {}; outside = {}
    for s in SCENARIOS:
        row = pair[s]; q = str(float(row['quantity_scale']))
        if not proof['quantity_support'].get(q, {}).get('supported'):
            return {'supported': False, 'reason': 'quantity_support'}
        x = feature_vector(row); model = proof['models'][s]
        outside[s] = [COLUMNS[i] for i, v in enumerate(x)
                      if v < model['minimum'][i]-1e-12 or v > model['maximum'][i]+1e-12]
        n = len(OWN+STATE) if recipe == 'own_state' else len(COLUMNS)
        z = [1.] + [(v-m)/sd for v, m, sd in zip(x[:n], model['center'][:n], model['scale'][:n])]
        values[s] = sum(a*b for a, b in zip(z, model['coefficients'][recipe]))
    if any(outside.values()):
        return {'supported': False, 'reason': 'prior_feature_envelope', 'outside': outside}
    return {'supported': True, 'values': values}


def risk_proof(records, sessions, first):
    """Count unique actual issuer-day exposures, retain failures and maturity."""
    allowed = allowed_days(sessions, first); grouped = {}
    for r in records:
        if not r['active_no_pending_both'] or not mature(r['day'], r['risk_available_at'], allowed):
            continue
        key = (r['day'], r['symbol'])
        grouped[key] = grouped.get(key, False) or r['unsafe']
    unsafe = sum(grouped.values()); n = len(grouped); days = len({k[0] for k in grouped})
    # The inherited zero-failure branch is used only when its premise holds.
    # A nonzero failure requires a separately qualified conditional risk fit.
    return {'issuer_days': n, 'mature_sessions': days, 'unsafe_issuer_days': unsafe,
            'probability': 1/(n+100) if unsafe == 0 else None,
            'supported': days >= 60 and unsafe == 0,
            'training_keys_hash': signature(sorted(grouped)),
            'interpretation': 'Conditional research estimate, not a certified true-risk bound.'}


def risk_pass(proof):
    return proof['supported'] and proof['probability'] < .001


def calibration(history, sessions, first, scenario):
    import numpy as np
    allowed = allowed_days(sessions, first); points = []; seen = set()
    for r in history:
        if r['day'] not in allowed:
            continue
        if r['valid_from'] > r['day']:
            raise ValueError('Nonforward calibration forecast')
        if not r['matched_predecision_both'] or not r['risk_pass'] or not r['raw']['supported']:
            continue
        if not all(r['labels'][s]['valid'] and mature(r['day'], r['labels'][s]['available_at'], allowed) for s in SCENARIOS):
            continue
        key = (r['day'], r['symbol'])
        if key in seen:
            raise ValueError('Duplicate selected issuer-day calibration')
        seen.add(key)
        points.append((key, r['raw']['values'][scenario], r['labels'][scenario]['value']))
    n = len(points); days = len({p[0][0] for p in points})
    result = {'supported': n >= 20 and days >= 8, 'issuer_days': n, 'sessions': days,
              'center': 0., 'offset': 0., 'slope': 1., 'prior': 100.,
              'points_hash': signature(points), 'embargo_cutoff': max(allowed, default=None)}
    if result['supported']:
        x = np.array([p[1] for p in points]); y = np.array([p[2] for p in points])
        center = float(x.mean()); dx = x-center; variance = float(np.mean(dx*dx))
        result['center'] = center; result['offset'] = float((y-center).sum()/(n+100.))
        if variance > 1e-12:
            result['slope'] = float(np.clip((dx @ (y-center)+100.*variance)/(dx @ dx+100.*variance), 0., 1.))
    return result


def calibrated(raw, maps, risk):
    if not raw['supported']:
        return {'supported': False, 'reason': raw['reason']}
    if not risk_pass(risk):
        return {'supported': False, 'reason': 'physical_risk_support_or_gate'}
    if not all(maps[s]['supported'] for s in SCENARIOS):
        return {'supported': False, 'reason': 'selected_forward_calibration_support'}
    values = {s: maps[s]['center'] + maps[s]['offset'] + maps[s]['slope']*(raw['values'][s]-maps[s]['center']) for s in SCENARIOS}
    return {'supported': True, 'values': values, 'positive_both': min(values.values()) > 0}


def native_row(comparison, context):
    """Actual integer state, not the isolated reference's quantity or price."""
    p = comparison['native']; held = p['held_quantity']; planned = p['planned_quantity']
    close = context['completed_close']; volume = context['consumed_bars'][-1]['volume']
    entry = p['mean_entry_fill_price']; atr = context['daily_atr_pct']
    features = None
    if held > 0 and planned > 0 and entry and entry > 0 and volume > 0 and atr > 0:
        features = {'log_held_notional': math.log(held*close),
                    'log_held_to_last_bar_volume': math.log1p(held/volume),
                    'held_to_planned_quantity': held/planned,
                    'pending_to_held_quantity': p['pending_buy_quantity']/held,
                    'marked_entry_return_atr': (close/entry-1)/atr}
    return {'opportunity_id': comparison['action_id'], 'day': comparison['day'], 'symbol': comparison['symbol'],
            'quantity_scale': comparison['quantity_scale'], 'held_quantity': held,
            'planned_quantity': planned, 'pending_buy_quantity': p['pending_buy_quantity'],
            'held_notional_krw': held*close, 'checkpoint_at': context['available_at'],
            'eligibility': 'parent_exit' if comparison['parent_exit_already_decided'] else 'eligible',
            'own_features': context['own_features'], 'context_features': context['context_features'],
            'state_features': features}


def exact_causal_match(native, reference):
    keys = ('held_quantity', 'planned_quantity', 'pending_buy_quantity', 'checkpoint_at')
    if not all(native[k] == reference[k] for k in keys):
        return False
    x = feature_vector(native); y = feature_vector(reference)
    return x is not None and y is not None and all(math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10) for a,b in zip(x,y))
