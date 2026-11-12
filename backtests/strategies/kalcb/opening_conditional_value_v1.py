"""Prior-only opening value correction in exact planned-notional units.

This is an economic research model, not a physical risk model or a policy.
Its two components are frozen before fitting. No calendar/native imports.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime
import hashlib
import json
import math

QUANTITIES = (.25, .5, 1.)
SCENARIOS = ('baseline', 'combined')
FEATURE_NAMES = ('log_atr', 'gap_atr', 'vwap_extension_atr', 'range_atr',
                 'relative_volume_log', 'path_efficiency', 'above_vwap_share',
                 'sector_breadth', 'market_relative_return_atr', 'return20_atr',
                 'last15_return_atr', 'acceleration_atr', 'quantity_scale',
                 'forecast_notional_bps', 'log_atr_x_efficiency', 'extension_x_vwap_share')
REQUIRED = ('daily_atr_pct','first30_gap','first30_vwap_ret','first30_range_atr',
            'first30_rel_volume_log','path_efficiency','path_above_vwap_share',
            'sector_intraday_breadth','market_adjusted_first30_ret','daily_return_20d',
            'path_last15_return','path_return_acceleration')


def signature(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def features(row, scenario):
    f = row['features']
    if any(k not in f for k in REQUIRED): raise ValueError('Missing declared causal feature')
    if any(f[k] is None for k in REQUIRED): return None
    if not all(math.isfinite(float(f[k])) for k in REQUIRED): return None
    atr = float(f['daily_atr_pct'])
    if atr <= 0 or row['notional_equity'] <= 0: return None
    logatr = math.log(atr)
    extension = float(f['first30_vwap_ret'])/atr
    efficiency = float(f['path_efficiency'])
    above = float(f['path_above_vwap_share'])
    return [logatr, f['first30_gap']/atr, extension, f['first30_range_atr'],
        f['first30_rel_volume_log'], efficiency, above, f['sector_intraday_breadth'],
        f['market_adjusted_first30_ret']/atr, f['daily_return_20d']/atr,
        f['path_last15_return']/atr, f['path_return_acceleration']/atr,
        row['quantity_scale'], row['calibrated_core'][scenario]/row['notional_equity'],
        logatr*efficiency, extension*above]


def eligible(row):
    return row['physical_risk_pass'] and row['capacity_pass'] and min(row['calibrated_core'].values()) > 0


def mature_rows(rows, sessions, first):
    allowed = set(sessions[:max(0, sessions.index(first)-1)])
    cutoff = datetime.fromisoformat(first+'T00:00:00+09:00')
    return [r for r in rows if r['day'] in allowed and eligible(r)
        and all(r['labels'][s]['label_valid'] and r['labels'][s].get('label_available_at')
                and datetime.fromisoformat(r['labels'][s]['label_available_at']) < cutoff
                for s in SCENARIOS)
        and all(features(r,s) is not None for s in SCENARIOS)]


def weights(rows):
    """One total weight per issuer-day, then balance sessions; not iid counts."""
    import numpy as np
    counts = Counter(r['base_id'] for r in rows)
    names = Counter(day for day, _ in {(r['day'],r['base_id']) for r in rows})
    w = np.array([1/counts[r['base_id']]/names[r['day']] for r in rows], dtype=float)
    return w*len(counts)/w.sum()


def fit(rows, sessions, first, last, *, minimum_sessions=60, minimum_issuer_days=100):
    import numpy as np
    train = mature_rows(rows, sessions, first)
    proof = {'valid_from':first,'valid_through':last,'feature_names':FEATURE_NAMES,
        'training_sessions':len({r['day'] for r in train}), 'training_issuer_days':len({r['base_id'] for r in train}),
        'training_actions':len(train), 'training_keys_sha256':signature([r['action_id'] for r in train]),
        'observed_through':max((r['day'] for r in train), default=None),
        'latest_maturity':max((r['labels'][s]['label_available_at'] for r in train for s in SCENARIOS), default=None),
        'status':'unsupported', 'models':{}, 'quantity_support':{},
        'scope':'Selected-management isolated exact-quantity full-cost labels; genuine forward core residuals, paired scenario maturity; physical model unchanged.'}
    if proof['training_sessions'] < minimum_sessions or proof['training_issuer_days'] < minimum_issuer_days:
        return proof
    w = weights(train)
    proof['training_weights_sha256'] = signature(w.tolist())
    for q in QUANTITIES:
        subset = [r for r in train if r['quantity_scale']==q]
        proof['quantity_support'][str(q)] = {'issuer_days':len({r['base_id'] for r in subset}),
            'sessions':len({r['day'] for r in subset}),
            'supported':len({r['base_id'] for r in subset})>=20 and len({r['day'] for r in subset})>=8}
    qmatrix = np.array([[float(r['quantity_scale']==q) for q in QUANTITIES] for r in train])
    for scenario in SCENARIOS:
        x = np.array([features(r,scenario) for r in train],dtype=float)
        mean = np.average(x,axis=0,weights=w)
        sd = np.sqrt(np.average((x-mean)**2,axis=0,weights=w))
        sd[sd<1e-12] = 1.
        y = np.array([(r['labels'][scenario]['net_value_equity_bps']-r['calibrated_core'][scenario])/r['notional_equity'] for r in train])
        proof['models'][scenario] = {'center':mean.tolist(),'scale':sd.tolist(),
            'minimum':x.min(axis=0).tolist(),'maximum':x.max(axis=0).tolist(),
            'training_target_sha256':signature(y.tolist()), 'coefficients':{}}
        for recipe in ('unit_mean','conditional'):
            z = qmatrix if recipe=='unit_mean' else np.column_stack((qmatrix,(x-mean)/sd))
            penalty = np.array([100.]*3 + ([] if recipe=='unit_mean' else [120.]*len(FEATURE_NAMES)))
            beta = np.linalg.solve(z.T@(w[:,None]*z)+np.diag(penalty), z.T@(w*y))
            proof['models'][scenario]['coefficients'][recipe] = beta.tolist()
    proof['status'] = 'supported'
    return proof


def forecast(row, proof, recipe):
    if recipe not in ('unit_mean','conditional'): raise ValueError('Unregistered recipe')
    if not proof['valid_from'] <= row['day'] <= proof['valid_through']: raise ValueError('Stale/future fit')
    if not eligible(row): return {'supported':False,'reason':'original_physical_capacity_or_value_gate'}
    if proof['status'] != 'supported': return {'supported':False,'reason':'mature_forward_support'}
    if not proof['quantity_support'][str(row['quantity_scale'])]['supported']:
        return {'supported':False,'reason':'exact_quantity_support'}
    values, domain = {}, {}
    for scenario in SCENARIOS:
        x = features(row,scenario)
        if x is None: return {'supported':False,'reason':'missing_features'}
        model = proof['models'][scenario]
        outside = [FEATURE_NAMES[i] for i,v in enumerate(x) if v < model['minimum'][i]-1e-12 or v > model['maximum'][i]+1e-12]
        domain[scenario] = outside
        # The same domain applies to both components to isolate context value.
        if outside: continue
        z = [float(row['quantity_scale']==q) for q in QUANTITIES]
        if recipe=='conditional': z += [(v-m)/s for v,m,s in zip(x,model['center'],model['scale'])]
        correction = sum(a*b for a,b in zip(z,model['coefficients'][recipe]))
        values[scenario] = {'original_notional_bps':row['calibrated_core'][scenario]/row['notional_equity'],
            'correction_notional_bps':correction,
            'corrected_notional_bps':row['calibrated_core'][scenario]/row['notional_equity']+correction,
            'corrected_equity_bps':row['calibrated_core'][scenario]+correction*row['notional_equity']}
    if any(domain.values()): return {'supported':False,'reason':'prior_feature_envelope','outside':domain}
    return {'supported':True,'values':values, 'positive_both':all(v['corrected_notional_bps']>0 for v in values.values())}
