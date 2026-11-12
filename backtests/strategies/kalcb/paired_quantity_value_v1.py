"""Research-only learning of adjacent quantity value differences.

The potential basis is exactly the parent's q * [1, frozen_state_map(x)].
Fit the paired difference loss directly. A difference of two fits on identical
design matrices is an algebraic alias, so that is explicitly not implemented.
"""
from collections import Counter,defaultdict
from datetime import datetime
import numpy as np

PAIRS=((.25,.5),(.5,1.))
PHYSICAL=('action_notional_equity','action_quantity_last_volume',
          'action_quantity_opening_volume','action_low_volume_exit_slices')

def potential_basis(quantity, transformed):
    q=np.asarray(quantity,dtype=float);z=np.asarray(transformed,dtype=float)
    if z.ndim!=2 or q.shape!=(len(z),) or not np.isfinite(z).all() or not np.isfinite(q).all():
        raise ValueError('Invalid potential matrix')
    if any(x not in (.25,.5,1.) for x in q):raise ValueError('Unknown quantity')
    return q[:,None]*np.column_stack([np.ones(len(z)),z])

def make_contrast(low,high,low_basis,high_basis,scenario):
    if (low['q'],high['q']) not in PAIRS or low['base_id']!=high['base_id'] or low['day']!=high['day']:
        raise ValueError('Unpaired issuer/day/quantity')
    if low['available_at']!=high['available_at'] or low['source']!=high['source']:
        raise ValueError('Different causal states')
    if high['qty']<=low['qty'] or high['notional']<=low['notional']:
        return None  # Explicit integer-dose alias; retain original source rows.
    valid=all(r['labels'][s]['label_valid'] for r in (low,high) for s in ('baseline','combined'))
    mature=max((r['labels'][s]['label_available_at'] for r in (low,high) for s in ('baseline','combined')
                if r['labels'][s]['label_available_at'] is not None),default=None)
    return {'id':low['base_id']+f"/q{low['q']:g}_q{high['q']:g}",'base_id':low['base_id'],
            'day':low['day'],'available_at':low['available_at'],'low':low['q'],'high':high['q'],
            'target_valid':valid,'mature_at':mature,
            'value':high['labels'][scenario]['net_value_equity_bps']-low['labels'][scenario]['net_value_equity_bps'] if valid else None,
            'x':np.asarray(high_basis)-np.asarray(low_basis),
            'physical':[high['features'][k]-low['features'][k] for k in PHYSICAL]}

def training_indices(records,first,sessions):
    if sessions!=sorted(set(sessions)) or first not in sessions:raise ValueError('Invalid calendar')
    allowed=set(sessions[:max(0,sessions.index(first)-1)])
    cutoff=datetime.fromisoformat(first+'T00:00:00+09:00');seen=set();use=[]
    for i,r in enumerate(records):
        if r['day'] not in allowed:continue
        if r['id'] in seen:raise ValueError('Duplicate dependent contrast')
        seen.add(r['id'])
        if not r['target_valid'] or r['mature_at'] is None:continue
        if datetime.fromisoformat(r['mature_at'])>=cutoff:continue
        if datetime.fromisoformat(r['mature_at'])<datetime.fromisoformat(r['available_at']):raise ValueError('Immature action')
        if not np.isfinite(r['value']):raise ValueError('Invalid target')
        use.append(i)
    return use

def fit(records,first,sessions):
    use=training_indices(records,first,sessions);selected=[records[i] for i in use]
    days=len({r['day'] for r in selected});issuers=len({r['base_id'] for r in selected})
    proof={'first':first,'mature_sessions':days,'issuer_days':issuers,'observed_through':sessions[sessions.index(first)-2] if sessions.index(first)>=2 else None,
           'supported':days>=60 and issuers>=100,'domains':{},'training_ids':[r['id'] for r in selected]}
    if not proof['supported']:return proof
    # One total issuer-day weight across adjacent contrasts, then equal days.
    perbase=Counter(r['base_id'] for r in selected);perday=defaultdict(set)
    for r in selected:perday[r['day']].add(r['base_id'])
    w=np.asarray([1/perbase[r['base_id']]/len(perday[r['day']]) for r in selected]);w*=issuers/w.sum()
    x=np.asarray([r['x'] for r in selected]);y=np.asarray([r['value'] for r in selected])
    if not np.isfinite(x).all():raise ValueError('Nonfinite features')
    # Fixed inherited ridge penalty; no intercept independent of q difference.
    coef=np.linalg.solve(x.T@(w[:,None]*x)+120.*np.eye(x.shape[1]),x.T@(w*y))
    proof.update(coefficients=coef.tolist(),weights=w.tolist(),penalty=120.)
    for low,high in PAIRS:
        rs=[r for r in selected if r['low']==low and r['high']==high]
        if len(rs)<20 or len({r['day'] for r in rs})<8:continue
        a=np.asarray([r['physical'] for r in rs]);ys=np.asarray([r['value'] for r in rs]);idx=[i for i,r in enumerate(selected) if r['low']==low and r['high']==high]
        ww=w[idx]
        proof['domains'][f'{low:g}_{high:g}']={'min':a.min(axis=0).tolist(),'max':a.max(axis=0).tolist(),
            'issuer_days':len(rs),'sessions':len({r['day'] for r in rs}),
            'mean_control':float(np.sum(ww*ys)/(np.sum(ww)+100.))}
    return proof

def predict(row,proof,recipe):
    if recipe not in ('paired_state','pair_mean'):raise ValueError('Undeclared recipe')
    if row['day']<proof['first']:raise ValueError('Future fit')
    d=proof['domains'].get(f"{row['low']:g}_{row['high']:g}")
    if not proof['supported'] or d is None:return {'supported':False,'value':None}
    if any(not low<=v<=high for v,low,high in zip(row['physical'],d['min'],d['max'])):
        return {'supported':False,'value':None}
    value=float(np.asarray(row['x'])@np.asarray(proof['coefficients'])) if recipe=='paired_state' else d['mean_control']
    if not np.isfinite(value):raise ValueError('Nonfinite forecast')
    return {'supported':True,'value':value}
