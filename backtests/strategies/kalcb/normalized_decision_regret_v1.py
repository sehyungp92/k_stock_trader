"""Research-only training-curvature-normalized paired decision loss.

This score is NOT a probability of profit or an expected return. Its sign selects
between two exact actions in research; the model cannot create an entry or fill.
"""
from collections import Counter, defaultdict
from datetime import datetime
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

PAIRS=((.25,.5),(.5,1.))
PHYSICAL=('action_notional_equity','action_quantity_last_volume',
          'action_quantity_opening_volume','action_low_volume_exit_slices')

def _spectrum(gram):
    """Keep roundoff null directions; reject non-PSD inputs beyond tolerance."""
    if not np.isfinite(gram).all():
        raise ValueError('Nonfinite training curvature')
    ev = np.linalg.eigvalsh(gram)
    scale = max(1., float(np.max(abs(ev))))
    if ev.min() < -1e-10 * scale:
        raise ValueError('Training curvature is not positive semidefinite')
    # This tolerance defines numerical rank, not a feature/outcome selection.
    ev[ev <= 1e-12 * scale] = 0.
    return ev


def normalize_penalty(x, base_weights, economic_weights, ridge_penalty=120.):
    """Match zero-score local df using only the already selected training rows.

    No fitted coefficients, validation outcomes or candidate forecasts enter
    this function. This inverts a monotone matrix diagnostic; it does not fit
    models or select a hyperparameter using held-out performance.
    """
    x = np.asarray(x, dtype=float)
    base = np.asarray(base_weights, dtype=float)
    economic = np.asarray(economic_weights, dtype=float)
    if (x.ndim != 2 or min(x.shape) == 0 or base.shape != (len(x),)
            or economic.shape != base.shape or not np.isfinite(x).all()
            or not np.isfinite(base).all() or not np.isfinite(economic).all()
            or (base <= 0).any() or (economic < 0).any()
            or not economic.sum() > 0 or not np.isfinite(ridge_penalty)
            or not ridge_penalty > 0):
        raise ValueError('Invalid training matrix, weights or ridge reference')
    r = _spectrum(x.T @ (base[:, None] * x))
    l = _spectrum(x.T @ ((economic / 4)[:, None] * x))
    target = float(np.sum(r / (r + ridge_penalty)))
    rank = int(np.sum(l > 0))
    if not 0 < target < rank:
        raise ValueError('No positive finite penalty can match reference df')
    def df(penalty):
        return float(np.sum(l / (l + penalty)))
    lo, hi = 0., float(ridge_penalty)
    for bracket_step in range(80):
        if df(hi) < target:
            break
        hi *= 2
    else:
        raise ValueError('Could not bound the normalization root')
    tolerance = 1e-10 * max(1., target)
    for root_step in range(160):
        penalty = (lo + hi) / 2
        achieved = df(penalty)
        if abs(achieved - target) <= tolerance:
            break
        if achieved > target:
            lo = penalty
        else:
            hi = penalty
    else:
        raise ValueError('Normalization root failed precision contract')
    if not np.isfinite(penalty) or penalty <= 0:
        raise ValueError('Invalid normalized penalty')
    return {'contract': 'train_only_zero_logistic_df_matches_ridge120_v1',
            'penalty': float(penalty), 'ridge_reference_penalty': float(ridge_penalty),
            'target_df': target, 'zero_score_df': achieved,
            'numerical_rank_ridge': int(np.sum(r > 0)),
            'numerical_rank_logistic': rank, 'basis_width': x.shape[1],
            'training_rows': len(x), 'root_iterations': root_step + 1,
            'bracket_iterations': bracket_step + 1,
            'df_tolerance': tolerance, 'spectrum_relative_rank_tolerance': 1e-12,
            'spectrum_relative_negative_tolerance': 1e-10}

def training_indices(records,first,sessions):
    if sessions!=sorted(set(sessions)) or first not in sessions:raise ValueError('Calendar')
    allowed=set(sessions[:max(0,sessions.index(first)-1)])
    cutoff=datetime.fromisoformat(first+'T00:00:00+09:00');seen=set();result=[]
    for i,r in enumerate(records):
        if r['day'] not in allowed:continue
        if r['id'] in seen:raise ValueError('Duplicate dependent contrast')
        seen.add(r['id'])
        if not r['target_valid'] or not r['mature_at']:continue
        mature=datetime.fromisoformat(r['mature_at'])
        if mature>=cutoff:continue
        if mature<datetime.fromisoformat(r['available_at']):raise ValueError('Maturity')
        if not np.isfinite(r['value']):raise ValueError('Target')
        result.append(i)
    return result

def objective(coef,x,target,weights,penalty=120.):
    score=x@coef
    loss=np.dot(weights,np.logaddexp(0,score)-target*score)+penalty/2*np.dot(coef,coef)
    gradient=x.T@(weights*(expit(score)-target))+penalty*coef
    return float(loss),gradient

def fit(records,first,sessions):
    rs=[records[i] for i in training_indices(records,first,sessions)]
    days=len({r['day'] for r in rs});n=len({r['base_id'] for r in rs})
    p={'first':first,'mature_sessions':days,'issuer_days':n,'training_ids':[r['id'] for r in rs],
       'observed_through':sessions[sessions.index(first)-2] if sessions.index(first)>=2 else None,
       'supported':days>=60 and n>=100,'domains':{},'score_units':'economic_weighted_log_odds_not_expected_return'}
    if not p['supported']:return p
    count=Counter(r['base_id'] for r in rs);daynames=defaultdict(set)
    for r in rs:daynames[r['day']].add(r['base_id'])
    base=np.array([n/days/len(daynames[r['day']])/count[r['base_id']] for r in rs])
    x=np.array([r['x'] for r in rs]);values=np.array([r['value'] for r in rs]);target=(values>0).astype(float)
    scale=float(np.dot(base,abs(values))/sum(base))
    if not scale>0:raise ValueError('No economic contrast')
    weight=base*abs(values)/scale
    if not np.isfinite(x).all():raise ValueError('Feature')
    normalization=normalize_penalty(x,base,weight)
    penalty=normalization["penalty"]
    result=minimize(objective,np.zeros(x.shape[1]),args=(x,target,weight,penalty),jac=True,method='L-BFGS-B',
                    options={'maxiter':2000,'ftol':1e-14,'gtol':1e-8,'maxls':50})
    # Numerical refinement of the SAME strongly convex objective. Relative
    # L-BFGS loss tolerance can stop before its absolute gradient is small.
    coef=result.x.copy();polish=0
    for polish in range(30):
        loss,gradient=objective(coef,x,target,weight,penalty)
        if max(abs(gradient))<=1e-6:break
        prob=expit(x@coef);hessian=x.T@((weight*prob*(1-prob))[:,None]*x)+penalty*np.eye(x.shape[1])
        step=np.linalg.solve(hessian,gradient);rate=1.
        while rate>=2**-30:
            candidate=coef-rate*step;candidate_loss,_=objective(candidate,x,target,weight,penalty)
            if candidate_loss<=loss-1e-4*rate*np.dot(gradient,step)+1e-12:break
            rate*=.5
        if rate<2**-30:raise ValueError('Newton refinement line search')
        coef=candidate
    _,gradient=objective(coef,x,target,weight,penalty)
    if max(abs(gradient))>1e-4:raise ValueError('Solver did not converge')
    p.update(coefficients=coef.tolist(),weights=weight.tolist(),base_weights=base.tolist(),penalty=penalty,normalization=normalization,
             training_absolute_value_scale=scale,optimization={'iterations':int(result.nit),'newton_refinement_steps':polish,'max_gradient':float(max(abs(gradient))),'converged':True})
    for low,high in PAIRS:
        subset=[r for r in rs if r['low']==low and r['high']==high]
        if len(subset)<20 or len({r['day'] for r in subset})<8:continue
        a=np.array([r['physical'] for r in subset])
        p['domains'][f'{low:g}_{high:g}']={'min':a.min(0).tolist(),'max':a.max(0).tolist(),
            'issuer_days':len(subset),'sessions':len({r['day'] for r in subset})}
    return p

def predict(row,proof):
    if row['day']<proof['first']:raise ValueError('Future fit')
    d=proof['domains'].get(f"{row['low']:g}_{row['high']:g}")
    if not proof['supported'] or not d:return {'supported':False,'score':None}
    if any(not lo<=v<=hi for v,lo,hi in zip(row['physical'],d['min'],d['max'])):return {'supported':False,'score':None}
    score=float(np.array(row['x'])@np.array(proof['coefficients']))
    if not np.isfinite(score):raise ValueError('Nonfinite score')
    return {'supported':True,'score':score}
