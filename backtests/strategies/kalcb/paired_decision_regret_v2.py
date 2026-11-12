"""Direct paired policy learning with full-cost economic mistake weights.

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
    result=minimize(objective,np.zeros(x.shape[1]),args=(x,target,weight),jac=True,method='L-BFGS-B',
                    options={'maxiter':2000,'ftol':1e-14,'gtol':1e-8,'maxls':50})
    # Numerical refinement of the SAME strongly convex objective. Relative
    # L-BFGS loss tolerance can stop before its absolute gradient is small.
    coef=result.x.copy();polish=0
    for polish in range(30):
        loss,gradient=objective(coef,x,target,weight)
        if max(abs(gradient))<=1e-6:break
        prob=expit(x@coef);hessian=x.T@((weight*prob*(1-prob))[:,None]*x)+120*np.eye(x.shape[1])
        step=np.linalg.solve(hessian,gradient);rate=1.
        while rate>=2**-30:
            candidate=coef-rate*step;candidate_loss,_=objective(candidate,x,target,weight)
            if candidate_loss<=loss-1e-4*rate*np.dot(gradient,step)+1e-12:break
            rate*=.5
        if rate<2**-30:raise ValueError('Newton refinement line search')
        coef=candidate
    _,gradient=objective(coef,x,target,weight)
    if max(abs(gradient))>1e-4:raise ValueError('Solver did not converge')
    p.update(coefficients=coef.tolist(),weights=weight.tolist(),base_weights=base.tolist(),penalty=120.,
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
