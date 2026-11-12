"""Regularised calibration on mature, feasible, out-of-fold action forecasts."""
from datetime import datetime


def score_support(row):
    return 'positive' if row['baseline_value_bps'] > 0 and row['combined_value_bps'] > 0 else 'nonpositive'


def fit_support_offsets(observations, labels, sessions, first, *, lookback=60, prior_count=100,
                        minimum_count=20, minimum_sessions=8):
    """Separate raw-score support by quantity; sparse groups shrink to identity.

    Each quantity cohort counts unique stock-days, with a separate session floor.
    No positive-only residual is extrapolated into the negative-score population.
    """
    stop=max(0,sessions.index(first)-1);allowed=set(sessions[max(0,stop-lookback):stop])
    cutoff=datetime.fromisoformat(first+'T00:00:00+09:00');offsets={};details={};latest=None
    for scenario in ('baseline','combined'):
        for quantity in (.25,.5,1.):
            for support in ('positive','nonpositive'):
                errors={};dates=set()
                for r in observations:
                    if (r['trade_date'] not in allowed or r['quantity_scale']!=quantity or score_support(r)!=support
                            or r['failure_probability']>=.001 or not r['capacity_ok']):continue
                    label=labels[scenario][r['opportunity_id']]
                    if not label['label_valid'] or datetime.fromisoformat(label['label_available_at'])>=cutoff:continue
                    key=r['opportunity_id']
                    if key in errors:raise ValueError('Duplicate calibration action')
                    errors[key]=label['net_value_equity_bps']-r[scenario+'_value_bps'];dates.add(r['trade_date'])
                    latest=max(latest or label['label_available_at'],label['label_available_at'])
                value=sum(errors.values())/(len(errors)+prior_count) if len(errors)>=minimum_count and len(dates)>=minimum_sessions else 0.
                offsets[(scenario,quantity,support)]=value
                details[f'{scenario}/q{quantity:g}/{support}']={'stock_days':len(errors),'sessions':len(dates),'offset_bps':value}
    return offsets,{'valid_from':first,'observed_through':max(allowed) if allowed else None,
        'latest_label_available_at':latest,'lookback_sessions':lookback,'embargo_sessions':1,
        'prior_count':prior_count,'minimum_count':minimum_count,'minimum_sessions':minimum_sessions,'cohorts':details}


def fit_affine(observations, labels, sessions, first, *, lookback=60, prior_count=100, minimum_count=20):
    import numpy as np
    stop=max(0,sessions.index(first)-1)
    allowed=set(sessions[max(0,stop-lookback):stop])
    cutoff=datetime.fromisoformat(first+'T00:00:00+09:00')
    models={};proof={'valid_from':first,'lookback_sessions':lookback,'embargo_sessions':1,
        'prior_count':prior_count,'minimum_count':minimum_count,'observed_through':max(allowed) if allowed else None,
        'latest_label_available_at':None,'cohorts':{}}
    for scenario in ('baseline','combined'):
        data=[]
        for r in observations:
            if r['trade_date'] not in allowed or r['failure_probability']>=.001 or not r['capacity_ok']:continue
            label=labels[scenario][r['opportunity_id']]
            if not label['label_valid'] or datetime.fromisoformat(label['label_available_at'])>=cutoff:continue
            q=r['quantity_scale']
            if q not in (.25,.5,1.):raise ValueError('Unsupported action quantity')
            data.append((r, label, q))
        unique=len({r['trade_date']+'/'+r['opportunity_id'].split('/')[1] for r,_,_ in data})
        model={'center':0.,'scale':1.,'coefficients':[0.]*5,'stock_days':unique,'observations':len(data)}
        if unique>=minimum_count:
            q=np.array([v[2] for v in data]);x=np.array([r[scenario+'_value_bps']/a for r,_,a in data]);y=np.array([l['net_value_equity_bps']/a for _,l,a in data]);w=q*q
            if not np.isfinite(np.r_[x,y]).all():raise ValueError('Nonfinite calibration observations')
            center=float(np.average(x,weights=w));scale=max(1.,float(np.sqrt(np.average((x-center)**2,weights=w))))
            z=np.column_stack([np.ones(len(x)),q==.25,q==.5,q==1,(x-center)/scale])
            # Identity prior: predict original value plus a shrunk residual.
            # q^2 makes the fit minimize errors in reference-equity cash units.
            penalty=np.diag([prior_count,prior_count*2,prior_count*2,prior_count*2,prior_count])
            lhs=z.T@(w[:,None]*z)+penalty;rhs=z.T@(w*(y-x))
            coef=np.linalg.solve(lhs,rhs)
            if coef[-1]<-scale:
                coef[-1]=-scale
                coef[:-1]=np.linalg.solve(lhs[:-1,:-1],rhs[:-1]-lhs[:-1,-1]*coef[-1])
            model.update(center=center,scale=scale,coefficients=coef.tolist())
        models[scenario]=model
        latest=max((l['label_available_at'] for _,l,_ in data),default=None)
        if latest:proof['latest_label_available_at']=max(proof['latest_label_available_at'] or latest,latest)
        proof['cohorts'][scenario]={**model,'slope':1+model['coefficients'][-1]/model['scale'],
            'negative_forecast_observations':sum(r[scenario+'_value_bps']<=0 for r,_,_ in data)}
    return models,proof


def affine_value(model, forecast, quantity):
    if quantity not in (.25,.5,1.):raise ValueError('Unsupported action quantity')
    c=model['coefficients'];group={.25:1,.5:2,1.:3}[quantity]
    return forecast+quantity*(c[0]+c[group]+c[-1]*(forecast/quantity-model['center'])/model['scale'])
