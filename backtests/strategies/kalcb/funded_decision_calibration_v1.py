"""Research-only causal value calibration for an unchanged parent decision.

Prediction is a planned-notional return rate, never rescaled counterfactual PnL.
The native target includes zero-filled submitted decisions. Callers must bind
source accounts, explicit order IDs, costs and settlement clocks independently.
"""
from collections import defaultdict
from datetime import datetime
import math
import statistics


def aware(x):
    d=datetime.fromisoformat(x)
    if d.tzinfo is None:raise ValueError('Naive clock')
    return d


def finite(x):
    x=float(x)
    if not math.isfinite(x):raise ValueError('Nonfinite value')
    return x


def fit(records, first, sessions, target, prior=100.):
    """One fixed identity-shrunk affine map; identical cohorts for both targets.

    Session-balanced weights preserve issuer-day support. Scenarios, parents
    and quantities never inflate one another's independent support counts.
    Original base20-session clock is enforced by the caller. No backfill.
    """
    if target not in ('isolated','funded') or prior != 100.:
        raise ValueError('Undeclared model')
    if sessions!=sorted(set(sessions)) or first not in sessions:raise ValueError('Calendar')
    index=sessions.index(first);allowed=set(sessions[:max(0,index-1)])
    cutoff=aware(first+'T00:00:00+09:00');history=[];seen=set()
    for r in records:
        if r['day'] not in allowed:continue
        if r['id'] in seen:raise ValueError('Duplicate decision')
        seen.add(r['id'])
        if r['forecast_valid_from']>r['day'] or aware(r['available_at']).date().isoformat()!=r['day']:
            raise ValueError('Future forecast')
        if not r['target_valid'] or aware(r['mature_at'])>=cutoff:continue
        if aware(r['mature_at'])<aware(r['available_at']):raise ValueError('Early maturity')
        finite(r['prediction']);finite(r['isolated']);finite(r['funded'])
        history.append(r)
    source_sessions=len({r['day'] for r in history});issuer_days=len({r['id'] for r in history})
    result={'first':first,'observed_through':max(allowed) if allowed else None,
            'target':target,'prior':prior,'mature_sessions':source_sessions,'issuer_days':issuer_days,'cells':{}}
    if source_sessions<60 or issuer_days<100:return result
    cells=defaultdict(list)
    for r in history:cells[(r['origin'],r['quantity_scale'])].append(r)
    for (origin,q),items in cells.items():
        if len(items)<20 or len({r['day'] for r in items})<8:continue
        perday=defaultdict(int)
        for r in items:perday[r['day']]+=1
        weights=[1/perday[r['day']] for r in items]
        normal=len(items)/sum(weights);weights=[w*normal for w in weights]
        xs=[r['prediction'] for r in items];ys=[r[target] for r in items]
        # Fit y-p = intercept + slope_delta * (p-mean)/scale, with a
        # dimensionless identity prior. Restrict resulting slope to [0,1].
        mean=sum(w*x for w,x in zip(weights,xs))/sum(weights)
        scale=max(1.,math.sqrt(sum(w*(x-mean)**2 for w,x in zip(weights,xs))/sum(weights)))
        zs=[(x-mean)/scale for x in xs];res=[(y-x)/scale for x,y in zip(xs,ys)]
        a=sum(weights)+prior;b=sum(w*z for w,z in zip(weights,zs));c=sum(w*z*z for w,z in zip(weights,zs))+prior
        d=sum(w*r for w,r in zip(weights,res));e=sum(w*z*r for w,z,r in zip(weights,zs,res))
        det=a*c-b*b;delta=(a*e-b*d)/det
        delta=max(-1.,min(0.,delta));intercept=(d-b*delta)/a
        result['cells'][f'{origin}/q{q:g}']={'intercept_standardized':intercept,'slope_delta':delta,
            'mean':mean,'scale':scale,'min':min(xs),'max':max(xs),'observations':len(items),
            'sessions':len(perday),'latest_mature_at':max(r['mature_at'] for r in items),
            'training_ids':[r['id'] for r in items]}
    return result


def predict(record,proof):
    if record['day']<proof['first']:raise ValueError('Future fit')
    p=finite(record['prediction']);cell=proof['cells'].get(f"{record['origin']}/q{record['quantity_scale']:g}")
    if cell is None or not cell['min']<=p<=cell['max']:
        return {'supported':False,'value':p,'reason':'unsupported_prior_cell_or_domain'}
    value=p+cell['scale']*cell['intercept_standardized']+cell['slope_delta']*(p-cell['mean'])
    return {'supported':True,'value':finite(value),'reason':'prior_selected_decision_calibration'}


def paired_abstention(baseline,combined):
    """No entry restoration, reranking, quantity change or unsupported veto."""
    return all(x['supported'] for x in (baseline,combined)) and min(baseline['value'],combined['value'])<=0


def native_target(exposure, positions, order_events, day):
    """Exact opening submitted-decision target, retaining no-fill zero outcomes."""
    action=exposure['action'];meta=action['metadata'];oid=exposure['order_id']
    if meta['entry_route']!='first30_open_anchor' or action['qty']<=0:raise ValueError('Wrong native action')
    if exposure['decision_available_at'][:10]!=day:raise ValueError('Wrong native day')
    planned=action['qty']*finite(meta['entry_price_ref'])
    if planned<=0:raise ValueError('Invalid planned notional')
    linked=[p for p in positions if oid in p['entry_order_ids']]
    if len(linked)>1 or any(p['entry_order_ids']!=[oid] for p in linked):
        raise ValueError('Multiple-order/lifetime action requires explicit reconstruction')
    if linked:
        p=linked[0]
        if p['live_qty'] or p['entry_day']!=day:raise ValueError('Unresolved target')
        pnl=finite(p['net_pnl_krw']);mature=p['exit_fill_time'];filled=p['qty']
    else:
        events=[e for e in order_events if e.get('order_id')==oid]
        if not events:raise ValueError('Unverified no-fill order')
        # The completed audited account proves terminal flat/no pending. The
        # conservative session-close maturity avoids an early reject retry leak.
        pnl=0.;filled=0;mature=day+'T15:35:00+09:00'
    return {'value':10000*pnl/planned,'net_pnl_krw':pnl,'planned_notional_krw':planned,
            'submitted_quantity':action['qty'],'filled_quantity':filled,'mature_at':mature,
            'order_id':oid,'lifetime_id':linked[0]['position_id'] if linked else None,
            'funded_counterfactual_portfolio_value':False}
