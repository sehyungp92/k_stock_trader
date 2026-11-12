from datetime import date,timedelta
from copy import deepcopy
import pytest
from backtests.strategies.kalcb.value_calibration import fit_affine,affine_value


def fixture():
    days=[(date(2025,1,1)+timedelta(days=i)).isoformat() for i in range(50)]
    rows=[];labels={'baseline':{},'combined':{}}
    for i,d in enumerate(days):
        for q in (.25,.5,1):
            key=d+'/A/q'+str(q);x=(i%10-5)*q
            rows.append(dict(trade_date=d,opportunity_id=key,quantity_scale=q,baseline_value_bps=x,
                combined_value_bps=x-1,failure_probability=0.,capacity_ok=True))
            for s in labels:labels[s][key]=dict(label_valid=True,label_available_at=d+'T15:35:00+09:00',net_value_equity_bps=2*x+3*q)
    return days,rows,labels


def test_affine_can_use_negative_forecasts_but_not_future_or_delayed_outcomes():
    days,rows,labels=fixture()
    for s in labels:labels[s][rows[0]['opportunity_id']]['label_available_at']=days[45]+'T15:35:00+09:00'
    model,proof=fit_affine(rows,labels,days,days[40])
    assert proof['cohorts']['baseline']['negative_forecast_observations']>0
    for r in rows:
        if r['trade_date']>=days[39] or r['opportunity_id']==rows[0]['opportunity_id']:
            for s in labels:labels[s][r['opportunity_id']]['net_value_equity_bps']=1e12
    assert fit_affine(rows,labels,days,days[40])==(model,proof)
    assert affine_value(model['baseline'],-1.,1.)>-1


def test_empty_history_preserves_raw_forecasts_and_slope_is_monotone():
    days,rows,labels=fixture();model,_=fit_affine([],labels,days,days[40])
    assert affine_value(model['baseline'],-10,.5)==-10
    for s in labels:
        for k,v in labels[s].items():v['net_value_equity_bps']=-1e5
    model,proof=fit_affine(rows,labels,days,days[40])
    for q in (.25,.5,1):
        assert affine_value(model['baseline'],10,q)>=affine_value(model['baseline'],-10,q)-1e-8


def test_risk_and_capacity_are_known_eligibility_not_future_fill_success():
    days,rows,labels=fixture()
    model,_=fit_affine(rows,labels,days,days[40])
    poisoned=deepcopy(rows)
    for r in poisoned:r['capacity_ok']=False
    empty,_=fit_affine(poisoned,labels,days,days[40])
    assert affine_value(empty['baseline'],3,.5)==3
    assert model!=empty
