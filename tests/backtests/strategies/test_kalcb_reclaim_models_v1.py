from copy import deepcopy
from datetime import date,timedelta
import numpy as np
import pytest
from backtests.strategies.kalcb import reclaim_models_v1 as m

def row(key='2026-01-05/000001/minute90/q0.25',value=1.):
    return {'opportunity_id':key,'trade_date':key[:10],'feature_available_at':key[:10]+'T10:30:00+09:00',
        'features':dict.fromkeys(set(m.PAYOFF)|set(m.PHYSICAL),value),'guard_at_rank1':True,
        'quarter_preview':{'ready':True},'action':{'quantity':1},'parent_snapshot_eligibility':{'100':True,'125':True}}

def label(day='2026-01-05',net=10):
    return {'label_available_at':day+'T15:35:00+09:00','risk_available_at':day+'T15:35:00+09:00',
        'label_valid':True,'net_value_planned_notional_bps':net,'physical_fill_failures':[],
        'deadline_violation':False,'unresolved':False,'independent_episode_execution_pass':True,'status':'filled'}

def score(r):
    return {'opportunity_id':r['opportunity_id'],'feature_available_at':r['feature_available_at'],
        'valid_from':r['trade_date'],'notional_forecast':{'baseline':100.,'combined':90.},
        'failure_probability':.0005,'domain_pass':True,'capacity_pass':True,
        'population_eligible':True,'calibration_supported':True}

def test_no_future_or_embargo_label_read():
    sessions=['2026-01-05','2026-01-06','2026-01-07']
    rows=[row(d+'/000001/minute90/q0.25') for d in sessions]
    assert m.mature_rows(rows,{rows[0]['opportunity_id']:label()},sessions,sessions[-1],'payoff')==rows[:1]

def test_late_available_outcome_embargoed():
    r=row();l=label();l['label_available_at']='2026-01-07T10:00:00+09:00'
    assert not m.mature_rows([r],{r['opportunity_id']:l},['2026-01-05','2026-01-06','2026-01-07'],'2026-01-07','payoff')

def test_auxiliary_payoff_forbidden():
    r=row();l=label();l['auxiliary_physical_only']=True
    with pytest.raises(ValueError):m.Payoff().fit([r],{r['opportunity_id']:l})
    with pytest.raises(ValueError):m.mature_rows([r],{r['opportunity_id']:l},['2026-01-05','2026-01-06','2026-01-07'],'2026-01-07','payoff')

def test_missing_geometry_name_not_silently_imputed():
    r=row();r['features'].pop('reclaim_pullback_depth')
    with pytest.raises(ValueError):m.features([r],m.PAYOFF)

def test_none_history_imputable_but_infinity_rejected():
    r=row();r['features']['prior_afternoon_return']=None
    assert np.isnan(m.features([r],m.PAYOFF)).sum()==1
    r['features']['daily_atr_pct']=float('inf')
    with pytest.raises(ValueError):m.features([r],m.PAYOFF)

def test_domain_uses_prior_bounds_only():
    inside,proof=m.domain([row(value=1),row(value=2)],[row(value=1.5),row(value=3)])
    assert inside.tolist()==[True,False] and proof['max']==[2.]*8

def test_either_scenario_failure_counts_once():
    r=row();a=label();b=label();b['independent_episode_execution_pass']=False
    result=m.physical_outcomes({'baseline':{r['opportunity_id']:a},'combined':{r['opportunity_id']:b}},[r])
    assert len(result)==1 and result[r['opportunity_id']]['unsafe']

def test_no_order_not_inflated_risk_count():
    r=row();r['quarter_preview']['ready']=False
    with pytest.raises(ValueError):m.physical_outcomes({'baseline':{},'combined':{}},[r])

def test_unchanged_zero_failure_prior_and_strict_threshold():
    model=m.av.FeasibilityModel().fit(np.zeros((308,8)),np.zeros(308,bool))
    assert model.predict(np.zeros((1,8)))[0]==1/408>.001
    s=score(row());s['failure_probability']=.001
    assert not m.admitted(s)

@pytest.mark.parametrize('gate',['domain_pass','capacity_pass','population_eligible','calibration_supported'])
def test_every_support_gate_abstains(gate):
    s=score(row());s[gate]=False
    if gate=='calibration_supported':s['notional_forecast']=None
    assert not m.admitted(s)

def test_both_full_cost_forecasts_must_exceed_reserve():
    s=score(row());s['notional_forecast']['combined']=48.
    assert not m.admitted(s)
    s['notional_forecast']['combined']=48.0001
    assert m.admitted(s)

def test_calibration_future_poison_and_unknown_membership():
    sessions=[(date(2025,10,1)+timedelta(days=i)).isoformat() for i in range(45)]
    rr=[row(d+'/000001/minute90/q0.25',i+1) for i,d in enumerate(sessions[:30])]
    records=[score(r) for r in rr];labels={r['opportunity_id']:label(r['trade_date'],i) for i,r in enumerate(rr)};by={r['opportunity_id']:r for r in rr}
    proof=m.calibration(records,labels,by,sessions,sessions[-1],'baseline',100)
    future={'opportunity_id':sessions[-1]+'/999999/minute90/q0.25'}
    assert m.calibration(records+[future],labels,by,sessions,sessions[-1],'baseline',100)==proof
    assert proof['supported'] and proof['issuer_days']==30
    for r in rr:r['parent_snapshot_eligibility']['100']=None
    assert not m.calibration(records,labels,by,sessions,sessions[-1],'baseline',100)['supported']

def test_payoff_fit_does_not_consume_future_targets():
    rr=[row(f'2026-01-{i:02d}/000001/minute90/q0.25',i) for i in range(1,5)]
    ll={r['opportunity_id']:label(r['trade_date'],i*20) for i,r in enumerate(rr)}
    a=m.Payoff().fit(rr[:3],ll).predict(rr[3:])
    poisoned=deepcopy(ll);poisoned[rr[3]['opportunity_id']]['net_value_planned_notional_bps']=-1e20
    b=m.Payoff().fit(rr[:3],poisoned).predict(rr[3:])
    assert np.array_equal(a,b)
