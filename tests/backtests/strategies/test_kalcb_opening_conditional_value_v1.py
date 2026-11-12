from copy import deepcopy
from datetime import date, timedelta

import pytest

from backtests.strategies.kalcb import opening_conditional_value_v1 as model


def records():
    sessions = [(date(2025,1,1)+timedelta(days=i)).isoformat() for i in range(90)]
    result=[]
    for i,day in enumerate(sessions):
        for n in range(3):
            for q in (.25,.5,1.):
                f = {k:.1 for k in model.REQUIRED}
                f.update(daily_atr_pct=.02+i*.0001, path_efficiency=.5, path_above_vwap_share=.8)
                result.append(dict(action_id=f'{day}/{n}/q{q}',base_id=f'{day}/{n}',day=day,
                    quantity_scale=q,notional_equity=.2*q, features=f, physical_risk_pass=True,
                    capacity_pass=True, calibrated_core={s:10*q for s in model.SCENARIOS},
                    labels={s:dict(label_valid=True,label_available_at=day+'T16:00:00+09:00',
                                    net_value_equity_bps=5*q) for s in model.SCENARIOS}))
    return sessions,result


def test_future_and_embargo_labels_cannot_change_fit():
    sessions, rows=records();first=sessions[70]
    a=model.fit(rows,sessions,first,sessions[80]); b=deepcopy(rows)
    for r in b:
        if r['day']>=sessions[69]:
            for label in r['labels'].values(): label['net_value_equity_bps']=1e9
    assert model.fit(b,sessions,first,sessions[80])==a
    assert a['observed_through']==sessions[68]


def test_late_maturity_and_invalid_cash_excluded():
    sessions,rows=records(); r=rows[0]
    r['labels']['combined']['label_available_at']=sessions[80]+'T16:00:00+09:00'
    rows[1]['labels']['combined']['label_valid']=False
    found=model.mature_rows(rows,sessions,sessions[70])
    assert r not in found and rows[1] not in found


def test_dependent_quantities_do_not_multiply_support():
    sessions,rows=records();proof=model.fit(rows,sessions,sessions[70],sessions[80])
    assert proof['training_actions']==proof['training_issuer_days']*3
    assert model.weights(model.mature_rows(rows,sessions,sessions[70])).sum()==pytest.approx(proof['training_issuer_days'])


def test_domain_and_unsupported_maps_abstain():
    sessions,rows=records();proof=model.fit(rows,sessions,sessions[70],sessions[80])
    row=deepcopy(rows[0]);row['day']=sessions[70];row['features']['daily_atr_pct']=100.
    assert model.forecast(row,proof,'conditional')['reason']=='prior_feature_envelope'
    small=model.fit(rows,sessions,sessions[10],sessions[20]);row['day']=sessions[10]
    assert model.forecast(row,small,'unit_mean')['reason']=='mature_forward_support'


def test_stale_fit_and_missing_schema_fail_closed():
    sessions,rows=records();proof=model.fit(rows,sessions,sessions[70],sessions[80])
    with pytest.raises(ValueError,match='Stale'): model.forecast(rows[0],proof,'conditional')
    del rows[0]['features']['path_efficiency']
    with pytest.raises(ValueError,match='Missing'):model.features(rows[0],'baseline')


def test_exact_equity_notional_bridge():
    sessions,rows=records();proof=model.fit(rows,sessions,sessions[70],sessions[80])
    row=deepcopy(rows[100]);row['day']=sessions[70]
    value=model.forecast(row,proof,'conditional')
    assert value['supported']
    for v in value['values'].values():
        assert v['corrected_equity_bps']/row['notional_equity']==pytest.approx(v['corrected_notional_bps'])


def test_scenario_pair_and_original_gates_retained():
    sessions,rows=records();proof=model.fit(rows,sessions,sessions[70],sessions[80])
    row=deepcopy(rows[100]);row['day']=sessions[70];row['physical_risk_pass']=False
    assert not model.forecast(row,proof,'conditional')['supported']
