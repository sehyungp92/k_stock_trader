import copy
from datetime import date, timedelta
import numpy as np
import pytest
from backtests.strategies.kalcb import checkpoint_holding_models_v1 as m


def sample(day='2025-05-01', symbol='A', q=.25, value=20.):
    row = {'day':day,'symbol':symbol,'opportunity_id':f'{day}/{symbol}/q{q:g}',
           'quantity_scale':q,'eligibility':'eligible','held_quantity':10,'planned_quantity':10,
           'pending_buy_quantity':0,'checkpoint_at':day+'T09:45:00+09:00',
           'own_features':dict.fromkeys(m.OWN,1.),'state_features':dict.fromkeys(m.STATE,1.),
           'context_features':dict.fromkeys(m.CONTEXT,1.),'current_holding_label_valid':True,
           'current_holding_advantage_bps':value,'label_available_at':day+'T15:35:00+09:00'}
    row['state_features']['pending_to_held_quantity']=0.
    return row


def fixture():
    sessions = [(date(2025,5,1)+timedelta(days=i)).isoformat() for i in range(66)]
    groups=[]
    for i, day in enumerate(sessions):
        for q in (.25,.5,1.):
            row=sample(day,q=q,value=float(i%3-1)*10)
            row['own_features'][m.OWN[0]]=float(i%3)
            groups.append({s:copy.deepcopy(row) for s in m.SCENARIOS})
    return groups,sessions


def test_one_session_embargo_and_late_label_exclusion():
    groups,sessions=fixture();first=sessions[61]
    p=m.fit(groups,sessions,first,sessions[-1]);assert p['mature_sessions']==60
    assert p['embargo_cutoff']==sessions[59]
    for s in m.SCENARIOS:groups[0][s]['label_available_at']=first+'T00:00:00+09:00'
    p=m.fit(groups,sessions,first,sessions[-1]);assert p['dependent_quantities']==179


def test_future_payoffs_cannot_change_fit():
    groups,sessions=fixture();first=sessions[61]
    before=m.fit(groups,sessions,first,sessions[-1])
    for g in groups:
        if g['baseline']['day']>=sessions[60]:
            for row in g.values():row['current_holding_advantage_bps']=1e20
    assert m.fit(groups,sessions,first,sessions[-1])==before


def test_sixty_sessions_cannot_be_created_by_quantities():
    groups,sessions=fixture();p=m.fit(groups,sessions,sessions[60],sessions[-1])
    assert p['mature_sessions']==59 and p['status']=='unsupported'


def test_pending_states_and_invalid_payoffs_not_training_labels():
    groups,sessions=fixture()
    groups[0]['combined']['pending_buy_quantity']=1
    groups[1]['baseline']['current_holding_label_valid']=False
    p=m.fit(groups,sessions,sessions[61],sessions[-1]);assert p['dependent_quantities']==178


def test_original_quantity_replication_does_not_add_weight():
    a=sample();b=sample(q=.5);c=sample(symbol='B');d=sample(day='2025-05-02')
    w=m.grouped_weights([a,b,c,d])
    assert w[0]+w[1]==pytest.approx(w[2])
    assert sum(w[:3])==pytest.approx(w[3]) and sum(w)==pytest.approx(3.)


@pytest.mark.parametrize('recipe',m.RECIPES)
def test_common_context_envelope_applies_to_both(recipe):
    groups,sessions=fixture();p=m.fit(groups,sessions,sessions[61],sessions[-1])
    pair=copy.deepcopy(groups[61*3]);pair['combined']['context_features'][m.CONTEXT[0]]=2.
    f=m.forecast(pair,p,recipe);assert not f['supported'] and f['reason']=='prior_feature_envelope'


def test_forecast_does_not_read_future_label():
    groups,sessions=fixture();p=m.fit(groups,sessions,sessions[61],sessions[-1]);pair=copy.deepcopy(groups[61*3])
    before=m.forecast(pair,p,'peer_context')
    for r in pair.values():r['current_holding_advantage_bps']=-1e99;r['current_holding_label_valid']=False
    assert m.forecast(pair,p,'peer_context')==before


def test_stale_model_rejected():
    groups,sessions=fixture();p=m.fit(groups,sessions,sessions[61],sessions[-1])
    with pytest.raises(ValueError):m.forecast(groups[0],p,'own_state')


def test_risk_counts_unique_exposures_and_retains_failures():
    _,sessions=fixture();rows=[]
    for day in sessions[:60]:
        for symbol in range(16):
            for q in (.25,.5,1.):
                rows.append({'day':day,'symbol':str(symbol),'risk_available_at':day+'T15:35:00+09:00',
                             'active_no_pending_both':True,'unsafe':False})
    p=m.risk_proof(rows,sessions,sessions[61]);assert p['issuer_days']==960 and m.risk_pass(p)
    rows[0]['unsafe']=True;p=m.risk_proof(rows,sessions,sessions[61])
    assert p['unsafe_issuer_days']==1 and not m.risk_pass(p)
    for r in rows:r['active_no_pending_both']=False
    assert m.risk_proof(rows,sessions,sessions[61])['issuer_days']==0


def calibration_rows():
    _,sessions=fixture();out=[]
    for i in range(20):
        day=sessions[i];out.append({'day':day,'symbol':'A','valid_from':sessions[0],
            'matched_predecision_both':True,'risk_pass':True,'raw':{'supported':True,'values':dict.fromkeys(m.SCENARIOS,float(i))},
            'labels':{s:{'valid':True,'available_at':day+'T15:35:00+09:00','value':-i} for s in m.SCENARIOS}})
    return out,sessions


def test_calibration_requires_twenty_unique_matched_forward_labels():
    rows,sessions=calibration_rows();p=m.calibration(rows,sessions,sessions[30],'baseline')
    assert p['supported'] and 0<=p['slope']<=1
    rows[0]['matched_predecision_both']=False
    assert not m.calibration(rows,sessions,sessions[30],'baseline')['supported']


def test_calibration_future_and_unknown_membership_cannot_train():
    rows,sessions=calibration_rows();p=m.calibration(rows,sessions,sessions[30],'combined')
    poisoned=copy.deepcopy(rows[0]);poisoned['day']=sessions[31];poisoned['labels']=None
    assert m.calibration(rows+[poisoned],sessions,sessions[30],'combined')==p
    rows[0]['valid_from']=sessions[1]
    with pytest.raises(ValueError):m.calibration(rows,sessions,sessions[30],'combined')


def test_duplicate_selected_issuer_day_is_not_independent():
    rows,sessions=calibration_rows()
    with pytest.raises(ValueError):m.calibration(rows+[rows[0]],sessions,sessions[30],'baseline')


def test_unsupported_calibration_cannot_force_admission():
    raw={'supported':True,'values':dict.fromkeys(m.SCENARIOS,1000.)}
    maps={s:{'supported':False} for s in m.SCENARIOS}
    result=m.calibrated(raw,maps,{'supported':True,'probability':.0001})
    assert not result['supported']


def test_native_features_use_actual_integer_holdings_and_entry_price():
    comparison={'day':'2025-05-01','symbol':'A','action_id':'a','quantity_scale':.25,'parent_exit_already_decided':False,
                'native':{'held_quantity':20,'planned_quantity':40,'pending_buy_quantity':0,'mean_entry_fill_price':100.}}
    context={'completed_close':110.,'daily_atr_pct':.1,'consumed_bars':[{'volume':100}],
             'own_features':dict.fromkeys(m.OWN,1.),'context_features':dict.fromkeys(m.CONTEXT,1.),
             'available_at':'2025-05-01T09:45:00+09:00'}
    row=m.native_row(comparison,context)
    assert row['held_notional_krw']==2200 and row['state_features']['held_to_planned_quantity']==.5
    assert row['state_features']['marked_entry_return_atr']==pytest.approx(1.)
    reference=copy.deepcopy(row);assert m.exact_causal_match(row,reference)
    reference['held_quantity']=19;assert not m.exact_causal_match(row,reference)


def test_missing_feature_and_naive_clock_fail_closed():
    row=sample();del row['context_features'][m.CONTEXT[0]]
    with pytest.raises(ValueError):m.feature_vector(row)
    with pytest.raises(ValueError):m.mature('2025-05-01','2025-05-01T15:35:00',{'2025-05-01'})
