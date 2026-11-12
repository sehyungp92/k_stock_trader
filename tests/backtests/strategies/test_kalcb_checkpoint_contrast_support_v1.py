import copy
import pytest
from backtests.strategies.kalcb.checkpoint_contrast_support_v1 import mature_support, native_prefix, joint_row, validate_source_context


def test_chronology_embargo_and_maturity_and_no_replication():
    sessions=['2026-01-05','2026-01-06','2026-01-07','2026-01-08']
    row={'action_id':'A/q1','day':sessions[0],'symbol':'A','joint_payoff_valid':True,
         'available_at':'2026-01-06T15:35:00+09:00'}
    rows=[row,{**row,'action_id':'A/q.5'},{**row,'day':sessions[2],'symbol':'B'}]
    result=mature_support(rows,sessions,[(sessions[2],sessions[3]),(sessions[3],sessions[3])])
    assert result[0]['mature_sessions']==0
    assert result[1]['mature_sessions']==1 and result[1]['mature_issuer_days']==1
    assert result[1]['dependent_action_rows']==2
    assert mature_support(rows,sessions,[(sessions[3],sessions[3])],set())[0]['mature_sessions']==0


def native_fixture():
    order={'order_id':'a','side':'BUY','symbol':'A','qty':10,
           'metadata':{'decision_available_at':'2026-01-05T09:30:00+09:00'}}
    fills=[{'order_id':'a','side':'BUY','symbol':'A','qty':4,'price':100.,'timestamp':'2026-01-05T09:30:00+09:00'},
           {'order_id':'a','side':'BUY','symbol':'A','qty':6,'price':101.,'timestamp':'2026-01-05T09:45:00+09:00'}]
    return fills,{'a':order},{},['a'],'A','2026-01-05T09:45:00+09:00'


def test_next_bar_fill_does_not_change_prior_held_or_pending():
    args=native_fixture();r=native_prefix(*args)
    assert (r['held_quantity'],r['planned_quantity'],r['pending_buy_quantity'])==(4,10,6)
    args[0][1]['price']=9999;args[0][1]['qty']=1
    assert native_prefix(*args)==r


def test_parent_cancellation_at_completed_checkpoint_cancels_pending():
    args=list(native_fixture());args[2]={'a':['2026-01-05T09:40:00+09:00']}
    assert native_prefix(*args)['pending_buy_quantity']==0
    args[2]={'a':['2026-01-05T09:45:00+09:00']}
    assert native_prefix(*args)['pending_buy_quantity']==6


@pytest.mark.parametrize('poison',['symbol','overfill','unbound','short'])
def test_native_prefix_rejects_invalid_lifetime(poison):
    args=list(native_fixture())
    if poison=='symbol':args[0][0]['symbol']='B'
    if poison=='overfill':args[0][0]['qty']=11
    if poison=='unbound':args[0][0]['order_id']='b'
    if poison=='short':args[0][0]['side']='SELL'
    with pytest.raises(ValueError):native_prefix(*args)


def pair_fixture():
    arm={'independent_execution_pass':True,'independent_account_pass':True,'deadline_violation':False,
         'checkpoint':{'action_submitted':True,'held_quantity':4},'actual_changed_exit_quantity':4}
    r={'opportunity_id':'2026-01-05/A/q1','day':'2026-01-05','symbol':'A','quantity_scale':1,
       'planned_quantity':10,'paired_label_valid':True,'label_available_at':'2026-01-05T15:35:00+09:00',
       'hold':copy.deepcopy(arm),'liquidate':copy.deepcopy(arm),'scenario':'baseline'}
    return r,{**copy.deepcopy(r),'scenario':'combined'}


def test_payoff_maturity_waits_for_both_scenarios():
    b,c=pair_fixture();c['label_available_at']='2026-01-06T15:35:00+09:00'
    assert joint_row(b,c)['available_at']==c['label_available_at']
    c['paired_label_valid']=False;c['label_available_at']=None
    assert joint_row(b,c)['available_at'] is None


def test_hold_deadline_is_separate_risk_not_missing_economic_maturity():
    b,c=pair_fixture();b['hold']['deadline_violation']=True
    r=joint_row(b,c);assert r['joint_payoff_valid'] and r['hold_deadline_observed']
    assert r['liquidation_physically_supported']


@pytest.mark.parametrize('poison',['deadline','execution','account','unfilled','inactive'])
def test_liquidation_risk_or_unfilled_action_not_supported(poison):
    b,c=pair_fixture();a=c['liquidate']
    if poison=='deadline':a['deadline_violation']=True
    if poison=='execution':a['independent_execution_pass']=False
    if poison=='account':a['independent_account_pass']=False
    if poison=='unfilled':a['actual_changed_exit_quantity']=3
    if poison=='inactive':a['checkpoint']['action_submitted']=False
    assert not joint_row(b,c)['liquidation_physically_supported']


def test_wrong_scenario_or_quantity_rejected():
    b,c=pair_fixture();c['quantity_scale']=.5
    with pytest.raises(ValueError):joint_row(b,c)


@pytest.mark.parametrize('poison',[None,'clock','features','quantity','denominator'])
def test_source_features_and_actual_integer_denominator(poison):
    context={'context_supported':True,'available_at':'2026-01-05T09:45:00+09:00',
             'own_features':{'x':1},'context_features':{'y':2},'completed_close':101.}
    row={'checkpoint_at':context['available_at'],'own_features':{'x':1},'context_features':{'y':2},
         'hold':{'checkpoint':{'held_quantity':4}},'planned_quantity':10,'held_notional_krw':404.}
    if poison=='clock':row['checkpoint_at']='2026-01-05T09:40:00+09:00'
    if poison=='features':row['context_features']['y']=3
    if poison=='quantity':row['hold']['checkpoint']['held_quantity']=4.5
    if poison=='denominator':row['held_notional_krw']=1010.
    if poison:
        with pytest.raises(ValueError):validate_source_context(row,context)
    else:validate_source_context(row,context)
