import copy
import pytest
from backtests.strategies.kalcb.checkpoint_episode_scope_v1 import lifetimes
from backtests.strategies.kalcb.checkpoint_holding_target_v1 import holding_at,derive


def fixture():
    cost={'commission_bps':2.,'tax_bps_on_sell':18.}
    h=[{'side':s,'symbol':'A','qty':q,'price':p,'timestamp':f'2026-01-05T{t}:00+09:00'} for s,q,p,t in
       [('BUY',10,100,'09:30'),('SELL',10,110,'10:00'),('BUY',2,90,'10:40'),('SELL',2,80,'11:10')]]
    l=copy.deepcopy(h[:2]);l[1].update(price=105,timestamp='2026-01-05T09:45:00+09:00')
    total=lambda fs:sum(x['net_cash_flow'] for x in lifetimes(fs,cost))
    arm={'checkpoint':{'held_quantity':10,'entry_price':100.,'action_submitted':False},'independent_execution_pass':True,'independent_account_pass':True,'deadline_violation':False}
    v={'opportunity_id':'a','day':'2026-01-05','symbol':'A','scenario':'baseline','quantity_scale':1,
       'checkpoint_at':'2026-01-05T09:45:00+09:00','planned_quantity':10,'held_notional_krw':1040.,'paired_label_valid':True,'eligibility':'eligible',
       'label_available_at':'2026-01-05T15:30:00+09:00','liquidate_minus_hold_bps':(total(l)-total(h))/1040*10000,
       'own_features':{'x':1},'context_features':{'y':2},'hold':arm,'liquidate':copy.deepcopy(arm)}
    v['liquidate']['checkpoint']['action_submitted']=True
    event={'hold':{'fills':h,'costs':cost,'checkpoint':{'state':{'orders':[]}}},'liquidate':{'fills':l,'costs':cost}}
    ctx={'completed_close':104.,'daily_atr_pct':.02,'consumed_bars':[{'volume':100}]}
    return v,event,ctx


def test_target_excludes_later_reentry_and_preserves_original_maturity():
    v,e,c=fixture();d=derive(v,e,c,lifetimes)
    assert d['current_holding_advantage_bps']<d['original_episode_advantage_bps']
    assert d['current_holding_advantage_bps']+d['other_lifetimes_advantage_bps']==pytest.approx(v['liquidate_minus_hold_bps'])
    assert d['label_available_at']==v['label_available_at'] and d['no_pending_entry_state']


def test_next_bar_exit_still_owned_at_decision_clock():
    v,e,c=fixture();p,q=holding_at(lifetimes(e['liquidate']['fills'],e['liquidate']['costs']),v['checkpoint_at'])
    assert p is not None and q==10


def test_later_position_is_not_first_lifetime():
    v,e,c=fixture();p,q=holding_at(lifetimes(e['hold']['fills'],e['hold']['costs']),'2026-01-05T10:45:00+09:00')
    assert q==2 and p['entry'].endswith('10:40:00+09:00')


def test_pending_entries_are_causal_and_not_fully_filled():
    v,e,c=fixture();e['hold']['checkpoint']['state']['orders']=[{'side':'BUY','symbol':'A','qty':3}]
    d=derive(v,e,c,lifetimes);assert not d['no_pending_entry_state'] and d['state_features']['pending_to_held_quantity']==.3


def test_invalid_pair_never_acquires_new_payoff_label():
    v,e,c=fixture();v['paired_label_valid']=False;v['label_available_at']=None
    d=derive(v,e,c,lifetimes);assert not d['current_holding_label_valid'] and d['current_holding_advantage_bps'] is None


@pytest.mark.parametrize('poison',['held','denominator','cost'])
def test_pair_or_integer_denominator_disagreement_rejected(poison):
    v,e,c=fixture()
    if poison=='held':v['hold']['checkpoint']['held_quantity']=9
    if poison=='denominator':v['held_notional_krw']=1000
    if poison=='cost':e['liquidate']['costs']={**e['liquidate']['costs'],'commission_bps':3}
    with pytest.raises(ValueError):derive(v,e,c,lifetimes)
