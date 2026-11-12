from dataclasses import replace
from datetime import date, datetime, timezone
import pytest
from strategy_kalcb.models import KALCBDailyCandidate, KALCBDailySnapshot
from backtests.strategies.kalcb.oos_ablation import candidate_quantity,transform_snapshot,KALCBOOSAblationPlugin

def candidate():
    return KALCBDailyCandidate(symbol='000100',trade_date=date(2026,6,1),prior_day_high=105,prior_day_low=95,
        prior_day_close=100,daily_atr=8,expected_5m_volume=10000,average_30m_volume=60000,
        metadata={'action_quantity_scale':1})

def test_atr_quantity_is_causal_bounded_and_does_not_upsize():
    c=candidate()
    assert candidate_quantity(c,'atr_quantity_4pct')==.5
    assert candidate_quantity(replace(c,daily_atr=2),'atr_quantity_4pct')==1
    assert candidate_quantity(replace(c,metadata={**c.metadata,'future_return':999}),'atr_quantity_4pct')==.5
    with pytest.raises(ValueError,match='prior, positive'):
        candidate_quantity(replace(c,prior_day_close=0),'atr_quantity_4pct')

def test_quantity_cap_preserves_smaller_selected_actions():
    assert candidate_quantity(candidate(),'q1_cap_half')==.5
    assert candidate_quantity(replace(candidate(),metadata={'action_quantity_scale':.25}),'q1_cap_half')==.25

def test_transformed_policy_is_hash_bound_and_does_not_mutate_parent():
    c=candidate();s=KALCBDailySnapshot(c.trade_date,(c,),'source',datetime(2026,6,1,0,30,tzinfo=timezone.utc),
        metadata={'active_symbols':['000100'],'active_symbol_count':1,'candidate_pool_count':1})
    raw=s.to_json_dict();changed=transform_snapshot(raw,'atr_quantity_4pct')
    assert changed['artifact_hash']!=raw['artifact_hash']
    assert raw['candidates'][0]['metadata']['action_quantity_scale']==1
    assert changed['candidates'][0]['metadata']['action_quantity_scale']==.5
    empty=transform_snapshot(raw,'no_support_additions',{'000100'})
    assert empty['candidates']==[] and empty['metadata']['active_symbol_count']==0
    raw['artifact_hash']='forged'
    with pytest.raises(ValueError,match='Changed parent'):
        transform_snapshot(raw,'q1_cap_half')

def test_unregistered_diagnostic_mutations_are_rejected():
    plugin=object.__new__(KALCBOOSAblationPlugin)
    plugin.spec={'recipes':{'control':{}}}
    with pytest.raises(ValueError,match='Undeclared'):
        plugin.canonicalize_mutations({'recipe':'best_after_seeing_oos'})
    with pytest.raises(ValueError,match='Undeclared'):
        plugin.canonicalize_mutations({'dates':'best_subperiod'})
