from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from deployment.olr_kalcb.intent_allocator import (
    AllocationAccount, EntryIntent, IntentAllocationConfig, allocate_entry_intents)

AT=datetime(2026,1,5,10,tzinfo=timezone.utc)


def account(**changes):
    return replace(AllocationAccount(1000,2000,1000,2000,1000,{'KALCB':2000,'OLR':1000}),**changes)


def intent(name, score=None, **changes):
    return replace(EntryIntent(name,'OLR',name,'TECH',10,100,AT,AT+timedelta(minutes=5),
        robust_net_bps=score,value_asof=AT-timedelta(days=1) if score is not None else None,
        value_source='frozen-model' if score is not None else ''),**changes)


def test_stronger_valid_intent_displaces_earlier_unsubmitted_intent():
    early=intent('weak',10,available_at=AT-timedelta(minutes=5))
    strong=intent('strong',40)
    result=allocate_entry_intents([early,strong],account(),AT,IntentAllocationConfig(rank_mode='supported_value'))
    assert [(x.intent_id,x.approved_qty) for x in result]==[('strong',10),('weak',0)]


def test_future_or_expired_signals_cannot_compete_with_current_signals():
    current=intent('current',5)
    future=intent('future',100,available_at=AT+timedelta(minutes=1))
    expired=intent('expired',100,available_at=AT-timedelta(minutes=10),expires_at=AT-timedelta(minutes=1))
    result=allocate_entry_intents([future,expired,current],account(),AT,IntentAllocationConfig(rank_mode='supported_value'))
    assert {x.intent_id:x.approved_qty for x in result}=={'future':0,'expired':0,'current':10}


def test_unknown_values_use_native_fallback_for_entire_cohort():
    rows=[intent('scored',100,native_rank=2),intent('unknown',None,native_rank=1)]
    result=allocate_entry_intents(rows,account(),AT,IntentAllocationConfig(rank_mode='supported_value'))
    assert result[0].intent_id=='unknown' and result[0].approved_qty==10
    assert all(x.rank_basis=='native_fallback' for x in result)


def test_reservations_leave_space_for_next_intent_after_symbol_clip():
    result=allocate_entry_intents([intent('A',native_rank=1),intent('B',native_rank=2)],
        account(symbol_limit=600),AT,IntentAllocationConfig())
    assert [x.approved_qty for x in result]==[6,4]
    assert sum(x.reserved_notional for x in result)==1000


def test_existing_pending_buy_obligations_and_overnight_budget_are_counted():
    state=account(gross_committed=500,strategy_committed={'OLR':500},funding_limit=500,
        strategy_limits={'KALCB':2000,'OLR':750})
    result=allocate_entry_intents([intent('A')],state,AT,IntentAllocationConfig())
    assert result[0].approved_qty==2
    assert result[0].binding_limit=='strategy_budget'


def test_same_symbol_can_share_exposure_but_does_not_create_capacity():
    state=account(gross_committed=600,symbol_committed={'A':600},strategy_committed={'KALCB':600},
        occupied=frozenset({('KALCB','A')}))
    assert allocate_entry_intents([intent('A')],state,AT,IntentAllocationConfig())[0].approved_qty==4
    assert allocate_entry_intents([intent('A')],replace(state,block_duplicates=True),AT,IntentAllocationConfig())[0].approved_qty==0


def test_net_value_after_capital_occupation_cost_and_capacity_matters():
    config=IntentAllocationConfig(rank_mode='supported_value',occupation_bps_per_session=10)
    long=intent('slow',20,expected_holding_sessions=3)
    fast=intent('fast',15,expected_holding_sessions=.25,capacity_qty=3)
    result=allocate_entry_intents([long,fast],account(),AT,config)
    assert [(x.intent_id,x.approved_qty) for x in result]==[('fast',3),('slow',0)]


@pytest.mark.parametrize('changes',[{'value_asof':AT+timedelta(seconds=1)}, {'robust_net_bps':float('nan')}, {'unit_cost':float('inf')}])
def test_invalid_or_noncausal_value_fails_closed(changes):
    with pytest.raises(ValueError):intent('A',20,**changes)


def test_native_slots_and_zero_equity_cannot_be_bypassed():
    state=account(occupied=frozenset({('OLR','old')}),max_positions={'OLR':1})
    assert allocate_entry_intents([intent('A')],state,AT,IntentAllocationConfig())[0].approved_qty==0
    assert allocate_entry_intents([intent('A')],account(equity=0),AT,IntentAllocationConfig())[0].approved_qty==0


def test_two_strategies_cannot_spend_the_same_entry_liquidity_forecast_twice():
    a=intent('A',20,symbol='S',strategy_id='KALCB',shared_entry_capacity_qty=5)
    b=intent('B',10,symbol='S',shared_entry_capacity_qty=5)
    result=allocate_entry_intents([b,a],account(),AT,IntentAllocationConfig(rank_mode='supported_value'))
    assert [(x.intent_id,x.approved_qty) for x in result]==[('A',5),('B',0)]
