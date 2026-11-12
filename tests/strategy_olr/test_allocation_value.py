from dataclasses import replace
from datetime import datetime, timedelta, time
import json

import pytest

from strategy_olr.core.allocation_value import quote, matured_labels
from strategy_olr.core.state import OLRState, OLRPositionState
from strategy_olr.core.logic import on_olr_fill,remember_submitted_order
from strategy_olr.core.core_models import OLRFillEvent,OLRPortfolioView
from strategy_olr.core.serializers import snapshot_state,restore_state
from strategy_olr.core.allocation import step_olr_batch
from strategy_common.actions import SubmitEntry
from strategy_common.clock import KST
from tests.strategy_olr.test_capacity_allocation import bar,config,snapshot,DAY


def learned_state():
    state=OLRState()
    at=datetime.combine(DAY,time(14,35),tzinfo=KST)
    for i in range(40):
        rank=1 if i%2==0 else 2
        closed=at-timedelta(days=40-i)
        key=str(i)
        p=OLRPositionState('999999',0,100,closed-timedelta(days=1),rank,50,entry_order_id=key,
            metadata={'capacity_value_cash':{'buy':1000.,'sell':1100. if rank==1 else 980.,
                'buy_qty':10,'sell_qty':10,'complete_costs':True,'last_fill':closed.isoformat()},
                'afternoon_score_band_rule':'a' if rank==1 else 'b'})
        state.symbol_state('999999').entry_history[key]=p
    return state,at


def test_value_uses_only_prior_closed_lifecycles_with_complete_costs_and_survives_restart():
    state,at=learned_state();cfg=config(capacity_joint_allocation=True,capacity_value_mode='rank')
    c=snapshot().candidates[1];scale,e=quote(state,c,at,cfg)
    assert .75<=scale<1 and e['status']=='supported' and e['strategy_support']==40
    future=replace(state.symbol_state('999999').entry_history['1'],entry_order_id='future')
    future.metadata={**future.metadata,'capacity_value_cash':{**future.metadata['capacity_value_cash'],
        'last_fill':at.isoformat(),'sell':1e12}}
    state.symbol_state('999999').entry_history['future']=future
    assert quote(state,c,at,cfg)==(scale,e)
    future.metadata['capacity_value_cash']['last_fill']=(at-timedelta(days=1)).isoformat()
    future.metadata['capacity_value_cash']['complete_costs']=False
    assert quote(state,c,at,cfg)==(scale,e)
    restored=restore_state(json.loads(json.dumps(snapshot_state(state))))
    assert quote(restored,c,at,cfg)==(scale,e)
    c=replace(c,metadata={**c.metadata,'close_to_close_label_pct':1e9,'next_session_mfe_label_pct':1e9})
    assert quote(restored,c,at,cfg)==(scale,e)


def test_incomplete_lifecycle_and_cold_start_keep_native_size():
    state,at=learned_state();cfg=config(capacity_joint_allocation=True,capacity_value_mode='rank')
    assert quote(OLRState(),snapshot().candidates[0],at,cfg)[0]==1
    state.symbol_state('999999').pending_entry_order_id='1'
    assert len(matured_labels(state,at))==39
    p=state.symbol_state('999999').entry_history['3']
    state.symbol_state('999999').position=replace(p,qty_open=2)
    assert len(matured_labels(state,at))==38


def test_value_changes_native_feasible_allocation_and_preserves_funding_ceiling():
    cfg=config(capacity_joint_allocation=True,capacity_value_mode='rank')
    state,at=learned_state()
    result=step_olr_batch(state,(bar('000660'),),cfg,snapshot(),OLRPortfolioView(10000,10000))
    entry=[a for a in result.actions if isinstance(a,SubmitEntry)][0]
    assert entry.metadata['capacity_value_qty_delta']<0 and entry.metadata['reserved_notional']<=10000


def test_fill_labels_deduplicate_and_aggregate_partials_before_maturing():
    cfg=config(capacity_joint_allocation=True,capacity_value_mode='rank')
    state=OLRState();stamp=datetime.combine(DAY,time(14,35),tzinfo=KST)
    action=SubmitEntry('OLR','005930',10,'MARKET',None,None,'entry',
        {'estimated_entry_price':100,'reserved_notional':1001,'candidate_rank':1,'candidate_score':50})
    remember_submitted_order(state,'buy',action)
    buy=OLRFillEvent('buy','005930','BUY',10,100,stamp,metadata={'fee':.2,'execution_id':'b','submitted_qty':10,'filled_qty_after':10})
    on_olr_fill(state,buy,cfg);on_olr_fill(state,buy,cfg)
    for i,price in enumerate((110,105)):
        on_olr_fill(state,OLRFillEvent('sale'+str(i),'005930','SELL',5,price,stamp+timedelta(minutes=5*(i+1)),
            metadata={'fee':1.,'execution_id':'s'+str(i)}),cfg)
    assert not matured_labels(state,stamp+timedelta(hours=2))
    labels=matured_labels(state,stamp+timedelta(days=1))
    assert len(labels)==1
    assert labels[0]['net_bps']==pytest.approx(10000*(1073/1000.2-1))
