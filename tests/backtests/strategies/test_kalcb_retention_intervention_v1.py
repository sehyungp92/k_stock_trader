from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timedelta
import pytest
from backtests.strategies.kalcb.retention_intervention_v1 import RetentionObserver
from backtests.engine.sim_broker import SimBroker, BrokerCosts
from backtests.engine.capacity import BarParticipationScenario
from strategy_common.actions import SubmitEntry, SubmitExit
from strategy_common.clock import KST
from strategy_common.market import MarketBar
from strategy_kalcb.config import KALCBConfig
from strategy_kalcb.core import logic
from strategy_kalcb.core.state import KALCBState, KALCBPositionState


def fixture():
    start = datetime(2025,8,7,9,0,tzinfo=KST)
    state = KALCBState(); symbol = state.symbol_state('A')
    for i,c in enumerate([110]*10+[103,102]):
        symbol.bars_today.append(MarketBar('A',start+timedelta(minutes=i*5),'5m',c,c+1,c-1,c,100000))
    position = KALCBPositionState(symbol='A',qty_entry=100,qty_open=100,entry_price=100,
        entry_time=start+timedelta(minutes=30),initial_stop=99,current_stop=99,risk_per_share=1,
        entry_type='KRX_FIRST30_OPEN',momentum_score=4,hold_bars=6,metadata={'daily_atr':10})
    position.update_mark(high=110,low=100);symbol.position=position
    cfg = KALCBConfig(exit_profit_retention_enabled=True,exit_profit_retention_start_atr=.75,
        exit_profit_retention_cost_multiple=4,use_partial_takes=False,exit_hard_stop_enabled=False,
        quick_exit_enabled=False,exit_failed_followthrough_bars=0,exit_target_r=0,
        exit_mfe_floor_enabled=False,adaptive_trail_enabled=False)
    return state,symbol,position,symbol.bars_today[-1],cfg


def native(observe, future_open=91, max_hold=0):
    state,symbol,position,bar,cfg=fixture()
    cfg=replace(cfg,exit_max_hold_bars=max_hold)
    broker=SimBroker(1000000,BrokerCosts(commission_bps=2,tax_bps_on_sell=18,slippage_bps=3),capacity_policy=BarParticipationScenario(.1,'test'))
    prior=position.entry_time-timedelta(minutes=5)
    broker.process_bar(MarketBar('A',prior,'5m',100,100,100,100,100000))
    broker.submit(SubmitEntry('KALCB','A',100,'MARKET',None,None,'entry'),prior)
    broker.process_bar(MarketBar('A',position.entry_time,'5m',100,100,100,100,100000))
    broker.process_bar(bar)
    observer=RetentionObserver({'exit_profit_retention_start_atr':1.25})
    if observe:observer.install()
    try:
        result=logic._manage_position(state,symbol,bar,cfg)
        exits=[a for a in result.actions if isinstance(a,SubmitExit)]
        assert len(exits)==1 and exits[0].reason=='profit_retention'
        for action in exits:broker.submit(action,bar.timestamp)
        before=len(broker.fills)
        broker.process_bar(MarketBar('A',bar.timestamp+timedelta(minutes=5),'5m',future_open,future_open+1,future_open-1,future_open,100000))
        assert len(broker.fills)==before+1
        fills=[asdict(f) for f in broker.fills]
        return {'cash':broker.cash,'fills':fills,'exit_action':exits[0]},observer.summarize(fills),observer
    finally:
        if observe:observer.restore()


def test_observer_preserves_native_actions_fills_and_cash_and_proves_gap_exit():
    plain,_,_=native(False)
    observed,support,_=native(True)
    assert observed==plain
    assert support['direct_changed_exit_orders']==1
    assert support['direct_changed_exit_issuer_days']==1
    assert support['direct_changed_exit_sessions']==1
    assert not support['preferred_support_passed']
    assert observed['fills'][-1]['price'] < 91


def test_future_gap_changes_execution_not_decision():
    a,_,oa=native(True,91);b,_,ob=native(True,80)
    assert a['exit_action']==b['exit_action']
    assert oa.probes[0]['actual_signal']==ob.probes[0]['actual_signal']
    assert a['fills'][-1]['price']>b['fills'][-1]['price']


def test_backout_metadata_cannot_contaminate_actual_position():
    _,symbol,position,bar,cfg=fixture();cfg=replace(cfg,exit_profit_retention_start_atr=1.25)
    observer=RetentionObserver({'exit_profit_retention_start_atr':.75});before=deepcopy(position.metadata)
    observer.install()
    try:
        assert not logic._exit_reason(symbol,position,bar,cfg,current_r=2,mfe_r=10)
        assert position.metadata==before
        assert observer.probes[0]['parent_signal_same_state']
    finally:observer.restore()


def test_recursive_overlay_has_one_probe_and_identity_no_effect():
    _,symbol,position,bar,cfg=fixture();cfg=replace(cfg,exit_profit_retention_recent_overlay_atr=2.5)
    observer=RetentionObserver({});observer.install()
    try:
        assert logic._exit_reason(symbol,position,bar,cfg,current_r=2,mfe_r=10)=='profit_retention'
        assert len(observer.probes)==1 and observer.calls==1
        assert observer.probes[0]['parent_signal_same_state']
        assert observer.summarize([])['direct_changed_exit_orders']==0
    finally:observer.restore()


def test_signal_without_fill_never_counts_as_broker_intervention():
    _,symbol,position,bar,cfg=fixture();observer=RetentionObserver({'exit_profit_retention_start_atr':1.25})
    observer.install()
    try:
        assert logic._exit_reason(symbol,position,bar,cfg,current_r=2,mfe_r=10)=='profit_retention'
        assert observer.summarize([])['direct_changed_exit_orders']==0
    finally:observer.restore()


def test_exit_retry_retains_initial_signal_lineage_without_inventing_support():
    _,symbol,position,bar,cfg=fixture()
    observer=RetentionObserver({'exit_profit_retention_start_atr':1.25});observer.install()
    try:
        assert logic._exit_reason(symbol,position,bar,cfg,current_r=2,mfe_r=10)=='profit_retention'
        broker=SimBroker(1000000,BrokerCosts())
        action=SubmitExit('KALCB','A',100,'MARKET',None,'profit_retention',metadata={'entry_price':100})
        broker.submit(action,bar.timestamp)
        broker.submit(action,bar.timestamp+timedelta(minutes=5))
        assert len(observer.orders)==2
        assert all(p['bar_timestamp']==bar.timestamp.isoformat() for p in observer.orders.values())
        assert observer.summarize([])['direct_changed_exit_orders']==0
    finally:observer.restore()


def test_another_parent_exit_at_the_same_decision_is_not_direct_support():
    _,support,observer=native(True,max_hold=6)
    assert support['filled_retention_orders']==1
    assert observer.probes[0]['parent_exit_reason']=='max_hold'
    assert support['direct_changed_exit_orders']==0
