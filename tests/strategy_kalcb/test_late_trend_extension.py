from dataclasses import replace
from datetime import datetime, time, timedelta
import pytest
from strategy_common.clock import KST
from strategy_common.market import MarketBar
from strategy_kalcb.config import KALCBConfig, CarryMode
from strategy_kalcb.core.state import KALCBState, KALCBPositionState
from strategy_kalcb.core.logic import _late_trend_extension_allowed, on_kalcb_timer, _exit_reason


def fixture():
    state=KALCBState(); s=state.symbol_state('A')
    start=datetime(2025,8,7,14,0,tzinfo=KST)
    for i,c in enumerate([100]*7+[102,103]):
        s.bars_today.append(MarketBar('A',start+timedelta(minutes=5*i),'5m',c,c+1,c-1,c,100000))
    p=KALCBPositionState(symbol='A',qty_entry=100,qty_open=100,entry_price=100,
        entry_time=start,initial_stop=99,current_stop=99,risk_per_share=1,
        entry_type='KRX_FIRST30_OPEN',momentum_score=4,hold_bars=20)
    s.position=p
    cfg=KALCBConfig(flatten_time=time(14,45),carry_mode=CarryMode.OFF,exit_late_trend_extension_minutes=15,
        exit_liquidity_guard_enabled=True,quick_exit_enabled=False,exit_failed_followthrough_bars=0)
    return state,s,p,start+timedelta(minutes=45),cfg


def test_extension_is_prefix_only_and_default_off():
    _,s,p,stamp,c=fixture()
    assert _late_trend_extension_allowed(s,p,stamp,c)
    s.bars_today.append(MarketBar('A',stamp,'5m',1,1,1,1,1))
    assert _late_trend_extension_allowed(s,p,stamp,c)
    assert not _late_trend_extension_allowed(s,p,stamp,replace(c,exit_late_trend_extension_minutes=0))
    assert p.metadata['late_trend_extension_evidence']['future_liquidity_verified'] is False


def test_capacity_is_reserved_after_extension_and_deadline_is_absolute():
    _,s,p,stamp,c=fixture()
    assert _late_trend_extension_allowed(s,p,stamp,c)
    assert p.metadata['late_trend_extension_evidence']['forecast_capacity_after_extension']==1600
    p.qty_open=1601
    assert not _late_trend_extension_allowed(s,p,stamp,c)
    p.qty_open=100
    assert not _late_trend_extension_allowed(s,p,stamp+timedelta(minutes=15),c)
    assert not _late_trend_extension_allowed(s,p,stamp+timedelta(minutes=5),c) # stale observation


def test_missing_bar_and_broken_trend_disable_extension():
    _,s,p,stamp,c=fixture();old=list(s.bars_today)
    s.bars_today.clear();s.bars_today.extend(old[:5]+old[6:])
    assert not _late_trend_extension_allowed(s,p,stamp,c)
    s.bars_today.clear();s.bars_today.extend(old)
    p.entry_price=103
    assert not _late_trend_extension_allowed(s,p,stamp,c)
    p.entry_price=100
    b=old[-1];s.bars_today[-1]=MarketBar('A',b.timestamp,'5m',99,100,98,99,100000)
    assert not _late_trend_extension_allowed(s,p,stamp,c)


def test_timer_extends_once_then_flattens_and_earlier_exits_still_win():
    state,s,p,stamp,c=fixture()
    assert not on_kalcb_timer(state,stamp,c).actions
    p.metadata['liquidation_reason']='liquidity_headroom'
    assert _exit_reason(s,p,s.bars_today[-1],c,current_r=3,mfe_r=4)=='liquidity_headroom'
    p.metadata.pop('liquidation_reason')
    result=on_kalcb_timer(state,stamp+timedelta(minutes=15),c)
    assert result.actions and result.actions[-1].reason=='eod_flatten'


def test_delayed_exchange_session_keeps_close_relative_deadlines():
    from strategy_common.execution_session import kalcb_session_config
    _,s,p,_,c=fixture()
    delayed=datetime(2025,11,13,15,0,tzinfo=KST)
    old=list(s.bars_today);s.bars_today.clear()
    s.bars_today.extend(replace(b,timestamp=delayed+timedelta(minutes=5*i)) for i,b in enumerate(old))
    c=kalcb_session_config(c,delayed.date())
    assert _late_trend_extension_allowed(s,p,delayed+timedelta(minutes=45),c)
    proof=p.metadata['late_trend_extension_evidence']
    assert proof['discretionary_deadline'].startswith('2025-11-13T16:00:00')
    assert proof['hard_deadline'].startswith('2025-11-13T16:20:00')


def test_cancelling_unfilled_entry_restores_enabled_stop_during_extension():
    from strategy_common.actions import CancelOrders,SubmitProtectiveStop,FlattenPosition
    state,s,p,stamp,c=fixture();c=replace(c,exit_hard_stop_enabled=True)
    s.pending_entry_order_id='entry-remainder';s.pending_entry_metadata={}
    result=on_kalcb_timer(state,stamp,c)
    assert any(isinstance(a,CancelOrders) for a in result.actions)
    assert any(isinstance(a,SubmitProtectiveStop) and a.qty==p.qty_open for a in result.actions)
    assert not any(isinstance(a,FlattenPosition) for a in result.actions)


@pytest.mark.parametrize('changes',[{'exit_late_trend_extension_minutes':10},
    {'exit_late_trend_extension_minutes':15},
    {'exit_late_trend_extension_minutes':15,'exit_liquidity_guard_enabled':True,'exit_profit_retention_roundtrip_cost_bps':float('nan')}])
def test_unsupported_or_unprotected_extension_rejected(changes):
    with pytest.raises(ValueError):KALCBConfig(**changes)
