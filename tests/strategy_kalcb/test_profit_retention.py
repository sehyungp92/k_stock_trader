from dataclasses import replace
from datetime import datetime,timedelta
import pytest
from strategy_common.clock import KST
from strategy_common.market import MarketBar
from strategy_kalcb.config import KALCBConfig
from strategy_kalcb.core.state import KALCBState,KALCBPositionState
from strategy_kalcb.core.logic import _profit_retention_trigger


def fixture():
    start=datetime(2025,8,7,9,0,tzinfo=KST)
    s=KALCBState().symbol_state('A')
    for i,c in enumerate([110]*10+[103,102]):
        s.bars_today.append(MarketBar('A',start+timedelta(minutes=i*5),'5m',c,c+1,c-1,c,1000))
    p=KALCBPositionState(symbol='A',qty_entry=100,qty_open=100,entry_price=100,
        entry_time=start+timedelta(minutes=30),initial_stop=99,current_stop=99,risk_per_share=1,
        entry_type='KRX_FIRST30_OPEN',momentum_score=4,hold_bars=6)
    return s,p,s.bars_today[-1],KALCBConfig(exit_profit_retention_enabled=True)


def test_completed_signal_requires_progress_giveback_and_confirmation():
    s,p,b,c=fixture()
    call=lambda **kw:_profit_retention_trigger(s,p,b,c,**kw)
    assert call(current_r=2,mfe_r=10)
    assert not call(current_r=6,mfe_r=10)
    assert not call(current_r=2,mfe_r=5)
    p.hold_bars=5
    assert not call(current_r=2,mfe_r=10)


def test_future_observations_cannot_change_signal_and_default_is_disabled():
    s,p,b,c=fixture()
    assert _profit_retention_trigger(s,p,b,c,current_r=2,mfe_r=10)
    s.bars_today.append(MarketBar('A',b.timestamp+timedelta(minutes=5),'5m',1,1,1,1,1e12))
    assert _profit_retention_trigger(s,p,b,c,current_r=2,mfe_r=10)
    assert not _profit_retention_trigger(s,p,b,replace(c,exit_profit_retention_enabled=False),current_r=2,mfe_r=10)


def test_missing_bar_cannot_supply_two_consecutive_confirmation_closes():
    s,p,b,c=fixture()
    old=list(s.bars_today);s.bars_today.clear();s.bars_today.extend(old[:-2]+old[-1:])
    assert not _profit_retention_trigger(s,p,b,c,current_r=2,mfe_r=10)


def test_each_close_uses_its_own_vwap():
    s,p,b,c=fixture()
    old=list(s.bars_today);s.bars_today.clear()
    # Previous close 111 is above its contemporaneous VWAP. The final high
    # volume bar can lift current VWAP above 111, but cannot rewrite that test.
    s.bars_today.extend(old[:-2])
    s.bars_today.append(MarketBar('A',old[-2].timestamp,'5m',111,111,111,111,100))
    b=MarketBar('A',b.timestamp,'5m',102,1000,102,102,1e6);s.bars_today.append(b)
    assert not _profit_retention_trigger(s,p,b,c,current_r=2,mfe_r=10)


@pytest.mark.parametrize('changes',[{'exit_profit_retention_fraction':0},
    {'exit_profit_retention_fraction':float('nan')},{'exit_profit_retention_start_r':float('inf')},
    {'exit_profit_retention_vwap_closes':0}])
def test_invalid_enabled_rules_fail(changes):
    with pytest.raises(ValueError):KALCBConfig(exit_profit_retention_enabled=True,**changes)


def test_economic_maturity_replaces_small_stop_r_threshold_and_requires_entry_atr():
    s,p,b,c=fixture();c=replace(c,exit_profit_retention_start_atr=1,exit_profit_retention_cost_multiple=4)
    assert not _profit_retention_trigger(s,p,b,c,current_r=2,mfe_r=10)
    p.metadata['daily_atr']=12
    assert not _profit_retention_trigger(s,p,b,c,current_r=2,mfe_r=10)
    p.metadata['daily_atr']=8
    assert _profit_retention_trigger(s,p,b,c,current_r=2,mfe_r=10)
    # A small ATR cannot undercut the known execution-cost floor.
    p.metadata['daily_atr']=.1
    assert not _profit_retention_trigger(s,p,b,c,current_r=.1,mfe_r=1)


def test_partial_retention_submits_once_and_preserves_runner_until_other_exit():
    from strategy_common.actions import SubmitPartialExit,SubmitExit,CancelOrders
    from strategy_kalcb.core.logic import _manage_position
    state=KALCBState();s,p,b,c=fixture();state.symbols['A']=s;s.position=p
    p.update_mark(high=110,low=100);p.metadata['daily_atr']=8
    c=replace(c,exit_profit_retention_start_atr=1,exit_profit_retention_reduce_fraction=.5,
        use_partial_takes=False,exit_hard_stop_enabled=False,quick_exit_enabled=False,
        exit_failed_followthrough_bars=0,exit_target_r=0,exit_mfe_floor_enabled=False)
    result=_manage_position(state,s,b,c)
    action=next(a for a in result.actions if isinstance(a,SubmitPartialExit))
    assert action.qty==50 and not p.exit_in_flight
    assert not _profit_retention_trigger(s,p,b,c,current_r=2,mfe_r=10)
    assert not p.metadata.get('liquidation_reason')
    assert not _manage_position(state,s,b,c).actions
    p.partial_order_id='partial-1'
    # A later full liquidation replaces the pending partial instead of overselling.
    c=replace(c,quick_exit_enabled=True,quick_exit_bars=p.hold_bars+1,quick_exit_min_r=3)
    result=_manage_position(state,s,b,c)
    assert any(isinstance(a,CancelOrders) for a in result.actions)
    assert any(isinstance(a,SubmitExit) and a.qty==100 for a in result.actions)


@pytest.mark.parametrize('changes',[{'exit_profit_retention_start_atr':-1},
    {'exit_profit_retention_cost_multiple':float('nan')},{'exit_profit_retention_roundtrip_cost_bps':-1},
    {'exit_profit_retention_reduce_fraction':0},{'exit_profit_retention_reduce_fraction':1.1}])
def test_invalid_maturity_parameters_fail(changes):
    with pytest.raises(ValueError):KALCBConfig(exit_profit_retention_enabled=True,**changes)
