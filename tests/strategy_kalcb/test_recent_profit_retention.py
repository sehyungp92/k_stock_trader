from dataclasses import replace
from datetime import datetime, timedelta
import pytest
from strategy_common.clock import KST
from strategy_common.market import MarketBar
from strategy_kalcb.config import KALCBConfig
from strategy_kalcb.core.state import KALCBState, KALCBPositionState
from strategy_kalcb.core.logic import _profit_retention_trigger


def state():
    start=datetime(2025,8,7,9,0,tzinfo=KST)
    s=KALCBState().symbol_state('A')
    for i,c in enumerate([100]*10+[120]*4+[113,112]):
        s.bars_today.append(MarketBar('A',start+timedelta(minutes=5*i),'5m',c,c+1,c-1,c,1000))
    p=KALCBPositionState(symbol='A',qty_entry=100,qty_open=100,entry_price=100,
        entry_time=start+timedelta(minutes=30),initial_stop=99,current_stop=99,risk_per_share=1,
        entry_type='KRX_FIRST30_OPEN',momentum_score=4,hold_bars=10,metadata={'daily_atr':10})
    c=KALCBConfig(exit_profit_retention_enabled=True,exit_profit_retention_start_atr=1,
        exit_profit_retention_fraction=.65,exit_profit_retention_confirmation='recent_vwap')
    return s,p,s.bars_today[-1],c


def test_recent_weakness_can_reduce_while_still_above_session_vwap():
    s,p,b,c=state()
    assert _profit_retention_trigger(s,p,b,c,current_r=12,mfe_r=20)
    assert not _profit_retention_trigger(s,p,b,replace(c,exit_profit_retention_confirmation='session_vwap'),current_r=12,mfe_r=20)
    # Remaining trend strength still prevents a reduction.
    assert not _profit_retention_trigger(s,p,b,c,current_r=16,mfe_r=20)


def test_future_volume_cannot_rewrite_recent_confirmation():
    s,p,b,c=state()
    before=_profit_retention_trigger(s,p,b,c,current_r=12,mfe_r=20)
    s.bars_today.append(MarketBar('A',b.timestamp+timedelta(minutes=5),'5m',1,1,1,1,1e12))
    assert _profit_retention_trigger(s,p,b,c,current_r=12,mfe_r=20)==before


def test_recent_window_requires_contiguous_observations():
    s,p,b,c=state();old=list(s.bars_today);s.bars_today.clear();s.bars_today.extend(old[:12]+old[13:])
    assert not _profit_retention_trigger(s,p,b,c,current_r=12,mfe_r=20)


def test_larger_move_overlay_preserves_the_original_session_rule():
    s,p,b,c=state()
    c=replace(c,exit_profit_retention_confirmation='session_vwap',exit_profit_retention_recent_overlay_atr=2.)
    assert _profit_retention_trigger(s,p,b,c,current_r=12,mfe_r=20)
    assert not _profit_retention_trigger(s,p,b,c,current_r=9,mfe_r=15)
    # A loss below session VWAP still triggers the original lower-maturity rule.
    s.bars_today[-2]=replace(s.bars_today[-2],open=95,high=96,low=94,close=95)
    s.bars_today[-1]=replace(b,open=94,high=95,low=93,close=94)
    assert _profit_retention_trigger(s,p,s.bars_today[-1],c,current_r=-6,mfe_r=15)


@pytest.mark.parametrize('change',[{'exit_profit_retention_confirmation':'future_vwap'}, {'exit_profit_retention_recent_bars':1}])
def test_invalid_confirmation_rejected(change):
    with pytest.raises(ValueError):KALCBConfig(exit_profit_retention_enabled=True,**change)
