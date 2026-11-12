from dataclasses import replace
from datetime import date, datetime, time, timedelta

import pytest

from strategy_common.actions import SubmitPartialExit, SubmitExit, ReplaceProtectiveStop
from strategy_common.clock import KST
from strategy_olr.core.stale_release import maybe_release_stale_inventory
from strategy_olr.core.state import OLRState, OLRPositionState
from strategy_olr.core.logic import remember_submitted_order, on_olr_fill
from strategy_olr.core.core_models import OLRFillEvent
from strategy_olr.core.serializers import snapshot_state, restore_state
from tests.strategy_olr.test_capacity_allocation import config, bar


def setup(day=date(2026,1,13), opening=time(9), quantity=51):
    cfg = config(capacity_stale_release=True)
    state = OLRState(); symbol = state.symbol_state('005930')
    symbol.position = OLRPositionState('005930',quantity,100,datetime(2026,1,9,14,35,tzinfo=KST),1,1,
        metadata={'current_stop_price':90, 'trade_exit_plan':{'name':'test','mode':'managed','hard_stop_enabled':True}})
    start = datetime.combine(day,opening,KST)
    symbol.session_bars = [replace(bar(day=day),timestamp=start+timedelta(minutes=5*i),
        open=100,high=100,low=99 if i==2 else 100,close=99 if i==2 else 100) for i in range(3)]
    return cfg,state,symbol.session_bars[-1]


def test_waits_until_second_exchange_session_and_complete_opening_prefix():
    cfg,state,last = setup(date(2026,1,12))  # Monday is next session after Friday.
    assert not maybe_release_stale_inventory(state,last,cfg).actions
    cfg,state,last = setup()
    saved = state.symbols['005930'].session_bars[:]
    for prefix in (saved[:2],saved[1:],saved[::2]):
        state.symbols['005930'].session_bars = prefix
        assert not maybe_release_stale_inventory(state,prefix[-1],cfg).actions
    state.symbols['005930'].session_bars = saved
    result = maybe_release_stale_inventory(state,last,cfg)
    assert [a.qty for a in result.actions if isinstance(a,SubmitPartialExit)] == [25]
    assert state.symbols['005930'].position.qty_open == 51  # Submission is not a fill.


def test_pending_exit_strong_inventory_and_disabled_policy_are_preserved():
    cfg,state,last = setup()
    symbol = state.symbols['005930']; symbol.pending_exit_order_id='auction'
    assert not maybe_release_stale_inventory(state,last,cfg).actions
    symbol.pending_exit_order_id=''
    assert not maybe_release_stale_inventory(state,last,replace(cfg,capacity_stale_release=False)).actions
    strong = replace(last,close=101,high=101)
    symbol.session_bars[-1]=strong
    assert not maybe_release_stale_inventory(state,strong,cfg).actions


def test_future_bars_cannot_change_decision_and_restart_preserves_daily_limit():
    cfg,state,last = setup()
    state.symbols['005930'].session_bars.append(replace(last,timestamp=last.timestamp+timedelta(minutes=5),close=200))
    assert maybe_release_stale_inventory(state,last,cfg).actions
    recovered=restore_state(snapshot_state(state))
    assert not maybe_release_stale_inventory(recovered,last,cfg).actions


def test_partial_fills_resize_protection_and_do_not_credit_unfilled_shares():
    cfg,state,last=setup()
    result=maybe_release_stale_inventory(state,last,cfg)
    action=next(a for a in result.actions if isinstance(a,SubmitPartialExit))
    remember_submitted_order(state,'slice',action)
    filled=on_olr_fill(state,OLRFillEvent('slice','005930','SELL',10,99,last.timestamp+timedelta(minutes=5),
        'stale_inventory_release',{'order_qty':25,'cumulative_filled_qty':10,'execution_id':'first'}),cfg)
    assert state.symbols['005930'].position.qty_open==41
    assert state.symbols['005930'].pending_exit_order_id=='slice'
    assert any(isinstance(a,ReplaceProtectiveStop) and a.qty==41 for a in filled.actions)
    assert not maybe_release_stale_inventory(state,last,cfg).actions


def test_single_share_uses_full_native_exit_and_incompatible_policies_reject():
    cfg,state,last=setup(quantity=1)
    assert any(isinstance(a,SubmitExit) and a.qty==1 for a in maybe_release_stale_inventory(state,last,cfg).actions)
    with pytest.raises(ValueError): config(capacity_stale_release=True,capacity_release_minutes=15)


def test_delayed_open_uses_exchange_hours():
    from kis_core.trading_calendar import get_trading_calendar
    day=date(2025,11,13); opening=get_trading_calendar().session_hours(day)[0]
    assert opening==time(10)
    cfg,state,last=setup(day,opening)
    state.symbols['005930'].position.entry_time=datetime(2025,11,10,14,35,tzinfo=KST)
    assert maybe_release_stale_inventory(state,last,cfg).actions
