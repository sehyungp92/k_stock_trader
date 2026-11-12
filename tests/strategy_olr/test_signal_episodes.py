from datetime import time,datetime
from dataclasses import replace
from copy import deepcopy
from strategy_olr.core.logic import step_olr_core
from strategy_olr.core.state import OLRState,OLRPositionState
from strategy_olr.core.core_models import OLRPortfolioView
from strategy_olr.core.serializers import snapshot_state,restore_state
from strategy_olr.core.signal_episodes import KEY
from strategy_olr.models import OLRDailySnapshot
from strategy_common.clock import KST
from tests.strategy_olr.test_core_runner import _candidate,_bar
from tests.strategy_olr.test_capacity_allocation import config,DAY


def test_signal_is_recorded_while_held_without_creating_an_order():
    c=replace(_candidate('005930',DAY,rank=1),prior_day_high=112)
    snap=OLRDailySnapshot(DAY,(c,),'fixture',datetime.combine(DAY,time(8),tzinfo=KST))
    cfg=config(trade_entry_plan=dict(name='confirm',mode='confirm_next_bar',max_signal_bars=6,
        min_bar_ret=-.001,min_vwap_ret=0,min_close_location=.5,max_vwap_extension_pct=.025))
    s=OLRState();s.meta['observe_structural_signals_v1']=True
    ss=s.symbol_state(c.symbol);ss.position=OLRPositionState(c.symbol,10,100,datetime.combine(DAY,time(13),tzinfo=KST),1,99)
    plain=deepcopy(s);plain.meta.clear()
    bars=[_bar(c.symbol,DAY,time(14,25),100,110,99,100),
          _bar(c.symbol,DAY,time(14,30),100,101,99,100),
          _bar(c.symbol,DAY,time(14,35),100,103,100,102.5)]
    for b in bars:
        r=step_olr_core(s,b,cfg,snap,OLRPortfolioView(10000,11000))
        old=step_olr_core(plain,b,cfg,snap,OLRPortfolioView(10000,11000))
        assert r.actions==old.actions
    episodes=s.meta[KEY][str(DAY)][c.symbol]['episodes']
    assert len([e for e in episodes if e['kind']=='native_confirmation'])==1
    assert not any(e['kind']=='level_breakout' for e in episodes)
    assert 'held' in episodes[0]['blockers'] and not episodes[0]['admission_authority']
    assert restore_state(snapshot_state(s)).meta==s.meta


def test_actual_breakout_and_reclaim_require_observed_sequences():
    from strategy_olr.core.signal_episodes import observe
    from strategy_olr.execution import OLREntryPlan
    s=OLRState();c=_candidate('005930',DAY,rank=1);ss=s.symbol_state(c.symbol)
    snap=OLRDailySnapshot(DAY,(c,),'fixture',datetime.combine(DAY,time(8),tzinfo=KST))
    plan=OLREntryPlan('p',mode='confirm_next_bar',max_signal_bars=6)
    rows=[_bar(c.symbol,DAY,time(14,25),100,101,98,99),_bar(c.symbol,DAY,time(14,30),99,103,99,103)]
    ss.session_bars=list(rows)
    observe(s,rows[-1],c,snap,config(),plan,OLRPortfolioView(0,1000))
    eps=s.meta[KEY][str(DAY)][c.symbol]['episodes']
    assert {'level_breakout','vwap_reclaim'}.issubset({e['kind'] for e in eps})
    assert all('no_cash' in e['blockers'] for e in eps)
    b=_bar(c.symbol,DAY,time(14,35),103,103,99,100);ss.session_bars.append(b)
    observe(s,b,c,snap,config(),plan,OLRPortfolioView(1000,1000))
    assert all(e['status']=='invalidated' for e in eps if e['kind']!='native_confirmation')
