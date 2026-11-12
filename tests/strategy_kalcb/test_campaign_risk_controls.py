from dataclasses import replace
from datetime import datetime,timedelta
import pytest
from strategy_common.market import MarketBar
from strategy_common.clock import KST
from strategy_kalcb.config import KALCBConfig
from strategy_kalcb.risk import governed_entry_qty
from strategy_kalcb.core.state import KALCBPositionState,KALCBState
from strategy_kalcb.core.logic import _exit_reason,_timed_acceptance_intact
from strategy_kalcb.core.logic import _governed_open_notional
from strategy_kalcb.core.core_models import KALCBPortfolioView
from strategy_kalcb.core.state import SymbolStage
from strategy_kalcb.risk import volatility_quantity_scale
from strategy_kalcb.core.logic import _update_portfolio_context

def test_governor_is_reduction_only_and_scales_after_action():
    cfg=KALCBConfig(intraday_leverage=2,risk_max_single_equity_pct=.35,risk_max_gross_equity_pct=1.2,
        risk_drawdown_start_pct=.03,risk_drawdown_full_pct=.07,risk_drawdown_min_mult=.5)
    f=lambda q,dd,used=0:governed_entry_qty(qty=q,entry_price=100,config=cfg,
        portfolio_equity=100000,open_notional=used,portfolio_drawdown_pct=dd)
    assert f(1000,0)==350
    assert f(200,0)==200
    assert f(200,.05)==150
    assert f(200,.1)==100
    assert f(200,0,115000)==50
    assert f(200,0,125000)==0
    assert governed_entry_qty(qty=100,entry_price=100,config=cfg)==0
    for q in (0,1,99,200,1000):
        assert governed_entry_qty(qty=q,entry_price=100,config=KALCBConfig())==q

@pytest.mark.parametrize('kwargs',[
    {'entry_quantity_multiplier':1.1},{'risk_drawdown_start_pct':.07,'risk_drawdown_full_pct':.03},
    {'risk_max_single_equity_pct':float('nan')},{'risk_drawdown_min_mult':0},
    {'exit_timed_acceptance_extension_bars':-1}])
def test_bad_risk_controls_are_rejected(kwargs):
    with pytest.raises(ValueError):KALCBConfig(**kwargs)

def fixture():
    state=KALCBState().symbol_state('A');state.or_low=98;state.or_high=102
    start=datetime(2025,8,7,9,0,tzinfo=KST)
    state.bars_today.extend(MarketBar('A',start+timedelta(minutes=5*i),'5m',100,101,99,100,1000) for i in range(12))
    pos=KALCBPositionState(symbol='A',qty_entry=100,qty_open=100,entry_price=101,
        entry_time=start+timedelta(minutes=30),initial_stop=99,current_stop=99,risk_per_share=2,
        entry_type='KRX_FIRST30_OPEN',momentum_score=4,hold_bars=6)
    cfg=KALCBConfig(exit_failed_followthrough_bars=6,exit_failed_followthrough_mfe_r=.75,
        exit_failed_followthrough_close_r=-.25,exit_timed_acceptance_extension_bars=3,
        quick_exit_enabled=False,mfe_conviction_enabled=False,flow_reversal_enabled=False)
    return state,pos,cfg,state.bars_today[-1]

def test_extension_expires_and_does_not_suppress_other_exits():
    state,pos,cfg,bar=fixture()
    assert _exit_reason(state,pos,bar,cfg,current_r=-.5,mfe_r=.1)==''
    assert pos.metadata['timed_acceptance_pending']['deadline_bar']==9
    pos.hold_bars=9
    assert _exit_reason(state,pos,bar,cfg,current_r=-.5,mfe_r=.1)=='failed_followthrough'
    state,pos,cfg,bar=fixture();cfg=replace(cfg,exit_max_hold_bars=6)
    assert _exit_reason(state,pos,bar,cfg,current_r=-.5,mfe_r=.1)=='max_hold'

def test_acceptance_break_cancels_deferral_and_future_bar_is_ignored():
    state,pos,cfg,bar=fixture()
    assert _exit_reason(state,pos,bar,cfg,current_r=-.5,mfe_r=.1)==''
    state.bars_today.append(MarketBar('A',bar.timestamp+timedelta(minutes=5),'5m',1e9,1e9,1e9,1e9,1e9))
    assert _timed_acceptance_intact(state,bar)
    broken=MarketBar('A',bar.timestamp,'5m',100,101,97,98,1000)
    pos.hold_bars=7
    assert _exit_reason(state,pos,broken,cfg,current_r=-1.5,mfe_r=.1)=='failed_followthrough'

def test_disabled_extension_preserves_original_timing():
    state,pos,cfg,bar=fixture();cfg=replace(cfg,exit_timed_acceptance_extension_bars=0)
    assert _exit_reason(state,pos,bar,cfg,current_r=-.5,mfe_r=.1)=='failed_followthrough'
    pos.hold_bars=7
    assert _exit_reason(state,pos,bar,cfg,current_r=-.5,mfe_r=.1)==''

def test_gross_cap_reserves_pending_and_partially_filled_entries():
    state=KALCBState();s=state.symbol_state('A');s.stage=SymbolStage.ENTRY_QUEUED
    s.pending_entry_order_id='o1'
    s.pending_entry_metadata={'risk_reserved_entry_qty':100,'entry_price_ref':100}
    view=KALCBPortfolioView(cash=100000)
    assert _governed_open_notional(state,view)==10000
    _,position,_,_=fixture();position.qty_entry=40;position.qty_open=40
    position.entry_price=100;position.entry_order_id='o1';s.position=position;s.stage=SymbolStage.IN_POSITION
    view.open_notional=4000
    assert _governed_open_notional(state,view)==10000
    s.pending_entry_order_id='';s.pending_entry_metadata.clear()
    assert _governed_open_notional(state,view)==4000

def test_volatility_scale_cannot_observe_future_returns():
    cfg=KALCBConfig(risk_daily_vol_target=.008,risk_volatility_lookback=2)
    history=[{'day':'2025-08-07','return':.0065,'scale':.65},
        {'day':'2025-08-08','return':-.0065,'scale':.65}]
    expected=volatility_quantity_scale(history,'2025-08-11',cfg)
    assert expected==pytest.approx(.8)
    history.extend([{'day':'2025-08-11','return':-999,'scale':.01},
        {'day':'2026-07-01','return':999,'scale':.01}])
    assert volatility_quantity_scale(history,'2025-08-11',cfg)==expected
    assert volatility_quantity_scale([],'2025-08-11',cfg)==.65

def test_volatility_history_updates_once_per_session_and_uses_cached_prior_equity():
    from datetime import date
    cfg=KALCBConfig(risk_daily_vol_target=.008,risk_volatility_lookback=2)
    state=KALCBState();state.session_date=date(2025,8,7)
    _update_portfolio_context(state,KALCBPortfolioView(cash=100,equity=100),cfg)
    _update_portfolio_context(state,KALCBPortfolioView(cash=101,equity=101),cfg)
    assert state.meta['portfolio_volatility_history']==[]
    state.session_date=date(2025,8,8)
    _update_portfolio_context(state,KALCBPortfolioView(cash=1000,equity=1000),cfg)
    history=state.meta['portfolio_volatility_history']
    assert len(history)==1 and history[0]['day']=='2025-08-07'
    assert history[0]['return']==pytest.approx(.01)
    _update_portfolio_context(state,KALCBPortfolioView(cash=1,equity=1),cfg)
    assert state.meta['portfolio_volatility_history']==history
