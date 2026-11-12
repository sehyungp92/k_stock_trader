from dataclasses import replace
from datetime import date, datetime, time, timedelta
import json

import pytest

from strategy_common.actions import SubmitEntry, SubmitExit
from strategy_common.clock import KST
from strategy_olr.config import OLRConfig
from strategy_olr.core.capacity import observe, forecast
from strategy_olr.core.allocation import step_olr_batch
from strategy_olr.core.core_models import OLRPortfolioView
from strategy_olr.core.logic import step_olr_core, remember_submitted_order, pending_entry_reservation
from strategy_olr.core.state import OLRState, OLRPositionState
from strategy_olr.core.serializers import snapshot_state, restore_state
from strategy_olr.models import OLRDailySnapshot
from strategy_olr.engine import OLREngine
from tests.strategy_olr.test_core_runner import _bar, _candidate

DAY = date(2026, 1, 8)

def test_capacity_settings_preserve_candidate_identity_but_bind_execution_identity():
    from types import SimpleNamespace
    from strategy_olr.research import final_candidate_config_fingerprint, _research_config_fingerprint

    base = OLRConfig()
    legacy_values = {key: getattr(base, key) for key in base.__dataclass_fields__
                     if not key.startswith('capacity_')}
    legacy = SimpleNamespace(**legacy_values, __dataclass_fields__=legacy_values)
    configured = replace(base, entry_mode='decision_next_open', capacity_entry_bars=6, capacity_exit_quantile=.5,
        capacity_participation=.005, capacity_min_exit_observations=5,
        capacity_min_position_pct=.0025, capacity_exit_guard=True,
        capacity_exit_guard_mode='excess_only', capacity_joint_allocation=True,
        capacity_defer_minutes=5, capacity_defer_min_fraction=.4,
        capacity_broad_auction_history=True)
    expected = final_candidate_config_fingerprint(legacy)
    assert final_candidate_config_fingerprint(base) == expected
    assert final_candidate_config_fingerprint(configured) == expected
    assert _research_config_fingerprint(base) != _research_config_fingerprint(configured)

def bar(symbol='005930', day=DAY, clock=time(14,30), volume=1000):
    return replace(_bar(symbol, day, clock, 100,100,100,100), volume=volume)

def config(**kw):
    return OLRConfig(entry_mode='decision_next_open', overnight_slot_count=2,
        allocation_mode='rank_weighted', max_position_pct=.65, target_gross_exposure=1.2,
        rank_decay=1.5, **kw)

def snapshot():
    return OLRDailySnapshot(DAY, (_candidate('005930',DAY,rank=1),_candidate('000660',DAY,rank=2)),
        'test',datetime.combine(DAY,time(8),tzinfo=KST))

def history(state,cfg):
    for day,volume in ((date(2026,1,5),1000),(date(2026,1,6),2000),(date(2026,1,7),3000)):
        observe(state,bar(day=day,clock=time(15,30),volume=volume),cfg)

def test_forecast_uses_past_sessions_and_completed_prefix_survives_restart():
    cfg=config(capacity_entry_bars=3,capacity_exit_quantile=.5)
    state=OLRState();history(state,cfg)
    now=bar();observe(state,now,cfg)
    expected=forecast(state,now,cfg)
    assert expected[0]==20 and expected[1]==20
    recovered=restore_state(json.loads(json.dumps(snapshot_state(state))))
    assert forecast(recovered,now,cfg)==expected
    observe(state,bar(clock=time(15,30),volume=999999),cfg)
    assert forecast(state,now,cfg)==expected
    assert all(datetime.fromisoformat(t).date()<DAY for t,_ in expected[2]['exit']['samples'])

def test_entry_forecast_cannot_assume_extra_intervals_after_continuous_close():
    cfg=config(capacity_entry_bars=12);state=OLRState()
    now=bar(clock=time(15,10));observe(state,now,cfg)
    quantity,_,evidence=forecast(state,now,cfg)
    assert quantity==10 and evidence['entry']['horizon']==1
    end=bar(clock=time(15,15));observe(state,end,cfg)
    assert forecast(state,end,cfg)[0]==0

def test_partial_history_does_not_invent_an_auction_forecast():
    cfg=config(capacity_entry_bars=3,capacity_exit_quantile=.5);state=OLRState()
    now=bar();observe(state,now,cfg)
    q,exit_q,e=forecast(state,now,cfg)
    assert q==30 and exit_q is None and e['exit']['status']=='insufficient_history'

def test_capacity_clips_reserved_request_and_is_used_by_live_engine():
    cfg=config(capacity_entry_bars=3);engine=OLREngine(config=cfg,candidate_snapshot=snapshot())
    sent=[]
    def submit(action):
        sent.append(action);return 'entry-1'
    engine.on_bar(bar(),OLRPortfolioView(1_000_000,1_000_000),submit)
    assert sent[0].qty==30
    q,reserved=pending_entry_reservation(engine.state.symbol_state('005930'))
    assert q==30 and reserved==pytest.approx(sent[0].metadata['reserved_notional'])
    assert reserved<4000

def test_joint_allocator_is_permutation_invariant_and_respects_existing_reserves():
    cfg=config(capacity_joint_allocation=True)
    bars=(bar('000660'),bar('005930'))
    view=OLRPortfolioView(cash=5000,equity=10000,open_notional=5000)
    paths=[]
    for sequence in (bars,tuple(reversed(bars))):
        result=step_olr_batch(OLRState(),sequence,cfg,snapshot(),view)
        entries=[a for a in result.actions if isinstance(a,SubmitEntry)]
        assert entries[0].symbol=='005930'
        assert sum(a.metadata['reserved_notional'] for a in entries)<=view.cash
        paths.append([(a.symbol,a.qty) for a in entries])
    assert paths[0]==paths[1]
    state=OLRState()
    reserve=SubmitEntry('OLR','012345',20,'MARKET',None,None,'existing',
        {'estimated_entry_price':100,'reserved_notional':2000})
    remember_submitted_order(state,'existing',reserve)
    result=step_olr_batch(state,bars,cfg,snapshot(),view)
    assert sum(a.metadata['reserved_notional'] for a in result.actions if isinstance(a,SubmitEntry))<=3000

def test_joint_minimum_rejects_tiny_residual_without_reserving_it():
    cfg=config(capacity_joint_allocation=True,capacity_min_position_pct=.02)
    result=step_olr_batch(OLRState(),(bar(),),cfg,snapshot(),OLRPortfolioView(100,10000,open_notional=9900))
    assert not any(isinstance(a,SubmitEntry) for a in result.actions)

def test_exit_guard_generates_future_market_order_only_when_bound_is_exceeded():
    cfg=config(capacity_exit_guard=True);state=OLRState();history(state,cfg)
    s=state.symbol_state('005930')
    s.position=OLRPositionState('005930',50,100,datetime(2026,1,7,14,35,tzinfo=KST),1,1)
    result=step_olr_core(state,bar(),cfg,None,OLRPortfolioView(0,5000,open_notional=5000))
    exits=[a for a in result.actions if isinstance(a,SubmitExit)]
    assert len(exits)==1 and exits[0].order_type=='MARKET'
    assert exits[0].reason=='capacity_exit_before_auction'
    assert exits[0].metadata['capacity_forecast']['exit']['quantity']==20
    # The decision creates an order; it does not fabricate a liquidation.
    assert s.position.qty_open==50

def test_batch_rejects_future_or_duplicate_observations():
    cfg=config(capacity_joint_allocation=True)
    for bars in ((bar(),bar()),(bar(),bar('000660',clock=time(14,35)))):
        with pytest.raises(ValueError):
            step_olr_batch(OLRState(),bars,cfg,snapshot(),OLRPortfolioView(10000,10000))

def test_native_replay_and_live_batch_emit_same_quantities_and_reserves():
    from backtests.engine.replay import drive_timestamp_batch
    from backtests.engine.sim_broker import SimBroker, BrokerCosts
    from backtests.engine.capacity import BarParticipationScenario
    from backtests.strategies.olr.runner import OLRReplayAdapter
    cfg=config(capacity_joint_allocation=True,capacity_entry_bars=3)
    rows=(bar('000660'),bar('005930'))
    sent=[]
    engine=OLREngine(config=cfg,candidate_snapshot=snapshot())
    engine.on_bars(rows,OLRPortfolioView(10000,10000),lambda a:sent.append(a) or a.symbol)
    adapter=OLRReplayAdapter(cfg,{DAY:snapshot()})
    broker=SimBroker(initial_equity=10000,costs=BrokerCosts(2,20,5),capacity_policy=BarParticipationScenario(.01,'test'))
    drive_timestamp_batch(broker,rows,(adapter,))
    assert not broker.fills
    assert [(o.symbol,o.qty,o.metadata['reserved_notional']) for o in broker.orders]==[
        (a.symbol,a.qty,a.metadata['reserved_notional']) for a in sent]
    drive_timestamp_batch(broker,tuple(replace(b,timestamp=b.timestamp+timedelta(minutes=5)) for b in rows),(adapter,))
    assert broker.fills and all(f.timestamp>rows[0].timestamp for f in broker.fills)
    assert all(pending_entry_reservation(s)[0]>=0 for s in adapter.state.symbols.values())

def test_joint_live_policy_refuses_silent_single_bar_fallback():
    engine=OLREngine(config=config(capacity_joint_allocation=True),candidate_snapshot=snapshot())
    with pytest.raises(ValueError,match='completed cohort'):
        engine.on_bar(bar(),OLRPortfolioView(10000,10000),lambda a:'id')

def test_unfunded_signal_waits_only_for_confirmed_cash_and_keeps_original_ttl():
    cfg=config(capacity_joint_allocation=True,capacity_defer_minutes=5)
    state=OLRState()
    state.symbol_state('012345').pending_exit_order_id='actual-exit'
    state.order_roles['actual-exit']={'reason':'capacity_exit_before_auction'}
    step_olr_batch(state,(bar(clock=time(14,25)),),cfg,snapshot(),OLRPortfolioView(100,10000,open_notional=9900))
    first=step_olr_batch(state,(bar(),),cfg,snapshot(),OLRPortfolioView(100,10000,open_notional=9900))
    assert not any(isinstance(a,SubmitEntry) for a in first.actions)
    assert any(e.decision_code=='ENTRY_DEFERRED' for e in first.decisions)
    pending=state.meta['capacity_deferred_intents']['005930']
    assert datetime.fromisoformat(pending['end']).time()==time(14,40)
    assert pending_entry_reservation(state.symbol_state('005930'))==(0,0)
    state=restore_state(json.loads(json.dumps(snapshot_state(state))))
    # Supply a refreshed account after a real fill, never predicted sale cash.
    second=step_olr_batch(state,(bar(clock=time(14,35)),),cfg,snapshot(),OLRPortfolioView(10000,10000))
    assert any(isinstance(a,SubmitEntry) and a.qty>1 for a in second.actions)
    assert not state.meta['capacity_deferred_intents']

def test_deferred_signal_expires_and_cannot_reset_its_validity():
    cfg=config(capacity_joint_allocation=True,capacity_defer_minutes=5)
    state=OLRState();state.symbol_state('012345').pending_exit_order_id='exit'
    state.order_roles['exit']={'reason':'capacity_exit_before_auction'}
    step_olr_batch(state,(bar(clock=time(14,25)),),cfg,snapshot(),OLRPortfolioView(0,10000,open_notional=10000))
    step_olr_batch(state,(bar(),),cfg,snapshot(),OLRPortfolioView(0,10000,open_notional=10000))
    result=step_olr_batch(state,(bar(clock=time(14,40)),),cfg,snapshot(),OLRPortfolioView(10000,10000))
    assert not any(isinstance(a,SubmitEntry) for a in result.actions)
    assert not state.meta['capacity_deferred_intents']

def test_closing_auction_cannot_justify_a_five_minute_funding_deferral():
    cfg=config(capacity_joint_allocation=True,capacity_defer_minutes=5)
    state=OLRState();state.symbol_state('012345').pending_exit_order_id='auction'
    state.order_roles['auction']={'reason':'next_close_exit'}
    result=step_olr_batch(state,(bar(),),cfg,snapshot(),OLRPortfolioView(100,10000,open_notional=9900))
    assert not any(e.decision_code=='ENTRY_DEFERRED' for e in result.decisions)
    assert not state.meta['capacity_deferred_intents']

def test_history_seed_rejects_future_auction_and_timer_expires_an_unobserved_symbol():
    from strategy_olr.core.capacity import seed_history
    from strategy_olr.core.logic import on_olr_timer
    cfg=config(capacity_broad_auction_history=True)
    state=OLRState()
    at=datetime.combine(DAY,time(0),tzinfo=KST)
    old=replace(bar(day=date(2026,1,7),clock=time(15,30)),source_fingerprint='bound-input')
    seed_history(state,[old],cfg,at)
    assert state.meta['capacity_seed_receipt']['observations']==1
    with pytest.raises(ValueError,match='prior-session'):
        seed_history(state,[replace(old,timestamp=at+timedelta(hours=15,minutes=30))],cfg,at)
    state.meta['capacity_deferred_intents']={'005930':{'end':at.isoformat()}}
    result=on_olr_timer(state,at+timedelta(minutes=1),cfg)
    assert len(result.decisions)==1 and not state.meta['capacity_deferred_intents']

def test_daily_warm_start_overlaps_idempotently_and_live_refuses_stale_history():
    cfg=config(capacity_broad_auction_history=True,capacity_joint_allocation=True)
    engine=OLREngine(config=cfg,candidate_snapshot=snapshot())
    old=replace(bar(day=date(2026,1,7),clock=time(15,30)),source_fingerprint='bound-input')
    at=datetime.combine(DAY,time(0),tzinfo=KST)
    engine.seed_capacity_history([old],at)
    engine.seed_capacity_history([old],at)
    assert len(engine.state.meta['capacity_history_v1']['005930']['auction'])==1
    engine.on_bars((bar(),),OLRPortfolioView(10000,10000),lambda a:'id')
    with pytest.raises(ValueError,match='refresh for this session'):
        engine.on_bars((bar(day=date(2026,1,9)),),OLRPortfolioView(10000,10000),lambda a:'id')

def test_capacity_slice_waits_for_confirmed_sale_then_auctions_only_remainder():
    from strategy_common.actions import SubmitPartialExit
    from strategy_olr.core.logic import on_olr_fill
    from strategy_olr.core.core_models import OLRFillEvent
    cfg=config(capacity_exit_guard=True,capacity_exit_guard_mode='excess_only')
    state=OLRState();history(state,cfg);s=state.symbol_state('005930')
    s.position=OLRPositionState('005930',50,100,datetime(2026,1,7,14,35,tzinfo=KST),1,1)
    first=step_olr_core(state,bar(),cfg,None,OLRPortfolioView(0,5000,open_notional=5000))
    partial=next(a for a in first.actions if isinstance(a,SubmitPartialExit))
    assert partial.qty==30
    remember_submitted_order(state,'slice',partial)
    waiting=step_olr_core(state,bar(clock=time(14,35)),cfg,None,OLRPortfolioView(0,5000,open_notional=5000))
    assert not any(isinstance(a,SubmitExit) for a in waiting.actions)
    on_olr_fill(state,OLRFillEvent('slice','005930','SELL',30,100,datetime.combine(DAY,time(14,40),tzinfo=KST),
        metadata={'order_qty':30,'filled_qty_after':30,'fee':0}),cfg)
    final=step_olr_core(state,bar(clock=time(14,40)),cfg,None,OLRPortfolioView(3000,5000,open_notional=2000))
    exits=[a for a in final.actions if isinstance(a,SubmitExit)]
    assert len(exits)==1 and exits[0].order_type=='CLOSE_AUCTION' and exits[0].qty==20
    assert s.position.qty_open==20
