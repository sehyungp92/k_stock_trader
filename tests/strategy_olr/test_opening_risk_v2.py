from dataclasses import replace
from datetime import date, datetime, timedelta, time
import json
import pytest
from strategy_common.actions import CancelOrders, SubmitExit, SubmitPartialExit
from strategy_olr.core.opening_tail import empty,advance,evidence,KEY,tail_mean
from strategy_olr.core.state import OLRState,OLRPositionState
from strategy_olr.core.logic import _maybe_submit_managed_exit,remember_submitted_order,on_olr_fill,on_olr_order_update,on_olr_order_expired
from strategy_olr.core.core_models import OLRFillEvent,OLROrderUpdateEvent,OLRExpiredOrderEvent
from strategy_olr.core.serializers import snapshot_state,restore_state
from strategy_olr.core.loss_budget import unit_loss,committed
from tests.strategy_olr.test_capacity_allocation import config,bar

def rows():
    return [{'symbol':s,'sector':'TECH','day':'2026-03-31','previous_day':'2026-03-30',
             'previous_close':100.,'eligible_open':p,'loss_fraction':max(0.,1-p/100),
             'available_at':'2026-03-31T09:05:00+09:00'} for s,p in [('005930',80.)]+[(f'{i:06d}',90.) for i in range(30)]]

def test_tail_is_strictly_prior_and_round_trip_restart_is_identical():
    state=OLRState();state.meta[KEY]=advance(empty(),rows(),'2026-03-31',['2026-03-31'])
    at=datetime.fromisoformat('2026-04-01T14:35:00+09:00')
    original=evidence(state,'005930','TECH',at)
    restored=restore_state(json.loads(json.dumps(snapshot_state(state))))
    assert evidence(restored,'005930','TECH',at)==original
    assert original['fraction']==pytest.approx((.2+20*.1)/21)
    assert original['peer_observations']==30
    with pytest.raises(ValueError):evidence(state,'005930','TECH',at-timedelta(days=1))

def test_tail_rejects_future_revised_and_unreconciled_observations():
    values=rows();h=advance(empty(),values,'2026-03-31',['2026-03-31'])
    with pytest.raises(ValueError):advance(h,values,'2026-03-30',['2026-03-30'])
    with pytest.raises(ValueError):advance(empty(),values,'2026-03-30',['2026-03-30'])
    with pytest.raises(ValueError):advance(h,[{**values[0],'eligible_open':70.,'loss_fraction':.3}],'2026-03-31',['2026-03-31'])
    with pytest.raises(ValueError):advance(empty(),[{**values[0],'loss_fraction':.1}],'2026-03-31',['2026-03-31'])

def test_tail_rolls_by_sessions_and_does_not_double_count_duplicate_ingestion():
    days=[(date(2025,1,1)+timedelta(days=i)).isoformat() for i in range(65)]
    observations=[{**rows()[0],'day':d,'previous_day':(date.fromisoformat(d)-timedelta(days=1)).isoformat(),
                   'available_at':d+'T09:05:00+09:00'} for d in days]
    h=advance(empty(),observations,days[-1],days)
    assert len(h['observations'])==len(h['sessions'])==60
    assert advance(h,observations,days[-1],days)==h
    assert tail_mean([0]*9+[.2])==.2

def test_tail_changes_sizing_scenario_not_risk_units_and_charges_pending_inventory():
    cfg=config(capacity_opening_tail=True,capacity_joint_allocation=True,capacity_loss_total_pct=.08,
               commission_bps=0,tax_bps_on_sell=0,slippage_bps=0)
    m={'risk_per_share':2.,'estimated_entry_price':100.,'opening_tail_evidence':{'fraction':.1}}
    assert unit_loss(m,cfg)==10
    assert unit_loss(m,replace(cfg,capacity_opening_tail=False))==2.5
    with pytest.raises(ValueError):unit_loss({'risk_per_share':2,'estimated_entry_price':100},cfg)
    s=OLRState();a=s.symbol_state('005930');a.position=OLRPositionState('005930',10,100,bar().timestamp,1,1.,metadata=m)
    a.pending_exit_order_id='sell'
    assert committed(s,cfg)[0]['005930']==100

def weak_setup():
    cfg=config(capacity_weak_partial_fraction=.5,capacity_exit_priority=True,
      trade_exit_plan={'name':'weak-test','mode':'managed','hard_stop_enabled':False,
      'failed_followthrough_bars':1,'failed_followthrough_close_r':-1.,'failed_followthrough_mfe_r':4.})
    current=replace(bar(clock=time(9,10)),open=95.,high=95.,low=95.,close=95.)
    state=OLRState();ss=state.symbol_state(current.symbol)
    ss.position=OLRPositionState(current.symbol,51,100,current.timestamp-timedelta(days=1),1,1.,
       entry_order_id='entry',metadata={'risk_per_share':10.,'trade_exit_plan':dict(cfg.trade_exit_plan)})
    ss.session_bars=[replace(current,timestamp=current.timestamp-timedelta(minutes=i)) for i in (10,5,0)]
    return cfg,state,current

def test_weak_partial_waits_for_three_current_session_bars_and_latches_once():
    cfg,state,current=weak_setup();ss=state.symbols[current.symbol]
    old=[replace(current,timestamp=current.timestamp-timedelta(days=1,minutes=i)) for i in range(10)]
    ss.session_bars=old+[current]
    assert not _maybe_submit_managed_exit(state,current,cfg).actions
    ss.session_bars=old+[replace(current,timestamp=current.timestamp-timedelta(minutes=i)) for i in (10,5,0)]
    action=_maybe_submit_managed_exit(state,current,cfg).actions[0]
    assert isinstance(action,SubmitPartialExit) and action.qty==25
    assert not _maybe_submit_managed_exit(state,current,cfg).actions
    restored=restore_state(json.loads(json.dumps(snapshot_state(state))))
    assert not _maybe_submit_managed_exit(restored,current,cfg).actions

def test_stronger_exit_preempts_partial_and_pending_partial_waits_for_cancel_ack():
    cfg,state,current=weak_setup();ss=state.symbols[current.symbol]
    partial=_maybe_submit_managed_exit(state,current,cfg).actions[0];remember_submitted_order(state,'partial',partial)
    crash=replace(current,close=88.,low=87.);ss.session_bars[-1]=crash
    result=_maybe_submit_managed_exit(state,crash,cfg)
    assert len(result.actions)==1 and isinstance(result.actions[0],CancelOrders)
    assert not _maybe_submit_managed_exit(state,crash,cfg).actions
    on_olr_fill(state,OLRFillEvent('partial',current.symbol,'SELL',7,92,current.timestamp,
                                 metadata={'submitted_qty':25}),cfg)
    ack=OLROrderUpdateEvent('partial',current.symbol,'CANCELLED',current.timestamp+timedelta(minutes=5))
    result=on_olr_order_update(state,ack,cfg)
    sells=[a for a in result.actions if isinstance(a,SubmitExit)]
    assert len(sells)==1 and sells[0].qty==44
    assert isinstance(result.actions[0],CancelOrders)
    assert not on_olr_order_update(state,ack,cfg).actions


@pytest.mark.parametrize('terminal',['REJECTED','EXPIRED'])
def test_partial_terminal_does_not_repeat_or_disable_full_guard(terminal):
    cfg,state,current=weak_setup();ss=state.symbols[current.symbol]
    partial=_maybe_submit_managed_exit(state,current,cfg).actions[0]
    remember_submitted_order(state,'partial',partial)
    if terminal=='EXPIRED':
        result=on_olr_order_expired(state,OLRExpiredOrderEvent('partial',current.symbol,'SELL','MARKET',25,current.timestamp),cfg)
    else:
        result=on_olr_order_update(state,OLROrderUpdateEvent('partial',current.symbol,terminal,current.timestamp),cfg)
    assert not result.actions and not ss.pending_exit_order_id
    assert not _maybe_submit_managed_exit(state,current,cfg).actions
    crash=replace(current,close=88.,low=87.);ss.session_bars[-1]=crash
    sells=[a for a in _maybe_submit_managed_exit(state,crash,cfg).actions if isinstance(a,SubmitExit)]
    assert len(sells)==1 and sells[0].qty==51

@pytest.mark.parametrize('change',['healthy','already_followed','same_day','pending_entry','disabled','one_share'])
def test_partial_preserves_non_targeted_positions(change):
    cfg,state,current=weak_setup();ss=state.symbols[current.symbol]
    if change=='healthy':current=replace(current,close=99.)
    if change=='already_followed':ss.position.max_favorable_price=106.
    if change=='same_day':ss.position.entry_time=current.timestamp-timedelta(minutes=30)
    if change=='pending_entry':ss.pending_entry_order_id='entry'
    if change=='disabled':cfg=replace(cfg,capacity_weak_partial_fraction=0.)
    if change=='one_share':ss.position.qty_open=1
    ss.session_bars[-1]=current
    assert not _maybe_submit_managed_exit(state,current,cfg).actions

@pytest.mark.parametrize('kw',[{'capacity_opening_tail':1},{'capacity_opening_tail':True},
 {'capacity_weak_partial_fraction':.6},{'capacity_weak_partial_fraction':float('nan')},
 {'capacity_weak_partial_fraction':.5}])
def test_invalid_feature_contract_fails(kw):
    with pytest.raises(ValueError):config(**kw)


def test_native_partial_is_next_open_and_retains_full_guard_for_remaining_inventory():
    from strategy_common.clock import KST
    from backtests.engine.replay import drive_timestamp_batch
    from backtests.engine.sim_broker import SimBroker,BrokerCosts
    from backtests.engine.capacity import BarParticipationScenario
    from backtests.strategies.olr.runner import OLRReplayAdapter
    from strategy_olr.models import OLRDailySnapshot
    from tests.strategy_olr.test_core_runner import _candidate
    from tests.strategy_olr.test_capacity_allocation import DAY
    yesterday=date(2026,1,7)
    snap=OLRDailySnapshot(yesterday,(_candidate('005930',yesterday,rank=1),),'test',datetime.combine(yesterday,time(8),tzinfo=KST))
    cfg,_,_=weak_setup();adapter=OLRReplayAdapter(cfg,{yesterday:snap})
    broker=SimBroker(initial_equity=10000,costs=BrokerCosts(2,20,5),capacity_policy=BarParticipationScenario(.01,'test'))
    def quote(day,t,price):return replace(bar(day=day,clock=t,volume=100000),open=price,high=price,low=price,close=price)
    for clock in (time(14,25),time(14,30),time(14,35)):
        drive_timestamp_batch(broker,(quote(yesterday,clock,100),),(adapter,))
    assert len(broker.fills)==1
    qty=broker.fills[0].qty;adapter.state.symbols['005930'].position.metadata['risk_per_share']=10.
    for clock in (time(9,0),time(9,5),time(9,10)):
        drive_timestamp_batch(broker,(quote(DAY,clock,95),),(adapter,))
    assert len(broker.fills)==1
    drive_timestamp_batch(broker,(quote(DAY,time(9,15),95),),(adapter,))
    sell=[f for f in broker.fills if f.side=='SELL']
    assert len(sell)==1 and sell[0].qty==qty//2 and sell[0].reason=='weak_followthrough_partial'
    for clock in (time(9,20),time(9,25)):
        drive_timestamp_batch(broker,(quote(DAY,clock,88),),(adapter,))
    sell=[f for f in broker.fills if f.side=='SELL']
    assert len(sell)==2 and sum(f.qty for f in sell)==qty and sell[-1].reason=='failed_followthrough'
    assert sell[-1].timestamp.time()==time(9,25) and broker.same_bar_fill_violations==0
    assert not broker.orders and not adapter.state.symbols['005930'].position
