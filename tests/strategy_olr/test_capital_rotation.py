from dataclasses import replace
from datetime import datetime, time, timedelta

from deployment.olr_kalcb.intent_allocator import AllocationAccount, EntryIntent
from strategy_common.actions import CancelOrders, SubmitEntry, SubmitExit, SubmitPartialExit
from strategy_olr.core.capital_rotation import rotation_actions
from strategy_olr.core.allocation import step_olr_batch
from strategy_olr.core.core_models import OLRPortfolioView, OLROrderUpdateEvent
from strategy_olr.core.logic import remember_submitted_order, on_olr_order_update
from strategy_olr.core.state import OLRState, OLRPositionState
from tests.strategy_olr.test_capacity_allocation import config, bar, snapshot, DAY
from strategy_common.clock import KST


def setup(pending=False):
    cfg=config(capacity_joint_allocation=True,capacity_rotation=True,capacity_defer_minutes=10,
        capacity_defer_min_fraction=1.0)
    state=OLRState();s=state.symbol_state('005930')
    s.position=OLRPositionState('005930',50,100,datetime(2026,1,7,14,35,tzinfo=KST),1,1)
    current=bar('005930',clock=time(14,35));current=replace(current,open=99,high=99,low=99,close=99)
    s.session_bars=[bar('005930',clock=time(14,25)),current]
    at=current.timestamp+timedelta(minutes=5)
    intent=EntryIntent('new','OLR','000660','TEST',15,100,at,at+timedelta(minutes=30),native_rank=2)
    account=AllocationAccount(5000,5000,5000,5000,0,{'OLR':5000},gross_committed=5000,strategy_committed={'OLR':5000})
    if pending:
        remember_submitted_order(state,'auction',SubmitExit('OLR','005930',50,'CLOSE_AUCTION',90,'next_close_exit',{}))
        s.exit_attempted_dates.add(DAY)
    return cfg,state,current,at,intent,account


def test_rotation_waits_for_cancel_ack_and_does_not_credit_predicted_cash():
    cfg,state,current,at,intent,account=setup(True)
    actions,events,waiting=rotation_actions(state,[current],cfg,[intent],account,[],at)
    assert waiting and len(actions)==1 and isinstance(actions[0],CancelOrders)
    deadline=state.meta['capacity_rotation_pending']['005930']['end']
    actions,_,waiting=rotation_actions(state,[current],cfg,[intent],account,[],at)
    assert waiting and not actions
    assert state.meta['capacity_rotation_pending']['005930']['end']==deadline
    on_olr_order_update(state,OLROrderUpdateEvent('auction','005930','CANCELLED',at+timedelta(minutes=1),side='SELL'),cfg)
    actions,_,waiting=rotation_actions(state,[current],cfg,[intent],account,[],at+timedelta(minutes=5))
    partials=[a for a in actions if isinstance(a,SubmitPartialExit)]
    assert waiting and len(partials)==1 and 0<partials[0].qty<=25
    assert account.funding_limit==0 and state.symbols['005930'].position.qty_open==50
    assert not state.meta['capacity_rotation_pending']


def test_invalidated_entry_restores_auction_after_cancel_ack():
    cfg,state,current,at,intent,account=setup(True)
    rotation_actions(state,[current],cfg,[intent],account,[],at)
    on_olr_order_update(state,OLROrderUpdateEvent('auction','005930','CANCELLED',at+timedelta(minutes=1),side='SELL'),cfg)
    actions,_,waiting=rotation_actions(state,[current],cfg,[],account,[],at+timedelta(minutes=5))
    exits=[a for a in actions if isinstance(a,SubmitExit)]
    assert len(exits)==1 and exits[0].order_type=='CLOSE_AUCTION' and exits[0].qty==50
    assert not waiting and not state.meta['capacity_rotation_pending']


def test_strong_incumbent_is_preserved_and_no_shortfall_means_no_rotation():
    cfg,state,current,at,intent,account=setup(False)
    strong=replace(current,open=101,high=101,low=101,close=101)
    state.symbols['005930'].session_bars[-1]=strong
    assert not rotation_actions(state,[strong],cfg,[intent],account,[],at)[0]
    assert not rotation_actions(state,[current],cfg,[intent],replace(account,funding_limit=2000,gross_committed=3000,strategy_committed={'OLR':3000}),[],at)[0]


def test_batch_replaces_unsubmitted_auction_and_defers_entry_without_phantom_reserve():
    cfg,state,current,at,intent,account=setup(False)
    # Build the normal pre-signal prefix used by the deferral revalidation.
    step_olr_batch(state,(bar('000660',clock=time(14,25)),bar('005930',clock=time(14,25))),
        cfg,snapshot(),OLRPortfolioView(0,5000,open_notional=5000))
    weak=replace(current,timestamp=current.timestamp-timedelta(minutes=5))
    result=step_olr_batch(state,(bar('000660'),weak),cfg,snapshot(),OLRPortfolioView(0,5000,open_notional=5000))
    assert any(isinstance(a,SubmitPartialExit) and a.reason=='opportunity_capital_rotation' for a in result.actions)
    assert not any(isinstance(a,SubmitExit) and a.order_type=='CLOSE_AUCTION' for a in result.actions)
    assert not any(isinstance(a,SubmitEntry) for a in result.actions)
    assert '000660' in state.meta['capacity_deferred_intents']
    assert not any(e.reason=='next_close_exit' and e.decision_code=='EXIT_SUBMITTED' for e in result.decisions)


def test_native_auction_cancel_sale_then_revalidated_entry_are_sequential():
    from datetime import date
    from backtests.engine.replay import drive_timestamp_batch
    from backtests.engine.sim_broker import SimBroker, BrokerCosts
    from backtests.engine.capacity import BarParticipationScenario
    from backtests.strategies.olr.runner import OLRReplayAdapter
    from strategy_olr.models import OLRDailySnapshot
    from tests.strategy_olr.test_core_runner import _candidate
    yesterday=date(2026,1,7)
    first=OLRDailySnapshot(yesterday,(_candidate('005930',yesterday,rank=1),),'test',datetime.combine(yesterday,time(8),tzinfo=KST))
    second=OLRDailySnapshot(DAY,(_candidate('000660',DAY,rank=1),_candidate('005930',DAY,rank=2)),
        'test',datetime.combine(DAY,time(8),tzinfo=KST))
    cfg=config(capacity_joint_allocation=True,capacity_rotation=True,capacity_defer_minutes=10,
        capacity_defer_min_fraction=1,trade_entry_plan={'name':'confirmation_test','mode':'confirm_next_bar','max_signal_bars':6,
            'min_bar_ret':.005,'min_close_location':.5})
    adapter=OLRReplayAdapter(cfg,{yesterday:first,DAY:second})
    broker=SimBroker(initial_equity=10000,costs=BrokerCosts(2,20,5),capacity_policy=BarParticipationScenario(.01,'test'))
    def quote(sym,d,t,op,cl):
        return replace(bar(sym,day=d,clock=t,volume=100000),open=op,high=max(op,cl),low=min(op,cl),close=cl)
    for t in [time(14,25),time(14,30),time(14,35)]:
        drive_timestamp_batch(broker,(quote('005930',yesterday,t,100,101),),(adapter,))
    initial=[f for f in broker.fills if f.side=='BUY']
    assert initial
    for minute in [25,30,35,40,45,50]:
        t=time(14,minute)
        old=quote('005930',DAY,t,100 if minute==25 else 99,100 if minute==25 else 99)
        new=quote('000660',DAY,t,100,100 if minute<=30 else 101)
        drive_timestamp_batch(broker,(old,new),(adapter,))
    sales=[f for f in broker.fills if f.side=='SELL']
    incoming=[f for f in broker.fills if f.side=='BUY' and f.symbol=='000660']
    assert sales and incoming
    assert sum(f.qty for f in sales)<=sum(f.qty for f in initial)*.5
    assert min(f.timestamp for f in incoming)>max(f.timestamp for f in sales)
    assert any(j['event']=='cancel' for j in broker.journal)
    assert broker.cash>=0
