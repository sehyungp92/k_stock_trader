from dataclasses import replace
from datetime import timedelta, time
import json
import pytest

from deployment.olr_kalcb.intent_allocator import AllocationAccount, EntryIntent, IntentAllocationConfig, allocate_entry_intents
from strategy_common.actions import CancelOrders, SubmitExit, SubmitEntry
from strategy_olr.core.logic import _maybe_submit_managed_exit, remember_submitted_order, on_olr_order_update, on_olr_fill
from strategy_olr.core.core_models import OLROrderUpdateEvent, OLRFillEvent, OLRPortfolioView
from strategy_olr.core.state import OLRState, OLRPositionState
from strategy_olr.core.serializers import snapshot_state, restore_state
from strategy_olr.core.loss_budget import allocate, committed, unit_loss
from tests.strategy_olr.test_capacity_allocation import config, bar, snapshot


def exit_setup(kind='CLOSE_AUCTION', enabled=True):
    cfg = config(capacity_exit_priority=enabled, trade_exit_plan={
        'name': 'priority_test', 'mode': 'managed', 'hard_stop_enabled': False, 'failed_followthrough_bars': 1,
        'failed_followthrough_close_r': -1., 'failed_followthrough_mfe_r': 4.})
    state = OLRState()
    current = replace(bar(clock=time(15, 0)), open=88, high=89, low=87, close=88)
    s = state.symbol_state(current.symbol)
    s.position = OLRPositionState(current.symbol, 50, 100, current.timestamp-timedelta(days=1),
        1, 1., entry_order_id='buy', metadata={'risk_per_share': 10})
    s.session_bars = [current]
    remember_submitted_order(state, 'auction', SubmitExit('OLR', current.symbol, 50, kind,
        95 if kind=='CLOSE_AUCTION' else None, 'next_close_exit' if kind=='CLOSE_AUCTION' else 'fallback'))
    return cfg, state, current


def test_priority_waits_for_matching_ack_persists_and_uses_remaining_qty():
    cfg, state, current = exit_setup()
    result = _maybe_submit_managed_exit(state, current, cfg)
    assert len(result.actions)==1 and isinstance(result.actions[0], CancelOrders)
    assert not _maybe_submit_managed_exit(state, current, cfg).actions
    state = restore_state(json.loads(json.dumps(snapshot_state(state))))
    # A partial execution during the cancellation race precedes terminal ack.
    on_olr_fill(state, OLRFillEvent('auction', current.symbol, 'SELL', 7, 96,
        current.timestamp+timedelta(minutes=6), metadata={'submitted_qty': 50}), cfg)
    wrong = OLROrderUpdateEvent('unrelated', current.symbol, 'CANCELLED', current.timestamp+timedelta(minutes=7))
    assert not on_olr_order_update(state, wrong, cfg).actions
    ack = replace(wrong, order_id='auction', timestamp=current.timestamp+timedelta(minutes=10))
    result = on_olr_order_update(state, ack, cfg)
    sells = [a for a in result.actions if isinstance(a, SubmitExit)]
    assert len(sells)==1 and sells[0].qty==43 and sells[0].order_type=='MARKET'
    assert sells[0].reason=='failed_followthrough'
    assert not on_olr_order_update(state, ack, cfg).actions
    remember_submitted_order(state, 'replacement', sells[0])
    assert not _maybe_submit_managed_exit(state, current, cfg).actions


@pytest.mark.parametrize('kind,enabled', [('MARKET',True),('CLOSE_AUCTION',False)])
def test_priority_preserves_market_orders_and_disabled_behavior(kind, enabled):
    cfg,state,current=exit_setup(kind,enabled)
    assert not _maybe_submit_managed_exit(state,current,cfg).actions


def test_priority_preserves_nontriggered_auction_and_full_fill_race():
    cfg,state,current=exit_setup()
    healthy=replace(current,close=99)
    state.symbols[current.symbol].session_bars=[healthy]
    assert not _maybe_submit_managed_exit(state,healthy,cfg).actions
    state.symbols[current.symbol].session_bars=[current]
    _maybe_submit_managed_exit(state,current,cfg)
    on_olr_fill(state,OLRFillEvent('auction',current.symbol,'SELL',50,96,current.timestamp,
        metadata={'submitted_qty':50}),cfg)
    ack=OLROrderUpdateEvent('auction',current.symbol,'CANCELLED',current.timestamp)
    assert not on_olr_order_update(state,ack,cfg).actions
    assert not state.meta['priority_exit_pending']


def risk_setup(**kw):
    cfg=config(capacity_joint_allocation=True,commission_bps=0,tax_bps_on_sell=0,slippage_bps=0,
        capacity_loss_atr_multiple=1.,**kw)
    at=bar().timestamp
    account=AllocationAccount(10000,10000,10000,10000,10000,{'OLR':10000})
    actions={};intents=[]
    for rank,symbol,sector in ((1,'005930','TECH'),(2,'000660','TECH'),(3,'035420','IT')):
        actions[symbol]=SubmitEntry('OLR',symbol,65,'MARKET',None,None,'test',
            {'estimated_entry_price':100.,'risk_per_share':10.,'sector':sector})
        intents.append(EntryIntent(symbol,'OLR',symbol,sector,65,100.,at,at+timedelta(minutes=5),native_rank=rank))
    return cfg,OLRState(),account,at,actions,intents


def test_joint_risk_caps_release_cash_to_lower_ranks_and_are_order_invariant():
    cfg,state,account,at,actions,intents=risk_setup(capacity_loss_position_pct=.04,
        capacity_loss_sector_pct=.06,capacity_loss_total_pct=.08)
    d=allocate(intents,account,at,cfg,state,actions)
    assert [x.approved_qty for x in d]==[40,20,20]
    assert allocate(list(reversed(intents)),account,at,cfg,state,actions)==d
    assert sum(x.reserved_notional for x in d)==8000


def test_disabled_risk_is_exact_native_allocation():
    cfg,state,account,at,actions,intents=risk_setup()
    assert allocate(intents,account,at,cfg,state,actions)==allocate_entry_intents(intents,account,at,IntentAllocationConfig())


def test_pending_buys_and_filled_positions_count_pending_sells_do_not_release():
    cfg,state,account,at,actions,intents=risk_setup(capacity_loss_total_pct=.08)
    s=state.symbol_state('111111')
    s.position=OLRPositionState('111111',20,100,at,1,1.,sector='TECH',metadata={'risk_per_share':10})
    s.pending_exit_order_id='pending-sale'
    buy=replace(actions['005930'],symbol='222222',qty=30)
    remember_submitted_order(state,'pending-buy',buy)
    state.symbols['222222'].pending_entry_metadata['entry_filled_qty']=5
    used,_=committed(state,cfg)
    assert sum(used.values())==450
    d=allocate(intents,account,at,cfg,state,actions)
    assert sum(x.approved_qty for x in d)==35


def test_redistribution_cannot_bypass_prior_cohort_risk():
    cfg,state,account,at,actions,intents=risk_setup(capacity_loss_position_pct=.04,
        capacity_loss_sector_pct=.06,capacity_loss_total_pct=.08)
    base=allocate(intents[:2],account,at,cfg,state,actions)
    residual=replace(account,funding_limit=4000,gross_committed=6000,strategy_committed={'OLR':6000})
    extra=allocate(intents,residual,at,cfg,state,actions,prior=base)
    assert [x.approved_qty for x in extra]==[0,0,20]


def test_batch_applies_loss_limits_after_liquidity_topups():
    from strategy_olr.core.allocation import step_olr_batch
    cfg=config(capacity_joint_allocation=True,capacity_redistribute=True,
        capacity_redistribution_evidence='entry',capacity_loss_position_pct=.015,
        capacity_loss_sector_pct=.025,capacity_loss_total_pct=.025)
    result=step_olr_batch(OLRState(),(bar('005930'),bar('000660')),cfg,snapshot(),OLRPortfolioView(10000,10000))
    entries=[a for a in result.actions if isinstance(a,SubmitEntry)]
    assert entries
    assert all(a.qty*unit_loss(a.metadata,cfg)<=150 for a in entries)
    assert sum(a.qty*unit_loss(a.metadata,cfg) for a in entries)<=250


@pytest.mark.parametrize('kw', [{'capacity_loss_position_pct':float('nan')},
    {'capacity_loss_sector_pct':-.1},{'capacity_loss_total_pct':.3}, {'capacity_loss_atr_multiple':.8},
    {'capacity_exit_priority':1}])
def test_invalid_risk_config_rejected(kw):
    with pytest.raises(ValueError):config(**kw)


def test_native_broker_cancel_ack_precedes_market_replacement_and_next_open_fill():
    from datetime import date, datetime
    from strategy_common.clock import KST
    from backtests.engine.replay import drive_timestamp_batch
    from backtests.engine.sim_broker import SimBroker, BrokerCosts
    from backtests.engine.capacity import BarParticipationScenario
    from backtests.strategies.olr.runner import OLRReplayAdapter
    from strategy_olr.models import OLRDailySnapshot
    from tests.strategy_olr.test_core_runner import _candidate
    from tests.strategy_olr.test_capacity_allocation import DAY
    yesterday=date(2026,1,7)
    first=OLRDailySnapshot(yesterday,(_candidate('005930',yesterday,rank=1),),'test',
        datetime.combine(yesterday,time(8),tzinfo=KST))
    cfg=config(capacity_joint_allocation=True,capacity_exit_priority=True,trade_exit_plan={
        'name':'priority_native','mode':'managed','hard_stop_enabled':False,
        'failed_followthrough_bars':1,'failed_followthrough_close_r':-1.,'failed_followthrough_mfe_r':4.})
    adapter=OLRReplayAdapter(cfg,{yesterday:first})
    broker=SimBroker(initial_equity=10000,costs=BrokerCosts(2,20,5),capacity_policy=BarParticipationScenario(.01,'test'))
    def quote(day,t,price):
        return replace(bar(day=day,clock=t,volume=100000),open=price,high=price,low=price,close=price)
    for t in (time(14,25),time(14,30),time(14,35)):
        drive_timestamp_batch(broker,(quote(yesterday,t,100),),(adapter,))
    assert len(broker.fills)==1
    p=adapter.state.symbols['005930'].position
    p.metadata['risk_per_share']=10.
    for t,price in ((time(14,25),100),(time(14,30),100),(time(14,35),100),
                    (time(14,55),100),(time(15,0),88),(time(15,5),89)):
        drive_timestamp_batch(broker,(quote(DAY,t,price),),(adapter,))
    sells=[f for f in broker.fills if f.side=='SELL']
    assert len(sells)==1 and sells[0].timestamp.time()==time(15,5)
    assert sells[0].qty==broker.fills[0].qty and sells[0].reason=='failed_followthrough'
    cancels=[i for i,j in enumerate(broker.journal) if j['event']=='cancel' and j['order']['order_type']=='CLOSE_AUCTION']
    markets=[i for i,j in enumerate(broker.journal) if j['event']=='submit' and j['order']['side']=='SELL' and j['order']['order_type']=='MARKET']
    assert len(cancels)==len(markets)==1 and cancels[0]<markets[0]
    assert not broker.orders and not adapter.state.symbols['005930'].pending_exit_order_id
    assert broker.same_bar_fill_violations==0
