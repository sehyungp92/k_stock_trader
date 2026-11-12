from dataclasses import replace
from datetime import datetime, timedelta

from strategy_common.actions import SubmitEntry, SubmitExit, CancelOrders
from strategy_common.clock import KST
from strategy_common.market import MarketBar
from strategy_kalcb.config import KALCBConfig
from strategy_kalcb.core import logic
from strategy_kalcb.core import pending_entry_revalidation_v1 as m
from strategy_kalcb.core.core_models import KALCBFillEvent, KALCBOrderUpdateEvent
from strategy_kalcb.core.serializers import snapshot_state, restore_state
from strategy_kalcb.core.state import KALCBState

AT = datetime(2026,1,5,9,30,tzinfo=KST)
CFG = replace(KALCBConfig(),exit_hard_stop_enabled=False)


def event(qty, before=0, side='BUY', oid='entry'):
    return KALCBFillEvent(oid,'005930',side,qty,99.,AT+timedelta(minutes=5),
        metadata={'execution_id':oid+str(before),'order_qty':10 if side=='BUY' else qty,
                  'filled_qty_before':before,'filled_qty_after':before+qty,'fee':1.})


def setup(filled=4):
    state=KALCBState();state.meta[m.KEY+'_enabled']=True
    a=SubmitEntry('KALCB','005930',10,'MARKET',None,None,'original',
        {'order_role':'ENTRY','entry_route':'first30_open_anchor','entry_price_ref':100.,
         'risk_reserved_entry_qty':10,'risk_per_share':2.,'stop_price':98.,
         'decision_available_at':AT.isoformat()})
    logic.remember_submitted_order(state,'entry',a)
    ss=state.symbol_state('005930');ss.pending_entry_metadata=dict(a.metadata)
    if filled:logic.on_kalcb_fill(state,replace(event(filled),timestamp=AT),CFG)
    for i in range(7):
        p=99. if i==6 else 100.
        ss.add_bar(MarketBar('005930',AT-timedelta(minutes=30)+timedelta(minutes=5*i),'5m',p,p,p,p,1000.))
    orders=[dict(strategy_id='KALCB',symbol='005930',order_id='entry',side='BUY',order_type='MARKET',qty=10-filled)]
    return state,orders


def act(state,orders,clock=AT+timedelta(minutes=5)):
    return m.revalidate(state,'005930',clock,orders)


def test_disabled_is_exact_and_parent_entry_unchanged():
    s,orders=setup();s.meta[m.KEY+'_enabled']=False;before=snapshot_state(s)
    assert act(s,orders)==([],'disabled') and snapshot_state(s)==before
    assert s.symbol_state('005930').pending_entry_metadata['risk_reserved_entry_qty']==10


def test_cancel_keeps_position_and_waits_for_ack_without_retry():
    s,orders=setup();p=s.symbol_state('005930').position;p.hold_bars=3
    actions,reason=act(s,orders)
    assert len(actions)==1 and isinstance(actions[0],CancelOrders) and reason=='cancel_requested'
    assert p.qty_open==4 and p.hold_bars==3 and not p.exit_in_flight
    assert s.symbol_state('005930').pending_entry_order_id=='entry'
    assert act(s,orders)==([],'already_evaluated')
    logic.on_kalcb_order_update(s,KALCBOrderUpdateEvent('entry','005930','CANCELLED',AT,role='ENTRY'))
    assert not s.symbol_state('005930').pending_entry_order_id and p.qty_open==4


def test_racing_fill_before_ack_is_managed_not_liquidated():
    s,orders=setup();act(s,orders);p=s.symbol_state('005930').position
    r=logic.on_kalcb_fill(s,event(2,4),CFG)
    assert p.qty_open==6 and p.qty_entry==6 and not p.exit_in_flight
    assert not any(isinstance(a,SubmitExit) for a in r.actions)
    before=snapshot_state(s);logic.on_kalcb_fill(s,event(2,4),CFG);assert snapshot_state(s)==before


def test_racing_fill_after_ack_and_snapshot_restore_preserves_lifetime():
    s,orders=setup();act(s,orders)
    logic.on_kalcb_order_update(s,KALCBOrderUpdateEvent('entry','005930','CANCELLED',AT,role='ENTRY'))
    restored=restore_state(snapshot_state(s))
    for state in [s,restored]:
        result=logic.on_kalcb_fill(state,event(6,4),CFG)
        assert not any(isinstance(a,SubmitExit) for a in result.actions)
        assert state.symbol_state('005930').position.qty_open==10
    assert snapshot_state(s)==snapshot_state(restored)


def test_existing_parent_exit_still_liquidates_racing_entry():
    s,orders=setup();act(s,orders);p=s.symbol_state('005930').position
    p.metadata['liquidation_reason']='parent_exit';p.exit_in_flight=True
    result=logic.on_kalcb_fill(s,event(2,4),CFG)
    assert any(isinstance(a,SubmitExit) and a.qty==6 for a in result.actions)


def test_other_symbol_orders_are_preserved_but_same_symbol_orders_block():
    s,orders=setup();orders.append({**orders[0],'symbol':'000000','order_id':'other'})
    assert act(s,orders)[1]=='cancel_requested'
    s,orders=setup();orders.append({**orders[0],'side':'SELL','order_id':'protective'})
    assert act(s,orders)==([],'symbol_cancellation_not_single_entry')


def test_missing_book_or_quantity_mismatch_fails_closed():
    s,orders=setup();assert act(s,None)==([],'unavailable_complete_order_view')
    orders[0]['qty']=7;assert act(s,orders)==([],'pending_quantity_not_reconciled')


def test_missing_or_late_clock_and_missing_prefix_abstain():
    s,orders=setup();assert act(s,orders,AT+timedelta(minutes=10))==([],'not_fixed_clock')
    s.symbol_state('005930')._market_bars.pop()
    assert act(s,orders)==([],'invalid_completed_prefix')


def test_original_management_sale_abstains():
    s,orders=setup();s.symbol_state('005930').position.partial_taken=True
    assert act(s,orders)==([],'parent_exit_sale_or_other_lifetime')


def test_unfilled_order_cancels_without_creating_position():
    s,orders=setup(0);assert act(s,orders)[1]=='cancel_requested'
    logic.on_kalcb_order_update(s,KALCBOrderUpdateEvent('entry','005930','CANCELLED',AT,role='ENTRY'))
    assert s.symbol_state('005930').position is None


def test_confirmed_price_keeps_original_pending_order():
    s,orders=setup();ss=s.symbol_state('005930')
    ss._market_bars[-1]=replace(ss._market_bars[-1],open=102,high=102,low=102,close=102)
    assert act(s,orders)==([],'confirmation_preserved')
    assert ss.pending_entry_order_id=='entry'


def test_unrelated_racing_order_is_still_rejected():
    import pytest
    s,orders=setup();act(s,orders)
    with pytest.raises(ValueError,match='Unrelated entry'):
        logic.on_kalcb_fill(s,event(1,oid='unrelated'),CFG)


def test_zero_volume_prefix_abstains():
    s,orders=setup();ss=s.symbol_state('005930')
    ss._market_bars[-1]=replace(ss._market_bars[-1],volume=0)
    assert act(s,orders)==([],'invalid_completed_prefix')


def test_parent_full_exit_before_racing_fill_remains_terminal():
    s,orders=setup();act(s,orders);ss=s.symbol_state('005930');p=ss.position
    p.metadata['liquidation_reason']='parent_exit';p.exit_in_flight=True
    a=SubmitExit('KALCB','005930',4,'MARKET',None,'parent_exit',{'order_role':'EXIT'})
    logic.remember_submitted_order(s,'exit',a)
    logic.on_kalcb_fill(s,event(4,side='SELL',oid='exit'),CFG)
    assert ss.position is None
    result=logic.on_kalcb_fill(s,event(2,4),CFG)
    assert ss.position.qty_open==2
    assert any(isinstance(a,SubmitExit) and a.qty==2 for a in result.actions)


def native_episode(scenario, enabled=None):
    from contextlib import ExitStack
    from dataclasses import asdict
    from unittest.mock import patch
    from tests.backtests.strategies.test_kalcb_checkpoint_contrast_labels_v1 import fixture
    from scripts.kalcb_executed_value_study import run_episode
    from backtests.strategies.kalcb.checkpoint_contrast_labels_v1 import independent_account
    from backtests.strategies.kalcb.runner import KALCBReplayAdapter
    from backtests.engine import sim_broker
    from backtests.engine.integrity import audit_broker
    row,bars,cfg,context,_=fixture();ref=bars[5].close
    q=row['action']['quantity']
    # OPEN capacity uses the previous completed bar, never future interval
    # volume. Constrain that causal reference so the first fill is partial.
    bars[5]=replace(bars[5],volume=max(200,int(q/.04)))
    bars[6]=replace(bars[6],open=ref,high=ref,low=ref*.99,close=ref*.99,
                    volume=max(200,int(q/.04)))
    indexes={row['trade_date']:{(b.symbol,b.timestamp.isoformat(),'5m'):
        (b.open,b.high,b.low,b.close,b.volume,True) for b in bars}}
    brokers=[];records=[];factory=sim_broker.SimBroker;end=KALCBReplayAdapter.on_timestamp_end
    def create(*args,**kw):
        b=factory(*args,**kw);brokers.append(b);return b
    def hook(adapter,timestamp,batch,broker):
        result=end(adapter,timestamp,batch,broker)
        if timestamp+timedelta(minutes=5)!=AT+timedelta(minutes=5):return result
        adapter.state.meta[m.KEY+'_enabled']=enabled
        actions,reason=act(adapter.state,[asdict(o) for o in broker.orders])
        if actions:adapter._submit_actions(actions,broker,timestamp)
        records.append(reason)
        return result
    with ExitStack() as stack:
        stack.enter_context(patch.object(sim_broker,'SimBroker',create))
        if enabled is not None:stack.enter_context(patch.object(KALCBReplayAdapter,'on_timestamp_end',hook))
        value,detail=run_episode(row,bars,cfg,scenario,'pending-revalidation-fixture',None,[row['trade_date']])
    broker=brokers[0];audit=audit_broker(broker);account=independent_account(broker,indexes)
    return value,detail,broker,records,audit,account


def test_native_baseline_disabled_parity_and_partial_cancellation():
    check_native('baseline')


def test_native_combined_disabled_parity_and_partial_cancellation():
    check_native('combined')


def check_native(scenario):
    original=native_episode(scenario);disabled=native_episode(scenario,False);candidate=native_episode(scenario,True)
    for key in ['fills','orders']:
        assert original[1][key]==disabled[1][key]
    assert original[2].journal==disabled[2].journal
    assert original[2].equity_curve==disabled[2].equity_curve
    assert original[2].cash==disabled[2].cash
    assert candidate[3]==['cancel_requested']
    for result in (original,disabled,candidate):
        assert result[0]['label_valid'] and result[4]['execution_validity_pass'] and result[5]['passed']
        assert result[5]['raw_close_marks']==78 and result[4]['audit_status']=='unverified'
    oldbuys=[f for f in original[2].fills if f.side=='BUY']
    buys=[f for f in candidate[2].fills if f.side=='BUY']
    assert buys and sum(f.qty for f in buys)<sum(f.qty for f in oldbuys)
    assert all(f.timestamp<AT+timedelta(minutes=5) for f in buys)
    assert any(f.timestamp>=AT+timedelta(minutes=5) for f in oldbuys)
    assert not any(f.reason==m.REASON for f in candidate[2].fills)
    assert any(f.side=='SELL' for f in candidate[2].fills)
