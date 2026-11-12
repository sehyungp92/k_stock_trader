from dataclasses import replace
from datetime import datetime, timedelta
import json
from pathlib import Path
import types
import pytest

from strategy_common.actions import SubmitEntry, SubmitExit, CancelOrders
from strategy_common.clock import KST
from strategy_common.market import MarketBar
from strategy_kalcb.config import KALCBConfig
from strategy_kalcb.core.core_models import KALCBCoreResult, KALCBFillEvent, KALCBPortfolioView, KALCBOrderUpdateEvent
from strategy_kalcb.core.state import KALCBState
from strategy_kalcb.core.serializers import snapshot_state, restore_state
from strategy_kalcb.core import staged_commitment_v1 as m

CORE=types.ModuleType('strategy_kalcb.core._staged_test')
CORE.__package__='strategy_kalcb.core'
exec(compile(m.derive_logic((Path(__file__).resolve().parents[1]/'strategy_kalcb/core/logic.py').read_text()), 'derived_core', 'exec'), CORE.__dict__)
AT=datetime(2026,1,5,9,30,tzinfo=KST)
CFG=replace(KALCBConfig(),exit_hard_stop_enabled=False)

def setup(q=10):
    s=KALCBState();s.meta['staged_commitment_enabled']=True
    a=SubmitEntry('KALCB','005930',q,'MARKET',None,None,'parent',
        {'entry_route':'first30_open_anchor','entry_price_ref':100.,'risk_per_share':2.,'stop_price':98.,'order_role':'ENTRY'})
    r=m.stage_initial(s,KALCBCoreResult(s,actions=[a]),AT)
    CORE.remember_submitted_order(s,'initial',r.actions[0])
    return s,r.actions[0]

def event(oid,qty,price=100.,side='BUY',before=0,total=None,stamp=AT):
    return KALCBFillEvent(oid,'005930',side,qty,price,stamp,metadata={
        'execution_id':oid+str(before),'filled_qty_before':before,'filled_qty_after':before+qty,
        'order_qty':total or qty,'fee':1.})

def fill_initial(s,q=5): CORE.on_kalcb_fill(s,event('initial',q),CFG)

def add_prefix(s,price=102.):
    ss=s.symbol_state('005930');ss.bars.clear();ss._market_bars.clear()
    ss.vwap_value=ss.vwap_volume=0
    for i in range(7):
        p=100. if i<6 else price
        ss.add_bar(MarketBar('005930',AT-timedelta(minutes=30)+timedelta(minutes=5*i),'5m',p,p,p,p,1000.))

def completion(s):
    add_prefix(s)
    a=m.completion_action(s,'005930',AT+timedelta(minutes=5),CFG,KALCBPortfolioView(1e8,equity=1e8))
    assert a is not None
    CORE.remember_submitted_order(s,'completion',a)
    return a

def test_disabled_and_other_routes_are_exact():
    s,a=setup();s2=KALCBState();r=KALCBCoreResult(s2,[a]);old=snapshot_state(s2)
    assert m.stage_initial(s2,r,AT) is r and r.actions==[a] and snapshot_state(s2)==old
    s2.meta['staged_commitment_enabled']=True
    b=replace(a,metadata={'entry_route':'checkpoint'})
    assert m.stage_initial(s2,KALCBCoreResult(s2,[b]),AT).actions==[b]

def test_partial_entry_prevents_completion_without_retry():
    s,a=setup();CORE.on_kalcb_fill(s,event('initial',2,total=5),CFG);add_prefix(s)
    assert m.completion_action(s,a.symbol,AT+timedelta(minutes=5),CFG,KALCBPortfolioView(1e8)) is None
    assert next(iter(m.contracts(s).values()))['events'][-1]['reason']=='initial_fill_incomplete'
    CORE.on_kalcb_fill(s,event('initial',3,before=2,total=5),CFG)
    assert m.completion_action(s,a.symbol,AT+timedelta(minutes=10),CFG,KALCBPortfolioView(1e8)) is None

def test_real_second_order_partial_fees_and_single_lifetime():
    s,a=setup();fill_initial(s);p=s.symbol_state(a.symbol).position
    p.hold_bars=4;p.update_mark(high=110.,low=95.);initial_time=p.entry_time
    addition=completion(s);assert addition.qty==4
    CORE.on_kalcb_fill(s,event('completion',2,102,before=0,total=4,stamp=AT+timedelta(minutes=5)),CFG)
    CORE.on_kalcb_fill(s,event('completion',2,103,before=2,total=4,stamp=AT+timedelta(minutes=10)),CFG)
    assert p.entry_order_id=='initial' and p.qty_entry==p.qty_open==9
    assert p.entry_time==initial_time and p.hold_bars==4 and p.observed_mfe_r==5.
    assert p.entry_price==pytest.approx((500+204+206)/9) and p.metadata['entry_fees']==3.
    assert len(m.contracts(s))==1
    assert 'completion' in json.dumps(s.meta['core_executions'])

def test_duplicate_fill_idempotent_overfill_and_unrelated_entry_rejected():
    s,a=setup();fill_initial(s);completion(s)
    f=event('completion',2,total=4);CORE.on_kalcb_fill(s,f,CFG)
    before=snapshot_state(s);CORE.on_kalcb_fill(s,f,CFG);assert snapshot_state(s)==before
    with pytest.raises(ValueError):CORE.on_kalcb_fill(s,event('completion',3,before=2,total=5),CFG)
    with pytest.raises(ValueError):CORE.on_kalcb_fill(s,event('unrelated',1),CFG)

def test_restart_preserves_authorization_and_quantity_accounting():
    s,a=setup();fill_initial(s);completion(s)
    restored=restore_state(json.loads(json.dumps(snapshot_state(s))))
    for state in (s,restored):CORE.on_kalcb_fill(state,event('completion',4,102),CFG)
    assert snapshot_state(s)==snapshot_state(restored)

def test_late_child_after_full_parent_exit_is_one_lifecycle_and_liquidates():
    s,a=setup();fill_initial(s);completion(s)
    ex=SubmitExit('KALCB',a.symbol,5,'MARKET',None,'parent_exit',{'order_role':'EXIT'})
    r=m.protect_exit(s,KALCBCoreResult(s,[ex]),AT+timedelta(minutes=5))
    assert isinstance(r.actions[0],CancelOrders)
    CORE.remember_submitted_order(s,'exit',ex)
    CORE.on_kalcb_fill(s,event('exit',5,99,side='SELL'),CFG)
    assert s.symbol_state(a.symbol).position is None
    r=CORE.on_kalcb_fill(s,event('completion',4,102),CFG)
    p=s.symbol_state(a.symbol).position
    assert p.entry_order_id=='initial' and p.qty_entry==9 and p.qty_open==4
    assert any(isinstance(x,SubmitExit) and x.qty==4 for x in r.actions)
    assert len(s.symbol_state(a.symbol).entry_history)==1

def test_cancelled_completion_late_fill_is_liquidated():
    s,a=setup();fill_initial(s);completion(s)
    CORE.on_kalcb_order_update(s,KALCBOrderUpdateEvent('completion',a.symbol,'CANCELLED',AT),)
    r=CORE.on_kalcb_fill(s,event('completion',4,102),CFG)
    assert any(isinstance(x,SubmitExit) for x in r.actions)

@pytest.mark.parametrize('failure',['confirmation','hole','zero_volume','parent_exit','late','heat'])
def test_completion_abstains_on_causal_guard(failure):
    s,a=setup();fill_initial(s);add_prefix(s);ss=s.symbol_state(a.symbol)
    clock=AT+timedelta(minutes=5);pv=KALCBPortfolioView(1e8,equity=1e8)
    if failure=='confirmation':add_prefix(s,99)
    elif failure=='hole':ss._market_bars[3]=replace(ss._market_bars[3],timestamp=AT)
    elif failure=='zero_volume':ss._market_bars[3]=replace(ss._market_bars[3],volume=0)
    elif failure=='parent_exit':ss.position.exit_in_flight=True
    elif failure=='late':clock+=timedelta(minutes=5)
    elif failure=='heat':pv.open_risk=1e12
    assert m.completion_action(s,a.symbol,clock,CFG,pv) is None

def test_one_share_is_unchanged_and_duplicate_plan_fails():
    s=KALCBState();s.meta['staged_commitment_enabled']=True
    a=SubmitEntry('KALCB','005930',1,'MARKET',None,None,'parent',{'entry_route':'first30_open_anchor'})
    assert m.stage_initial(s,KALCBCoreResult(s,[a]),AT).actions==[a]
    s,a=setup()
    original=replace(a,qty=10,metadata={**a.metadata,'entry_price_ref':100.})
    with pytest.raises(ValueError):m.stage_initial(s,KALCBCoreResult(s,[original]),AT)

@pytest.mark.parametrize('scenario',['baseline','combined'])
def test_native_disabled_parity_gap_cost_and_timestamped_mtm(scenario):
    from contextlib import ExitStack
    from unittest.mock import patch
    from tests.backtests.strategies.test_kalcb_checkpoint_contrast_labels_v1 import fixture
    from backtests.strategies.kalcb import runner,executed_value as ev
    from backtests.strategies.kalcb.staged_commitment_pilot_v1 import episode
    from strategy_kalcb.core import logic
    row,bars,cfg,_,indexes=fixture()
    close=bars[5].close+1
    bars[6]=replace(bars[6],open=close,high=close,low=close,close=close)
    bars[7]=replace(bars[7],open=close+2,high=close+2,low=close+2,close=close+2)
    indexes={row['trade_date']:{(b.symbol,b.timestamp.isoformat(),'5m'):(b.open,b.high,b.low,b.close,b.volume,True) for b in bars}}
    snap=ev.snapshot_for(row['trade_date'],[(row,0)],source='fixture')
    spec={'id':'fixture','day':row['trade_date'],'symbol':row['symbol'],'available_at':row['feature_available_at'],
          'parent_qty':row['action']['quantity'],'initial_equity':1e8,'scenario':scenario}
    with ExitStack() as stack:
        for name in CORE._staged_parent_api:stack.enter_context(patch.object(runner,name,getattr(CORE,name)))
        stack.enter_context(patch.object(logic,'_staged_parent_api',CORE._staged_parent_api,create=True))
        control,cd=episode(spec,snap,bars,cfg,None,[spec['day']],indexes,'control')
        original,od=episode(spec,snap,bars,cfg,None,[spec['day']],indexes,'control',bridge=True)
        staged,sd=episode(spec,snap,bars,cfg,None,[spec['day']],indexes,'staged')
    for key in ('fills','orders','journal','equity_curve','timestamps','terminal_cash'):assert cd[key]==od[key]
    assert control['net_value_krw']==original['net_value_krw']
    assert staged['label_valid'] and staged['independent_account_pass'] and staged['independent_execution_pass']
    assert staged['completion_filled_quantity']>0
    assert staged['initial_quantity']==spec['parent_qty']//2
    assert staged['full_portfolio_audit_status']=='unverified'
    c=next(iter(sd['staged_commitments'].values()))
    buys=[f for f in sd['fills'] if f['order_id']==c['completion_id']]
    assert buys and all(f['price']>=close+2 for f in buys)
    assert all(f['timestamp']>=datetime.fromisoformat(c['completion_at']) for f in buys)
    assert sd['independent_account']['raw_close_marks']==78
