from copy import deepcopy
from dataclasses import asdict, field, make_dataclass, replace
from datetime import datetime, timedelta
import inspect
import pytest
from strategy_common.clock import KST
from strategy_common.market import MarketBar
from strategy_common.actions import SubmitEntry, SubmitExit
from strategy_kalcb.config import KALCBConfig
from strategy_kalcb.core import logic
from strategy_kalcb.core.state import KALCBState, KALCBPositionState
from strategy_kalcb.opening_failure_v1 import evidence
from backtests.engine.sim_broker import SimBroker, BrokerCosts
from backtests.strategies.kalcb.mechanism_observer_v2 import MechanismObserver
from backtests.strategies.kalcb.mechanism_io_v1 import patch_logic, FIELD

Config=make_dataclass('MechanismConfig',[(FIELD,int,field(default=0))],bases=(KALCBConfig,),frozen=True,slots=True)


def setup(monkeypatch, hold=6):
    text=inspect.getsource(logic._exit_reason)+'\n\ndef _sentinel(): pass\n'
    namespace=dict(logic.__dict__); exec(compile(patch_logic(text),'<frozen-research-overlay>','exec'),namespace)
    monkeypatch.setattr(logic,'_exit_reason',namespace['_exit_reason'])
    start=datetime(2025,8,7,9,0,tzinfo=KST)
    state=KALCBState(); s=state.symbol_state('A')
    for i,c in enumerate([100]*10+[96,95]):
        s.add_bar(MarketBar('A',start+timedelta(minutes=5*i),'5m',c,102 if i<6 else c+1,98 if i<6 else c-1,c,100000))
    p=KALCBPositionState('A',100,100,100,start+timedelta(minutes=30),99,99,1,'KRX_FIRST30_OPEN',4,
        hold_bars=6,metadata={'entry_route':'first30_open_anchor'})
    s.position=p
    cfg=Config(research_opening_failure_min_hold_bars=hold,exit_hard_stop_enabled=False,
        quick_exit_enabled=False,exit_failed_followthrough_bars=0,exit_profit_retention_enabled=False,
        exit_mfe_floor_enabled=False,adaptive_trail_enabled=False,use_partial_takes=False)
    return state,s,p,s.bars_today[-1],cfg


def test_completed_own_vwap_and_opening_structure(monkeypatch):
    _,s,p,b,_=setup(monkeypatch)
    proof=evidence(s,p,b,6)
    assert proof and proof['confirmations'][0]['own_vwap'] != proof['confirmations'][1]['own_vwap']
    assert evidence(s,p,b,8) is None
    p.hold_bars=8; assert evidence(s,p,b,8)


@pytest.mark.parametrize('change',['winner','other_route','gap','pending','early','uncompleted'])
def test_required_conditions(monkeypatch,change):
    _,s,p,b,_=setup(monkeypatch)
    if change=='winner':p.entry_price=94
    if change=='other_route':p.metadata['entry_route']='checkpoint_continuation'
    if change=='gap':s._market_bars.pop(8)
    if change=='pending':p.exit_in_flight=True
    if change=='early':p.entry_time=b.timestamp
    if change=='uncompleted':
        with pytest.raises(ValueError):evidence(s,p,replace(b,is_completed=False),6)
        return
    assert evidence(s,p,b,6) is None


def test_future_bars_do_not_affect_predicate(monkeypatch):
    _,s,p,b,_=setup(monkeypatch); before=evidence(s,p,b,6)
    s.add_bar(MarketBar('A',b.timestamp+timedelta(minutes=5),'5m',200,201,199,200,999999))
    assert evidence(s,p,b,6)==before


def test_parent_exit_has_priority(monkeypatch):
    _,s,p,b,cfg=setup(monkeypatch)
    cfg=replace(cfg,exit_max_hold_bars=6)
    ob=MechanismObserver(); ob.install()
    try:
        assert logic._exit_reason(s,p,b,cfg,current_r=-5,mfe_r=0)=='max_hold'
        assert ob.exit_probes[0]['parent_exit_reason']=='max_hold'
        assert ob.summarize([])['observed_opportunities']['6']['parent_no_exit_positions']==0
    finally:ob.restore()


def test_disabled_overlay_is_observation_only(monkeypatch):
    _,s,p,b,cfg=setup(monkeypatch,0); before=deepcopy(p)
    ob=MechanismObserver(); ob.install()
    try:
        assert logic._exit_reason(s,p,b,cfg,current_r=-5,mfe_r=0)==''
        assert p==before
        assert ob.summarize([])['observed_opportunities']['6']['parent_no_exit_positions']==1
        assert ob.summarize([])['direct_filled_positions']==0
    finally:ob.restore()


def test_new_exit_has_actual_next_bar_gap_fill(monkeypatch):
    state,s,p,b,cfg=setup(monkeypatch)
    broker=SimBroker(1000000,BrokerCosts())
    at=p.entry_time-timedelta(minutes=5)
    broker.process_bar(MarketBar('A',at,'5m',100,101,99,100,100000))
    broker.submit(SubmitEntry('KALCB','A',100,'MARKET',None,None,'entry'),at)
    broker.process_bar(MarketBar('A',p.entry_time,'5m',100,101,99,100,100000))
    p.entry_price=broker.fills[-1].price
    observer=MechanismObserver();observer.install()
    try:
        result=logic._manage_position(state,s,b,cfg)
        exits=[a for a in result.actions if isinstance(a,SubmitExit)]
        assert len(exits)==1 and exits[0].reason=='opening_thesis_failure'
        broker.process_bar(b); broker.submit(exits[0],b.timestamp)
        assert observer.summarize([asdict(f) for f in broker.fills])['direct_filled_positions']==0
        broker.process_bar(MarketBar('A',b.timestamp+timedelta(minutes=5),'5m',80,81,79,80,100000))
        result=observer.summarize([asdict(f) for f in broker.fills])
        assert result['direct_filled_positions']==1
        assert broker.fills[-1].price<80
    finally:observer.restore()


def test_broker_clone_keeps_funding_and_does_not_mutate_original():
    from backtests.strategies.kalcb.mechanism_observer_v2 import clone_broker
    broker=SimBroker(1000000,BrokerCosts(),buying_power_leverage=2)
    trial=clone_broker(broker)
    assert trial.buying_power_leverage==2 and trial.cash==broker.cash
    trial.cash-=100; assert broker.cash==1000000


def test_broker_clone_after_real_completed_trade_preserves_receipts_and_pending_orders():
    from backtests.strategies.kalcb.mechanism_observer_v2 import clone_broker
    at=datetime(2025,8,7,9,0,tzinfo=KST)
    broker=SimBroker(1000000,BrokerCosts(),buying_power_leverage=2)
    def bar(i):return MarketBar('A',at+timedelta(minutes=5*i),'5m',100,101,99,100,100000)
    broker.process_bar(bar(0))
    broker.submit(SubmitEntry('KALCB','A',100,'MARKET',None,None,'entry',metadata={'nested':{'x':[1]}}),at)
    broker.process_bar(bar(1))
    broker.submit(SubmitExit('KALCB','A',100,'MARKET',None,'exit'),bar(1).timestamp)
    broker.process_bar(bar(2))
    broker.submit(SubmitEntry('KALCB','A',25,'MARKET',None,None,'pending'),bar(2).timestamp)
    assert len(broker.trades)==1 and broker.orders
    with pytest.raises(TypeError,match='mappingproxy'):
        deepcopy(broker,{id(broker.source_events):broker.source_events.copy()})
    clone=clone_broker(broker)
    assert clone.trades[0].to_json_dict()==broker.trades[0].to_json_dict()
    assert clone.cash==broker.cash and clone.journal==broker.journal and clone.orders==broker.orders
    clone.trades[0].route_metadata['nested']['x'].append(2)
    assert broker.trades[0].route_metadata['nested']['x']==[1]
    clone.process_bar(bar(3))
    assert clone.positions and not broker.positions
    assert broker.orders and len(clone.fills)==len(broker.fills)+1
    assert clone.source_events.hexdigest()!=broker.source_events.hexdigest()


def test_broker_clone_includes_dependent_accounts_and_independent_hashes():
    from backtests.strategies.kalcb.mechanism_observer_v2 import clone_broker
    broker=SimBroker(1000000); child=SimBroker(123000)
    broker.capacity_shadow_brokers=[child]
    clone=clone_broker(broker)
    clone.capacity_shadow_brokers[0].cash=1
    clone.capacity_shadow_brokers[0].source_events.update(b'copy only')
    assert child.cash==123000
    assert child.source_events.hexdigest()!=clone.capacity_shadow_brokers[0].source_events.hexdigest()


def test_overlay_patch_retains_complete_parent_cascade():
    text=inspect.getsource(logic._exit_reason)+'\n\ndef _sentinel(): pass\n'
    new=patch_logic(text)
    assert new.index('if config.research_opening_failure_min_hold_bars:')>new.index('liquidation_evidence(')
    assert new.count('return "opening_thesis_failure"')==1
    assert new.replace(new[new.index('    if config.research_opening_failure_min_hold_bars:'):new.index('    return ""\n\n\ndef _sentinel')],'')==text


@pytest.mark.parametrize('cash',[1000000,0])
def test_late_preview_uses_actual_funding_and_never_submits_to_parent(monkeypatch,cash):
    from tests.strategy_kalcb.test_checkpoint_entry import setup as checkpoint
    from backtests.strategies.kalcb.runner import KALCBReplayAdapter
    from backtests.engine.replay import drive_timestamp_batch
    from backtests.strategies.kalcb import late_entry_v1
    snap,bars,cfg,at=checkpoint(); at=at.replace(hour=10,minute=0)
    # Isolate funding in this synthetic strongly extended price path.
    cfg=replace(cfg,frontier_rotation_enabled=False,breakout_distance_cap_r=0,orb_entry_range_cap_r=0)
    candidate=replace(snap.candidates[0],metadata={**snap.candidates[0].metadata,
        'entry_checkpoint_at':at.isoformat(),'feature_available_at':at.isoformat()})
    snap=replace(snap,candidates=(candidate,),generated_at=at)
    base=replace(snap,candidates=(),generated_at=at.replace(hour=0))
    monkeypatch.setattr(late_entry_v1,'append_snapshot',lambda *a:snap)
    costs=BrokerCosts(0,0,0); broker=SimBroker(cash,costs)
    adapter=KALCBReplayAdapter(cfg,{snap.trade_date:base},initial_equity=cash,costs=costs,
        observation_symbols=(candidate.symbol,))
    row={'trade_date':str(snap.trade_date),'symbol':candidate.symbol,
         'feature_available_at':at.isoformat(),'opportunity_id':str(snap.trade_date)+'/'+candidate.symbol+'/minute60/q0.25',
         'opening_bars':[b.to_json_dict() for b in bars[:12]]}
    ob=MechanismObserver([(row,[{'baseline':1,'combined':1}])]);ob.install()
    try:
        for b in bars[:12]:drive_timestamp_batch(broker,(b,),(adapter,))
        assert len(ob.entry_probes)==1
        assert ob.entry_probes[0]['native_order_submitted']==bool(cash),ob.entry_probes
        assert not broker.orders and not broker.fills and broker.cash==cash
    finally:ob.restore()


def test_real_frozen_overlay_is_scoped_and_compiles(tmp_path):
    from pathlib import Path
    import ast,json,hashlib
    from backtests.strategies.kalcb.mechanism_io_v1 import make_overrides
    source=Path('E:/CodexResearch/k_stock_trader/kalcb_late_entry_20261011/v10_signal/cases/control/oos/input_profile.json')
    item=json.loads(source.read_text(encoding='utf-8'));code=Path(item['manifest']['path']).parent/'files'
    before={rel:hashlib.sha256((code/rel).read_bytes()).hexdigest() for rel in ('strategy_kalcb/core/logic.py','strategy_kalcb/config.py')}
    paths=make_overrides(code,tmp_path/'overrides',{})
    assert set(paths)==set(before)
    for rel in before:
        assert hashlib.sha256((code/rel).read_bytes()).hexdigest()==before[rel]
        ast.parse(Path(paths[rel]['path']).read_text(encoding='utf-8'))


def test_finite_design_retains_two_controls_before_four_variants():
    from scripts.kalcb_mechanism_study_v2 import recipes
    r=recipes();assert list(r)[:2]==['control','control_prior125']
    assert len(r)*2*2==24
    assert {(v['prior'],v['hold']) for v in r.values()}=={(p,h) for p in (100,125) for h in (0,6,8)}
