from dataclasses import asdict, replace
from datetime import date
from copy import deepcopy
import pytest
from backtests.engine.sim_broker import SimBroker, BrokerCosts
from backtests.engine.replay import drive_timestamp_batch
from backtests.strategies.kalcb.runner import KALCBReplayAdapter
from backtests.strategies.kalcb.population_removal_observer_v1 import PopulationRemovalObserver
from strategy_kalcb.config import KALCBConfig
from strategy_kalcb.core.serializers import snapshot_state
from tests.strategy_kalcb.test_core_logic import _snapshot, _or_bars
from tests.strategy_kalcb.test_checkpoint_entry import setup as checkpoint_setup
from scripts.kalcb_population_removal_worker_v1 import removed_opening_entries, economic_bridge


def native(observe, *, removed=True, fill=True, future_open=101., checkpoint=False):
    day=date(2026,1,5)
    snapshot=_snapshot(day)
    cfg=KALCBConfig(entry_plan_mode='first30_open',rvol_threshold=0,cpr_threshold=0,
                    momentum_score_min=0,entry_score_blocklist=(),entry_require_complete_prefix=False)
    bars=_or_bars(day)
    updates={}
    if checkpoint:
        cp,bars,cfg,_=checkpoint_setup()
        snapshot=replace(cp,candidates=(),generated_at=cp.generated_at.replace(hour=0,minute=0))
        updates={day:[cp]}
    elif fill:
        bars[-1]=replace(bars[-1],open=future_open,high=max(future_open+2,bars[-1].high),low=min(future_open-1,bars[-1].low))
    if not fill:
        bars=bars[:6]
    costs=BrokerCosts(2,18,3)
    broker=SimBroker(1_000_000,costs)
    adapter=KALCBReplayAdapter(cfg,{day:snapshot},initial_equity=1_000_000,costs=costs,
                              timed_snapshots=updates,observation_symbols=('005930',))
    observer=PopulationRemovalObserver({str(day)+'/005930'} if removed else set())
    if observe:
        observer.install()
    try:
        for bar in bars:
            drive_timestamp_batch(broker,(bar,),(adapter,))
        fills=[asdict(f) for f in broker.fills]
        output={'fills':fills,'cash':broker.cash,'equity':broker.equity_curve,
                'state':snapshot_state(adapter.state),'orders':[asdict(o) for o in broker.orders]}
        return output,observer.summarize(fills),observer
    finally:
        if observe:
            observer.restore()


def test_same_state_removal_observer_preserves_real_core_fills_cash_and_state():
    plain,_,_=native(False)
    observed,support,_=native(True)
    assert plain==observed
    assert support['direct_blockable_entry_issuer_days']==1
    assert support['filled_parent_orders_blockable_by_gate']==1
    assert support['months']=={'2026-01':1}
    assert all(f['timestamp'].minute>=30 for f in observed['fills'])


def test_no_fill_does_not_count_as_effective_gate_support():
    _,support,observer=native(True,fill=False)
    assert observer.probes
    assert support['submitted_parent_entries']==1
    assert support['direct_blockable_entry_issuer_days']==0


def test_future_gap_changes_fill_not_parent_or_shadow_decision():
    a,_,oa=native(True,future_open=101)
    b,_,ob=native(True,future_open=108)
    assert oa.probes==ob.probes
    assert a['fills'][0]['price']!=b['fills'][0]['price']


def test_checkpoint_readmission_is_never_counted_as_opening_removal_support():
    a,support,observer=native(True,checkpoint=True)
    assert any(f['side']=='BUY' for f in a['fills'])
    assert not observer.probes
    assert support['direct_blockable_entry_issuer_days']==0


def test_identity_gate_has_no_support_and_does_not_change_account():
    a,_,_=native(False)
    b,support,_=native(True,removed=False)
    assert a==b
    assert support['same_state_parent_entry_probes']==0


def test_duplicate_physical_execution_cannot_inflate_support():
    output,_,observer=native(True)
    with pytest.raises(ValueError,match='duplicate'):
        observer.summarize(output['fills']+output['fills'])


def test_gate_derivation_rejects_added_or_repriced_actions_and_changed_calendar():
    day=date(2026,1,5)
    snapshot=_snapshot(day)
    empty=replace(snapshot,candidates=())
    assert removed_opening_entries({day:snapshot},{day:empty})=={str(day)+'/005930'}
    with pytest.raises(ValueError,match='additions'):
        removed_opening_entries({day:empty},{day:snapshot})
    changed=replace(snapshot,candidates=(replace(snapshot.candidates[0],selection_score=999),))
    with pytest.raises(ValueError,match='changed incumbent'):
        removed_opening_entries({day:snapshot},{day:changed})
    with pytest.raises(ValueError,match='calendar'):
        removed_opening_entries({day:snapshot},{})


def test_exact_parent_bridge_rejects_fee_or_mtm_drift(tmp_path):
    native_account,_,_=native(False)
    reference=tmp_path/'reference'
    actual=tmp_path/'actual'
    files={'fills.json':native_account['fills'],'equity.json':native_account['equity'],
           'daily.json':[{'cash':native_account['cash']}]}
    values={folder/name:deepcopy(value) for folder in (reference,actual) for name,value in files.items()}
    assert economic_bridge(actual,reference,values.__getitem__)['passed']
    values[actual/'equity.json'][-1]+=1
    with pytest.raises(ValueError,match='account bridge'):
        economic_bridge(actual,reference,values.__getitem__)
    values[actual/'equity.json']=deepcopy(files['equity.json'])
    values[actual/'fills.json'][0]['metadata']['fee']+=1
    with pytest.raises(ValueError,match='physical fill bridge'):
        economic_bridge(actual,reference,values.__getitem__)
