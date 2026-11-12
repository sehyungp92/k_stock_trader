from copy import deepcopy
from dataclasses import replace,asdict
from datetime import timedelta
import pytest
from tests.backtests.strategies.test_kalcb_reclaim_labels_v1 import native_fixture
from backtests.strategies.kalcb.reclaim_observer_v1 import ReclaimObserver,source_prefix


def setup():
    from backtests.engine.sim_broker import SimBroker,BrokerCosts
    from backtests.strategies.kalcb.runner import KALCBReplayAdapter
    snap,bars,cfg,at=native_fixture();cfg=replace(cfg,frontier_shadow_enabled=False,frontier_rotation_enabled=False)
    base=replace(snap,candidates=(),generated_at=at.replace(hour=0,minute=0))
    broker=SimBroker(1000000,BrokerCosts(1,2,3));adapter=KALCBReplayAdapter(cfg,{snap.trade_date:base},initial_equity=1000000,costs=broker.costs,observation_symbols=('005930',))
    row={'trade_date':str(at.date()),'symbol':'005930','opportunity_id':str(at.date())+'/005930/minute90/q0.25',
        'feature_available_at':at.isoformat(),'guard_at_rank1':True,'quarter_preview':{'ready':True},
        'candidate':snap.candidates[0].to_json_dict(),'opening_bars':[dict(symbol=b.symbol,timestamp=b.timestamp.isoformat(),**{k:getattr(b,k) for k in ('open','high','low','close','volume')}) for b in bars]}
    return adapter,broker,bars,row,snap,at


def drive(adapter,broker,bars,spent=False):
    from backtests.engine.replay import drive_timestamp_batch
    for b in bars[:12]:drive_timestamp_batch(broker,(b,),(adapter,))
    if spent:adapter.state.meta['entry_route_session_counts']={'checkpoint_continuation':1}
    for b in bars[12:]:drive_timestamp_batch(broker,(replace(b,symbol='000100'),),(adapter,))


def test_copied_native_order_is_not_parent_order_and_marks_remain_exact():
    a,b,bars,row,snap,at=setup();other,control,_,_,_,_=setup()
    observer=ReclaimObserver([(row,[])]);observer.install()
    try:drive(a,b,bars)
    finally:observer.restore()
    drive(other,control,bars)
    assert observer.previews[0]['native_order_submitted'] is True
    assert observer.previews[0]['copied_watch_bars']==5
    assert not b.orders and not b.fills and not observer.entries
    assert b.equity_curve==control.equity_curve and b.timestamps==control.timestamps and b.cash==control.cash
    assert a.state==other.state


@pytest.mark.parametrize('block',['cash','checkpoint'])
def test_actual_account_constraints_are_preserved(block):
    a,b,bars,row,_,_=setup()
    if block=='cash':b.cash=0
    observer=ReclaimObserver([(row,[])]);observer.install()
    try:drive(a,b,bars,spent=block=='checkpoint')
    finally:observer.restore()
    assert not observer.previews[0]['native_order_submitted']
    assert not b.orders and not b.fills


@pytest.mark.parametrize('poison',['clock','missing','future','price'])
def test_observer_rejects_invalid_prefix(poison):
    a,b,bars,row,_,at=setup()
    if poison=='clock':row['feature_available_at']=(at+timedelta(minutes=5)).isoformat()
    elif poison=='missing':row['opening_bars'].pop()
    elif poison=='future':row['opening_bars'][-1]['timestamp']=at.isoformat()
    else:row['opening_bars'][0]['close']+=1
    observer=ReclaimObserver([(row,[])]);observer.install()
    try:
        if poison=='clock':
            with pytest.raises(ValueError,match='clock'):source_prefix(row,at)
        else:
            with pytest.raises(ValueError):drive(a,b,bars)
    finally:observer.restore()


def test_real_entry_census_binds_actual_fill_not_shadow_submission():
    from backtests.engine.replay import drive_timestamp_batch
    a,b,bars,row,snap,at=setup();a.timed_snapshots={at.date():[snap]}
    observer=ReclaimObserver([(row,[])]);observer.install()
    try:
        for bar in bars:drive_timestamp_batch(b,(bar,),(a,))
        nxt=replace(bars[-1],timestamp=at,open=104.5,high=105,low=104.25,close=104.75)
        drive_timestamp_batch(b,(nxt,),(a,))
    finally:observer.restore()
    assert observer.previews[0]['reason']=='already_parent_candidate'
    assert len(observer.entries)==1 and b.fills
    summary=observer.summarize([asdict(f) for f in b.fills])
    assert summary['actual_filled_entry_orders']==1


def test_population_cannot_silently_include_non_geometry_or_duplicates():
    row=setup()[3]
    with pytest.raises(ValueError,match='Duplicate'):ReclaimObserver([(row,[]),(row,[])])
    bad=deepcopy(row);bad['guard_at_rank1']=False
    with pytest.raises(ValueError,match='population'):ReclaimObserver([(bad,[])])


def test_full_account_lifecycle_and_source_contract_are_retained():
    import inspect
    from scripts.kalcb_reclaim_observer_worker_v1 import main,raw_proof
    from scripts.kalcb_reclaim_observer_study_v1 import run
    assert 'PhaseRunner(' in inspect.getsource(run)
    assert 'run(plan,a.policies' in inspect.getsource(main)
    assert 'bridge_accounts(a.output,a.bridge)' in inspect.getsource(main)
    assert 'verify_rows(population[day],rows,index' in inspect.getsource(raw_proof)
    assert 'native_worker' in inspect.getsource(run)
