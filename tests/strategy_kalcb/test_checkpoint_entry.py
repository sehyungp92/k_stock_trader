from dataclasses import replace
from datetime import date, datetime, timedelta
import pytest
from strategy_common.clock import KST
from strategy_common.actions import SubmitEntry
from strategy_kalcb.config import KALCBConfig
from strategy_kalcb.core.core_models import KALCBPortfolioView
from strategy_kalcb.core.preview import preview_entry
from strategy_kalcb.core.state import KALCBState
from strategy_kalcb.core.candidate_updates import activate_candidate_snapshot, observe_candidate_bar
from tests.strategy_kalcb.test_core_logic import _snapshot, _pathproof_bars


def setup():
    day=date(2026,1,5);snap=_snapshot(day);at=datetime(2026,1,5,9,45,tzinfo=KST)
    c=replace(snap.candidates[0],average_30m_volume=3000,metadata={'entry_checkpoint_at':at.isoformat(),
        'feature_available_at':at.isoformat(),'frontier_initial_active':True,'candidate_rank':1})
    snap=replace(snap,candidates=(c,),generated_at=at)
    cfg=KALCBConfig(entry_checkpoint_enabled=True,entry_plan_mode='first30_open',
        rvol_threshold=0,cpr_threshold=0,momentum_score_min=0,entry_score_blocklist=(),
        entry_require_complete_prefix=False)
    return snap,_pathproof_bars(day),cfg,at


def test_checkpoint_has_no_early_or_later_reentry_and_ignores_future_bars():
    snap,bars,cfg,at=setup();account=KALCBPortfolioView(1000000,equity=1000000)
    seen=preview_entry(bars,snap,cfg,account,available_at=at)
    assert seen['ready'],seen['decisions']
    assert all(x['available_at']==at.isoformat() for x in seen['intents'])
    poisoned=bars[:9]+[replace(bars[9],close=1e9,volume=1e12)]
    assert preview_entry(poisoned,snap,cfg,account,available_at=at)==seen
    # The same complete history cannot produce a second checkpoint intent.
    later=preview_entry(bars,snap,cfg,account,available_at=at+timedelta(minutes=15))
    assert len(later['intents'])==1


def test_intraday_activation_cannot_precede_feature_clock():
    snap,bars,cfg,at=setup();state=KALCBState()
    with pytest.raises(ValueError):activate_candidate_snapshot(state,snap,at-timedelta(seconds=1))
    activate_candidate_snapshot(state,snap,at)
    assert state.snapshot_hash==snap.artifact_hash
    c=replace(snap.candidates[0],metadata={**snap.candidates[0].metadata,'feature_available_at':(at+timedelta(minutes=5)).isoformat()})
    with pytest.raises(ValueError):activate_candidate_snapshot(state,replace(snap,candidates=(c,)),at)


def test_watch_observations_do_not_lose_the_prefix_when_candidate_activates():
    snap,bars,cfg,at=setup();state=KALCBState();base=replace(snap,candidates=(),generated_at=at.replace(hour=0,minute=0))
    for b in bars[:8]:observe_candidate_bar(state,b,base)
    activate_candidate_snapshot(state,snap,at)
    result=preview_entry(bars[8:9],snap,cfg,KALCBPortfolioView(1000000,equity=1000000),available_at=at,state=state)
    assert result['ready'],result['decisions']


def test_native_broker_never_fills_checkpoint_before_information_is_complete():
    from backtests.strategies.kalcb.runner import KALCBReplayAdapter
    from backtests.engine.sim_broker import SimBroker, BrokerCosts
    from backtests.engine.replay import drive_timestamp_batch
    snap,bars,cfg,at=setup()
    base=replace(snap,candidates=(),generated_at=at.replace(hour=0,minute=0))
    costs=BrokerCosts(0,0,0)
    broker=SimBroker(1000000,costs)
    adapter=KALCBReplayAdapter(cfg,{snap.trade_date:base},initial_equity=1000000,costs=costs,
        timed_snapshots={snap.trade_date:[snap]},observation_symbols=(bars[0].symbol,))
    for b in bars[:9]:drive_timestamp_batch(broker,(b,),(adapter,))
    assert len(broker.qualification_snapshots)==2
    assert broker.qualification_snapshots[-1]['artifact_hash']==snap.artifact_hash
    assert not broker.fills
    assert broker.orders
    for b in bars[9:]:drive_timestamp_batch(broker,(b,),(adapter,))
    entries=[f for f in broker.fills if f.side=='BUY']
    assert entries
    assert all(f.timestamp>=at for f in entries)
    assert not broker.same_bar_fill_violations


def test_watch_prefix_is_independent_of_future_candidate_membership():
    from scripts.kalcb_executed_value_replay import observation_prefix
    snap,bars,cfg,at=setup()
    other=[replace(b,symbol='other') for b in bars]
    small=observation_prefix(bars+other,snap.trade_date,{bars[0].symbol},45)
    large=observation_prefix(bars+other,snap.trade_date,{bars[0].symbol,'other'},45)
    prefix=lambda rows:[b for b in rows if b.timestamp+timedelta(minutes=5)<=at]
    assert prefix(small)==prefix(large)
    assert all(b.symbol==bars[0].symbol or b.timestamp+timedelta(minutes=5)<=at for b in small)


def test_watch_universe_cannot_silently_expand_audited_market_scope():
    from scripts.kalcb_executed_value_replay import validate_watch_universe
    validate_watch_universe(('000100',),('000100','000270'))
    with pytest.raises(ValueError,match='outside declared source universe'):
        validate_watch_universe(('000100','999999'),('000100','000270'))


def test_timed_source_audit_cannot_use_the_shorter_opening_feature_horizon():
    from backtests.engine.qualification_evidence import validate_source_scope
    scope={'version':'causal-timed-candidates-v1','strategy':'kalcb',
        'feature_intraday_before':'09:45','candidate_intraday_before':'15:20'}
    validate_source_scope(scope)
    for changes in ({'feature_intraday_before':'09:30'},{'candidate_intraday_before':'09:45'},{'strategy':'olr'}):
        with pytest.raises(ValueError):validate_source_scope({**scope,**changes})
