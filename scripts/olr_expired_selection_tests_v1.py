"""0039 clock, reservation, native allocation and checkpoint integration."""
from dataclasses import replace
from datetime import datetime, date, time, timedelta
import json
import pytest
from strategy_common.actions import SubmitEntry
from strategy_common.clock import KST
from strategy_olr.config import OLRConfig
from strategy_olr.core.state import OLRState
from strategy_olr.core.logic import _sync_snapshot_state, _candidate_weights
from strategy_olr.core.allocation import step_olr_batch
from strategy_olr.core.core_models import OLRPortfolioView
from strategy_olr.core.serializers import snapshot_state,restore_state
from strategy_olr.core.expired_selection import prepare
from strategy_olr.models import OLRDailySnapshot
from tests.strategy_olr.test_core_runner import _bar,_candidate


def setup(day=date(2026,1,8), enabled=True):
    from strategy_common.execution_session import session_cutoff
    cutoff=session_cutoff(day,time(14,30))
    start=datetime.combine(day,cutoff,KST)
    cfg=OLRConfig(entry_mode='decision_next_open',capacity_joint_allocation=True,trade_entry_plan=dict(name='test',mode='confirm_next_bar',max_signal_bars=6,
        min_bar_ret=-.001,min_vwap_ret=0.,min_close_location=.5,
        max_vwap_extension_pct=.025),overnight_slot_count=6,
        allocation_mode='rank_weighted',max_position_pct=.65,target_gross_exposure=1.2,rank_decay=1.5)
    candidates=tuple(_candidate(str(i).zfill(6),day,rank=i) for i in range(1,9))
    snapshot=OLRDailySnapshot(day,candidates,'test',start-timedelta(hours=7))
    state=OLRState();_sync_snapshot_state(state,snapshot)
    state.meta['expired_selection_enabled_v1']=enabled
    current=[]
    for c in candidates:
        for i in range(-1,6):
            t=start+timedelta(minutes=5*i)
            close=100 if c.rank>=7 or i<0 else 99
            b=_bar(c.symbol,day,t.time(),100,101,98,close)
            if i<5:state.symbol_state(c.symbol).session_bars.append(b)
            else:current.append(b)
    return cfg,state,snapshot,current


def test_expiry_creates_current_proposal_without_changing_original_weights():
    cfg,s,snapshot,bars=setup()
    original=_candidate_weights(snapshot.candidates[:6],cfg)
    result=step_olr_batch(s,bars,cfg,snapshot,OLRPortfolioView(1e8,1e8))
    actions=[a for a in result.actions if isinstance(a,SubmitEntry)]
    assert [a.symbol for a in actions]==['000007','000008']
    assert all(a.reason=='expired_selection_entry' for a in actions)
    denom=sum(i**-1.5 for i in range(1,7))
    assert actions[0].metadata['allocation_weight']==pytest.approx(1.2*7**-1.5/denom)
    assert _candidate_weights(snapshot.candidates[:6],cfg)==original
    assert actions[0].metadata['risk_per_share']>0
    assert all(s.symbol_state(a.symbol).entry_attempted for a in actions)
    assert sum(a.metadata['reserved_notional'] for a in actions)<1e8
    restored=restore_state(json.loads(json.dumps(snapshot_state(s))))
    assert not any(isinstance(a,SubmitEntry) for a in step_olr_batch(restored,bars,cfg,snapshot,OLRPortfolioView(1e8,1e8)).actions)


@pytest.mark.parametrize('mode',['disabled','early','late','missing_extra','invalid_extra','no_slots'])
def test_exclusions(mode):
    cfg,s,snapshot,bars=setup(enabled=mode!='disabled')
    if mode in ('early','late'):
        bars=[replace(b,timestamp=b.timestamp+timedelta(minutes=-5 if mode=='early' else 5)) for b in bars]
    elif mode=='missing_extra':
        for i in (7,8):s.symbol_state(str(i).zfill(6)).session_bars.pop(0)
    elif mode=='invalid_extra':bars=[replace(b,close=98) if int(b.symbol)>6 else b for b in bars]
    elif mode=='no_slots':
        for c in snapshot.candidates[:6]:s.symbol_state(c.symbol).entry_attempted=True
    result=step_olr_batch(s,bars,cfg,snapshot,OLRPortfolioView(1e8,1e8))
    assert not any(isinstance(a,SubmitEntry) and a.reason=='expired_selection_entry' for a in result.actions)


def test_original_current_confirmation_and_missing_source_keep_reservations():
    cfg,s,snapshot,bars=setup()
    for c in snapshot.candidates[:3]:s.symbol_state(c.symbol).entry_attempted=True
    s.symbol_state('000004').pending_entry_order_id='native'
    # Fifth confirms for the first time in the same cohort; sixth has no source.
    bars=[replace(b,close=100) if b.symbol=='000005' else b for b in bars if b.symbol!='000006']
    prepare(s,bars,cfg,snapshot)
    c=next(iter(s.meta['expired_selection_cohorts_v1'].values()))
    assert len(c['reserved'])==6 and not c['candidates'] and c['missing']==['000006']


def test_one_expired_place_no_fallback_after_unfunded_first_extra():
    cfg,s,snapshot,bars=setup()
    for c in snapshot.candidates[:5]:s.symbol_state(c.symbol).entry_attempted=True
    result=step_olr_batch(s,bars,cfg,snapshot,OLRPortfolioView(0,1e8,open_notional=1e8))
    assert not any(isinstance(a,SubmitEntry) for a in result.actions)
    assert s.symbol_state('000007').entry_attempted and not s.symbol_state('000008').entry_attempted
    result=step_olr_batch(s,bars,cfg,snapshot,OLRPortfolioView(1e8,1e8))
    assert not any(isinstance(a,SubmitEntry) for a in result.actions)


def test_shifted_calendar_and_permutation():
    paths=[]
    for reverse in (False,True):
        cfg,s,snapshot,bars=setup(day=date(2025,11,13))
        result=step_olr_batch(s,list(reversed(bars)) if reverse else bars,cfg,snapshot,OLRPortfolioView(1e8,1e8))
        actions=[a for a in result.actions if isinstance(a,SubmitEntry)]
        assert actions and actions[0].metadata['expired_selection_v1']['available_at'][11:16]=='16:00'
        paths.append([(a.symbol,a.qty) for a in actions])
    assert paths[0]==paths[1]


def test_original_native_topup_precedes_extra_allocation(monkeypatch):
    from strategy_olr.core import allocation,logic
    evidence=dict(exit=dict(quantity=10000),entry=dict(samples=[1],quantity=10000))
    monkeypatch.setattr(allocation,'forecast',lambda *args:(10000,10000,evidence))
    monkeypatch.setattr(logic,'capacity_forecast',lambda *args:(10000,10000,evidence))
    quantities=[]
    for enabled in (False,True):
        cfg,s,snapshot,bars=setup(enabled=enabled)
        cfg=replace(cfg,capacity_redistribute=True,capacity_redistribution_evidence='auction',capacity_redistribute_max_rank=2)
        for i in (1,3,4,5):s.symbol_state(str(i).zfill(6)).entry_attempted=True
        bars=[replace(b,close=100) if b.symbol=='000002' else b for b in bars]
        result=step_olr_batch(s,bars,cfg,snapshot,OLRPortfolioView(65000,100000,open_notional=35000))
        entries=[a for a in result.actions if isinstance(a,SubmitEntry)]
        original=next(a for a in entries if a.symbol=='000002')
        assert original.metadata['allocation_redistributed_qty']>0
        quantities.append(original.qty)
        assert sum(a.metadata['reserved_notional'] for a in entries)<=65000
    assert quantities[0]==quantities[1]
