"""Focused tests executed against the newly frozen research engine."""
from dataclasses import replace
from datetime import datetime,timedelta
import json
from types import SimpleNamespace
import pytest
from strategy_olr.config import OLRConfig
from strategy_common.actions import SubmitExit
from strategy_olr.core.state import OLRState
from strategy_olr.core.serializers import snapshot_state,restore_state
from strategy_olr.core.liquidation_tail_v3 import empty,advance,scenario,evidence,KEY
from strategy_olr.core.precision_v3 import entry_plan,entry_evidence,observe_entry_day,structural_evidence
from strategy_olr.core.logic import _maybe_submit_managed_exit
from strategy_olr.core.loss_budget import unit_loss
from strategy_olr.execution import OLREntryPlan,find_olr_entry_signal
from strategy_olr.research import final_candidate_config_fingerprint
from tests.strategy_olr.test_opening_risk_v2 import weak_setup
from tests.strategy_olr.test_capacity_allocation import config,bar
from tests.strategy_olr.test_core_runner import _candidate


def row(symbol='005930',prices=(90.,110.,80.)):
    return {'symbol':symbol,'sector':'TECH','day':'2026-03-31','previous_day':'2026-03-30',
      'reference_price':100.,'reference_at':'2026-03-30T14:25:00+09:00','available_at':'2026-03-31T15:35:00+09:00',
      'liquidation':[['2026-03-31T09:%02d:00+09:00'%m,p,100.,'2026-03-31T09:%02d:00+09:00'%(m-5)]
                     for m,p in zip((5,10,15),prices)]}

def test_quantity_dependent_scenario_is_conservative_monotone_and_does_not_guess_unfilled_prices():
    r=row();values=[scenario(r,q*100,.1)['fraction'] for q in range(1,70)]
    assert values==sorted(values)
    assert scenario(r,1000,.1)['fraction']==pytest.approx(.1)
    assert scenario(r,2000,.1)['fraction']==pytest.approx(.1)
    assert scenario(r,4000,.1)['unfilled_shares']==10
    assert scenario(r,4000,.1)['fraction']==pytest.approx(.3)
    assert scenario({**r,'liquidation':[]},100,.1)['fraction']==1

def test_liquidation_uses_prior_bar_participation_and_stress_can_extend_horizon():
    r=row(prices=(100.,90.,80.))
    assert scenario(r,1000,.1)['last_fill'].endswith('09:05:00+09:00')
    assert scenario(r,1000,.05)['last_fill'].endswith('09:10:00+09:00')
    assert scenario(r,1000,.05)['fraction']>scenario(r,1000,.1)['fraction']

def test_liquidation_checkpoint_round_trip_and_strict_cutoff():
    rows=[row()]+[row(f'{i:06d}',(95.,95.,95.)) for i in range(30)]
    state=OLRState();state.meta[KEY]=advance(empty(),rows,'2026-03-31',['2026-03-31'])
    at=datetime.fromisoformat('2026-04-01T14:35:00+09:00')
    expected=evidence(state,'005930','TECH',at,1000,.1)
    restored=restore_state(json.loads(json.dumps(snapshot_state(state))))
    assert evidence(restored,'005930','TECH',at,1000,.1)==expected
    assert expected['peer_observations']==30
    with pytest.raises(ValueError):evidence(state,'005930','TECH',at-timedelta(days=1),1000,.1)

@pytest.mark.parametrize('defect',['future','revised','noncontiguous','negative_volume','wrong_reference_day'])
def test_liquidation_rejects_unsupported_or_revised_source_rows(defect):
    r=row();h=advance(empty(),[r],'2026-03-31',['2026-03-31']);bad=json.loads(json.dumps(r))
    if defect=='future':bad['day']='2026-04-01'
    if defect=='revised':bad['reference_price']=101.
    if defect=='noncontiguous':bad['liquidation'][0][3]='2026-03-30T15:15:00+09:00'
    if defect=='negative_volume':bad['liquidation'][0][2]=-1.
    if defect=='wrong_reference_day':bad['reference_at']='2026-03-31T14:25:00+09:00'
    with pytest.raises(ValueError):advance(h,[bad],'2026-03-31',['2026-03-31'])

def test_liquidation_risk_keeps_atr_floor_and_requires_evidence():
    cfg=config(capacity_liquidation_tail=True,capacity_joint_allocation=True,capacity_loss_total_pct=.08,
               commission_bps=0,tax_bps_on_sell=0,slippage_bps=0)
    m={'risk_per_share':2.,'estimated_entry_price':100.,'liquidation_tail_evidence':{'fraction':.1}}
    assert unit_loss(m,cfg)==10.
    assert unit_loss({**m,'liquidation_tail_evidence':{'fraction':.01}},cfg)==2.5
    with pytest.raises(ValueError):unit_loss({'risk_per_share':2.,'estimated_entry_price':100.},cfg)

def structural_setup():
    cfg,state,current=weak_setup();cfg=replace(cfg,capacity_structural_partial=True)
    ss=state.symbols[current.symbol];ss.position.metadata['structural_entry_day_low']=96.
    ss.session_bars=[replace(current,open=98.,high=100.,low=94.,close=95.,timestamp=current.timestamp-timedelta(minutes=i)) for i in (10,5,0)]
    return cfg,state,ss.session_bars[-1]

def test_structural_partial_requires_low_break_vwap_persistence_and_preserves_recovery():
    cfg,state,current=structural_setup();ss=state.symbols[current.symbol]
    assert structural_evidence(ss,ss.position,current,10.)
    altered=replace(current,close=97.);ss.session_bars[-1]=altered
    assert structural_evidence(ss,ss.position,altered,10.) is None
    ss.session_bars[-1]=current
    ss.session_bars[-2]=replace(ss.session_bars[-2],timestamp=current.timestamp-timedelta(minutes=10))
    assert structural_evidence(ss,ss.position,current,10.) is None

def test_structural_full_guard_precedes_half_and_half_latch_survives_restart():
    cfg,state,current=structural_setup();ss=state.symbols[current.symbol]
    result=_maybe_submit_managed_exit(state,current,cfg)
    assert result.actions[0].qty==25
    assert 'structural_partial_evidence' in result.actions[0].metadata
    recovered=restore_state(json.loads(json.dumps(snapshot_state(state))))
    assert not _maybe_submit_managed_exit(recovered,current,cfg).actions
    cfg,state,current=structural_setup();crash=replace(current,close=88.,low=87.);state.symbols[current.symbol].session_bars[-1]=crash
    result=_maybe_submit_managed_exit(state,crash,cfg)
    sells=[a for a in result.actions if isinstance(a,SubmitExit)]
    assert len(sells)==1 and sells[0].qty==51 and sells[0].reason!='weak_followthrough_partial'

def test_entry_day_low_includes_preentry_and_later_bars_but_freezes_next_session():
    cfg,state,current=structural_setup();p=state.symbols[current.symbol].position
    old=replace(current,timestamp=p.entry_time,low=90.)
    observe_entry_day(p,old);assert p.metadata['structural_entry_day_low']==90.
    observe_entry_day(p,replace(current,low=80.));assert p.metadata['structural_entry_day_low']==90.
    evidence=entry_evidence([replace(old,low=85.)],old,cfg)
    assert evidence['structural_entry_day_low']==85.

def test_late_confirmation_preserves_early_signal_and_adds_only_bars_seven_to_nine():
    base=OLREntryPlan('native','confirm_next_bar',max_signal_bars=6,min_bar_ret=0.,min_close_location=.5)
    enabled=entry_plan(base,SimpleNamespace(capacity_late_confirmation=True))
    first=bar();pre=replace(first,timestamp=first.timestamp-timedelta(minutes=5))
    bars=[pre]+[replace(first,timestamp=first.timestamp+timedelta(minutes=i*5),open=100.,high=101.,low=98.,close=99.) for i in range(9)]
    bars[7]=replace(bars[7],close=101.)
    c=_candidate(first.symbol,first.timestamp.date(),rank=1)
    assert find_olr_entry_signal(bars,c,base,require_fill_bar=False) is None
    assert find_olr_entry_signal(bars,c,enabled,require_fill_bar=False).signal_index==7
    bars[2]=replace(bars[2],close=101.)
    assert find_olr_entry_signal(bars,c,base,require_fill_bar=False)==find_olr_entry_signal(bars,c,enabled,require_fill_bar=False)
    assert find_olr_entry_signal(bars[:7],c,base,require_fill_bar=False)==find_olr_entry_signal(bars[:7],c,enabled,require_fill_bar=False)


def test_entry_day_observation_survives_later_buy_fragment_metadata_and_restart():
    cfg,state,current=structural_setup();p=state.symbols[current.symbol].position
    first=replace(current,timestamp=p.entry_time,low=90.)
    observe_entry_day(p,first)
    # The real fill handler merges the original pending-order metadata over
    # existing position metadata on subsequent partial entry fills.
    p.metadata.update(structural_entry_day_low=96.)
    recovered=restore_state(json.loads(json.dumps(snapshot_state(state))))
    p=recovered.symbols[current.symbol].position
    observe_entry_day(p,replace(first,timestamp=first.timestamp+timedelta(minutes=5),low=94.))
    assert p.metadata['structural_entry_day_low']==90.
    assert p.metadata['structural_observed_entry_day_low']==90.

def test_flags_are_off_by_default_do_not_change_candidate_identity_and_validate_dependencies():
    cfg=OLRConfig()
    assert not cfg.capacity_liquidation_tail and not cfg.capacity_structural_partial and not cfg.capacity_late_confirmation
    alt=replace(cfg,capacity_late_confirmation=True)
    assert final_candidate_config_fingerprint(cfg)==final_candidate_config_fingerprint(alt)
    with pytest.raises(ValueError):replace(cfg,capacity_liquidation_tail=True)
    with pytest.raises(ValueError):replace(cfg,capacity_structural_partial=True)
    with pytest.raises(ValueError):replace(cfg,capacity_late_confirmation=1)
