"""Boundary and lifecycle integration tests for the isolated research patch."""
from dataclasses import replace
from datetime import timedelta
import json
import pytest
from strategy_common.actions import CancelOrders, SubmitExit, SubmitPartialExit
from strategy_olr.core.logic import _maybe_submit_managed_exit, remember_submitted_order, on_olr_order_update, on_olr_order_expired
from strategy_olr.core.core_models import OLROrderUpdateEvent, OLRExpiredOrderEvent
from strategy_olr.core.serializers import snapshot_state, restore_state
from tests.strategy_olr.test_opening_risk_v2 import weak_setup


def setup(fraction=.5):
    cfg,state,current=weak_setup()
    cfg=replace(cfg,capacity_weak_partial_fraction=0.)
    current=replace(current,timestamp=current.timestamp.replace(hour=9,minute=25))
    ss=state.symbols[current.symbol]
    ss.session_bars=[replace(current,timestamp=current.timestamp-timedelta(minutes=i)) for i in (25,20,15,10,5,0)]
    evidence=dict(available_at=(current.timestamp+timedelta(minutes=5)).isoformat(),
        source_last_label=current.timestamp.isoformat(),close=current.close,own_return=-.05,
        market_median=.001,peer_count=102)
    state.meta['relative_context_v1']=dict(fraction=fraction,rows={current.symbol:evidence},feature_sha256='test')
    return cfg,state,current


@pytest.mark.parametrize('fraction,quantity,action',[(.5,25,SubmitPartialExit),(1.,51,SubmitExit)])
def test_action_latches_and_preserves_original_risk_through_checkpoint(fraction,quantity,action):
    cfg,state,bar=setup(fraction)
    result=_maybe_submit_managed_exit(state,bar,cfg)
    assert isinstance(result.actions[0],action) and result.actions[0].qty==quantity
    assert result.actions[0].metadata['risk_per_share']==10.
    restored=restore_state(json.loads(json.dumps(snapshot_state(state))))
    assert not _maybe_submit_managed_exit(restored,bar,cfg).actions
    assert restored.symbols[bar.symbol].position.metadata['risk_per_share']==10.


@pytest.mark.parametrize('change',['disabled','pending_buy','pending_sell','same_day','positive_own','falling_market','thin_peers','wrong_clock','single_half'])
def test_exclusions_do_not_create_orders(change):
    cfg,state,bar=setup();ss=state.symbols[bar.symbol];b=state.meta['relative_context_v1'];e=b['rows'][bar.symbol]
    if change=='disabled':b['fraction']=0
    elif change=='pending_buy':ss.pending_entry_order_id='pending'
    elif change=='pending_sell':ss.pending_exit_order_id='pending'
    elif change=='same_day':ss.position.entry_time=bar.timestamp-timedelta(minutes=20)
    elif change=='positive_own':e['own_return']=.001
    elif change=='falling_market':e['market_median']=-.001
    elif change=='thin_peers':e['peer_count']=67
    elif change=='wrong_clock':bar=replace(bar,timestamp=bar.timestamp+timedelta(minutes=5))
    elif change=='single_half':ss.position.qty_open=1
    assert not _maybe_submit_managed_exit(state,bar,cfg).actions


@pytest.mark.parametrize('field,value',[('available_at','2027-01-01T09:30:00+09:00'),('source_last_label','2027-01-01T09:25:00+09:00'),('close',96.)])
def test_misaligned_sources_fail_closed(field,value):
    cfg,state,bar=setup();state.meta['relative_context_v1']['rows'][bar.symbol][field]=value
    with pytest.raises(ValueError,match='not aligned'):_maybe_submit_managed_exit(state,bar,cfg)


def test_full_risk_preempts_context_and_replaces_outstanding_partial_after_ack():
    cfg,state,bar=setup();ss=state.symbols[bar.symbol]
    partial=_maybe_submit_managed_exit(state,bar,cfg).actions[0]
    remember_submitted_order(state,'relative',partial)
    crash=replace(bar,close=88.,low=87.);ss.session_bars[-1]=crash
    result=_maybe_submit_managed_exit(state,crash,cfg)
    assert isinstance(result.actions[0],CancelOrders)
    assert not _maybe_submit_managed_exit(state,crash,cfg).actions
    result=on_olr_order_update(state,OLROrderUpdateEvent('relative',bar.symbol,'CANCELLED',bar.timestamp),cfg)
    assert any(isinstance(a,SubmitExit) and a.qty==51 and a.reason=='failed_followthrough' for a in result.actions)
    cfg,state,bar=setup();crash=replace(bar,close=88.,low=87.);state.symbols[bar.symbol].session_bars[-1]=crash
    actions=_maybe_submit_managed_exit(state,crash,cfg).actions
    assert any(isinstance(a,SubmitExit) and a.qty==51 and a.reason=='failed_followthrough' for a in actions)
    assert not any(a.reason.startswith('relative_context') for a in actions)


def test_context_partial_expiry_releases_pending_state_without_repeat():
    cfg,state,bar=setup();ss=state.symbols[bar.symbol]
    action=_maybe_submit_managed_exit(state,bar,cfg).actions[0]
    remember_submitted_order(state,'relative',action)
    expired=OLRExpiredOrderEvent('relative',bar.symbol,'SELL','MARKET',25,bar.timestamp)
    result=on_olr_order_expired(state,expired,cfg)
    assert not ss.pending_exit_order_id and not result.actions
    assert not _maybe_submit_managed_exit(state,bar,cfg).actions
