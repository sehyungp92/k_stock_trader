from dataclasses import replace
from datetime import date, datetime, time, timedelta

import pytest

from strategy_common.actions import SubmitEntry, SubmitExit, SubmitPartialExit
from strategy_common.clock import KST
from strategy_olr.core.capacity import observe_routed, observe, forecast
from strategy_olr.core.logic import step_olr_core, remember_submitted_order
from strategy_olr.core.allocation import step_olr_batch
from strategy_olr.core.core_models import OLRPortfolioView, OLRFillEvent
from strategy_olr.core.state import OLRState, OLRPositionState
from strategy_olr.engine import OLREngine
from tests.strategy_olr.test_capacity_allocation import bar, config, snapshot, DAY


def test_irrelevant_feed_does_not_enter_orderable_history():
    cfg=config(capacity_exit_quantile=.5,capacity_history_scope='orderable')
    state=OLRState()
    observe_routed(state,bar('999999',clock=time(15,30)),cfg,snapshot())
    assert not state.meta.get('capacity_history_v1')
    observe_routed(state,bar('005930',clock=time(15,30)),cfg,snapshot())
    assert set(state.meta['capacity_history_v1'])=={'005930'}
    state.symbol_state('999999').pending_entry_order_id='working'
    observe_routed(state,bar('999999',clock=time(15,30)),cfg,snapshot())
    assert '999999' in state.meta['capacity_history_v1']


def test_sparse_fallback_bounds_entry_without_inventing_exit_evidence():
    cfg=config(capacity_exit_quantile=.5,capacity_sparse_exit_mode='continuous')
    state=OLRState();now=bar(clock=time(15,10),volume=1200)
    observe(state,now,cfg)
    q,exit_q,e=forecast(state,now,cfg)
    assert q==12 and exit_q is None and e['exit']['status']=='insufficient_history'
    assert e['entry']['horizon']==1


@pytest.mark.parametrize('minutes,trigger',[(30,time(14,55)),(60,time(14,25)),(90,time(13,55))])
def test_scheduled_release_waits_for_actual_clock_and_previous_session(minutes,trigger):
    cfg=config(capacity_release_minutes=minutes)
    state=OLRState();s=state.symbol_state('005930')
    s.position=OLRPositionState('005930',50,100,datetime(2026,1,7,14,35,tzinfo=KST),1,1)
    clock=datetime.combine(DAY,trigger,tzinfo=KST)
    before=replace(bar(),timestamp=clock-timedelta(minutes=5))
    assert not step_olr_core(state,before,cfg,None,OLRPortfolioView(0,5000,open_notional=5000)).actions
    result=step_olr_core(state,replace(bar(),timestamp=clock),cfg,None,OLRPortfolioView(0,5000,open_notional=5000))
    exits=[a for a in result.actions if isinstance(a,SubmitExit)]
    assert len(exits)==1 and exits[0].order_type=='MARKET' and exits[0].qty==50
    # A request is not a fill and does not release cash or inventory.
    assert s.position.qty_open==50
    remember_submitted_order(state,'release',exits[0])
    assert not step_olr_core(state,replace(bar(),timestamp=clock+timedelta(minutes=5)),cfg,None,OLRPortfolioView(0,5000,open_notional=5000)).actions
    fresh=OLRState();fresh.symbol_state('005930').position=OLRPositionState('005930',50,100,clock-timedelta(minutes=20),1,1)
    assert not step_olr_core(fresh,replace(bar(),timestamp=clock),cfg,None,OLRPortfolioView(0,5000,open_notional=5000)).actions


def test_partial_release_has_one_working_sell_and_no_phantom_entry_budget():
    cfg=config(capacity_release_minutes=60,capacity_release_fraction=.5,capacity_joint_allocation=True)
    state=OLRState();s=state.symbol_state('005930')
    s.position=OLRPositionState('005930',50,100,datetime(2026,1,7,14,35,tzinfo=KST),1,1)
    result=step_olr_batch(state,(bar('005930',clock=time(14,25)),),cfg,snapshot(),OLRPortfolioView(0,5000,open_notional=5000))
    partials=[a for a in result.actions if isinstance(a,SubmitPartialExit)]
    assert len(partials)==1 and partials[0].qty==25
    remember_submitted_order(state,'partial',partials[0])
    result=step_olr_batch(state,(bar('000660'),bar('005930')),cfg,snapshot(),OLRPortfolioView(0,5000,open_notional=5000))
    assert not any(isinstance(a,(SubmitEntry,SubmitExit,SubmitPartialExit)) for a in result.actions)


def test_redistribution_preserves_native_allocations_and_requires_forecast():
    cfg=config(capacity_joint_allocation=True,capacity_exit_quantile=.5)
    def run(top_up,liquidity):
        state=OLRState();policy=replace(cfg,capacity_redistribute=top_up)
        if liquidity:
            for d in (date(2026,1,5),date(2026,1,6),date(2026,1,7)):
                observe(state,bar('000660',day=d,clock=time(15,30),volume=100000),policy)
        return [a for a in step_olr_batch(state,(bar('000660'),),policy,snapshot(),OLRPortfolioView(10000,10000)).actions if isinstance(a,SubmitEntry)]
    native=run(False,True)[0];extra=run(True,True)[0]
    assert extra.qty>native.qty
    assert extra.metadata['reserved_notional']<=6500
    assert extra.metadata['allocation_before_redistribution']==native.qty
    assert run(True,False)[0].qty==run(False,False)[0].qty


def test_new_sizing_fields_preserve_candidates_and_bind_execution():
    from strategy_olr.research import final_candidate_config_fingerprint, _research_config_fingerprint
    a=config();b=replace(a,capacity_history_scope='orderable',capacity_release_minutes=60,
        capacity_joint_allocation=True,capacity_redistribute=True,capacity_sparse_exit_mode='continuous')
    assert final_candidate_config_fingerprint(a)==final_candidate_config_fingerprint(b)
    assert _research_config_fingerprint(a)!=_research_config_fingerprint(b)


@pytest.mark.parametrize('fraction',[.5,1.0])
def test_native_release_fills_after_decision_and_cash_changes_only_on_fill(fraction):
    from backtests.engine.replay import drive_timestamp_batch
    from backtests.engine.sim_broker import SimBroker, BrokerCosts
    from backtests.engine.capacity import BarParticipationScenario
    from backtests.strategies.olr.runner import OLRReplayAdapter
    from strategy_olr.models import OLRDailySnapshot
    from tests.strategy_olr.test_core_runner import _candidate
    yesterday=date(2026,1,7)
    first=OLRDailySnapshot(yesterday,(_candidate('005930',yesterday,rank=1),),
        'test',datetime.combine(yesterday,time(8),tzinfo=KST))
    cfg=config(capacity_joint_allocation=True,capacity_history_scope='orderable',capacity_exit_quantile=.5,
        capacity_release_minutes=60,capacity_release_fraction=fraction)
    adapter=OLRReplayAdapter(cfg,{yesterday:first,DAY:snapshot()})
    broker=SimBroker(initial_equity=10000,costs=BrokerCosts(2,20,5),capacity_policy=BarParticipationScenario(.01,'test'))
    for t in [time(14,30),time(14,35)]:
        drive_timestamp_batch(broker,(bar(day=yesterday,clock=t,volume=100000),),(adapter,))
    bought=sum(f.qty for f in broker.fills if f.side=='BUY')
    assert bought>0
    before=broker.cash
    drive_timestamp_batch(broker,(bar(clock=time(14,25),volume=100000),),(adapter,))
    assert broker.cash==before and not any(f.side=='SELL' for f in broker.fills)
    drive_timestamp_batch(broker,(bar(clock=time(14,30),volume=100000),),(adapter,))
    sales=[f for f in broker.fills if f.side=='SELL']
    assert sales and sum(f.qty for f in sales)==int(bought*fraction)
    assert all(f.timestamp.time()==time(14,30) for f in sales)
    assert broker.cash>before
    assert sum(p.qty for p in broker.positions.values())==bought-sum(f.qty for f in sales)


def test_acceptance_rejects_profitable_but_risk_or_frequency_failing_policy():
    from backtests.strategies.olr.allocation_recovery import cell_failures
    base={'return_pct':50.,'drawdown_pct':7.,'trades':176,'scenario':'baseline',
        'execution_validity_pass':True,'terminal_inventory':[],'terminal_pending':[]}
    assert not cell_failures({**base,'return_pct':54.},base,True)
    assert 'drawdown' in cell_failures({**base,'return_pct':90.,'drawdown_pct':9.},base,True)
    assert 'frequency' in cell_failures({**base,'return_pct':90.,'trades':150},base,True)
    assert 'material_gain' in cell_failures({**base,'return_pct':50.5},base,True)


def test_history_contract_rejects_unconditional_seed():
    from strategy_olr.core.capacity import seed_history
    with pytest.raises(ValueError,match='observed state'):
        seed_history(OLRState(),[],config(capacity_history_scope='orderable'),datetime.combine(DAY,time(8),tzinfo=KST))


def test_framework_baseline_uses_current_mutations():
    from backtests.strategies.olr.allocation_recovery import AllocationRecoveryPlugin
    from backtests.auto.shared.types import Experiment
    plugin=object.__new__(AllocationRecoveryPlugin);plugin.names=['release60']
    seen=[];plugin.batch=lambda names:seen.extend(names)
    plugin.evaluate=lambda name:{'recipe':name,'score':1,'failures':[],'metrics':{'qualified':True}}
    rows=plugin.create_evaluate_batch(1,{'recipe':'median'})([Experiment('__baseline__',{})],{'recipe':'median'})
    assert seen==['median','median'] and len(rows)==1


def test_every_registered_profile_obeys_native_override_contract():
    from backtests.strategies.olr.allocation_recovery import PROFILES
    from strategy_olr.config import OLRConfig
    for settings in PROFILES.values():
        assert all(k.startswith('capacity_') for k in settings)
        replace(OLRConfig(entry_mode='decision_next_open'),**settings)


def test_entry_supported_topup_respects_native_multiple_and_does_not_invent_exit_history():
    cfg=config(capacity_joint_allocation=True,capacity_exit_quantile=.5,capacity_entry_bars=12,
        capacity_redistribute=True,capacity_redistribution_evidence='entry',
        capacity_redistribution_max_multiplier=1.5)
    state=OLRState()
    result=step_olr_batch(state,(bar('000660',volume=100000),),cfg,snapshot(),OLRPortfolioView(10000,10000))
    entries=[a for a in result.actions if isinstance(a,SubmitEntry)]
    assert len(entries)==1
    action=entries[0];m=action.metadata
    assert m['capacity_forecast']['exit']['quantity'] is None
    assert 0<m['allocation_redistributed_qty']
    assert action.qty<=int(m['allocation_native_qty']*1.5)
    assert action.qty<=m['capacity_forecast']['entry']['quantity']
    assert m['reserved_notional']<=10000


@pytest.mark.parametrize('relative,exponent,expected',[(4,.5,40),(.25,.5,10),(1,1,20),(100,1,40)])
def test_volume_conditioning_is_causal_clipped_and_ignores_future_labels(relative,exponent,expected):
    from tests.strategy_olr.test_capacity_allocation import history
    cfg=config(capacity_exit_quantile=.5,capacity_broad_auction_history=True,capacity_volume_exponent=exponent)
    state=OLRState();history(state,cfg)
    c=replace(snapshot().candidates[0],metadata={'afternoon_features':{'rel_volume':relative},
        'close_to_close_label_pct':100,'next_session_mfe_label_pct':200})
    state.symbol_state(c.symbol).candidate=c
    first=forecast(state,bar(),cfg)
    assert first[1]==expected and first[2]['exit']['volume_conditioning']['base_quantity']==20
    state.symbol_state(c.symbol).candidate=replace(c,metadata={**c.metadata,
        'close_to_close_label_pct':-100,'next_session_mfe_label_pct':-200})
    assert forecast(state,bar(),cfg)==first
    with pytest.raises(ValueError,match='cutoff'):
        forecast(state,bar(clock=time(14,20)),cfg)
    state.symbol_state(c.symbol).candidate=replace(c,metadata={})
    assert forecast(state,bar(),cfg)[1]==20


def test_volume_support_measures_feasible_same_account_quantity_not_only_larger_demand():
    from tests.strategy_olr.test_capacity_allocation import history
    cfg=config(capacity_joint_allocation=True,capacity_exit_quantile=.5,
        capacity_broad_auction_history=True,capacity_volume_exponent=1)
    c=replace(snapshot().candidates[0],metadata={'afternoon_features':{'rel_volume':2}})
    snap=replace(snapshot(),candidates=(c,))
    def run(cash):
        state=OLRState();history(state,cfg)
        return step_olr_batch(state,(bar(),),cfg,snap,OLRPortfolioView(cash,10000,open_notional=10000-cash))
    funded=[a for a in run(10000).actions if isinstance(a,SubmitEntry)][0]
    assert funded.qty==40 and funded.metadata['capacity_unconditioned_approved_qty']==20
    assert funded.metadata['capacity_conditioned_qty_delta']==20
    constrained=[a for a in run(1000).actions if isinstance(a,SubmitEntry)][0]
    assert constrained.qty<20 and constrained.metadata['capacity_conditioned_qty_delta']==0


def test_session_budget_preserves_real_cash_limits_and_survives_restart():
    import json
    from strategy_olr.core.serializers import snapshot_state,restore_state
    cfg=config(capacity_joint_allocation=True,capacity_session_budget_fraction=.5)
    state=OLRState()
    result=step_olr_batch(state,(bar(),),cfg,snapshot(),OLRPortfolioView(10000,10000))
    entry=[a for a in result.actions if isinstance(a,SubmitEntry)][0]
    assert entry.metadata['reserved_notional']<=5000 and entry.metadata['capacity_budget_qty_delta']<0
    remember_submitted_order(state,'first',entry)
    state=restore_state(json.loads(json.dumps(snapshot_state(state))))
    next_result=step_olr_batch(state,(bar('000660',clock=time(14,35)),),cfg,snapshot(),OLRPortfolioView(11000,11000))
    other=[a for a in next_result.actions if isinstance(a,SubmitEntry)]
    assert sum(a.metadata['reserved_notional'] for a in other)+entry.metadata['reserved_notional']<=5000
    assert state.meta['capacity_session_budget_v1']['equity']==10000
    no_cash=step_olr_batch(OLRState(),(bar(),),cfg,snapshot(),OLRPortfolioView(0,10000,open_notional=10000))
    assert not any(isinstance(a,SubmitEntry) for a in no_cash.actions)


def test_session_budget_counts_fills_and_remainder_once_and_does_not_recycle_same_day_sales():
    from strategy_olr.core.logic import on_olr_fill
    cfg=config(capacity_joint_allocation=True,capacity_session_budget_fraction=.5)
    state=OLRState()
    first=step_olr_batch(state,(bar(),),cfg,snapshot(),OLRPortfolioView(10000,10000))
    entry=[a for a in first.actions if isinstance(a,SubmitEntry)][0]
    remember_submitted_order(state,'first',entry)
    fill=OLRFillEvent('first','005930','BUY',10,100,datetime.combine(DAY,time(14,35),tzinfo=KST),
        metadata={'fee':.2,'submitted_qty':entry.qty,'execution_id':'buy1'})
    on_olr_fill(state,fill,cfg)
    next_result=step_olr_batch(state,(bar('000660',clock=time(14,35)),),cfg,snapshot(),OLRPortfolioView(8999.8,9999.8,open_notional=1000))
    evidence=[e.metadata['capacity_session_budget'] for e in next_result.decisions if 'capacity_session_budget' in e.metadata][0]
    assert evidence['acquisition_cost']==pytest.approx(1000.2)
    assert evidence['pending_reserve']==pytest.approx(entry.metadata['reserved_notional']*(entry.qty-10)/entry.qty)
    assert evidence['remaining']==pytest.approx(max(0,5000-evidence['acquisition_cost']-evidence['pending_reserve']))
    # Cancelling the working remainder frees only that remainder; selling the
    # acquired shares does not replenish the session's acquisition allowance.
    s=state.symbol_state('005930');s.pending_entry_order_id='';s.pending_entry_metadata.clear()
    on_olr_fill(state,OLRFillEvent('sale','005930','SELL',10,100,
        datetime.combine(DAY,time(14,40),tzinfo=KST),metadata={'execution_id':'sell1'}),cfg)
    from strategy_olr.core.allocation import session_acquisition_cost
    assert session_acquisition_cost(state,DAY,cfg)==pytest.approx(1000.2)


def test_budget_uses_actual_debits_when_late_buy_follows_partial_sale():
    from strategy_olr.core.logic import on_olr_fill
    from strategy_olr.core.allocation import session_acquisition_cost
    cfg=config(capacity_joint_allocation=True,capacity_session_budget_fraction=.5)
    state=OLRState();stamp=datetime.combine(DAY,time(14,35),tzinfo=KST)
    action=SubmitEntry('OLR','005930',10,'MARKET',None,None,'entry',
        {'estimated_entry_price':200,'reserved_notional':2000,'candidate_rank':1})
    remember_submitted_order(state,'buy',action)
    for order,side,quantity,price,after,minute in [
        ('buy','BUY',6,100,6,0),('sale','SELL',3,110,3,5),('buy','BUY',4,200,10,10)]:
        on_olr_fill(state,OLRFillEvent(order,'005930',side,quantity,price,stamp+timedelta(minutes=minute),
            metadata={'fee':0.,'execution_id':str(minute),'filled_qty_after':after,
                'submitted_qty':10 if side=='BUY' else 3}),cfg)
    assert session_acquisition_cost(state,DAY,cfg)==1400
    assert state.symbol_state('005930').position.qty_open==7


@pytest.mark.parametrize('liquidity_constrained',[False,True])
def test_recycling_requires_measured_displaced_funding_and_keeps_actual_cash_bound(liquidity_constrained):
    import json
    from strategy_olr.core.serializers import snapshot_state,restore_state
    cfg=config(capacity_joint_allocation=True,capacity_exit_quantile=.5,capacity_entry_bars=12,
        capacity_redistribute=True,capacity_redistribution_evidence='entry',
        capacity_redistribution_max_multiplier=1.5,capacity_recycle_liquidity_only=True)
    state=OLRState()
    for day in (date(2026,1,5),date(2026,1,6),date(2026,1,7)):
        observe(state,bar(day=day,clock=time(15,30),volume=1000 if liquidity_constrained else 100000),cfg)
    first=step_olr_batch(state,(bar(volume=100000),),cfg,snapshot(),OLRPortfolioView(10000,10000))
    action=[a for a in first.actions if isinstance(a,SubmitEntry)][0]
    remember_submitted_order(state,'first',action)
    state=restore_state(json.loads(json.dumps(snapshot_state(state))))
    second=step_olr_batch(state,(bar('000660',clock=time(14,35),volume=100000),),cfg,snapshot(),OLRPortfolioView(10000,10000))
    extra=[a for a in second.actions if isinstance(a,SubmitEntry)][0]
    m=extra.metadata
    assert (m['allocation_redistributed_qty']>0)==liquidity_constrained
    assert m['capacity_recycling']['assigned_extras']<=m['capacity_recycling']['displaced_request_funding']+1e-9
    assert extra.metadata['reserved_notional']+action.metadata['reserved_notional']<=10000


def test_conditional_liquidity_shrinks_sparse_selected_sessions_toward_broad_history():
    import json
    from strategy_olr.core.serializers import snapshot_state,restore_state
    cfg=config(capacity_joint_allocation=True,capacity_exit_quantile=.5,
        capacity_broad_auction_history=True,capacity_history_scope='conditional_shrink')
    state=OLRState()
    for day,volume in [(date(2026,1,5),1000),(date(2026,1,6),2000)]:
        observe(state,bar(day=day,clock=time(15,30),volume=volume),cfg)
    prior=date(2026,1,7)
    snap=replace(snapshot(),trade_date=prior,generated_at=datetime.combine(prior,time(14,30),tzinfo=KST),
        candidates=tuple(replace(c,trade_date=prior) for c in snapshot().candidates))
    auction=bar(day=prior,clock=time(15,30),volume=6000)
    observe_routed(state,auction,cfg,snap);observe_routed(state,auction,cfg,snap)
    q,_,e=forecast(state,bar(),cfg)
    assert q==30
    conditioning=e['exit']['selection_conditioning']
    assert conditioning['broad_quantity']==20 and conditioning['selected_observations']==1
    assert conditioning['weight']==.25
    observe_routed(state,bar('999999',day=prior,clock=time(15,30),volume=999999),cfg,snap)
    assert forecast(state,bar(),cfg)==(q,q,e)
    restored=restore_state(json.loads(json.dumps(snapshot_state(state))))
    assert forecast(restored,bar(),cfg)==(q,q,e)
    with pytest.raises(ValueError,match='before the auction'):
        observe_routed(state,auction,cfg,replace(snap,generated_at=datetime.combine(prior,time(16),tzinfo=KST)))


def test_conditional_liquidity_falls_back_to_real_broad_evidence_not_uncapped_size():
    from tests.strategy_olr.test_capacity_allocation import history
    cfg=config(capacity_joint_allocation=True,capacity_exit_quantile=.5,
        capacity_broad_auction_history=True,capacity_history_scope='conditional_shrink')
    state=OLRState();history(state,cfg)
    q,_,e=forecast(state,bar(),cfg)
    assert q==20 and e['exit']['selection_conditioning']['selected_observations']==0


def test_exit_conditioning_matches_next_session_target_and_preserves_causal_membership():
    import json
    from strategy_olr.core.serializers import snapshot_state,restore_state
    cfg=config(capacity_joint_allocation=True,capacity_exit_quantile=.5,
        capacity_broad_auction_history=True,capacity_history_scope='conditional_exit')
    state=OLRState();selected=date(2026,1,9);exit_day=date(2026,1,12)
    snap=replace(snapshot(),trade_date=selected,
        generated_at=datetime.combine(selected,time(14,30),tzinfo=KST),
        candidates=tuple(replace(c,trade_date=selected) for c in snapshot().candidates))
    # A replay-loaded snapshot cannot seal future membership in the morning.
    observe_routed(state,bar(day=selected,clock=time(9)),cfg,snap)
    assert not state.meta['capacity_exit_cohorts_v1']
    observe_routed(state,bar(day=selected,clock=time(14,30)),cfg,snap)
    assert state.meta['capacity_exit_cohorts_v1'][str(exit_day)]['symbols']==['000660','005930']
    observe_routed(state,bar(day=selected,clock=time(15,30),volume=1000),cfg,snap)
    assert not state.meta.get('capacity_selected_history_v1')
    state=restore_state(json.loads(json.dumps(snapshot_state(state))))
    # No order or position is needed: every selected issuer contributes its
    # next-session auction, regardless of whether the original entry filled.
    auction=bar(day=exit_day,clock=time(15,30),volume=6000)
    observe_routed(state,auction,cfg,None);observe_routed(state,auction,cfg,None)
    rows=state.meta['capacity_selected_history_v1']['005930']['auction']
    assert rows==[[auction.timestamp.isoformat(),6000.]]
    observe_routed(state,bar('999999',day=exit_day,clock=time(15,30),volume=999999),cfg,None)
    assert '999999' not in state.meta['capacity_selected_history_v1']
    # Today's auction is not a prior-session forecast observation at a decision
    # earlier that day, even if a caller incorrectly supplies advanced state.
    assert forecast(state,bar(day=exit_day),cfg)[2]['exit']['quantity'] is None
    for d,v in [(date(2026,1,7),2000),(date(2026,1,8),3000)]:
        # Add warm history through the real seeding API; never insert labels.
        from strategy_olr.core.capacity import seed_history
        seed_history(state,[replace(bar(day=d,clock=time(15,30),volume=v),source_fingerprint='fixture-auction')],cfg,
            datetime.combine(date(2026,1,13),time(0),tzinfo=KST))
    _,q,e=forecast(state,bar(day=date(2026,1,13)),cfg)
    assert e['exit']['selection_conditioning']['target']=='next_session_exit'
    assert e['exit']['selection_conditioning']['selected_observations']==1
    assert q==33


def test_exit_conditioning_rejects_changed_sealed_membership():
    cfg=config(capacity_joint_allocation=True,capacity_exit_quantile=.5,
        capacity_broad_auction_history=True,capacity_history_scope='conditional_exit')
    state=OLRState();snap=snapshot()
    observe_routed(state,bar(),cfg,snap)
    with pytest.raises(ValueError,match='membership changed'):
        observe_routed(state,bar(clock=time(14,35)),cfg,replace(snap,candidates=snap.candidates[:1]))


def test_direct_support_does_not_count_carried_fill_metadata_on_later_days(tmp_path):
    import gzip,json
    from backtests.strategies.olr.allocation_recovery import support
    events=[]
    for day,code in [(1,'ENTRY_SUBMITTED'),(1,'ENTRY_FILLED'),(2,'ENTRY_FILLED'),(3,'EXIT_FILLED')]:
        events.append({'timestamp':f'2026-01-0{day}T14:30:00+09:00','symbol':'005930',
            'decision_code':code,'reason':'entry','metadata':{'allocation_redistributed_qty':1}})
    for day,code in [(4,'CAPITAL_ROTATION_SUBMITTED'),(5,'EXIT_FILLED')]:
        events.append({'timestamp':f'2026-01-0{day}T14:30:00+09:00','symbol':'005930',
            'decision_code':code,'reason':'opportunity_capital_rotation','metadata':{}})
    with gzip.open(tmp_path/'decisions.jsonl.gz','wt',encoding='utf-8') as stream:
        for event in events:
            stream.write(json.dumps(event)+'\n')
    result=support(tmp_path)
    assert result['distinct_issuer_days']==2
    assert result['by_mechanism']['redistribute']==result['by_mechanism']['rotation']==1


def test_recovery_selection_restores_median_when_high_score_fails_one_capital_cell():
    from backtests.strategies.olr.allocation_recovery import AllocationRecoveryPlugin
    from backtests.auto.shared.greedy_optimizer import run_greedy
    from backtests.auto.shared.types import Experiment
    plugin=object.__new__(AllocationRecoveryPlugin)
    plugin.names=['anchor_winner'];plugin.batch=lambda names:None
    def evaluate(name):
        failed=name!='median'
        return {'score':200 if failed else 100,
            'failures':['10000000_combined_return'] if failed else [],
            'metrics':{'qualified':not failed}}
    plugin.evaluate=evaluate
    result=run_greedy([Experiment('anchor_winner',{'recipe':'anchor_winner'})],
        {'recipe':'median'},plugin.create_evaluate_batch(1,{'recipe':'median'}),max_rounds=1)
    assert result.final_mutations=={'recipe':'median'}
    assert result.accepted_count==0


@pytest.mark.parametrize('volume,has_extra',[(100,False),(100000,True)])
def test_extra_only_entry_forecast_never_shrinks_median_native_allocation(volume,has_extra):
    base=config(capacity_joint_allocation=True,capacity_exit_quantile=.5)
    extra=replace(base,capacity_redistribute=True,capacity_redistribution_evidence='entry',
        capacity_redistribution_max_multiplier=1.5)
    current=bar('000660',volume=volume)
    def run(cfg):
        result=step_olr_batch(OLRState(),(current,),cfg,snapshot(),OLRPortfolioView(10000,10000))
        return next(a for a in result.actions if isinstance(a,SubmitEntry))
    original=run(base);augmented=run(extra);m=augmented.metadata
    assert m['allocation_before_redistribution']==original.qty
    assert augmented.qty>=original.qty
    assert bool(m['allocation_redistributed_qty'])==has_extra
    assert augmented.qty<=int(original.qty*1.5)
    assert m['reserved_notional']<=10000
    entry_capacity=m['capacity_forecast']['entry']['quantity']
    if has_extra:
        assert augmented.qty<=entry_capacity
    else:
        assert original.qty>entry_capacity and augmented.qty==original.qty
