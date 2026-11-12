from dataclasses import replace
from datetime import timedelta

import pytest

from backtests.engine.capacity import BarParticipationScenario
from backtests.engine.integrity import audit_broker
from backtests.engine.replay import drive_timestamp_batch
from backtests.strategies.portfolio_synergy.intent_replay import (
    IntentSharedBroker,IntentOLRReplayAdapter,audit_intent_allocations)
from deployment.olr_kalcb.intent_allocator import IntentAllocationConfig
from deployment.olr_kalcb.portfolio import PortfolioPolicyConfig
from strategy_common.actions import SubmitEntry,SubmitExit
from strategy_olr.config import OLRConfig
from strategy_olr.core.logic import pending_entry_reservation
from strategy_olr.models import OLRDailySnapshot
from tests.backtests.test_strategy_integrity_repairs import ZERO,bar,ts


def setup(config=None, **policy_changes):
    policy=PortfolioPolicyConfig(max_gross_notional=2000,max_symbol_notional=600,
        max_sector_notional=2000,same_symbol_policy='aggregate',funding_mode='native_buying_power')
    account=IntentSharedBroker(1000,strategy_costs={'KALCB':ZERO,'OLR':ZERO},
        strategy_leverage={'KALCB':2,'OLR':1},portfolio_policy=replace(policy,**policy_changes),
        allocation_config=config or IntentAllocationConfig(),capacity_policy=BarParticipationScenario(.01,'test'))
    snap=OLRDailySnapshot(ts().date(),(),'test',ts())
    adapter=IntentOLRReplayAdapter(OLRConfig(commission_bps=0,slippage_bps=0,tax_bps_on_sell=0),{snap.trade_date:snap})
    return account,adapter


def queue(account,adapter,symbol,qty=10,rank=1,at=None):
    stamp=at or ts(30,hour=14);account._decision_available_at=stamp+timedelta(minutes=5)
    adapter._submit_actions([SubmitEntry('OLR',symbol,qty,'MARKET',None,None,'test',
        metadata={'estimated_entry_price':100,'reserved_notional':qty*100,'candidate_rank':rank})],account,stamp)
    adapter.state.symbol_state(symbol).entry_attempted=True


def test_only_approved_quantities_are_reserved_and_partial_fills_release_them():
    account,adapter=setup()
    queue(account,adapter,'000001',rank=1);queue(account,adapter,'000002',rank=2)
    assert not account.orders and pending_entry_reservation(adapter.state.symbol_state('000001'))==(0,0)
    batch=[bar(s,ts(30,hour=14),volume=300) for s in ('000001','000002')]
    drive_timestamp_batch(account,batch,[adapter])
    assert [o.qty for o in account.orders]==[6,4]
    assert pending_entry_reservation(adapter.state.symbol_state('000001'))==(6,600)
    drive_timestamp_batch(account,[bar(s,ts(35,hour=14),volume=300) for s in ('000001','000002')],[adapter])
    assert pending_entry_reservation(adapter.state.symbol_state('000001'))==(3,300)
    assert pending_entry_reservation(adapter.state.symbol_state('000002'))==(1,100)
    assert audit_broker(account)['execution_validity_pass']
    assert audit_intent_allocations(account)['pass']


def test_unfilled_exit_does_not_fund_new_intents_and_expired_deferral_is_recorded():
    account,adapter=setup(IntentAllocationConfig(defer_minutes=5),max_symbol_notional=1000)
    queue(account,adapter,'000001')
    drive_timestamp_batch(account,[bar('000001',ts(30,hour=14))],[adapter])
    drive_timestamp_batch(account,[bar('000001',ts(35,hour=14))],[adapter])
    adapter._submit_actions([SubmitExit('OLR','000001',10,'LIMIT',200,'future_sale')],account,ts(35,hour=14))
    queue(account,adapter,'000002',at=ts(35,hour=14))
    drive_timestamp_batch(account,[bar('000001',ts(40,hour=14)),bar('000002',ts(40,hour=14))],[adapter])
    assert not any(o.side=='BUY' for o in account.orders)
    assert any(x['event']=='intent_cancel' and x['reason']=='insufficient_strategy_budget' for x in account.allocation_events)


def test_capacity_forecast_uses_only_completed_prior_observations():
    account,adapter=setup(IntentAllocationConfig(entry_capacity_bars=2))
    queue(account,adapter,'000001')
    drive_timestamp_batch(account,[bar('000001',ts(30,hour=14),volume=100)],[adapter])
    assert account.orders[0].qty==2
    forecast=next(x for x in account.allocation_events if x['event']=='allocation_cycle')['forecasts']
    assert next(iter(forecast.values()))['capacity']['entry']['samples'][0][0]==ts(30,hour=14).isoformat()
    assert audit_intent_allocations(account)['pass']
    entry=next(iter(forecast.values()))['capacity']['entry']
    stamp,volume=entry['samples'][0];entry['samples'][0]=(stamp,volume+1)
    assert 'allocation_unbound_capacity_observation' in audit_intent_allocations(account)['errors']


def test_independent_allocation_audit_rejects_reservation_corruption():
    account,adapter=setup();queue(account,adapter,'000001')
    drive_timestamp_batch(account,[bar('000001',ts(30,hour=14))],[adapter])
    cycle=next(x for x in account.allocation_events if x['event']=='allocation_cycle')
    cycle['decisions'][0]['reserved_notional']+=100
    assert 'allocation_reservation_mismatch' in audit_intent_allocations(account)['errors']


def test_external_olr_view_preserves_virtual_sizing_equity():
    account,adapter=setup(IntentAllocationConfig(olr_sizing_fraction=.5))
    view=adapter._portfolio_view(account)
    assert view.equity==view.cash==500 and view.external_entry_allocation


def test_native_olr_intent_is_visible_even_when_account_funding_is_exhausted():
    from strategy_olr.core.core_models import OLRPortfolioView
    from strategy_olr.core.state import OLRState
    from strategy_olr.core.logic import step_olr_core
    from tests.strategy_olr.test_core_runner import _candidate
    at=ts(30,hour=14);candidate=_candidate('000001',at.date(),rank=1)
    snapshot=OLRDailySnapshot(at.date(),(candidate,),'test',at)
    config=OLRConfig(commission_bps=0,slippage_bps=0,
        trade_entry_plan={'name':'intent_test','mode':'decision_next_open'})
    native=step_olr_core(OLRState(),bar('000001',at),config,snapshot,OLRPortfolioView(0,1000))
    proposed=step_olr_core(OLRState(),bar('000001',at),config,snapshot,
        OLRPortfolioView(0,1000,external_entry_allocation=True))
    assert not native.actions and native.decisions[0].reason=='qty_zero'
    assert len(proposed.actions)==1 and proposed.actions[0].qty>0


def test_research_sizing_reference_keeps_actual_borrowing_limit():
    from backtests.strategies.portfolio_synergy.intent_replay import ReferenceSizingBroker
    b=ReferenceSizingBroker(1000,strategy_costs={'KALCB':ZERO,'OLR':ZERO},
        strategy_leverage={'KALCB':2,'OLR':1},reference_sizing=1.5)
    assert b.strategy_sizing_equity()==b.strategy_risk_cash('KALCB')==1500
    assert b.strategy_buying_power('KALCB')==2000


def test_larger_kalcb_sizing_cannot_spend_virtual_cash_as_physical_cash():
    from backtests.strategies.portfolio_synergy.intent_replay import IntentKALCBReplayAdapter
    from strategy_kalcb.config import KALCBConfig
    from strategy_kalcb.models import KALCBDailySnapshot
    account,olr=setup(IntentAllocationConfig(kalcb_sizing_fraction=1.5),max_symbol_notional=2000)
    queue(account,olr,'000001',qty=5)
    drive_timestamp_batch(account,[bar('000001',ts(30,hour=14))],[olr])
    drive_timestamp_batch(account,[bar('000001',ts(35,hour=14))],[olr])
    snapshot=KALCBDailySnapshot(ts().date(),(),'test',ts())
    kalcb=IntentKALCBReplayAdapter(KALCBConfig(frontier_enabled=False),{snapshot.trade_date:snapshot},initial_equity=1000,costs=ZERO)
    view=kalcb._portfolio_view(account)
    assert view.equity==view.cash==1500
    account._decision_available_at=ts(45,hour=14)
    kalcb._submit_actions([SubmitEntry('KALCB','000002',30,'MARKET',None,None,'test',
        metadata={'entry_price_ref':100,'candidate_rank':1})],account,ts(40,hour=14))
    drive_timestamp_batch(account,[bar('000002',ts(40,hour=14))],[kalcb,olr])
    assert account.orders[0].qty==15
    assert account.cash==500 and account._portfolio_equity()==1000
    assert audit_intent_allocations(account)['pass']


def test_native_kalcb_submission_preserves_working_state_and_has_audit_coverage():
    from backtests.strategies.portfolio_synergy.intent_replay import IntentKALCBReplayAdapter
    from strategy_kalcb.config import KALCBConfig
    from strategy_kalcb.models import KALCBDailySnapshot
    from strategy_kalcb.core.state import SymbolStage
    account,olr=setup(IntentAllocationConfig(preserve_kalcb_submission=True),max_symbol_notional=2000)
    drive_timestamp_batch(account,[bar('000001',ts(25,hour=14))],[olr])
    snapshot=KALCBDailySnapshot(ts().date(),(),'test',ts())
    kalcb=IntentKALCBReplayAdapter(KALCBConfig(frontier_enabled=False),{snapshot.trade_date:snapshot},initial_equity=1000,costs=ZERO)
    state=kalcb.state.symbol_state('000002');state.stage=SymbolStage.ENTRY_QUEUED
    account._decision_available_at=ts(35,hour=14)
    kalcb._submit_actions([SubmitEntry('KALCB','000002',10,'MARKET',None,None,'test',
        metadata={'entry_price_ref':100,'risk_reserved_entry_qty':10,'candidate_rank':1})],account,ts(30,hour=14))
    assert len(account.orders)==1 and account.orders[0].qty==10
    assert not account.entry_intents and state.stage==SymbolStage.ENTRY_QUEUED
    assert state.pending_entry_order_id==account.orders[0].order_id
    audit=audit_intent_allocations(account)
    assert audit['pass'] and audit['native_kalcb_submissions']==1
    account.allocation_events[-1]['native_qty']+=1
    assert 'native_kalcb_submit_mismatch' in audit_intent_allocations(account)['errors']


def test_native_kalcb_mode_rejects_an_external_value_ranker():
    with pytest.raises(ValueError,match='external value ranking'):
        IntentAllocationConfig(preserve_kalcb_submission=True,rank_mode='supported_value')


def test_execution_rechecks_strategy_budget_after_price_moves_and_audits_it():
    account,adapter=setup(IntentAllocationConfig(olr_budget_fraction=.5),max_symbol_notional=1000)
    queue(account,adapter,'000001')
    drive_timestamp_batch(account,[bar('000001',ts(30,hour=14))],[adapter])
    assert account.orders[0].qty==5
    drive_timestamp_batch(account,[bar('000001',ts(35,hour=14),o=200,h=200,l=200,c=200)],[adapter])
    assert account.fills[0].qty==2
    assert audit_broker(account)['execution_validity_pass']
    row=next(x for x in account.journal if x['event']=='strategy_budget_admission')
    row['approved_qty']+=1
    assert 'incorrect_strategy_budget_admission' in audit_broker(account)['audit_receipt']['errors']
