"""Fixed one-minute exit clock: native fills, partials and causal observations."""
from dataclasses import replace
from datetime import date,datetime,time,timedelta
import pytest

from strategy_common.clock import KST
from strategy_common.actions import SubmitEntry,SubmitExit
from strategy_olr.models import OLRDailySnapshot
from backtests.strategies.olr.runner import OLRReplayAdapter
from backtests.engine.sim_broker import SimBroker,BrokerCosts
from backtests.engine.olr_minute_v5 import (MinuteParticipationScenario,MinuteOLRDriver,
    continuous_minute,fixed_exit_plan)
from tests.strategy_olr.test_capacity_allocation import config,bar

DAY=date(2026,1,8)
PREVIOUS=date(2026,1,7)
PLAN=dict(name='fixed_minute_v5',mode='managed',hard_stop_enabled=False,
    failed_followthrough_bars=1,failed_followthrough_close_r=-1.,failed_followthrough_mfe_r=4.,
    mfe_fade_start_r=1.,mfe_fade_gap_r=1.25,mfe_fade_floor_r=0.)


def minute(day=DAY,t=time(9),price=100,volume=1000,**kw):
    return replace(bar(day=day,clock=t,volume=volume),timeframe='1m',open=price,high=price,
        low=price,close=price,source_fingerprint='minute_fixture',**kw)


def coarse(rows):
    assert len(rows)==5
    return replace(rows[0],timeframe='5m',open=rows[0].open,high=max(b.high for b in rows),
        low=min(b.low for b in rows),close=rows[-1].close,volume=sum(b.volume for b in rows),
        source_fingerprint='coarse_fixture')


def setup(fast=True,qty=20,costs=None):
    cfg=config(capacity_joint_allocation=True,capacity_exit_priority=True,trade_exit_plan=PLAN)
    snaps={d:OLRDailySnapshot(d,(),'fixture',datetime.combine(d,time(8),tzinfo=KST)) for d in (PREVIOUS,DAY)}
    adapter=OLRReplayAdapter(cfg,snaps)
    broker=SimBroker(initial_equity=10000,costs=costs or BrokerCosts(0,0,0),
        capacity_policy=MinuteParticipationScenario(.01,'minute-test'))
    driver=MinuteOLRDriver(broker,adapter,fast_exit=fast)
    at=datetime.combine(PREVIOUS,time(14,24),tzinfo=KST)
    adapter._submit_actions([SubmitEntry('OLR','005930',qty,'MARKET',None,None,'fixture',
        dict(risk_per_share=10,trade_exit_plan=PLAN,estimated_entry_price=100))],broker,at)
    rows=[minute(PREVIOUS,time(14,25+i)) for i in range(5)]
    driver.run_day(rows,[coarse(rows)])
    assert sum(f.qty for f in broker.fills)==qty
    return broker,adapter,driver


def test_capacity_uses_only_prior_contiguous_minute_and_fixed_stress():
    prior=minute(volume=1234);current=minute(t=time(9,1),volume=987654)
    assert MinuteParticipationScenario(.01,'base').quote(current,'open',prior).budget==12
    assert MinuteParticipationScenario(.005,'stress').quote(current,'open',prior).budget==6
    policy=MinuteParticipationScenario(.01,'base')
    for bad in (None,replace(prior,timestamp=prior.timestamp-timedelta(minutes=1)),
                replace(prior,timeframe='5m'),replace(prior,symbol='000660'),
                replace(prior,timestamp=prior.timestamp-timedelta(days=1))):
        assert policy.quote(current,'open',bad).budget==0
    assert policy.quote(current,'intrabar',prior).budget==0


def test_session_boundaries_and_real_auction_volume():
    assert not continuous_minute(minute(t=time(8,59)))
    assert continuous_minute(minute(t=time(15,19)))
    assert not continuous_minute(minute(t=time(15,20)))
    policy=MinuteParticipationScenario(.01,'base')
    auction=minute(t=time(15,30),volume=900)
    assert policy.quote(auction,'auction',None).budget==9
    assert policy.quote(replace(auction,high=101),'auction',None).budget==0
    assert policy.quote(auction,'open',minute(t=time(15,29))).budget==0


@pytest.mark.parametrize('change',[{'failed_followthrough_close_r':-.5},{'mfe_fade_gap_r':1},
    {'hard_stop_enabled':True},{'no_mfe_bars':1},{'max_hold_bars':2},{'vwap_fail_bars':1}])
def test_threshold_or_extra_rule_change_rejected(change):
    with pytest.raises(ValueError):fixed_exit_plan(config(trade_exit_plan={**PLAN,**change}))


def test_native_faster_loss_exit_fills_next_open_with_partial_remainders():
    broker,adapter,driver=setup()
    rows=[minute(t=time(9,i),price=p) for i,p in enumerate((100,88,89,90,100))]
    driver.run_day(rows,[coarse(rows)])
    signals=[r for r in broker.journal if r['event']=='minute_exit_signal_v5']
    sells=[f for f in broker.fills if f.side=='SELL']
    assert len(signals)==1 and signals[0]['available_at']==rows[2].timestamp.isoformat()
    assert [(f.timestamp.time(),f.qty,f.price) for f in sells]==[(time(9,2),10,89),(time(9,3),10,90)]
    assert all(f.reason=='failed_followthrough' for f in sells)
    assert not broker.orders and not adapter.state.symbols['005930'].position
    assert broker.cash==9790 and broker.same_bar_fill_violations==0


def test_control_defers_risk_to_completed_coarse_and_never_reads_early_close():
    broker,adapter,driver=setup(fast=False)
    rows=[minute(t=time(9,i),price=p) for i,p in enumerate((100,88,89,90,100))]
    observed=[];original=adapter.on_bars
    def watch(batch,b):
        observed.append((batch[0].timestamp,b._decision_available_at,len(b.timestamps),b.last_prices['005930']))
        return original(batch,b)
    adapter.on_bars=watch
    driver.run_day(rows,[coarse(rows)])
    assert observed==[(rows[0].timestamp,rows[-1].timestamp+timedelta(minutes=1),10,100)]
    assert not [f for f in broker.fills if f.side=='SELL']
    assert not [r for r in broker.journal if r['event']=='minute_exit_signal_v5']


def test_no_minute_exit_on_entry_day_and_no_volume_across_gap():
    broker,adapter,driver=setup()
    rows=[minute(PREVIOUS,time(14,30+i),price=80) for i in range(5)]
    driver.run_day(rows,[coarse(rows)])
    assert not [r for r in broker.journal if r['event']=='minute_exit_signal_v5']
    # Overnight loss triggers at09:01; missing09:01 prevents09:02 fill.
    rows=[minute(t=time(9,i),price=80) for i in (0,2,3,4)]
    summary=replace(coarse([minute(t=time(9,i),price=80) for i in range(5)]),volume=4000)
    driver.run_day(rows,[summary])
    assert [f.timestamp.time() for f in broker.fills if f.side=='SELL']==[time(9,3),time(9,4)]


def test_priority_auction_cancel_ack_before_single_market_replacement():
    broker,adapter,driver=setup()
    at=datetime.combine(DAY,time(8,59),tzinfo=KST)
    adapter._submit_actions([SubmitExit('OLR','005930',20,'CLOSE_AUCTION',95,'next_close_exit')],broker,at)
    rows=[minute(t=time(9,i),price=88) for i in range(5)]
    driver.run_day(rows,[coarse(rows)])
    cancels=[i for i,r in enumerate(broker.journal) if r['event']=='cancel' and r['order']['order_type']=='CLOSE_AUCTION']
    replacements=[i for i,r in enumerate(broker.journal) if r['event']=='submit' and r['order']['side']=='SELL' and r['order']['order_type']=='MARKET']
    assert len(cancels)==len(replacements)==1 and cancels[0]<replacements[0]
    assert sum(f.qty for f in broker.fills if f.side=='SELL')==20
    assert not adapter.state.meta.get('priority_exit_pending')


def test_mfe_fade_uses_observed_minutes_and_costs_are_native():
    broker,adapter,driver=setup(costs=BrokerCosts(2,20,5))
    rows=[minute(t=time(9,i),price=p) for i,p in enumerate((120,105,106,107,108))]
    driver.run_day(rows,[coarse(rows)])
    signals=[r for r in broker.journal if r['event']=='minute_exit_signal_v5']
    assert len(signals)==1 and signals[0]['reason']=='mfe_fade'
    sells=[f for f in broker.fills if f.side=='SELL']
    assert [f.timestamp.time() for f in sells]==[time(9,2),time(9,3)]
    assert all(f.price<f.metadata['base_price'] and f.metadata['fee']>0 for f in broker.fills if f.side=='SELL')
    debit=sum(f.qty*f.price+f.metadata['fee'] for f in broker.fills if f.side=='BUY')
    credit=sum(f.qty*f.price-f.metadata['fee'] for f in sells)
    assert broker.cash==pytest.approx(10000-debit+credit)


def test_final_auction_timer_never_moves_clock_backwards():
    broker,adapter,driver=setup(fast=False)
    row=minute(t=time(15,30))
    driver.run_day([row],[replace(row,timeframe='5m')])
    assert broker.minute_clock_schedule[-1]['available_at']==row.timestamp.replace(minute=35).isoformat()
    checkpoints=[r for r in broker.journal if r['event']=='strategy_state']
    assert len(checkpoints)==len(broker.expected_batches)


def test_missing_final_continuous_minute_rejects_lookahead():
    broker,adapter,driver=setup(fast=False)
    rows=[minute(t=time(9,i)) for i in range(5)]
    with pytest.raises(ValueError,match='Unobserved continuous'):
        driver.run_day(rows[:3],[coarse(rows)])


def test_periodic_closing_observation_does_not_gain_execution_capacity(monkeypatch):
    from kis_core.trading_calendar import get_trading_calendar
    from backtests.engine.olr_minute_audit_v5 import audit_minute_clock,audit_minute_capacity
    calendar=get_trading_calendar();original=calendar.periodic_auction_sources
    monkeypatch.setattr(calendar,'periodic_auction_sources',lambda symbol,day: ['fixture'] if symbol=='086520' else original(symbol,day))
    broker,adapter,driver=setup(fast=False)
    row=minute(t=time(15,30));periodic=replace(row,symbol='086520')
    before=len(broker.fills)
    driver.run_day([row],[replace(row,timeframe='5m'),replace(periodic,timeframe='5m')])
    assert broker.minute_clock_schedule[-1]['available_at']==row.timestamp.replace(minute=35).isoformat()
    assert len(broker.fills)==before
    assert broker.capacity_policy.quote(periodic,'auction',None).budget==0
    assert audit_minute_capacity(broker)['errors']==[]
    assert audit_minute_clock(broker)['errors']==[]


def test_session_final_timer_audit_allows_next_session_without_early_delivery():
    from backtests.engine.olr_minute_audit_v5 import audit_minute_clock
    broker,adapter,driver=setup(fast=False)
    for day in (PREVIOUS,DAY):
        row=minute(day,t=time(15,30))
        driver.run_day([row],[replace(row,timeframe='5m')])
    assert audit_minute_clock(broker)['errors']==[]
    timer=next(r for r in broker.journal if r['event']=='coarse_observation_v5' and r['label'].startswith(str(PREVIOUS)+'T15:30'))
    timer['delivered_by']=timer['label']
    assert 'minute_coarse_clock' in audit_minute_clock(broker)['errors']


def test_multiple_terminal_clocks_deliver_at_their_own_availability(monkeypatch):
    from kis_core.trading_calendar import get_trading_calendar
    from backtests.engine.olr_minute_audit_v5 import audit_minute_clock
    calendar=get_trading_calendar()
    monkeypatch.setattr(calendar,'closing_auctions',{**calendar.closing_auctions,('000660',DAY):time(15,32)})
    broker,adapter,driver=setup(fast=False)
    row=minute(t=time(15,30));late=replace(minute(t=time(15,32)),symbol='000660',timeframe='5m')
    driver.run_day([row],[replace(row,timeframe='5m'),late])
    schedule=broker.minute_clock_schedule[-2:]
    assert [r['available_at'][11:16] for r in schedule]==['15:35','15:37']
    assert all(r['delivered_by']==r['available_at'] for r in schedule)
    assert audit_minute_clock(broker)['errors']==[]


@pytest.mark.parametrize('change',[{'high':101},{'volume':0},{'timestamp':datetime.combine(DAY,time(15,31),tzinfo=KST)}])
def test_terminal_observation_does_not_accept_nonpoint_or_invalid_source(change):
    broker,adapter,driver=setup(fast=False)
    row=minute(t=time(15,30));invalid=replace(row,timeframe='5m',**change)
    with pytest.raises(ValueError,match='Unobserved continuous'):
        driver.run_day([row],[invalid])


@pytest.mark.parametrize('fast',[False,True])
def test_independent_capacity_and_clock_audits_reconstruct_native_path(fast):
    from backtests.engine.olr_minute_audit_v5 import audit_minute_capacity,audit_minute_clock
    broker,adapter,driver=setup(fast=fast)
    rows=[minute(t=time(9,i),price=p) for i,p in enumerate((100,88,89,90,100))]
    driver.run_day(rows,[coarse(rows)])
    capacity=audit_minute_capacity(broker);clock=audit_minute_clock(broker)
    assert capacity['errors']==[]
    assert clock['errors']==[]
    assert clock['reconstructed_required_signals']==int(fast)


@pytest.mark.parametrize('tamper',['prior_volume','available_clock','missing_signal','high_water','coarse_early'])
def test_independent_audits_reject_causal_or_exit_evidence_tampering(tamper):
    from backtests.engine.olr_minute_audit_v5 import audit_minute_capacity,audit_minute_clock
    broker,adapter,driver=setup()
    rows=[minute(t=time(9,i),price=p) for i,p in enumerate((100,88,89,90,100))]
    driver.run_day(rows,[coarse(rows)])
    if tamper=='prior_volume':
        row=next(r for r in broker.journal if r['event']=='capacity' and r['quote']['budget'])
        row['quote']['predictor_volume']+=100
        assert 'capacity_quote_mismatch' in audit_minute_capacity(broker)['errors']
        return
    if tamper=='available_clock':
        row=next(r for r in broker.journal if r['event']=='submit' and r['order']['side']=='SELL')
        row['order']['metadata']['decision_available_at']=rows[1].timestamp.isoformat()
    elif tamper=='missing_signal':broker.journal[:]=[r for r in broker.journal if r['event']!='minute_exit_signal_v5']
    elif tamper=='high_water':
        row=next(r for r in broker.journal if r['event']=='minute_exit_signal_v5');row['high_water']+=20
    elif tamper=='coarse_early':
        row=next(r for r in broker.journal if r['event']=='coarse_observation_v5');row['delivered_by']=row['label']
    assert audit_minute_clock(broker)['errors']


def test_native_ledger_audit_keeps_fine_marks_and_authenticates_costs():
    from backtests.engine.integrity import audit_broker
    broker,adapter,driver=setup(costs=BrokerCosts(2,20,5))
    rows=[minute(t=time(9,i),price=p) for i,p in enumerate((100,88,89,90,100))]
    driver.run_day(rows,[coarse(rows)])
    audit=audit_broker(broker)
    assert audit['execution_validity_pass'],audit['audit_receipt']['errors']
    assert audit['capacity_model_audit']['status']=='model_consistent'
    assert audit['audit_receipt']['minute_clock_audit']['status']=='pass'
    assert audit['audit_receipt']['source_event_count']==10
    # Synthetic fixture bars have no market-source qualification. Never promote
    # execution-only test success to a native historical research result.
    assert not audit['audit_pass']
    fill=next(r for r in broker.journal if r['event']=='fill')
    fill['cash_after']+=1
    assert not audit_broker(broker)['execution_validity_pass']
