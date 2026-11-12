from datetime import datetime,timedelta
from copy import deepcopy
import pytest
from backtests.strategies.olr.decision_account import ObservedAccount
from tests.strategy_olr.test_decision_response import base


def events():
    r,j=base()
    for e in j:
        if e['event']=='fill':e['fill']['strategy_id']='OLR'
    return j


def test_account_peak_includes_unrealized_giveback_and_costs_incrementally():
    j=events();at=datetime.fromisoformat('2025-09-01T09:01:00+09:00');a=ObservedAccount(100_000_000)
    a.consume(j[:3],at,actual_cash=99_999_989,actual_equity=100_000_009)
    z=a.consume(j,at,actual_cash=100_000_003,actual_equity=100_000_003)
    assert z['high_water']==100_000_009
    assert z['drawdown_fraction']==pytest.approx(6/100_000_009)
    assert a.consume(j,at,actual_cash=100_000_003,actual_equity=100_000_003)==z


def test_account_cannot_silently_reset_peak_or_accept_future_or_bad_cash():
    j=events();at=datetime.fromisoformat('2025-09-01T09:01:00+09:00')
    with pytest.raises(ValueError,match='Future'):ObservedAccount(100_000_000).consume(j,at-timedelta(minutes=1),actual_cash=0,actual_equity=0)
    bad=deepcopy(j);bad[-1]['cash_before']+=1
    with pytest.raises(ValueError,match='Cash-before'):ObservedAccount(100_000_000).consume(bad,at,actual_cash=100_000_003,actual_equity=100_000_003)
    with pytest.raises(ValueError,match='disagrees'):ObservedAccount(100_000_000).consume(j,at,actual_cash=100_000_003,actual_equity=100_000_002)


def test_replay_provider_observes_acknowledged_native_account_and_defaults_to_parity():
    from datetime import time
    from backtests.strategies.olr.runner import OLRReplayAdapter
    from backtests.engine.sim_broker import SimBroker,BrokerCosts
    from tests.strategy_olr.test_capacity_allocation import config,bar,snapshot,DAY
    cfg=config(capacity_joint_allocation=True);seen=[]
    def provider(req):
        seen.append(req)
        assert req['observed_account']['equity']==req['account']['equity']
        assert req['observed_account']['high_water']>=req['account']['equity']
        return None
    def run(callback):
        a=OLRReplayAdapter(cfg,{DAY:snapshot()},decision_context_provider=callback)
        b=SimBroker(10000,costs=BrokerCosts(2,18,5))
        for current in (bar(),bar(clock=time(14,35)),bar(clock=time(14,40))):
            b.process_bar(current);a.on_bars((current,),b)
        return a,b
    control,cb=run(None);treatment,tb=run(provider)
    assert seen and tb.journal==cb.journal and tb.cash==cb.cash
    assert treatment.action_payloads==control.action_payloads
