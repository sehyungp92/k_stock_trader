from dataclasses import asdict, replace
from datetime import date, datetime
import pytest
from scripts.kalcb_atr_intervention_observer import Observer, feasible_extra
from backtests.engine.sim_broker import SimBroker, BrokerCosts
from backtests.engine.capacity import BarParticipationScenario
from strategy_common.actions import SubmitEntry
from strategy_common.clock import KST
from strategy_common.market import MarketBar
from strategy_kalcb.config import KALCBConfig
from strategy_kalcb.models import KALCBDailyCandidate
from strategy_kalcb.core import logic


@pytest.mark.parametrize('actual,parent,capacity,realized,affordable,expected', [
    (25, 100, 20, 100, True, 0),
    (25, 100, 50, 100, True, 25),
    (25, 100, 100, 99, True, 0),
    (25, 100, 100, 100, False, 0),
    (25, 100, 100, 100, True, 75),
])
def test_same_state_feasibility_is_conservative(actual,parent,capacity,realized,affordable,expected):
    assert feasible_extra(actual,parent,capacity,realized,affordable) == expected


def native_case(observe, actual_volume=100000):
    candidate = KALCBDailyCandidate('005930',date(2026,1,5),110,90,100,20,100000,100000,
        metadata={'action_quantity_scale':.25,'diagnostic_original_quantity':1.,'diagnostic_quantity_rule':'atr_quantity_5pct'})
    cfg = replace(KALCBConfig(),entry_action_quantity_scale_enabled=True)
    observer = Observer()
    if observe: observer.install()
    try:
        broker = SimBroker(1_000_000,BrokerCosts(0,0,0),capacity_policy=BarParticipationScenario(.01,'observer-test'))
        first = datetime(2026,1,5,9,0,tzinfo=KST); following=first.replace(minute=5)
        broker.process_bar(MarketBar('005930',first,'5m',100,100,100,100,100000))
        qty = logic.compute_entry_qty(cash=broker.cash,open_notional=0,portfolio_equity=broker.cash,
            entry_price=100,stop_price=90,config=cfg,candidate=candidate,entry_type='KRX_FIRST30_OPEN',momentum_score=3)
        assert qty > 0
        broker.submit(SubmitEntry('KALCB','005930',qty,'MARKET',None,None,'entry'),first)
        broker.process_bar(MarketBar('005930',following,'5m',100,100,100,100,actual_volume))
        return {'cash':broker.cash,'fills':[asdict(f) for f in broker.fills],'orders':[asdict(o) for o in broker.orders]},observer
    finally:
        if observe: observer.restore()


def test_observer_preserves_native_fills_and_cash_and_confirms_actual_quantity_effect():
    plain,_ = native_case(False)
    observed,probe = native_case(True)
    assert observed == plain
    assert probe.summary()['direct_feasible_issuer_days'] == 1
    assert probe.summary()['direct_feasible_sessions'] == 1
    assert not probe.summary()['preferred_support_passed']


def test_future_realized_volume_never_changes_native_decision_and_cannot_inflate_support():
    normal,normal_probe = native_case(True)
    thin,thin_probe = native_case(True,actual_volume=1)
    assert normal_probe.probes[0]['actual_qty'] == thin_probe.probes[0]['actual_qty']
    assert normal_probe.probes[0]['restored_qty'] == thin_probe.probes[0]['restored_qty']
    assert len(normal['fills']) == len(thin['fills']) == 1
    assert thin_probe.summary()['direct_feasible_issuer_days'] == 0
