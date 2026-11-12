from dataclasses import replace

import pytest

from backtests.engine.integrity import audit_broker
from backtests.engine.replay import drive_timestamp_batch
from backtests.strategies.portfolio_synergy.native_budget import NativeBudgetSharedBroker
from backtests.strategies.portfolio_synergy.source_replay import NativeRiskSharedBroker
from deployment.olr_kalcb.native_budget import NativeBudgetConfig
from deployment.olr_kalcb.portfolio import PortfolioPolicyConfig
from strategy_common.actions import SubmitEntry, SubmitExit, CancelOrders
from tests.backtests.test_strategy_integrity_repairs import ZERO, bar, ts


def account(config=None, native=True, **changes):
    policy = PortfolioPolicyConfig(max_gross_notional=2000, max_symbol_notional=1000,
        max_sector_notional=2000, funding_mode='native_buying_power', same_symbol_policy='aggregate')
    cls = NativeBudgetSharedBroker if native else NativeRiskSharedBroker
    extra = {'native_budget': config or NativeBudgetConfig()} if native else {}
    return cls(1000, strategy_costs={'KALCB': ZERO, 'OLR': ZERO},
        strategy_leverage={'KALCB': 2, 'OLR': 1}, portfolio_policy=replace(policy, **changes), **extra)


def submit(b, sid, sym, qty, stamp=None, **meta):
    return b.submit(SubmitEntry(sid, sym, qty, 'MARKET', None, None, 'entry',
        metadata={'estimated_entry_price': 100, 'risk_per_share': 2, **meta}), stamp or ts())


def test_nonbinding_native_fill_path_and_costs_are_identical():
    accounts = [account(native=n) for n in (False, True)]
    for b in accounts:
        submit(b, 'KALCB', '000001', 4); submit(b, 'OLR', '000002', 4)
        drive_timestamp_batch(b, [bar('000001'), bar('000002')], [])
        assert audit_broker(b)['execution_validity_pass']
    assert accounts[0].fills == accounts[1].fills
    assert accounts[0].equity_curve == accounts[1].equity_curve
    assert accounts[0].cash == accounts[1].cash


def test_native_order_quantities_are_preserved_while_reservations_are_feasible():
    b = account(max_gross_notional=1000)
    submit(b, 'KALCB', '000001', 8); submit(b, 'OLR', '000002', 8)
    assert [o.qty for o in b.orders] == [8, 8]
    assert b.strategy_account_view('OLR') == (1000, 200)
    drive_timestamp_batch(b, [bar('000001'), bar('000002')], [])
    assert [(f.strategy_id, f.qty) for f in b.fills] == [('KALCB', 8), ('OLR', 2)]
    assert audit_broker(b)['execution_validity_pass']


def test_olr_view_does_not_subtract_its_own_pending_reservation_twice():
    b = account(NativeBudgetConfig(olr_fraction=.5))
    submit(b, 'OLR', '000001', 4)
    assert b.strategy_account_view('OLR') == (500, 500)
    drive_timestamp_batch(b, [bar('000001')], [])
    assert b.strategy_account_view('OLR') == (500, 100)
    assert b.strategy_sizing_equity() == 1000


def test_sleeve_budget_is_enforced_at_execution_price():
    b = account(NativeBudgetConfig(olr_fraction=.5))
    submit(b, 'OLR', '000001', 10)
    drive_timestamp_batch(b, [bar('000001', o=125, h=130, l=120, c=125)], [])
    assert b.fills[0].qty == 4
    assert audit_broker(b)['execution_validity_pass']


def test_unfilled_exit_cannot_supply_olr_funding():
    b = account(NativeBudgetConfig(olr_fraction=.5))
    submit(b, 'OLR', '000001', 5)
    drive_timestamp_batch(b, [bar('000001')], [])
    b.submit(SubmitExit('OLR', '000001', 5, 'LIMIT', 200, 'exit'), ts(5))
    assert b.strategy_account_view('OLR') == (500, 0)
    submit(b, 'OLR', '000002', 5, ts(5))
    drive_timestamp_batch(b, [bar('000001', ts(10)), bar('000002', ts(10))], [])
    assert len(b.fills) == 1
    assert audit_broker(b)['execution_validity_pass']


def test_partial_fill_and_acknowledged_cancel_release_only_remaining_reservation():
    from backtests.engine.capacity import BarParticipationScenario
    b = account(max_gross_notional=1000)
    b.capacity_policy = BarParticipationScenario(.01, 'test')
    drive_timestamp_batch(b, [bar('000001', ts(0), volume=300)], [])
    submit(b, 'KALCB', '000001', 8)
    drive_timestamp_batch(b, [bar('000001', ts(5), volume=300)], [])
    assert b.fills[0].qty == 3 and b.orders[0].qty == 5
    assert b.strategy_account_view('OLR') == (1000, 200)
    b.submit(CancelOrders('KALCB', '000001'), ts(5))
    assert b.strategy_account_view('OLR') == (1000, 700)
    assert audit_broker(b)['execution_validity_pass']


def test_aggregate_planned_risk_includes_pending_and_held_kalcb():
    b = account(NativeBudgetConfig(max_kalcb_stop_risk_fraction=.02))
    submit(b, 'KALCB', '000001', 8); submit(b, 'KALCB', '000002', 8)
    drive_timestamp_batch(b, [bar('000001'), bar('000002')], [])
    assert [f.qty for f in b.fills] == [8, 2]
    assert audit_broker(b)['execution_validity_pass']


@pytest.mark.parametrize('corruption', ['cash', 'reserved_qty', 'approval', 'unit'])
def test_integrity_audit_rejects_corrupt_native_funding_evidence(corruption):
    b = account(); submit(b, 'OLR', '000001', 4)
    drive_timestamp_batch(b, [bar('000001')], [])
    row = next(x for x in b.journal if x['event'] == 'native_budget_admission')
    if corruption == 'cash': row['ledger']['equity'] += 1
    elif corruption == 'reserved_qty': row['reservation']['qty'] += 1
    elif corruption == 'approval': row['approved_qty'] += 1
    else: row['unit_cost'] += 1
    assert not audit_broker(b)['execution_validity_pass']


def test_budget_limits_do_not_grow_under_no_upward_compounding():
    b = account(NativeBudgetConfig(olr_fraction=.5))
    b.sizing_equity_cap = 1000
    submit(b, 'KALCB', '000001', 5)
    drive_timestamp_batch(b, [bar('000001')], [])
    drive_timestamp_batch(b, [bar('000001', ts(10), o=200, h=200, l=200, c=200)], [])
    assert b.strategy_sizing_equity() == 1000
    submit(b, 'OLR', '000002', 10, ts(10))
    drive_timestamp_batch(b, [bar('000001', ts(15), o=200, h=200, l=200, c=200), bar('000002', ts(15))], [])
    assert b.fills[-1].qty == 5
    assert audit_broker(b)['execution_validity_pass']


def test_fractional_costs_and_concurrent_marks_reconcile_without_rounding_shares():
    from backtests.engine.sim_broker import BrokerCosts
    policy = PortfolioPolicyConfig(max_gross_notional=50_000_000, max_symbol_notional=25_000_000,
        max_sector_notional=50_000_000, funding_mode='native_buying_power', same_symbol_policy='aggregate')
    b=NativeBudgetSharedBroker(25_000_000, strategy_costs={'KALCB':BrokerCosts(2,20,3),'OLR':BrokerCosts(2,20,5)},
        strategy_leverage={'KALCB':2,'OLR':1},portfolio_policy=policy,native_budget=NativeBudgetConfig())
    symbols=[f'{i:06}' for i in range(1,9)]
    prices={s:23731.23+i*143.17 for i,s in enumerate(symbols)}
    for i,s in enumerate(symbols):
        submit(b,'KALCB' if i<4 else 'OLR',s,17+i,estimated_entry_price=prices[s])
    def bars(at,offset):
        return [bar(s,at,o=p+offset,h=p+offset+10,l=p+offset-10,c=p+offset+.07) for s,p in prices.items()]
    drive_timestamp_batch(b,bars(ts(5),0),[])
    submit(b,'KALCB','000009',23,ts(5),estimated_entry_price=32891.37)
    drive_timestamp_batch(b,[*bars(ts(10),117.03),bar('000009',ts(10),o=32891.37,h=32895,l=32890,c=32892)],[])
    audit=audit_broker(b)
    assert audit['execution_validity_pass'] and audit['native_budget_audit']['checked_admissions']==9


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -.1, 1.1, True])
def test_invalid_olr_fraction_fails_closed(value):
    with pytest.raises(ValueError): NativeBudgetConfig(olr_fraction=value)
