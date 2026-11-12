from datetime import date
from types import SimpleNamespace

import pytest

from backtests.engine.corporate_action_guard import verify_zero_exposure

EVENTS = [{"symbol": "068270", "ex_date": "2026-06-04", "type": "bonus_issue"}]


def test_flat_account_needs_no_entitlement_adjustment():
    broker = SimpleNamespace(positions={}, orders=[])
    result = verify_zero_exposure(broker, date(2026, 6, 4), EVENTS)
    assert result[0]["held_shares"] == 0
    assert result[0]["account_adjustment_required"] is False
    assert verify_zero_exposure(broker, date(2026, 6, 5), EVENTS) == []


@pytest.mark.parametrize("exposure", ["holding", "pending_order"])
def test_unsupported_accounting_aborts_without_changing_orders_or_positions(exposure):
    broker = SimpleNamespace(positions={}, orders=[])
    if exposure == "holding": broker.positions[("OLR", "068270")] = SimpleNamespace(qty=23)
    else: broker.orders.append(SimpleNamespace(symbol="068270", order_id="pending"))
    before = (dict(broker.positions), list(broker.orders))
    with pytest.raises(ValueError, match="Unsupported corporate-action exposure"):
        verify_zero_exposure(broker, date(2026, 6, 4), EVENTS)
    assert (broker.positions, broker.orders) == before


def test_unrelated_holdings_are_unaffected():
    broker = SimpleNamespace(positions={("OLR", "005930"): SimpleNamespace(qty=23)}, orders=[])
    assert verify_zero_exposure(broker, date(2026, 6, 4), EVENTS)[0]["held_shares"] == 0
