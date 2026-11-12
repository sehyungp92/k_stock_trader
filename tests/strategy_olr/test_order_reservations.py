from copy import deepcopy
import json
from datetime import date, datetime, time

import pytest

from backtests.engine.capacity import BarParticipationScenario
from backtests.engine.integrity import audit_broker
from backtests.engine.replay import drive_timestamp_batch
from backtests.engine.sim_broker import BrokerCosts, SimBroker
from backtests.strategies.olr.runner import OLRReplayAdapter
from strategy_common.actions import CancelOrders, SubmitEntry, SubmitExit
from strategy_common.clock import KST
from strategy_olr.config import OLRConfig
from strategy_olr.core.logic import on_olr_fill, pending_entry_reservation
from strategy_olr.core.core_models import OLRFillEvent
from strategy_olr.core.serializers import snapshot_state, restore_state
from strategy_olr.models import OLRDailySnapshot
from tests.strategy_olr.test_core_runner import _bar


DAY = date(2026, 1, 5)
SYMBOL = "005930"


def setup(cash=10_000):
    cfg = OLRConfig(commission_bps=0, slippage_bps=0, tax_bps_on_sell=0)
    adapter = OLRReplayAdapter(cfg, {DAY: OLRDailySnapshot(DAY, (), "test", datetime.combine(DAY, time(8), tzinfo=KST))})
    broker = SimBroker(initial_equity=cash, costs=BrokerCosts(0, 0, 0),
                       capacity_policy=BarParticipationScenario(.01, "test"))
    return adapter, broker


def advance(adapter, broker, minute, price=100):
    bar = _bar(SYMBOL, DAY, time(14, minute), price, price, price, price)
    return drive_timestamp_batch(broker, (bar,), (adapter,))


def enter(adapter, broker, qty=20):
    advance(adapter, broker, 30)
    adapter._submit_actions([SubmitEntry("OLR", SYMBOL, qty, "MARKET", None, None, "test_entry",
        metadata={"estimated_entry_price": 100, "reserved_notional": qty * 100})],
        broker, datetime.combine(DAY, time(14, 30), tzinfo=KST))
    adapter.state.symbol_state(SYMBOL).entry_attempted = True


def test_rejected_entry_releases_reservation_once_without_retry():
    adapter, broker = setup(cash=900)
    enter(adapter, broker)
    decisions = advance(adapter, broker, 35)
    state = adapter.state.symbol_state(SYMBOL)
    assert len(broker.rejected_orders) == 1 and not broker.fills
    assert not state.pending_entry_order_id
    assert pending_entry_reservation(state) == (0, 0)
    assert state.entry_attempted
    updates = [d for d in decisions if d.decision_code == "ORDER_UPDATE"]
    assert len(updates) == 1 and updates[0].timestamp.minute == 35
    assert adapter._sync_order_updates(broker, updates[0].timestamp) == []
    assert audit_broker(broker)["audit_receipt"]["coverage"]["strategy_state"] == "pass"


def test_partial_reservation_survives_restart_duplicate_and_partial_sale():
    adapter, broker = setup()
    enter(adapter, broker, qty=30)
    advance(adapter, broker, 35)
    state = adapter.state.symbol_state(SYMBOL)
    assert pending_entry_reservation(state) == (20, 2000)
    adapter.state = restore_state(json.loads(json.dumps(snapshot_state(adapter.state))))
    state = adapter.state.symbol_state(SYMBOL)
    fill = broker.fills[0]
    on_olr_fill(adapter.state, OLRFillEvent(fill.order_id, fill.symbol, fill.side,
        fill.qty, fill.price, fill.timestamp, metadata=fill.metadata), adapter.config)
    assert pending_entry_reservation(state) == (20, 2000)
    # Inventory reduction is not an entry cancellation acknowledgement.
    state.position.qty_open = 3
    assert pending_entry_reservation(state) == (20, 2000)
    state.position.qty_open = 10
    advance(adapter, broker, 40)
    assert pending_entry_reservation(state) == (10, 1000)
    advance(adapter, broker, 45)
    assert pending_entry_reservation(state) == (0, 0)
    assert state.position.qty_open == 30
    assert audit_broker(broker)["audit_receipt"]["coverage"]["strategy_state"] == "pass"


@pytest.mark.parametrize("terminal", ["reject", "cancel", "expire"])
def test_partial_terminal_releases_only_remainder_keeps_position(terminal):
    adapter, broker = setup(cash=1000 if terminal == "reject" else 10_000)
    enter(adapter, broker)
    advance(adapter, broker, 35)
    state = adapter.state.symbol_state(SYMBOL)
    assert state.position.qty_open == 10
    assert pending_entry_reservation(state) == (10, 1000)
    if terminal == "cancel":
        adapter._submit_actions([CancelOrders("OLR", SYMBOL, "test_cancel")], broker,
            datetime.combine(DAY, time(14, 35), tzinfo=KST))
    elif terminal == "expire":
        # Set a real declared expiry before submission in the broker evidence.
        # Use a fresh run so the immutable submit record also contains it.
        adapter, broker = setup()
        advance(adapter, broker, 30)
        adapter._submit_actions([SubmitEntry("OLR", SYMBOL, 20, "MARKET", None, None, "test_entry", metadata={
            "estimated_entry_price": 100, "reserved_notional": 2000,
            "expiry_ts": datetime.combine(DAY, time(14, 36), tzinfo=KST).isoformat()})],
            broker, datetime.combine(DAY, time(14, 30), tzinfo=KST))
        advance(adapter, broker, 35)
        state = adapter.state.symbol_state(SYMBOL)
    advance(adapter, broker, 40)
    assert state.position.qty_open == 10
    assert pending_entry_reservation(state) == (0, 0)
    assert audit_broker(broker)["audit_receipt"]["coverage"]["strategy_state"] == "pass"


@pytest.mark.parametrize("corruption,expected", [
    ("reserve", "strategy_state_entry_reservation_mismatch"),
    ("ghost", "strategy_state_entry_order_mismatch"),
    ("position", "strategy_state_position_mismatch"),
    ("omit", "strategy_state_missing_checkpoint"),
])
def test_independent_audit_rejects_state_disagreement(corruption, expected):
    adapter, broker = setup()
    enter(adapter, broker)
    advance(adapter, broker, 35)
    checkpoint = broker.journal[-1]
    assert checkpoint["event"] == "strategy_state"
    key = next(iter(checkpoint["pending_entries"]))
    if corruption == "reserve":
        checkpoint["pending_entries"][key]["reserved_notional"] = 0
    elif corruption == "ghost":
        checkpoint["pending_entries"]["ghost"] = deepcopy(checkpoint["pending_entries"][key])
    elif corruption == "position":
        checkpoint["positions"][SYMBOL] = 11
    else:
        broker.journal.pop()
    audit = audit_broker(broker)
    assert expected in audit["audit_receipt"]["errors"]
    assert audit["execution_validity_pass"] is False


def test_entry_conversion_counts_lifecycles_not_fill_fragments():
    from backtests.strategies.olr.plugin import _augment_trade_alpha_metrics
    metrics = {"selected_candidate_count": 10, "entry_fill_count": 12,
               "entry_lifecycle_count": 4, "entry_level_trade_count": 3}
    _augment_trade_alpha_metrics(metrics, [])
    assert metrics["entry_conversion_rate"] == .4


def test_position_close_acknowledges_cancelled_entry_remainder():
    adapter, broker = setup()
    enter(adapter, broker)
    advance(adapter, broker, 35)
    adapter._submit_actions([SubmitExit("OLR", SYMBOL, 10, "MARKET", None, reason="test_exit")],
        broker, datetime.combine(DAY, time(14, 35), tzinfo=KST))
    advance(adapter, broker, 40)
    state = adapter.state.symbol_state(SYMBOL)
    assert state.position is None and not broker.orders
    assert pending_entry_reservation(state) == (0, 0)
    assert not state.pending_exit_order_id
    assert audit_broker(broker)["execution_validity_pass"] is True


def test_state_audit_uses_declared_adapter_scope_in_shared_replay():
    adapter, broker = setup()
    advance(adapter, broker, 30)
    other = _bar("000660", DAY, time(14, 35), 100, 100, 100, 100)
    drive_timestamp_batch(broker, (other,), (adapter,), accepts=lambda owner, bar: bar.symbol == SYMBOL)
    assert "strategy_state_missing_checkpoint" not in audit_broker(broker)["audit_receipt"]["errors"]
