from dataclasses import replace
from datetime import datetime, timedelta

from strategy_common.actions import CancelOrders, SubmitExit, SubmitProtectiveStop
from strategy_common.clock import KST
from strategy_common.market import MarketBar
from strategy_kalcb.config import KALCBConfig
from strategy_kalcb.core.core_models import KALCBFillEvent, KALCBOrderUpdateEvent
from strategy_kalcb.core.logic import on_kalcb_fill, on_kalcb_timer, on_kalcb_order_update, _manage_position, remember_submitted_order
from strategy_kalcb.core.serializers import restore_state, snapshot_state
from strategy_kalcb.core.state import KALCBState, KALCBPositionState
from strategy_kalcb.liquidation import liquidation_evidence


def setup(qty=100):
    cfg = replace(KALCBConfig(), exit_hard_stop_enabled=False,
                  exit_liquidity_guard_enabled=True)
    state = KALCBState()
    ss = state.symbol_state("005930")
    ts = datetime(2025, 6, 2, 14, 15, tzinfo=KST)
    ss.position = KALCBPositionState("005930", qty, qty, 50000, ts.replace(hour=9, minute=30),
                                    49000, 49000, 1000, "KRX_FIRST30_OPEN", 5,
                                    entry_order_id="entry")
    for i in range(6):
        ss.add_bar(MarketBar("005930", ts + timedelta(minutes=i*5), "5m",
                             50100, 50200, 50000, 50100, 1000))
    return cfg, state, ss


def test_headroom_uses_only_completed_volume_and_actual_remaining_slices():
    cfg, state, ss = setup()
    evidence = liquidation_evidence(ss, ss.position, cfg)
    assert evidence["available_at"].endswith("14:45:00+09:00")
    assert evidence["remaining_open_slices"] == 7
    assert evidence["forecast_capacity"] == 28  # floor(1000*.005)*7*.8
    assert liquidation_evidence(ss, ss.position, replace(cfg, exit_liquidity_guard_enabled=False)) is None
    ss.position.qty_open = 28
    assert liquidation_evidence(ss, ss.position, cfg) is None


def test_invalid_prefix_is_not_used_as_liquidity_forecast():
    cfg, state, ss = setup()
    ss._market_bars[-1] = replace(ss._market_bars[-1], timestamp=ss._market_bars[-1].timestamp + timedelta(minutes=5))
    assert liquidation_evidence(ss, ss.position, cfg) is None


def test_zero_prints_request_exit_but_do_not_create_fills():
    cfg, state, ss = setup()
    ss._market_bars = [replace(b, volume=0) for b in ss.bars_today]
    evidence = liquidation_evidence(ss, ss.position, cfg)
    assert evidence["forecast_capacity"] == 0
    cfg = replace(cfg, flatten_time=datetime.strptime("15:15", "%H:%M").time())
    result = _manage_position(state, ss, ss.bars_today[-1], cfg)
    assert any(isinstance(a, SubmitExit) and a.reason == "liquidity_headroom" for a in result.actions)
    assert ss.position.qty_open == 100


def test_timer_cancels_queued_only_entry_and_waits_for_ack():
    cfg, state, ss = setup()
    ss.position = None
    ss.pending_entry_order_id = "entry"
    ss.pending_entry_metadata = {"expiry_timestamp": "2025-06-02T09:35:00+09:00"}
    ts = datetime(2025, 6, 2, 9, 40, tzinfo=KST)
    result = on_kalcb_timer(state, ts, cfg)
    assert len(result.actions) == 1 and isinstance(result.actions[0], CancelOrders)
    assert ss.pending_entry_order_id == "entry"
    assert on_kalcb_timer(state, ts, cfg).actions == []
    on_kalcb_order_update(state, KALCBOrderUpdateEvent("entry", "005930", "CANCELLED", ts))
    assert ss.pending_entry_order_id == ""


def test_late_entry_after_cancel_ack_and_restart_is_liquidated():
    cfg, state, ss = setup()
    ss.position = None
    ss.pending_entry_order_id = "entry"
    ss.pending_entry_metadata = {"expiry_timestamp": "2025-06-02T09:35:00+09:00"}
    state.order_roles["entry"] = {"order_role": "ENTRY", "risk_per_share": 1000}
    ts = datetime(2025, 6, 2, 9, 40, tzinfo=KST)
    on_kalcb_timer(state, ts, cfg)
    on_kalcb_order_update(state, KALCBOrderUpdateEvent("entry", "005930", "CANCELLED", ts))
    state = restore_state(snapshot_state(state))
    result = on_kalcb_fill(state, KALCBFillEvent("entry", "005930", "BUY", 4, 50000, ts,
        metadata={"order_qty": 4, "filled_qty_before": 0, "filled_qty_after": 4}), cfg)
    assert isinstance(result.actions[0], CancelOrders)
    assert isinstance(result.actions[1], SubmitExit) and result.actions[1].qty == 4


def test_partial_liquidation_does_not_submit_duplicate_exit():
    cfg, state, ss = setup()
    ss.position.exit_in_flight = True
    ss.position.metadata["liquidation_reason"] = "liquidity_headroom"
    ts = ss.bars_today[-1].timestamp
    on_kalcb_fill(state, KALCBFillEvent("exit", "005930", "SELL", 10, 50100, ts,
        metadata={"order_role": "EXIT", "order_qty": 100, "filled_qty_before": 0, "filled_qty_after": 10}), cfg)
    assert ss.position.qty_open == 90 and ss.position.exit_in_flight
    assert _manage_position(state, ss, ss.bars_today[-1], cfg).actions == []


def test_new_settings_survive_config_round_trip():
    cfg = KALCBConfig.from_mapping(mutations={"kalcb.entry.order_expiry_bars": 2,
        "kalcb.exit.liquidity_guard_enabled": True, "kalcb.exit.liquidity_guard_start": "13:30"})
    assert cfg.entry_order_expiry_bars == 2
    assert cfg.exit_liquidity_guard_start.hour == 13 and cfg.exit_liquidity_guard_start.minute == 30


def test_entry_expiry_preserves_protection_for_already_filled_quantity():
    cfg, state, ss = setup(qty=25)
    cfg = replace(cfg, exit_hard_stop_enabled=True)
    ss.pending_entry_order_id = "entry"
    ss.pending_entry_metadata = {"expiry_timestamp": "2025-06-02T09:35:00+09:00"}
    result = on_kalcb_timer(state, datetime(2025, 6, 2, 9, 40, tzinfo=KST), cfg)
    assert isinstance(result.actions[0], CancelOrders)
    assert isinstance(result.actions[1], SubmitProtectiveStop)
    assert result.actions[1].qty == 25 and result.actions[1].stop_price == 49000


def test_broker_expiry_ack_also_marks_late_entry_for_liquidation():
    cfg, state, ss = setup()
    ss.position = None
    ss.pending_entry_order_id = "entry"
    state.order_roles["entry"] = {"order_role": "ENTRY", "risk_per_share": 1000}
    ts = datetime(2025, 6, 2, 9, 40, tzinfo=KST)
    on_kalcb_order_update(state, KALCBOrderUpdateEvent("entry", "005930", "EXPIRED", ts))
    result = on_kalcb_fill(state, KALCBFillEvent("entry", "005930", "BUY", 4, 50000, ts,
        metadata={"order_qty": 4, "filled_qty_after": 4}), cfg)
    assert any(isinstance(a, SubmitExit) and a.reason == "entry_window_expired" for a in result.actions)


def test_replacement_exit_remains_in_flight_after_old_cancel_ack():
    cfg, state, ss = setup()
    action = SubmitExit("KALCB", ss.symbol, 100, "MARKET", None, "liquidity_headroom")
    remember_submitted_order(state, "old_exit", action)
    remember_submitted_order(state, "replacement_exit", action)
    on_kalcb_order_update(state, KALCBOrderUpdateEvent("old_exit", ss.symbol, "CANCELLED", ss.bars_today[-1].timestamp))
    assert ss.position.exit_in_flight
    on_kalcb_order_update(state, KALCBOrderUpdateEvent("replacement_exit", ss.symbol, "REJECTED", ss.bars_today[-1].timestamp))
    assert not ss.position.exit_in_flight
