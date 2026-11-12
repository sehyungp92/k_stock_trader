from datetime import datetime, timedelta
from importlib import import_module

import pytest

from strategy_common.clock import KST
from strategy_common.actions import SubmitPartialExit, ReplaceProtectiveStop
from strategy_common.oms_adapter import action_to_intent


def test_action_identity_distinguishes_fragments_and_preserves_retries():
    def target(ref):
        return SubmitPartialExit("OLR", "005930", 1, "LIMIT", 110, "target", {"provisional_order_ref": ref})
    first = action_to_intent(target("fill-1:action:0"))
    assert first.idempotency_key != action_to_intent(target("fill-2:action:0")).idempotency_key
    assert first.idempotency_key == action_to_intent(target("fill-1:action:0")).idempotency_key
    stop = ReplaceProtectiveStop("OLR", "005930", 95, qty=4, metadata={"action_ref": "same-stop"})
    assert action_to_intent(stop).idempotency_key == action_to_intent(stop).idempotency_key


@pytest.mark.parametrize("strategy", ["kalcb", "olr"])
def test_flat_cancel_restart_late_fragment_keeps_original_lifecycle(strategy):
    prefix = strategy.upper()
    state_module = import_module(f"strategy_{strategy}.core.state")
    core = import_module(f"strategy_{strategy}.core.logic")
    models = import_module(f"strategy_{strategy}.core.core_models")
    serializer = import_module(f"strategy_{strategy}.core.serializers")
    config = getattr(import_module(f"strategy_{strategy}.config"), f"{prefix}Config")()
    state = getattr(state_module, f"{prefix}State")()
    Fill = getattr(models, f"{prefix}FillEvent")
    Update = getattr(models, f"{prefix}OrderUpdateEvent")
    on_fill = getattr(core, f"on_{strategy}_fill")
    on_update = getattr(core, f"on_{strategy}_order_update")
    start = datetime(2026, 1, 5, 9, tzinfo=KST)
    context = {"execution_id": "first", "account_id": "a", "order_date": "2026-01-05", "order_qty": 10,
        "filled_qty_before": 0, "filled_qty_after": 4, "candidate_rank": 7,
        "protective_stop_price": 95, "stop_price": 95, "risk_per_share": 5,
        "trade_exit_plan": {"name": "probe", "mode": "managed", "hard_stop_enabled": True}}
    ss = state.symbol_state("005930")
    ss.pending_entry_order_id = "entry"
    ss.pending_entry_metadata = dict(context)
    state.order_roles["entry"] = {**context, "role": "entry"}
    on_fill(state, Fill("entry", "005930", "BUY", 4, 100, start, metadata=context), config)
    on_fill(state, Fill("exit", "005930", "SELL", 4, 90, start + timedelta(minutes=1),
                       metadata={"execution_id": "exit"}), config)
    update = Update("entry", "005930", "CANCELLED", start + timedelta(minutes=1))
    on_update(state, update, config) if strategy == "olr" else on_update(state, update)
    state = serializer.restore_state(serializer.snapshot_state(state))
    pending = state.symbol_state("005930")
    pending.pending_entry_order_id = "different-pending-entry"
    pending.pending_entry_metadata = {"candidate_rank": 99, "stop_price": 80}
    late = Fill("entry", "005930", "BUY", 6, 110, start + timedelta(minutes=2), metadata={
        "execution_id": "late", "account_id": "a", "order_date": "2026-01-05", "order_qty": 10,
        "filled_qty_before": 4, "filled_qty_after": 10})
    result = on_fill(state, late, config)
    pos = state.symbol_state("005930").position
    assert pos.qty_open == 6 and pos.entry_price == 110 and pos.entry_time == start
    assert getattr(pos, "qty_entry", pos.metadata.get("acquired_qty")) == 10
    assert getattr(pos, "candidate_rank", pos.metadata.get("candidate_rank")) == 7
    assert any("ProtectiveStop" in type(action).__name__ for action in result.actions)
    assert not on_fill(state, late, config).actions
    assert pos.qty_open == 6
    assert pending.pending_entry_order_id == "different-pending-entry"
    assert pending.pending_entry_metadata["candidate_rank"] == 99


@pytest.mark.parametrize("strategy", ["kalcb", "olr"])
def test_unrelated_late_entry_is_rejected_without_consuming_identity(strategy):
    # The core's explicit reconciliation boundary prevents silent trade merging.
    prefix = strategy.upper()
    state = getattr(import_module(f"strategy_{strategy}.core.state"), f"{prefix}State")()
    Fill = getattr(import_module(f"strategy_{strategy}.core.core_models"), f"{prefix}FillEvent")
    handler = getattr(import_module(f"strategy_{strategy}.core.logic"), f"on_{strategy}_fill")
    cfg = getattr(import_module(f"strategy_{strategy}.config"), f"{prefix}Config")()
    ts = datetime(2026, 1, 5, 9, tzinfo=KST)
    handler(state, Fill("new-entry", "005930", "BUY", 4, 100, ts, metadata={"execution_id": "new"}), cfg)
    old = Fill("old-entry", "005930", "BUY", 6, 110, ts, metadata={"execution_id": "old"})
    before = dict(state.meta["core_executions"])
    with pytest.raises(ValueError, match="Unrelated"):
        handler(state, old, cfg)
    assert state.meta["core_executions"] == before
