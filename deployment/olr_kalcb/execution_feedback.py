"""Catch up the durable OMS stream before consuming new market observations."""
from datetime import datetime
from types import SimpleNamespace
import importlib
import json
import os
from pathlib import Path


def checkpoint_path(driver):
    return Path(driver.recorder.paths.root) / f"{driver.descriptor.strategy_id.lower()}_feedback_checkpoint.json"


def save_checkpoint(driver):
    serializer = importlib.import_module(f"strategy_{driver.descriptor.strategy_id.lower()}.core.serializers")
    payload = {"state": serializer.snapshot_state(driver.descriptor.engine.state),
        "checkpoint_version": "canonical-execution-v2",
        "order_identity": driver.order_identity, "order_metadata": driver.order_metadata,
        "pending_event": driver.pending_event_payload,
        "config": repr(driver.descriptor.engine.config)}
    path = checkpoint_path(driver)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def restore_checkpoint(driver):
    path = checkpoint_path(driver)
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("checkpoint_version") != "canonical-execution-v2":
        raise ValueError("Legacy feedback checkpoint requires explicit canonical-stream reconciliation")
    if payload["config"] != repr(driver.descriptor.engine.config):
        raise ValueError("Feedback checkpoint configuration mismatch")
    serializer = importlib.import_module(f"strategy_{driver.descriptor.strategy_id.lower()}.core.serializers")
    driver.descriptor.engine.state = serializer.restore_state(payload["state"])
    driver.order_identity = dict(payload.get("order_identity", {}))
    driver.order_metadata = dict(payload.get("order_metadata", {}))
    driver.pending_event_payload = payload.get("pending_event")


async def drain_execution_feedback(plan):
    if plan.mode not in {"paper", "live"}:
        return 0
    if plan.action_router is None:
        raise RuntimeError("Execution feedback requires an action router")
    client = plan.action_router.oms_client
    reader = getattr(client, "get_execution_feedback", None)
    if not callable(reader):
        raise RuntimeError("Paper/live execution requires durable OMS execution feedback")
    count = 0
    for strategy_id, driver in sorted(plan.drivers.items()):
        state = driver.descriptor.engine.state
        if state.meta.get("feedback_recovery_required"):
            raise RuntimeError("Interrupted feedback requires restart from the last durable checkpoint")
        cursor = int(state.meta.get("oms_feedback_cursor", 0))
        snapshot_reader = getattr(client, "get_execution_snapshot", None)
        if callable(snapshot_reader):
            _validate_stream(state, cursor, await snapshot_reader(strategy_id, cursor))
        await driver.resume_pending_event()
        packet = None
        while True:
            if callable(snapshot_reader):
                packet = await snapshot_reader(strategy_id, cursor)
                _validate_stream(state, cursor, packet)
                events = packet["events"]
            else:
                events = await reader(strategy_id, cursor)
            if not events:
                break
            for row in events:
                sequence = int(row["sequence"])
                if sequence <= cursor:
                    raise ValueError("OMS feedback is not ordered after the saved cursor")
                values = {key: value for key, value in row.items() if key not in {"kind", "sequence"}}
                values["timestamp"] = datetime.fromisoformat(values["timestamp"])
                values.setdefault("role", "")
                event = SimpleNamespace(**values)
                # The driver persists this cursor with core state and pending
                # actions before any new external submission.
                state.meta["oms_feedback_cursor"] = sequence
                try:
                    if row["kind"] == "fill":
                        await driver.handle_fill(event)
                    elif row["kind"] == "order":
                        await driver.handle_order_event(event)
                    else:
                        raise ValueError("Unknown OMS feedback event")
                    save_checkpoint(driver)
                except Exception:
                    state.meta["oms_feedback_cursor"] = cursor
                    state.meta["feedback_recovery_required"] = True
                    raise
                cursor = sequence
                count += 1
            if (packet is not None and not packet["has_more"]) or (packet is None and len(events) < 500):
                break
        await driver.portfolio_context.refresh()
        if not driver.portfolio_context.last_refresh_ok:
            raise RuntimeError("Cannot reconcile OMS execution feedback to fresh allocations")
        expected = packet["allocations"] if packet is not None else driver.portfolio_context.portfolio_view(strategy_id).positions
        actual = {symbol: item.position.qty_open for symbol, item in state.symbols.items() if item.position is not None}
        if {str(k).zfill(6): int(v) for k, v in expected.items() if v} != actual:
            raise RuntimeError(f"{strategy_id} core quantities disagree with OMS after feedback catch-up")
        if packet is not None and not checkpoint_path(driver).is_file():
            # A verified empty canonical stream also needs a durable starting
            # cursor. Otherwise cutover would require a trade before readiness.
            state.meta["oms_feedback_cursor"] = cursor
            save_checkpoint(driver)
    return count


def _validate_stream(state, cursor, packet):
    if cursor > int(packet["watermark"]):
        raise ValueError("Saved feedback cursor exceeds the canonical stream; reconcile before resuming")
    stream = packet.get("stream_id")
    previous_stream = state.meta.get("oms_feedback_stream")
    if not stream or str(stream).startswith("postgres-execution-v1:") or (cursor and not previous_stream):
        raise ValueError("Canonical feedback requires a generation-bound stream; reconcile before resuming")
    if previous_stream and previous_stream != stream:
        raise ValueError("OMS feedback stream changed; reconcile before resuming")
    if stream:
        state.meta["oms_feedback_stream"] = stream
