"""Read-only cutover checks; never infer or write a starting allocation."""
import json
import math
from datetime import datetime, timezone


async def execution_preflight(store, broker_positions, *, checkpoints=(), broker_snapshot=None, risk_contract=None, now=None):
    failures = []
    if not await store.execution_schema_ready():
        return {"passed": False, "failures": ["execution_schema_missing"]}
    async with store.pool.acquire() as db:
        async with db.transaction(isolation="repeatable_read", readonly=True):
            generation = await db.fetchval("SELECT generation FROM execution_generations WHERE oms_id=$1", store.oms_id)
            allocations = await db.fetch("SELECT symbol,strategy_id,qty,cost_basis FROM allocations WHERE oms_id=$1", store.oms_id)
            stops = await db.fetch("SELECT symbol,strategy_id,qty,status FROM protective_stops WHERE oms_id=$1", store.oms_id)
            orders = await db.fetch("SELECT symbol,strategy_id,qty,filled_qty,status FROM orders WHERE oms_id=$1", store.oms_id)
            trades = await db.fetch("SELECT symbol,strategy_id,entry_qty,exit_qty,status FROM trades WHERE oms_id=$1 AND status='OPEN'", store.oms_id)
            streams = await db.fetch("SELECT strategy_id,MAX(sequence) AS watermark FROM execution_feedback WHERE oms_id=$1 GROUP BY strategy_id", store.oms_id)
            legacy = await db.fetchval("""SELECT COUNT(*) FROM fills f WHERE f.oms_id=$1 AND NOT EXISTS
                (SELECT 1 FROM execution_commits c WHERE c.oms_id=f.oms_id AND c.identity=f.kis_exec_id)""", store.oms_id)
    if generation is None:
        failures.append("execution_generation_missing")
    if legacy:
        failures.append("legacy_fills_require_reconciliation")
    aggregate = {}
    for row in allocations:
        aggregate[row["symbol"]] = aggregate.get(row["symbol"], 0) + row["qty"]
        if row["qty"] < 0 or (row["qty"] and float(row["cost_basis"] or 0) <= 0):
            failures.append("invalid_allocation_cost_or_quantity")
    actual = {str(symbol).zfill(6): int(qty) for symbol, qty in broker_positions.items() if qty}
    if {s: q for s, q in aggregate.items() if q} != actual:
        failures.append("broker_allocation_quantity_mismatch")
    quantity = {(r["strategy_id"], r["symbol"]): r["qty"] for r in allocations}
    trade_quantity = {}
    for row in trades:
        key = (row["strategy_id"], row["symbol"])
        if key in trade_quantity:
            failures.append("ambiguous_open_trade_identity")
        trade_quantity[key] = trade_quantity.get(key, 0) + row["entry_qty"] - (row["exit_qty"] or 0)
    if {k: v for k, v in trade_quantity.items() if v} != {k: v for k, v in quantity.items() if v}:
        failures.append("open_trade_allocation_quantity_mismatch")
    for row in stops:
        if row["status"] in {"PENDING", "ACTIVE", "TRIGGERED", "TRIGGERED_PENDING_EXECUTION", "EXIT_SUBMITTED"} and row["qty"] != quantity.get((row["strategy_id"], row["symbol"]), 0):
            failures.append("protective_stop_quantity_mismatch")
    if any(not 0 <= r["filled_qty"] <= r["qty"] for r in orders):
        failures.append("working_order_progress_mismatch")
    watermarks = {r["strategy_id"]: int(r["watermark"]) for r in streams}
    from deployment.olr_kalcb.execution_feedback import _validate_stream
    from types import SimpleNamespace
    for checkpoint in checkpoints:
        sid = checkpoint["strategy_id"]
        meta = checkpoint["meta"]
        try:
            _validate_stream(SimpleNamespace(meta=dict(meta)), int(meta.get("oms_feedback_cursor", 0)),
                {"watermark": watermarks.get(sid, 0), "stream_id": f"postgres-execution-v2:{store.oms_id}:{sid}:{generation}"})
        except ValueError:
            failures.append(f"checkpoint_reconciliation_required:{sid}")
    ledger_passed = not failures
    readiness = snapshot_requirements(broker_snapshot, risk_contract, oms_id=store.oms_id,
        allocations=allocations, stops=stops, watermarks=watermarks, checkpoints=checkpoints,
        broker_positions=actual, now=now)
    failures.extend(readiness)
    return {"passed": not failures, "failures": sorted(set(failures)),
        "ledger_passed": ledger_passed,
        "generation": str(generation) if generation else None, "watermarks": watermarks,
        "allocation_count": len(allocations), "orders_checked": len(orders), "stops_checked": len(stops),
        "scope": "fresh_account_positions_cost_basis_required_protection_and_checkpoint_streams",
        "unverified": ["cash_and_realized_net_reconciliation", "intended_adapter_paper_recovery", "current_calendar_coverage"],
        "deployment_qualified": False}


def snapshot_requirements(snapshot, risk, *, oms_id, allocations, stops, watermarks, checkpoints, broker_positions, now=None):
    """Validate the supplied account/risk boundary without broker or database writes."""
    if not isinstance(snapshot, dict) or not isinstance(risk, dict):
        return ["fresh_broker_snapshot_and_risk_contract_required"]
    failures = []
    try:
        if snapshot.get("oms_id") != oms_id or risk["oms_id"] != oms_id or not risk["account_alias"] or snapshot.get("account_alias") != risk["account_alias"]:
            failures.append("broker_snapshot_account_mismatch")
        if snapshot.get("source") == "KISExecutionAdapter":
            if (snapshot.get("queries_complete") is not True or snapshot.get("mode") != "paper"
                    or not risk.get("account_fingerprint")
                    or risk["account_fingerprint"] != snapshot.get("account_fingerprint")):
                failures.append("broker_snapshot_route_or_account_fingerprint_mismatch")
        stamp = datetime.fromisoformat(snapshot["as_of"])
        now = now or datetime.now(timezone.utc)
        max_age = float(risk.get("max_snapshot_age_seconds", 300))
        if stamp.tzinfo is None or now.tzinfo is None or not 0 < max_age <= 900 or not 0 <= (now - stamp).total_seconds() <= max_age:
            failures.append("broker_snapshot_stale_or_invalid_timestamp")
        if {str(k).zfill(6): int(v) for k, v in snapshot["positions"].items() if v} != broker_positions:
            failures.append("snapshot_position_argument_mismatch")
        basis = snapshot["cost_basis"]
        aggregate = {}
        for row in allocations:
            qty, cost = aggregate.get(row["symbol"], (0, 0))
            aggregate[row["symbol"]] = (qty + row["qty"], cost + row["qty"] * float(row["cost_basis"] or 0))
        tolerance = float(risk.get("cost_basis_tolerance_krw", .01))
        if not math.isfinite(tolerance) or not 0 <= tolerance <= 1:
            raise ValueError("Invalid cost tolerance")
        for symbol, (qty, cost) in aggregate.items():
            if qty:
                observed = float(basis[symbol])
                if not math.isfinite(observed) or abs(cost / qty - observed) > tolerance:
                    failures.append(f"broker_cost_basis_mismatch:{symbol}")
        required = set(risk["required_stop_strategies"])
        active = {row["strategy_id"] for row in allocations if row["qty"]}
        enabled = set(risk["enabled_strategies"])
        if not active.issubset(enabled) or not required.issubset(enabled):
            failures.append("risk_contract_strategy_scope_mismatch")
        for row in allocations:
            if row["qty"] and row["strategy_id"] in required:
                protection = [stop for stop in stops if stop["symbol"] == row["symbol"]
                    and stop["strategy_id"] == row["strategy_id"] and stop["status"] in {"PENDING", "ACTIVE"}]
                if len(protection) != 1 or protection[0]["qty"] != row["qty"]:
                    failures.append(f"required_protection_missing_or_ambiguous:{row['strategy_id']}:{row['symbol']}")
        checkpoint_map = {row["strategy_id"]: row for row in checkpoints}
        if len(checkpoint_map) != len(checkpoints):
            failures.append("duplicate_checkpoint")
        for strategy in enabled | set(watermarks):
            checkpoint = checkpoint_map.get(strategy)
            if checkpoint is None or int(checkpoint["meta"].get("oms_feedback_cursor", -1)) != watermarks.get(strategy, 0):
                failures.append(f"checkpoint_not_caught_up:{strategy}")
    except (ValueError, TypeError, KeyError, OverflowError):
        failures.append("incomplete_or_invalid_broker_snapshot_or_risk_contract")
    return failures
