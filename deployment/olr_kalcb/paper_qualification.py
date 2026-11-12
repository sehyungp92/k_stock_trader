"""Join existing parity, account reconciliation and observed paper lifecycle evidence."""
from collections import defaultdict
from datetime import date, datetime
from hashlib import sha256
import json
import math
from pathlib import Path


def _rows(root, name):
    path = root / name
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.is_file() else []


def qualify_paper_session(session, *, parity, snapshot_path, cutover_path):
    root = Path(session)
    manifest = json.loads((root / "session_manifest.json").read_text(encoding="utf-8"))
    snapshot_path, cutover_path = Path(snapshot_path), Path(cutover_path)
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    cutover = json.loads(cutover_path.read_text(encoding="utf-8"))
    failures = []
    if parity.get("behavior_parity_passed") is not True:
        failures.append("captured_behavior_parity_failed")
    if manifest.get("mode") != "paper":
        failures.append("actual_paper_session_required")
    if manifest.get("startup_preflight", {}).get("passed") is not True:
        failures.append("current_calendar_artifacts_and_runtime_preflight_required")
    health = manifest.get("startup_oms_health") or {}
    fingerprint = snapshot.get("account_fingerprint")
    if (snapshot.get("source") != "KISExecutionAdapter" or snapshot.get("mode") != "paper"
            or snapshot.get("queries_complete") is not True or not fingerprint
            or health.get("broker_mode") != "paper" or health.get("broker_account_fingerprint") != fingerprint
            or health.get("broker_reconciliation_ready") is not True):
        failures.append("captured_paper_account_identity_mismatch_or_unverified")
    if (cutover.get("passed") is not True or cutover.get("oms_id") != snapshot.get("oms_id")
            or cutover.get("broker_snapshot_sha256") != sha256(snapshot_path.read_bytes()).hexdigest()
            or cutover.get("account_fingerprint") != fingerprint):
        failures.append("matching_account_cutover_required")

    day = date.fromisoformat(manifest["trade_date"])
    broker = {(str(row.get("order_date")), str(row.get("order_id"))): row
              for row in snapshot.get("execution_orders", [])}
    submitted = {str(row.get("order_id")) for row in _rows(root, "order_events.jsonl")
                 if row.get("actually_submitted_to_broker") is True and row.get("order_id")}
    aggregate = defaultdict(lambda: [0, 0.0])
    sides = set()
    strategies = set()
    for row in _rows(root, "fill_events.jsonl"):
        event = row.get("event") or {}
        meta = event.get("metadata") or {}
        oid = str(meta.get("broker_order_id") or row.get("broker_order_id") or "")
        raw_day = str(meta.get("order_date") or day.isoformat()).replace("-", "")
        observed = broker.get((raw_day, oid))
        qty, price = int(event.get("qty", 0)), float(event.get("price", 0))
        if (oid not in submitted or observed is None or qty <= 0 or not math.isfinite(price) or price <= 0
                or not observed.get("execution_price_observed")
                or event.get("side") != observed.get("side") or event.get("symbol") != observed.get("symbol")):
            failures.append("captured_fill_not_matched_to_broker")
            continue
        key = (raw_day, oid)
        aggregate[key][0] += qty
        aggregate[key][1] += qty * price
        sides.add(event["side"])
        strategies.add(row.get("strategy_id"))
    for key, (qty, notional) in aggregate.items():
        observed = broker[key]
        if qty != int(observed["filled_qty"]) or abs(notional - qty * float(observed["price"])) > 1:
            failures.append("captured_fill_quantity_or_notional_mismatch")
    if sides != {"BUY", "SELL"}:
        failures.append("observed_entry_and_exit_required")
    enabled = set(manifest.get("strategy_ids") or [])
    if not enabled or not enabled.issubset(strategies):
        failures.append("enabled_strategy_fill_evidence_required")
    cancelled = any(row.get("status") == "CANCELLED" and str(row.get("order_id")) in submitted
                    and str(row.get("order_date")) == day.strftime("%Y%m%d") for row in broker.values())
    if not cancelled:
        failures.append("observed_cancel_required")
    try:
        if (datetime.fromisoformat(snapshot["as_of"]) < datetime.fromisoformat(manifest["generated_at"])
                or not date.fromisoformat(snapshot["history_start"]) <= day <= date.fromisoformat(snapshot["history_end"])):
            failures.append("post_session_broker_snapshot_required")
    except (KeyError, TypeError, ValueError):
        failures.append("post_session_broker_snapshot_required")
    return {"paper_readiness_passed": not failures, "failures": sorted(set(failures)),
            "scope": "configured_paper_session_parity_account_cutover_and_observed_entry_exit_cancel",
            "observed_fill_orders": len(aggregate), "enabled_strategies": sorted(enabled),
            "recovery_evidence": "separate deterministic durable restart tests; no broker outage certification",
            "live_activation": False, "real_money_fill_calibration": False}
