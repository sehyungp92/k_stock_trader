import hashlib
import json

import pytest

from deployment.olr_kalcb.paper_qualification import qualify_paper_session


def evidence(tmp_path, mutate=None):
    manifest = {"mode": "paper", "trade_date": "2026-10-02", "generated_at": "2026-10-02T06:31:00+00:00",
                "strategy_ids": ["KALCB"], "startup_preflight": {"passed": True},
                "startup_oms_health": {"broker_mode": "paper", "broker_account_fingerprint": "fixture",
                                       "broker_reconciliation_ready": True}}
    snapshot = {"source": "KISExecutionAdapter", "mode": "paper", "oms_id": "unit-test", "queries_complete": True,
                "account_fingerprint": "fixture", "as_of": "2026-10-02T06:32:00+00:00",
                "history_start": "2026-10-02", "history_end": "2026-10-02", "execution_orders": [
                    {"order_id": oid, "order_date": "20261002", "symbol": "005930", "side": side,
                     "filled_qty": 1 if oid != "cancel" else 0, "price": 100,
                     "execution_price_observed": oid != "cancel", "status": status}
                    for oid, side, status in [("entry", "BUY", "FILLED"), ("exit", "SELL", "FILLED"), ("cancel", "BUY", "CANCELLED")]]}
    orders = [{"order_id": oid, "actually_submitted_to_broker": True} for oid in ("entry", "exit", "cancel")]
    fills = [{"strategy_id": "KALCB", "event": {"symbol": "005930", "side": side, "qty": 1, "price": 100,
              "metadata": {"broker_order_id": oid, "order_date": "2026-10-02"}}} for oid, side in [("entry", "BUY"), ("exit", "SELL")]]
    if mutate:
        mutate(manifest, snapshot, orders, fills)
    for name, payload in [("session_manifest.json", manifest), ("broker.json", snapshot)]:
        (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")
    for name, rows in [("order_events.jsonl", orders), ("fill_events.jsonl", fills)]:
        (tmp_path / name).write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    cutover = {"passed": True, "oms_id": "unit-test", "account_fingerprint": "fixture",
               "broker_snapshot_sha256": hashlib.sha256((tmp_path / "broker.json").read_bytes()).hexdigest()}
    (tmp_path / "cutover.json").write_text(json.dumps(cutover), encoding="utf-8")
    return qualify_paper_session(tmp_path, parity={"behavior_parity_passed": True},
                                snapshot_path=tmp_path / "broker.json", cutover_path=tmp_path / "cutover.json")


def test_bound_observed_session_receives_only_the_scoped_paper_verdict(tmp_path):
    report = evidence(tmp_path)
    assert report["paper_readiness_passed"]
    assert report["live_activation"] is False
    assert report["real_money_fill_calibration"] is False


@pytest.mark.parametrize("failure", ["empty", "dry_run", "account", "preflight", "price", "cancel", "scope", "before_session"])
def test_parity_alone_cannot_qualify_missing_or_conflicting_broker_evidence(tmp_path, failure):
    def mutate(manifest, snapshot, orders, fills):
        if failure == "empty":
            orders.clear(); fills.clear()
        elif failure == "dry_run":
            manifest["mode"] = "dry_run"
        elif failure == "account":
            snapshot["account_fingerprint"] = "another-account"
        elif failure == "preflight":
            manifest["startup_preflight"]["passed"] = False
        elif failure == "price":
            fills[0]["event"]["price"] = 999
        elif failure == "cancel":
            snapshot["execution_orders"].pop()
        elif failure == "scope":
            manifest["strategy_ids"].append("OLR")
        else:
            snapshot["as_of"] = "2026-10-02T06:30:00+00:00"
    assert not evidence(tmp_path, mutate)["paper_readiness_passed"]
