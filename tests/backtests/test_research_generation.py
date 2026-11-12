from copy import deepcopy
from dataclasses import replace
import os

import pytest

from kis_core.input_manifest import InputManifest, read_input_bytes
from backtests.engine.capacity import BarParticipationScenario
from backtests.engine.integrity import audit_broker, receipt_matches
from backtests.engine.contracts import AUDIT_VERSION, METRIC_VERSION, EXECUTION_VERSION
from backtests.engine.integrity import REQUIRED_COVERAGE
from backtests.auto.shared.cache_keys import stable_signature, source_generation_hash
from backtests.auto.shared.plugin_utils import CachedBatchEvaluator, mutation_signature
from backtests.auto.shared.types import Experiment, ScoredCandidate
from backtests.engine.replay import drive_timestamp_batch
from backtests.engine.sim_broker import SimBroker
from tests.backtests.test_strategy_integrity_repairs import bar, buy, ts, ZERO


def test_frozen_reads_ignore_mutable_original_and_reject_changed_snapshot(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    path = root / "input"
    path.write_bytes(b"1234")
    manifest = InputManifest.create([path], source_root=root, destination=tmp_path / "generation")
    stamp = path.stat()
    path.write_bytes(b"5678")
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    with manifest.activate():
        assert read_input_bytes(path) == b"1234"
    other = InputManifest.create([path], source_root=root, destination=tmp_path / "other")
    (other.root / "input").write_bytes(b"bad!")
    with other.activate(), pytest.raises(ValueError, match="bytes changed"):
        read_input_bytes(path)


def receipt(mutations):
    identity = {"request": "request-a", "mutations": mutation_signature(mutations)}
    payload = {"audit_version": AUDIT_VERSION, "metric_version": METRIC_VERSION,
        "execution_version": EXECUTION_VERSION, "source_generation_hash": source_generation_hash(),
        "identity": identity, "coverage": dict.fromkeys(REQUIRED_COVERAGE, "pass"), "errors": []}
    return {"audit_pass": True, "audit_status": "passed", "evaluation_request": "request-a",
        "evaluation_mutations": identity["mutations"], "audit_receipt": {**payload, "receipt_hash": stable_signature(payload)}}


def test_copied_receipt_rejected_at_fresh_and_cached_evaluator():
    metrics = receipt({"a": 1})
    assert receipt_matches(metrics, expected_request="request-a")
    assert not receipt_matches(metrics, expected_request="request-b")
    delegate = lambda items, base: [ScoredCandidate(item.name, 1, metrics=metrics) for item in items]
    evaluator = CachedBatchEvaluator(delegate)
    for _ in range(2):
        result = evaluator([Experiment("different", {"a": 2})], {})[0]
        assert result.rejected


def test_partial_fills_share_budget_and_preserve_remaining_order():
    broker = SimBroker(10000, ZERO, capacity_policy=BarParticipationScenario(.1, "registered-ten-percent"))
    drive_timestamp_batch(broker, [bar(at=ts(), volume=50)], [])
    first = buy(broker, qty=8)
    second = buy(broker, qty=8)
    drive_timestamp_batch(broker, [bar(at=ts(5), volume=100)], [])
    assert [fill.qty for fill in broker.fills] == [5]
    assert [(o.order_id, o.qty) for o in broker.orders] == [(second, 8), (first, 3)]
    drive_timestamp_batch(broker, [bar(at=ts(10), volume=100)], [])
    assert [(f.order_id, f.qty) for f in broker.fills] == [(first, 5), (first, 3), (second, 7)]
    metrics = audit_broker(broker)
    assert metrics["execution_validity_pass"] and metrics["audit_receipt"]["coverage"]["capacity"] == "unverified"
    assert broker.orders[0].qty == 1


def test_independent_events_marks_and_terminal_equity_detect_tampering():
    broker = SimBroker(10000, ZERO)
    buy(broker)
    drive_timestamp_batch(broker, [bar()], [])
    assert audit_broker(broker)["execution_validity_pass"]
    broker.expected_batches.clear()
    errors = audit_broker(broker)["audit_receipt"]["errors"]
    assert "source_event_disposition_mismatch" in errors and "fill_source_event_mismatch" in errors


def test_sourced_exceptional_session_windows():
    from datetime import date, datetime, time, timedelta
    from kis_core.trading_calendar import get_trading_calendar
    from strategy_common.clock import KST
    from strategy_kalcb.first30 import completed_first30_bars
    from strategy_olr.research import _intraday_selection_bars
    from strategy_common.execution_session import kalcb_session_config
    from strategy_kalcb.config import KALCBConfig
    day = date(2025, 11, 13)
    calendar = get_trading_calendar()
    assert calendar.session_hours(day) == (time(10), time(16, 30))
    assert calendar.has_session_coverage(day) and not calendar.has_session_coverage(date(2026, 10, 3))
    opening = datetime.combine(day, time(10), tzinfo=KST)
    bars = [bar(at=opening + timedelta(minutes=5*i)) for i in range(70)]
    assert [b.timestamp for b in completed_first30_bars(bars)] == [b.timestamp for b in bars[:6]]
    assert _intraday_selection_bars(bars, day)[-1].timestamp.time() == time(15, 25)
    config = kalcb_session_config(KALCBConfig(), day)
    assert config.session_close == time(16, 30)


def test_partial_execution_identity_retains_order_date_across_sessions():
    from datetime import timedelta
    from strategy_common.fill_lifecycle import record_fill_delta
    broker = SimBroker(10000, ZERO, capacity_policy=BarParticipationScenario(.1, "overnight"))
    drive_timestamp_batch(broker, [bar(at=ts(), volume=50)], [])
    buy(broker, qty=8)
    drive_timestamp_batch(broker, [bar(at=ts(5), volume=100)], [])
    drive_timestamp_batch(broker, [bar(at=ts(5) + timedelta(days=1), volume=100)], [])
    assert [f.qty for f in broker.fills] == [5]
    drive_timestamp_batch(broker, [bar(at=ts(10) + timedelta(days=1), volume=100)], [])
    assert [f.qty for f in broker.fills] == [5, 3]
    assert len({f.metadata["order_date"] for f in broker.fills}) == 1
    state = {}
    assert all(record_fill_delta(state, fill) for fill in broker.fills)
