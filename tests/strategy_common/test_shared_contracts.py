from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from strategy_common.actions import SubmitEntry, SubmitExit, action_to_json_dict
from strategy_common.clock import KST, ClockContext
from strategy_common.events import DecisionEvent
from strategy_common.market import MarketBar, require_completed_bar
from strategy_common.oms_adapter import action_to_intent


def test_market_bar_rejects_incomplete_for_replay():
    bar = MarketBar("000001", datetime(2026, 1, 5, 9, 0, tzinfo=KST), "1m", 1, 2, 1, 2, 100, is_completed=False)
    with pytest.raises(ValueError):
        require_completed_bar(bar)


def test_neutral_action_has_no_oms_shape():
    action = SubmitEntry("alpha", "000001", 10, "MARKET", None, 100.0, "test", {"risk_per_share": 1.0})
    payload = action_to_json_dict(action)
    assert payload["strategy_id"] == "ALPHA"
    assert payload["action_type"] == "SubmitEntry"
    assert "intent_type" not in payload


def test_close_auction_action_carries_execution_style_to_intent():
    entry = SubmitEntry("olr", "005930", 10, "CLOSE_AUCTION", 72000.0, None, "close_auction", {"expiry_ts": 1234.0})
    intent = action_to_intent(entry)
    assert intent.constraints.execution_style == "CLOSE_AUCTION"
    assert intent.constraints.limit_price == 72000.0
    assert intent.constraints.expiry_ts == 1234.0
    assert intent.urgency.name == "LOW"

    exit_action = SubmitExit("olr", "005930", 10, "CLOSE_AUCTION", 71900.0, "next_close")
    exit_intent = action_to_intent(exit_action)
    assert exit_intent.constraints.execution_style == "CLOSE_AUCTION"


def test_synthetic_stop_entry_intent_separates_trigger_from_protective_stop():
    entry = SubmitEntry(
        "delta",
        "005930",
        10,
        "STOP",
        None,
        104.0,
        "opening_range_stop_breakout_trigger",
        {"execution_style": "SYNTHETIC_STOP", "entry_trigger_price": 104.0, "protective_stop_price": 100.0},
    )

    intent = action_to_intent(entry)

    assert intent.constraints.execution_style == "SYNTHETIC_STOP"
    assert intent.constraints.stop_price == 104.0
    assert intent.risk_payload.entry_px == 104.0
    assert intent.risk_payload.stop_px == 100.0
    assert intent.risk_payload.hard_stop_px == 100.0


def test_synthetic_stop_entry_intent_parses_iso_expiry_as_kst_epoch():
    entry = SubmitEntry(
        "delta",
        "005930",
        10,
        "STOP",
        None,
        104.0,
        "opening_range_stop_breakout_trigger",
        {
            "execution_style": "SYNTHETIC_STOP",
            "entry_trigger_price": 104.0,
            "protective_stop_price": 100.0,
            "expiry_timestamp": "2026-01-05T09:45:00",
        },
    )

    intent = action_to_intent(entry)

    expected = datetime(2026, 1, 5, 9, 45, tzinfo=ZoneInfo("Asia/Seoul")).timestamp()
    assert intent.constraints.expiry_ts == expected


def test_intent_preserves_causal_action_metadata():
    entry = SubmitEntry(
        "kalcb",
        "005930",
        10,
        "LIMIT",
        72000.0,
        70000.0,
        "entry",
        {
            "action_ref": "action-1",
            "strategy_action_hash": "action-hash",
            "source_artifact_hash": "artifact-hash",
            "source_fingerprint": "source-fingerprint",
            "candidate_hash": "candidate-hash",
            "decision_ref": "decision-1",
        },
    )

    intent = action_to_intent(entry)

    assert intent.metadata["action_ref"] == "action-1"
    assert intent.metadata["source_artifact_hash"] == "artifact-hash"
    assert intent.metadata["strategy_action_hash"] == "action-hash"
    assert intent.metadata["candidate_hash"] == "candidate-hash"


def test_decision_event_json_round_trip_shape():
    action = SubmitEntry("beta", "000002", 5, "LIMIT", 100.0, 95.0, "reclaim")
    event = DecisionEvent(datetime(2026, 1, 5, 9, 30, tzinfo=KST), "beta", "000002", "entry", "reclaim", actions=(action,))
    payload = event.to_json_dict()
    assert payload["strategy_id"] == "BETA"
    assert payload["actions"][0]["action_type"] == "SubmitEntry"


def test_clock_context_exposes_kst_utc_and_epoch():
    clock = ClockContext.fixed(datetime(2026, 1, 5, 9, 0, tzinfo=KST))
    assert clock.now_kst.tzinfo == KST
    assert clock.now_utc.hour == 0
    assert clock.now_epoch > 0
