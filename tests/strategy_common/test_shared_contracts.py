from __future__ import annotations

from datetime import datetime

import pytest

from strategy_common.actions import SubmitEntry, action_to_json_dict
from strategy_common.clock import KST, ClockContext
from strategy_common.events import DecisionEvent
from strategy_common.market import MarketBar, require_completed_bar


def test_market_bar_rejects_incomplete_for_replay():
    bar = MarketBar("000001", datetime(2026, 1, 5, 9, 0, tzinfo=KST), "1m", 1, 2, 1, 2, 100, is_completed=False)
    with pytest.raises(ValueError):
        require_completed_bar(bar)


def test_neutral_action_has_no_oms_shape():
    action = SubmitEntry("kmp", "000001", 10, "MARKET", None, 100.0, "test", {"risk_per_share": 1.0})
    payload = action_to_json_dict(action)
    assert payload["strategy_id"] == "KMP"
    assert payload["action_type"] == "SubmitEntry"
    assert "intent_type" not in payload


def test_decision_event_json_round_trip_shape():
    action = SubmitEntry("kpr", "000002", 5, "LIMIT", 100.0, 95.0, "reclaim")
    event = DecisionEvent(datetime(2026, 1, 5, 9, 30, tzinfo=KST), "kpr", "000002", "entry", "reclaim", actions=(action,))
    payload = event.to_json_dict()
    assert payload["strategy_id"] == "KPR"
    assert payload["actions"][0]["action_type"] == "SubmitEntry"


def test_clock_context_exposes_kst_utc_and_epoch():
    clock = ClockContext.fixed(datetime(2026, 1, 5, 9, 0, tzinfo=KST))
    assert clock.now_kst.tzinfo == KST
    assert clock.now_utc.hour == 0
    assert clock.now_epoch > 0

