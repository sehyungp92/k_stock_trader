from datetime import datetime, timezone

import pytest

from scripts.kalcb_executed_value_replay import equity_records


def test_equity_export_preserves_both_observed_marks_and_terminal_equity():
    times = [datetime(2025, 8, 7, 0, 30, tzinfo=timezone.utc),
             datetime(2025, 8, 8, 6, 30, tzinfo=timezone.utc)]
    records = equity_records(times, [100_000_000, 98_000_000, 104_000_000])
    assert records == [(str(times[0]), 98_000_000), (str(times[1]), 104_000_000)]


def test_cash_before_any_market_observation_is_not_a_timestamped_mark():
    assert equity_records([], [100_000_000]) == []


def test_equity_export_rejects_incomplete_timestamp_alignment():
    with pytest.raises(ValueError, match="one mark per timestamp"):
        equity_records([datetime(2025, 8, 7)], [100_000_000])
