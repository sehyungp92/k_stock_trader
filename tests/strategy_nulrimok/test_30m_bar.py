"""Tests for fetch_30m_bar completed-bar selection."""

from datetime import datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest

from strategy_nulrimok.main import fetch_30m_bar


def _make_minute_bars(start: datetime, end: datetime) -> pd.DataFrame:
    timestamps = pd.date_range(start=start, end=end, freq="min")
    rows = []
    for idx, ts in enumerate(timestamps):
        base = 100 + idx
        rows.append({
            "timestamp": ts.to_pydatetime(),
            "open": base,
            "high": base + 1,
            "low": base - 1,
            "close": base + 0.5,
            "volume": 1,
        })
    return pd.DataFrame(rows)


def _drop_timestamp(df: pd.DataFrame, ts: datetime) -> pd.DataFrame:
    return df[df["timestamp"] != ts].reset_index(drop=True)


class TestFetch30mBar:
    @pytest.mark.asyncio
    async def test_returns_none_when_no_completed_30m_bucket(self):
        api = MagicMock()
        api.get_minute_bars.return_value = _make_minute_bars(
            datetime(2026, 4, 24, 9, 0),
            datetime(2026, 4, 24, 9, 10),
        )

        result = await fetch_30m_bar(api, "005930")

        assert result is None
        api.get_minute_bars.assert_called_once_with("005930", minutes=60)

    @pytest.mark.asyncio
    async def test_returns_previous_completed_bucket_at_exact_boundary(self):
        api = MagicMock()
        api.get_minute_bars.return_value = _make_minute_bars(
            datetime(2026, 4, 24, 9, 1),
            datetime(2026, 4, 24, 10, 0),
        )

        result = await fetch_30m_bar(api, "005930")

        assert result is not None
        assert result["timestamp"] == datetime(2026, 4, 24, 9, 30)
        assert result["volume"] == 30

    @pytest.mark.asyncio
    async def test_returns_latest_completed_bucket_without_next_bucket(self):
        api = MagicMock()
        api.get_minute_bars.return_value = _make_minute_bars(
            datetime(2026, 4, 24, 9, 30),
            datetime(2026, 4, 24, 10, 29),
        )

        result = await fetch_30m_bar(api, "005930")

        assert result is not None
        assert result["timestamp"] == datetime(2026, 4, 24, 10, 0)
        assert result["volume"] == 30

    @pytest.mark.asyncio
    async def test_skips_bucket_with_missing_minute_even_if_boundary_passed(self):
        api = MagicMock()
        api.get_minute_bars.return_value = _drop_timestamp(
            _make_minute_bars(
                datetime(2026, 4, 24, 9, 1),
                datetime(2026, 4, 24, 10, 0),
            ),
            datetime(2026, 4, 24, 9, 45),
        )

        result = await fetch_30m_bar(api, "005930")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_for_empty_minute_bars(self):
        api = MagicMock()
        api.get_minute_bars.return_value = pd.DataFrame()

        result = await fetch_30m_bar(api, "005930")

        assert result is None
