"""Tests for Nulrimok Active Set Rotation."""

import pytest
from datetime import datetime, timedelta
from strategy_nulrimok.iepe.rotation import rotate_active_set
from strategy_nulrimok.iepe.entry import TickerEntryState, EntryState


class TestRotateActiveSet:
    """Tests for rotate_active_set: time-gated rotation between active set and overflow."""

    def test_no_rotation_if_too_soon(self):
        """Rotation interval not elapsed (< 60 min) -> no change."""
        now = datetime(2024, 1, 15, 10, 0)
        last = datetime(2024, 1, 15, 9, 30)  # 30 min ago, need 60
        active, overflow, ts = rotate_active_set(
            ["A", "B"], ["C"], {}, {}, {}, now, last
        )
        assert active == ["A", "B"]
        assert overflow == ["C"]
        assert ts == last  # timestamp unchanged

    def test_no_rotation_if_no_overflow(self):
        """Empty overflow -> nothing to promote, no change."""
        now = datetime(2024, 1, 15, 10, 0)
        last = datetime(2024, 1, 15, 8, 0)
        active, overflow, ts = rotate_active_set(
            ["A", "B"], [], {}, {}, {}, now, last
        )
        assert active == ["A", "B"]
        assert overflow == []

    def test_rotation_swaps(self):
        """Lowest-ranked idle ticker gets demoted; first overflow promoted."""
        now = datetime(2024, 1, 15, 10, 0)
        last = datetime(2024, 1, 15, 8, 0)  # > 60 min ago
        entry_states = {}
        daily_ranks = {"A": 0.2, "B": 0.8}  # A has lowest rank
        active, overflow, ts = rotate_active_set(
            ["A", "B"], ["C"], entry_states, {}, daily_ranks, now, last
        )
        assert "C" in active
        assert "A" in overflow
        assert "B" in active
        assert ts == now

    def test_armed_ticker_not_demoted(self):
        """ARMED ticker is protected from demotion; next lowest rank is demoted."""
        now = datetime(2024, 1, 15, 10, 0)
        last = datetime(2024, 1, 15, 8, 0)
        entry_states = {"A": TickerEntryState(ticker="A", state=EntryState.ARMED)}
        daily_ranks = {"A": 0.1, "B": 0.5}  # A has lowest rank but is ARMED
        active, overflow, ts = rotate_active_set(
            ["A", "B"], ["C"], entry_states, {}, daily_ranks, now, last
        )
        assert "A" in active  # Protected
        assert "B" in overflow  # B gets demoted instead
        assert "C" in active  # C promoted

    def test_triggered_ticker_not_demoted(self):
        """TRIGGERED ticker is also protected from demotion."""
        now = datetime(2024, 1, 15, 10, 0)
        last = datetime(2024, 1, 15, 8, 0)
        entry_states = {"A": TickerEntryState(ticker="A", state=EntryState.TRIGGERED)}
        daily_ranks = {"A": 0.1, "B": 0.5}
        active, overflow, ts = rotate_active_set(
            ["A", "B"], ["C"], entry_states, {}, daily_ranks, now, last
        )
        assert "A" in active
        assert "B" in overflow

    def test_near_band_recently_protects(self):
        """Ticker flagged as near_band_recently is not demoted."""
        now = datetime(2024, 1, 15, 10, 0)
        last = datetime(2024, 1, 15, 8, 0)
        near_band = {"A": True}
        daily_ranks = {"A": 0.1, "B": 0.5}
        active, overflow, ts = rotate_active_set(
            ["A", "B"], ["C"], {}, near_band, daily_ranks, now, last
        )
        assert "A" in active  # Protected by near_band
        assert "B" in overflow

    def test_all_protected_no_demotion(self):
        """If all active tickers are protected, no swap occurs."""
        now = datetime(2024, 1, 15, 10, 0)
        last = datetime(2024, 1, 15, 8, 0)
        entry_states = {
            "A": TickerEntryState(ticker="A", state=EntryState.ARMED),
            "B": TickerEntryState(ticker="B", state=EntryState.TRIGGERED),
        }
        daily_ranks = {"A": 0.1, "B": 0.5}
        active, overflow, ts = rotate_active_set(
            ["A", "B"], ["C"], entry_states, {}, daily_ranks, now, last
        )
        assert active == ["A", "B"]
        assert overflow == ["C"]
        assert ts == now  # Timestamp updated even without swap

    def test_rotation_exactly_at_interval(self):
        """Exactly at 60 min boundary: 60 min is not < 60 -> rotation proceeds."""
        now = datetime(2024, 1, 15, 10, 0)
        last = datetime(2024, 1, 15, 9, 0)  # exactly 60 min ago
        daily_ranks = {"A": 0.2, "B": 0.8}
        active, overflow, ts = rotate_active_set(
            ["A", "B"], ["C"], {}, {}, daily_ranks, now, last
        )
        assert "C" in active
        assert "A" in overflow

    def test_demoted_ticker_goes_to_end_of_overflow(self):
        """Demoted ticker is appended at end of overflow list."""
        now = datetime(2024, 1, 15, 10, 0)
        last = datetime(2024, 1, 15, 8, 0)
        daily_ranks = {"A": 0.1, "B": 0.9}
        active, overflow, ts = rotate_active_set(
            ["A", "B"], ["C", "D"], {}, {}, daily_ranks, now, last
        )
        # C promoted (first in overflow), A demoted (lowest rank)
        assert overflow == ["D", "A"]
