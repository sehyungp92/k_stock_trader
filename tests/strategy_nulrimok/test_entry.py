"""Tests for Nulrimok Intraday Entry Logic."""

import pytest
from strategy_nulrimok.iepe.entry import check_entry_conditions, check_confirmation, TickerEntryState, EntryState
from strategy_nulrimok.dse.artifact import TickerArtifact


class TestCheckEntryConditions:
    """Tests for check_entry_conditions: requires in_band, is_dip, and vol_ratio < 0.60."""

    def test_all_conditions_met(self):
        """All three conditions satisfied: in_band, is_dip, vol_dryup."""
        artifact = TickerArtifact(ticker="005930", band_lower=95, band_upper=105, avwap_ref=100)
        bar = {'close': 98, 'high': 102, 'low': 96, 'volume': 400}
        assert check_entry_conditions(artifact, bar, sma5=100, vol_avg=1000) is True

    def test_not_in_band(self):
        """Bar entirely above band_upper -> not in_band."""
        artifact = TickerArtifact(ticker="005930", band_lower=95, band_upper=105, avwap_ref=100)
        bar = {'close': 110, 'high': 115, 'low': 108, 'volume': 400}
        assert check_entry_conditions(artifact, bar, sma5=115, vol_avg=1000) is False

    def test_not_dip(self):
        """close > sma5 -> not a dip."""
        artifact = TickerArtifact(ticker="005930", band_lower=95, band_upper=105, avwap_ref=100)
        bar = {'close': 102, 'high': 105, 'low': 96, 'volume': 400}
        # close=102 > sma5=100
        assert check_entry_conditions(artifact, bar, sma5=100, vol_avg=1000) is False

    def test_volume_too_high(self):
        """vol_ratio = 0.8 > ENTRY_VOL_DRYUP_PCT (0.60) -> fails."""
        artifact = TickerArtifact(ticker="005930", band_lower=95, band_upper=105, avwap_ref=100)
        bar = {'close': 98, 'high': 102, 'low': 96, 'volume': 800}
        # vol_ratio = 800/1000 = 0.8 > 0.60
        assert check_entry_conditions(artifact, bar, sma5=100, vol_avg=1000) is False

    def test_bar_below_band(self):
        """Bar entirely below band_lower -> not in_band (high < band_lower)."""
        artifact = TickerArtifact(ticker="005930", band_lower=95, band_upper=105, avwap_ref=100)
        bar = {'close': 90, 'high': 93, 'low': 88, 'volume': 400}
        assert check_entry_conditions(artifact, bar, sma5=100, vol_avg=1000) is False

    def test_zero_vol_avg(self):
        """vol_avg=0 -> vol_ratio defaults to 1.0 which exceeds threshold."""
        artifact = TickerArtifact(ticker="005930", band_lower=95, band_upper=105, avwap_ref=100)
        bar = {'close': 98, 'high': 102, 'low': 96, 'volume': 400}
        assert check_entry_conditions(artifact, bar, sma5=100, vol_avg=0) is False

    def test_zero_sma5(self):
        """sma5=0 -> is_dip is False."""
        artifact = TickerArtifact(ticker="005930", band_lower=95, band_upper=105, avwap_ref=100)
        bar = {'close': 98, 'high': 102, 'low': 96, 'volume': 400}
        assert check_entry_conditions(artifact, bar, sma5=0, vol_avg=1000) is False

    def test_edge_volume_exactly_at_threshold(self):
        """vol_ratio = exactly 0.60 is NOT less than threshold -> False."""
        artifact = TickerArtifact(ticker="005930", band_lower=95, band_upper=105, avwap_ref=100)
        bar = {'close': 98, 'high': 102, 'low': 96, 'volume': 600}
        # vol_ratio = 600/1000 = 0.60, not < 0.60
        assert check_entry_conditions(artifact, bar, sma5=100, vol_avg=1000) is False


class TestCheckConfirmation:
    """Tests for check_confirmation: returns (confirmed, reason)."""

    def test_invalidation(self):
        """close < band_lower * 0.998 -> (False, 'INVALIDATED')."""
        artifact = TickerArtifact(ticker="005930", band_lower=100, band_upper=110, avwap_ref=105)
        entry_state = TickerEntryState(ticker="005930")
        bar = {'close': 99, 'low': 98}  # close < 100 * 0.998 = 99.8
        confirmed, reason = check_confirmation(entry_state, artifact, bar)
        assert confirmed is False
        assert reason == "INVALIDATED"

    def test_reclaim(self):
        """close > avwap_ref -> (True, 'RECLAIM')."""
        artifact = TickerArtifact(ticker="005930", band_lower=100, band_upper=110, avwap_ref=105)
        entry_state = TickerEntryState(ticker="005930")
        bar = {'close': 106, 'low': 103}  # close > avwap_ref
        confirmed, reason = check_confirmation(entry_state, artifact, bar)
        assert confirmed is True
        assert reason == "RECLAIM"

    def test_higher_low(self):
        """low > last_30m_low and close in band range -> (True, 'HIGHER_LOW')."""
        artifact = TickerArtifact(ticker="005930", band_lower=100, band_upper=110, avwap_ref=115)
        entry_state = TickerEntryState(ticker="005930")
        entry_state.last_30m_low = 99
        bar = {'close': 103, 'low': 100}  # low > last_30m_low, close in band range
        confirmed, reason = check_confirmation(entry_state, artifact, bar)
        assert confirmed is True
        assert reason == "HIGHER_LOW"

    def test_no_confirmation(self):
        """low < last_30m_low -> no confirmation, updates last_30m_low."""
        artifact = TickerArtifact(ticker="005930", band_lower=100, band_upper=110, avwap_ref=115)
        entry_state = TickerEntryState(ticker="005930")
        entry_state.last_30m_low = 102
        bar = {'close': 103, 'low': 101}  # low < last_30m_low
        confirmed, reason = check_confirmation(entry_state, artifact, bar)
        assert confirmed is False
        assert reason == ""

    def test_no_confirmation_updates_last_30m_low(self):
        """When no confirmation, last_30m_low is updated to min of current."""
        artifact = TickerArtifact(ticker="005930", band_lower=100, band_upper=110, avwap_ref=115)
        entry_state = TickerEntryState(ticker="005930")
        entry_state.last_30m_low = 102
        bar = {'close': 103, 'low': 101}
        check_confirmation(entry_state, artifact, bar)
        assert entry_state.last_30m_low == 101

    def test_invalidation_takes_priority_over_reclaim(self):
        """If close is both < band_lower*0.998 and > avwap_ref, invalidation wins."""
        # This can happen if avwap_ref < band_lower*0.998
        artifact = TickerArtifact(ticker="005930", band_lower=100, band_upper=110, avwap_ref=90)
        entry_state = TickerEntryState(ticker="005930")
        bar = {'close': 95, 'low': 94}  # close < 99.8, but close > avwap_ref=90
        confirmed, reason = check_confirmation(entry_state, artifact, bar)
        assert confirmed is False
        assert reason == "INVALIDATED"


class TestTickerEntryState:
    """Tests for TickerEntryState dataclass and reset behavior."""

    def test_defaults(self):
        """Default state is IDLE with no arm_time."""
        s = TickerEntryState(ticker="005930")
        assert s.state == EntryState.IDLE
        assert s.arm_time is None
        assert s.confirm_bars_remaining == 0
        assert s.last_30m_low == float('inf')
        assert s.pending_fill_cycles == 0
        assert s.conf_type == ""
        assert s.anchor_date == ""

    def test_reset(self):
        """reset() restores all fields to their defaults."""
        s = TickerEntryState(ticker="005930")
        s.state = EntryState.ARMED
        s.arm_time = "2024-01-01"
        s.confirm_bars_remaining = 3
        s.last_30m_low = 50.0
        s.pending_fill_cycles = 2
        s.conf_type = "RECLAIM"
        s.anchor_date = "2024-01-01"
        s.reset()
        assert s.state == EntryState.IDLE
        assert s.arm_time is None
        assert s.confirm_bars_remaining == 0
        assert s.last_30m_low == float('inf')
        assert s.pending_fill_cycles == 0
        assert s.conf_type == ""
        assert s.anchor_date == ""

    def test_entry_state_enum_values(self):
        """EntryState enum has the expected members."""
        assert EntryState.IDLE is not None
        assert EntryState.ARMED is not None
        assert EntryState.PENDING_FILL is not None
        assert EntryState.TRIGGERED is not None
        assert EntryState.DONE is not None
