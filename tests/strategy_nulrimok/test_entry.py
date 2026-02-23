"""Tests for Nulrimok Intraday Entry Logic."""

import pytest
from unittest.mock import AsyncMock
from datetime import datetime
from zoneinfo import ZoneInfo

from strategy_nulrimok.iepe.entry import (
    check_entry_conditions, check_confirmation, process_entry,
    TickerEntryState, EntryState,
)
from strategy_nulrimok.dse.artifact import TickerArtifact
from oms.intent import IntentResult, IntentStatus


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


class TestProcessEntryExposure:
    """Tests for process_entry exposure headroom pre-check and qty scaling."""

    def _make_armed_state(self):
        s = TickerEntryState(ticker="005930")
        s.state = EntryState.ARMED
        s.confirm_bars_remaining = 3
        s.last_30m_low = 99.0
        return s

    def _make_artifact(self):
        return TickerArtifact(
            ticker="005930", band_lower=95, band_upper=105,
            avwap_ref=100, recommended_risk=0.005,
            atr30m_est=2.0, acceptance_pass=True,
        )

    @pytest.mark.asyncio
    async def test_skips_entry_when_exposure_exhausted(self):
        """Entry skipped when gross exposure >= regime cap (no headroom)."""
        entry_state = self._make_armed_state()
        artifact = self._make_artifact()
        # RECLAIM bar: close > avwap_ref
        bar = {'close': 101, 'high': 102, 'low': 100, 'volume': 300}
        oms = AsyncMock()
        now = datetime(2026, 2, 23, 10, 0, tzinfo=ZoneInfo("Asia/Seoul"))

        result = await process_entry(
            entry_state, artifact, bar, sma5=100, vol_avg=1000,
            now=now, equity=100_000_000, oms=oms,
            gross_exposure_pct=0.82, regime_exposure_cap=0.80,
        )

        assert result is None
        # OMS should NOT be called — skipped before submission
        oms.submit_intent.assert_not_called()
        # State stays ARMED (not reset, can retry if exposure frees)
        assert entry_state.state == EntryState.ARMED
        # Confirmation bar NOT consumed
        assert entry_state.confirm_bars_remaining == 3

    @pytest.mark.asyncio
    async def test_scales_qty_to_fit_headroom(self):
        """Qty scaled down to fit within exposure headroom."""
        entry_state = self._make_armed_state()
        artifact = self._make_artifact()
        bar = {'close': 101, 'high': 102, 'low': 100, 'volume': 300}
        oms = AsyncMock()
        oms.submit_intent = AsyncMock(return_value=IntentResult(
            intent_id="test", status=IntentStatus.EXECUTED, message="ok",
        ))
        now = datetime(2026, 2, 23, 10, 0, tzinfo=ZoneInfo("Asia/Seoul"))

        # 5% headroom on 100M equity = 5M max notional
        # close=101, so max_qty_by_exposure = 5M/101 = ~49504
        # Normal risk-based qty would be much larger
        result = await process_entry(
            entry_state, artifact, bar, sma5=100, vol_avg=1000,
            now=now, equity=100_000_000, oms=oms,
            gross_exposure_pct=0.75, regime_exposure_cap=0.80,
        )

        # Should have submitted with scaled qty
        assert oms.submit_intent.call_count == 1
        submitted_intent = oms.submit_intent.call_args[0][0]
        # Qty should be capped to headroom: 5% of 100M / 101 = 49504
        assert submitted_intent.desired_qty <= 49505

    @pytest.mark.asyncio
    async def test_normal_entry_with_ample_headroom(self):
        """Normal entry when plenty of headroom exists."""
        entry_state = self._make_armed_state()
        artifact = self._make_artifact()
        bar = {'close': 101, 'high': 102, 'low': 100, 'volume': 300}
        oms = AsyncMock()
        oms.submit_intent = AsyncMock(return_value=IntentResult(
            intent_id="test", status=IntentStatus.EXECUTED, message="ok",
        ))
        now = datetime(2026, 2, 23, 10, 0, tzinfo=ZoneInfo("Asia/Seoul"))

        result = await process_entry(
            entry_state, artifact, bar, sma5=100, vol_avg=1000,
            now=now, equity=100_000_000, oms=oms,
            gross_exposure_pct=0.20, regime_exposure_cap=0.80,
        )

        assert oms.submit_intent.call_count == 1
        assert entry_state.state == EntryState.PENDING_FILL

    @pytest.mark.asyncio
    async def test_default_exposure_params_allow_entry(self):
        """Default exposure params (0.0, 1.0) don't block entries."""
        entry_state = self._make_armed_state()
        artifact = self._make_artifact()
        bar = {'close': 101, 'high': 102, 'low': 100, 'volume': 300}
        oms = AsyncMock()
        oms.submit_intent = AsyncMock(return_value=IntentResult(
            intent_id="test", status=IntentStatus.EXECUTED, message="ok",
        ))
        now = datetime(2026, 2, 23, 10, 0, tzinfo=ZoneInfo("Asia/Seoul"))

        # Default params — backwards compatible
        result = await process_entry(
            entry_state, artifact, bar, sma5=100, vol_avg=1000,
            now=now, equity=100_000_000, oms=oms,
        )

        assert oms.submit_intent.call_count == 1
