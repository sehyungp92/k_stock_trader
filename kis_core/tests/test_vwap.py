import pytest
from datetime import date, datetime
from kis_core.vwap import VWAPLedger, compute_anchored_daily_vwap, vwap_band

class TestVWAPLedger:
    def test_initial_vwap_zero(self):
        ledger = VWAPLedger()
        assert ledger.vwap == 0.0

    def test_update_from_tick(self):
        ledger = VWAPLedger()
        ledger.update_from_tick(100.0, 10)
        assert ledger.vwap == 100.0
        ledger.update_from_tick(200.0, 10)
        assert ledger.vwap == pytest.approx(150.0)

    def test_update_from_tick_zero_volume(self):
        ledger = VWAPLedger()
        ledger.update_from_tick(100.0, 0)
        assert ledger.vwap == 0.0

    def test_update_from_bar(self):
        ledger = VWAPLedger()
        bar = {'high': 110, 'low': 90, 'close': 100, 'volume': 30}
        ledger.update_from_bar(bar)
        assert ledger.vwap == pytest.approx(100.0)  # typical = (110+90+100)/3 = 100

    def test_update_from_bar_zero_volume(self):
        ledger = VWAPLedger()
        bar = {'high': 110, 'low': 90, 'close': 100, 'volume': 0}
        ledger.update_from_bar(bar)
        assert ledger.vwap == 0.0

    def test_reset(self):
        ledger = VWAPLedger()
        ledger.update_from_tick(100.0, 10)
        ledger.reset(anchor=date(2024, 1, 15))
        assert ledger.vwap == 0.0
        assert ledger.anchor_date == date(2024, 1, 15)

class TestComputeAnchoredDailyVwap:
    def test_basic_computation(self):
        bars = [
            {'date': '20240110', 'high': 110, 'low': 90, 'close': 100, 'volume': 1000},
            {'date': '20240111', 'high': 115, 'low': 95, 'close': 105, 'volume': 2000},
            {'date': '20240112', 'high': 120, 'low': 100, 'close': 110, 'volume': 1500},
        ]
        result = compute_anchored_daily_vwap(bars, date(2024, 1, 10))
        assert result > 0

    def test_anchor_filters_earlier_bars(self):
        bars = [
            {'date': '20240108', 'high': 50, 'low': 40, 'close': 45, 'volume': 5000},
            {'date': '20240110', 'high': 110, 'low': 90, 'close': 100, 'volume': 1000},
        ]
        result = compute_anchored_daily_vwap(bars, date(2024, 1, 10))
        # Should only use the second bar
        assert result == pytest.approx(100.0)  # (110+90+100)/3

    def test_no_bars_returns_zero(self):
        assert compute_anchored_daily_vwap([], date(2024, 1, 10)) == 0.0

    def test_zero_volume_bars_skipped(self):
        bars = [
            {'date': '20240110', 'high': 110, 'low': 90, 'close': 100, 'volume': 0},
        ]
        assert compute_anchored_daily_vwap(bars, date(2024, 1, 10)) == 0.0

class TestVwapBand:
    def test_default_band(self):
        lower, upper = vwap_band(100.0)
        assert lower == pytest.approx(99.5)
        assert upper == pytest.approx(100.5)

    def test_custom_band(self):
        lower, upper = vwap_band(1000.0, band_pct=0.01)
        assert lower == pytest.approx(990.0)
        assert upper == pytest.approx(1010.0)

    def test_zero_vwap(self):
        lower, upper = vwap_band(0.0)
        assert lower == 0.0
        assert upper == 0.0
