import pytest
from strategy_kpr.signals.micro import MicroSignal, MicroPressureProvider

class TestMicroPressureProvider:
    def test_neutral_on_zero_range(self):
        mp = MicroPressureProvider()
        bar = {'high': 100, 'low': 100, 'open': 100, 'close': 100, 'volume': 1000}
        assert mp.update("005930", bar) == MicroSignal.NEUTRAL

    def test_neutral_on_zero_volume(self):
        mp = MicroPressureProvider()
        bar = {'high': 110, 'low': 90, 'open': 100, 'close': 105, 'volume': 0}
        assert mp.update("005930", bar) == MicroSignal.NEUTRAL

    def test_accumulate_bullish_close_above_open(self):
        mp = MicroPressureProvider()
        # bar_strength = (105-90)/(110-90) = 0.75 >= 0.6 (BAR_STRENGTH_BULL)
        # close > open -> ACCUMULATE even without vol surge
        bar = {'high': 110, 'low': 90, 'open': 100, 'close': 105, 'volume': 1000}
        assert mp.update("005930", bar) == MicroSignal.ACCUMULATE

    def test_distribute_surging_bearish(self):
        mp = MicroPressureProvider()
        # Fill history with low-vol bars
        for i in range(20):
            mp.update("005930", {'high': 110, 'low': 90, 'open': 100, 'close': 100, 'volume': 100})
        # Now a high-vol bearish bar
        # bar_strength = (92-90)/(110-90) = 0.1 <= 0.3 (BAR_STRENGTH_BEAR)
        # vol_ratio = 500/100 = 5.0 >= 1.5 (VOL_SURGE_THRESHOLD)
        bar = {'high': 110, 'low': 90, 'open': 100, 'close': 92, 'volume': 500}
        assert mp.update("005930", bar) == MicroSignal.DISTRIBUTE

    def test_neutral_weak_bar_no_surge(self):
        mp = MicroPressureProvider()
        # bar_strength = (95-90)/(110-90) = 0.25 - not bullish enough for accumulate
        # not surging, close < open
        bar = {'high': 110, 'low': 90, 'open': 100, 'close': 95, 'volume': 1000}
        assert mp.update("005930", bar) == MicroSignal.NEUTRAL
