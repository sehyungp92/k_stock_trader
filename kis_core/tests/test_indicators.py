import pytest
import math
from kis_core.indicators import sma, ema, atr, zscore, percentile_rank, RollingSMA, RollingATR

class TestSMA:
    def test_basic_sma(self):
        result = sma([1, 2, 3, 4, 5], 3)
        assert result == pytest.approx([2.0, 3.0, 4.0])

    def test_insufficient_data(self):
        assert sma([1, 2], 3) == []

    def test_period_equals_length(self):
        result = sma([10, 20, 30], 3)
        assert result == pytest.approx([20.0])

    def test_single_value_period_1(self):
        result = sma([5, 10, 15], 1)
        assert result == pytest.approx([5, 10, 15])

class TestEMA:
    def test_basic_ema(self):
        result = ema([1, 2, 3, 4, 5], 3)
        assert len(result) == 5
        assert result[0] == 1.0  # first value = first input

    def test_empty_input(self):
        assert ema([], 3) == []

    def test_single_value(self):
        assert ema([42.0], 10) == [42.0]

class TestATR:
    def test_basic_atr(self):
        highs = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
        lows = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
        closes = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
        result = atr(highs, lows, closes, period=14)
        # 16 bars -> 15 TRs -> SMA(14) -> 15-14+1=2 values
        assert len(result) == 2

    def test_insufficient_data(self):
        assert atr([10], [5], [7]) == []

class TestZscore:
    def test_basic_zscore(self):
        values = list(range(25))
        result = zscore(values, lookback=20)
        assert len(result) == 6  # 25 - 20 + 1

    def test_insufficient_data(self):
        assert zscore([1, 2, 3], lookback=20) == []

class TestPercentileRank:
    def test_basic_rank(self):
        dist = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        assert percentile_rank(5, dist) == pytest.approx(40.0)

    def test_empty_distribution(self):
        assert percentile_rank(5, []) == 50.0

    def test_highest_value(self):
        assert percentile_rank(100, [1, 2, 3]) == pytest.approx(100.0)

    def test_lowest_value(self):
        assert percentile_rank(0, [1, 2, 3]) == pytest.approx(0.0)

class TestRollingSMA:
    def test_returns_none_before_period(self):
        rsma = RollingSMA(period=3)
        assert rsma.update(1.0) is None
        assert rsma.update(2.0) is None

    def test_returns_value_at_period(self):
        rsma = RollingSMA(period=3)
        rsma.update(1.0)
        rsma.update(2.0)
        result = rsma.update(3.0)
        assert result == pytest.approx(2.0)

    def test_rolling_window(self):
        rsma = RollingSMA(period=3)
        for v in [1, 2, 3]:
            rsma.update(v)
        result = rsma.update(4.0)  # window is [2, 3, 4]
        assert result == pytest.approx(3.0)

class TestRollingATR:
    def test_first_bar_uses_range(self):
        ratr = RollingATR(period=2)
        result = ratr.update_bar(110, 90, 100)
        assert result is None  # period not met yet

    def test_returns_value_at_period(self):
        ratr = RollingATR(period=2)
        ratr.update_bar(110, 90, 100)  # TR = 20 (H-L for first bar)
        result = ratr.update_bar(120, 95, 110)  # TR = max(25, |120-100|, |95-100|) = 25
        assert result == pytest.approx(22.5)  # avg(20, 25)
