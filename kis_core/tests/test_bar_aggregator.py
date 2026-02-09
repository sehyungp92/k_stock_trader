import pytest
from datetime import datetime, timedelta
from kis_core.bar_aggregator import Bar, BarAggregator, aggregate_bars

class TestBar:
    def test_creation(self):
        ts = datetime(2024, 1, 15, 9, 30)
        bar = Bar(ts, 100, 110, 90, 105, 1000)
        assert bar.open == 100
        assert bar.high == 110
        assert bar.low == 90
        assert bar.close == 105
        assert bar.volume == 1000

class TestBarAggregator:
    def test_first_tick_creates_bar(self):
        agg = BarAggregator(interval_minutes=1)
        ts = datetime(2024, 1, 15, 9, 30, 15)
        result = agg.update_tick(ts, 100.0, 50)
        assert result is None  # first tick, no completed bar
        assert agg._current_bar is not None
        assert agg._current_bar.open == 100.0

    def test_tick_in_same_bar_updates(self):
        agg = BarAggregator(interval_minutes=1)
        ts1 = datetime(2024, 1, 15, 9, 30, 10)
        ts2 = datetime(2024, 1, 15, 9, 30, 30)
        agg.update_tick(ts1, 100.0, 50)
        agg.update_tick(ts2, 105.0, 30)
        assert agg._current_bar.high == 105.0
        assert agg._current_bar.close == 105.0
        assert agg._current_bar.volume == 80

    def test_new_bar_returns_completed(self):
        agg = BarAggregator(interval_minutes=1)
        ts1 = datetime(2024, 1, 15, 9, 30, 10)
        ts2 = datetime(2024, 1, 15, 9, 31, 10)
        agg.update_tick(ts1, 100.0, 50)
        completed = agg.update_tick(ts2, 105.0, 30)
        assert completed is not None
        assert completed.open == 100.0
        assert completed.close == 100.0
        assert completed.volume == 50

    def test_5_minute_bars(self):
        agg = BarAggregator(interval_minutes=5)
        ts1 = datetime(2024, 1, 15, 9, 30, 0)
        ts2 = datetime(2024, 1, 15, 9, 33, 0)
        ts3 = datetime(2024, 1, 15, 9, 35, 0)
        agg.update_tick(ts1, 100.0, 50)
        result = agg.update_tick(ts2, 105.0, 30)
        assert result is None  # still in same 5-min bar
        result = agg.update_tick(ts3, 110.0, 20)
        assert result is not None  # new bar started

    def test_get_completed_bars(self):
        agg = BarAggregator(interval_minutes=1)
        for i in range(5):
            ts = datetime(2024, 1, 15, 9, 30 + i, 10)
            agg.update_tick(ts, 100.0 + i, 50)
        bars = agg.get_completed_bars()
        assert len(bars) == 4  # 5 ticks = 4 completed bars

    def test_get_completed_bars_with_n(self):
        agg = BarAggregator(interval_minutes=1)
        for i in range(5):
            ts = datetime(2024, 1, 15, 9, 30 + i, 10)
            agg.update_tick(ts, 100.0 + i, 50)
        bars = agg.get_completed_bars(n=2)
        assert len(bars) == 2

class TestAggregateBars:
    def test_basic_aggregation(self):
        bars = [
            {'timestamp': datetime(2024, 1, 15, 9, 30), 'open': 100, 'high': 110, 'low': 90, 'close': 105, 'volume': 1000},
            {'timestamp': datetime(2024, 1, 15, 9, 31), 'open': 105, 'high': 115, 'low': 95, 'close': 110, 'volume': 1200},
            {'timestamp': datetime(2024, 1, 15, 9, 35), 'open': 110, 'high': 120, 'low': 100, 'close': 115, 'volume': 800},
        ]
        result = aggregate_bars(bars, 5)
        assert len(result) == 2  # 9:30 bar and 9:35 bar

    def test_empty_input(self):
        assert aggregate_bars([], 5) == []

    def test_single_bar(self):
        bars = [
            {'timestamp': datetime(2024, 1, 15, 9, 30), 'open': 100, 'high': 110, 'low': 90, 'close': 105, 'volume': 1000},
        ]
        result = aggregate_bars(bars, 5)
        assert len(result) == 1
        assert result[0].open == 100
