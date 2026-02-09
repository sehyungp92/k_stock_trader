import pytest
from strategy_kmp.core.tick_imbalance import TickImbalance, ImbalanceBucket

class TestImbalanceBucket:
    def test_creation(self):
        b = ImbalanceBucket(ts_sec=1000)
        assert b.ts_sec == 1000
        assert b.buy_val == 0.0
        assert b.sell_val == 0.0

class TestTickImbalance:
    def test_initial_state(self):
        ti = TickImbalance(window_sec=90)
        assert ti.last_px is None
        assert ti.last_dir == 0
        assert len(ti.buckets) == 0

    def test_first_tick_no_direction(self):
        ti = TickImbalance()
        ti.update(1000.0, 100.0, 10)
        # First tick has no direction (d=0), so no buy/sell
        assert ti.compute(1000.0) == 0.0

    def test_uptick_is_buy(self):
        ti = TickImbalance()
        ti.update(1000.0, 100.0, 10)  # first tick
        ti.update(1001.0, 101.0, 10)  # uptick = buy
        result = ti.compute(1001.0)
        assert result > 0  # positive = buy pressure

    def test_downtick_is_sell(self):
        ti = TickImbalance()
        ti.update(1000.0, 100.0, 10)
        ti.update(1001.0, 99.0, 10)  # downtick = sell
        result = ti.compute(1001.0)
        assert result < 0  # negative = sell pressure

    def test_zero_tick_inherits_direction(self):
        ti = TickImbalance()
        ti.update(1000.0, 100.0, 10)
        ti.update(1001.0, 101.0, 10)  # uptick
        ti.update(1002.0, 101.0, 10)  # zero tick, inherits buy
        result = ti.compute(1002.0)
        assert result > 0

    def test_balanced_imbalance(self):
        ti = TickImbalance()
        ti.update(1000.0, 100.0, 10)
        ti.update(1001.0, 101.0, 10)  # buy
        ti.update(1002.0, 99.0, 10)  # sell (same value)
        result = ti.compute(1002.0)
        # buy_val = 101*10 = 1010, sell_val = 99*10 = 990 - not exactly balanced
        assert abs(result) < 0.1  # approximately balanced

    def test_window_expiry(self):
        ti = TickImbalance(window_sec=5)
        ti.update(1000.0, 100.0, 10)
        ti.update(1001.0, 101.0, 10)  # buy
        # Query after window expired
        result = ti.compute(1010.0)
        assert result == 0.0

    def test_invalid_price_ignored(self):
        ti = TickImbalance()
        ti.update(1000.0, 0.0, 10)  # price <= 0
        assert len(ti.buckets) == 0

    def test_invalid_volume_ignored(self):
        ti = TickImbalance()
        ti.update(1000.0, 100.0, 0)  # volume <= 0
        assert len(ti.buckets) == 0

    def test_reset(self):
        ti = TickImbalance()
        ti.update(1000.0, 100.0, 10)
        ti.update(1001.0, 101.0, 10)
        ti.reset()
        assert ti.last_px is None
        assert ti.last_dir == 0
        assert len(ti.buckets) == 0
