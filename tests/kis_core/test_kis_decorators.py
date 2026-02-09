"""Tests for KIS API rate limiting."""
import pytest
import time
from kis_core.kis_decorators import RateLimiter, rate_limit


class TestRateLimiter:
    """Tests for the RateLimiter class."""

    def test_init_defaults(self):
        rl = RateLimiter()
        assert rl.min_interval == 0.05
        assert rl.max_rate == pytest.approx(20.0)

    def test_init_custom_interval(self):
        rl = RateLimiter(min_interval=0.1, name="test")
        assert rl.min_interval == 0.1
        assert rl.max_rate == pytest.approx(10.0)

    def test_init_custom_name(self):
        rl = RateLimiter(name="my_limiter")
        stats = rl.get_stats()
        assert stats['name'] == "my_limiter"

    def test_default_name(self):
        rl = RateLimiter()
        stats = rl.get_stats()
        assert stats['name'] == "RateLimiter"

    def test_negative_interval_raises(self):
        with pytest.raises(ValueError):
            RateLimiter(min_interval=-1)

    def test_negative_fraction_raises(self):
        with pytest.raises(ValueError):
            RateLimiter(min_interval=-0.001)

    def test_zero_interval_infinite_rate(self):
        rl = RateLimiter(min_interval=0)
        assert rl.max_rate == float('inf')

    def test_first_call_no_wait(self):
        rl = RateLimiter(min_interval=1.0)
        rl.reset()
        waited = rl.wait()
        assert waited == 0.0

    def test_rapid_calls_throttled(self):
        rl = RateLimiter(min_interval=0.1)
        rl.wait()  # First call - no wait
        start = time.time()
        rl.wait()  # Should wait ~0.1s
        elapsed = time.time() - start
        assert elapsed >= 0.08  # Allow small tolerance

    def test_spaced_calls_no_wait(self):
        rl = RateLimiter(min_interval=0.05)
        rl.wait()
        time.sleep(0.1)  # Wait longer than min_interval
        waited = rl.wait()
        assert waited == 0.0

    def test_reset_clears_state(self):
        rl = RateLimiter(min_interval=0.05)
        # Make two rapid calls so at least one actually waits
        rl.wait()
        rl.wait()
        rl.reset()
        stats = rl.get_stats()
        assert stats['total_waits'] == 0
        assert stats['total_wait_time'] == 0.0

    def test_reset_allows_immediate_call(self):
        rl = RateLimiter(min_interval=10.0)  # Very long interval
        rl.wait()  # First call
        rl.reset()
        # After reset, should not need to wait
        start = time.time()
        waited = rl.wait()
        elapsed = time.time() - start
        assert elapsed < 0.1
        assert waited == 0.0

    def test_get_stats(self):
        rl = RateLimiter(min_interval=0.05, name="test_limiter")
        rl.wait()
        stats = rl.get_stats()
        assert stats['name'] == "test_limiter"
        assert stats['min_interval'] == 0.05
        assert 'total_waits' in stats
        assert 'avg_wait_time' in stats
        assert 'total_wait_time' in stats

    def test_stats_accumulate_waits(self):
        rl = RateLimiter(min_interval=0.05)
        rl.wait()  # No wait (first call)
        rl.wait()  # Should wait ~0.05s
        rl.wait()  # Should wait ~0.05s
        stats = rl.get_stats()
        assert stats['total_waits'] >= 1
        assert stats['total_wait_time'] > 0

    def test_repr(self):
        rl = RateLimiter(min_interval=0.1, name="test")
        r = repr(rl)
        assert "0.1" in r
        assert "test" in r

    def test_repr_contains_class_name(self):
        rl = RateLimiter()
        assert "RateLimiter" in repr(rl)


class TestRateLimitDecorator:
    """Tests for the rate_limit decorator."""

    def test_decorator_preserves_function(self):
        @rate_limit(min_interval=0.01)
        def my_func(x):
            return x * 2
        assert my_func(5) == 10

    def test_decorator_preserves_name(self):
        @rate_limit(min_interval=0.01)
        def my_named_func():
            """My docstring."""
            pass
        assert my_named_func.__name__ == "my_named_func"
        assert my_named_func.__doc__ == "My docstring."

    def test_decorator_attaches_limiter(self):
        @rate_limit(min_interval=0.1)
        def my_func():
            pass
        assert hasattr(my_func, '_rate_limiter')
        assert isinstance(my_func._rate_limiter, RateLimiter)

    def test_custom_limiter(self):
        custom = RateLimiter(min_interval=0.5, name="custom")

        @rate_limit(limiter=custom)
        def my_func():
            pass

        assert my_func._rate_limiter is custom

    def test_decorator_with_kwargs(self):
        @rate_limit(min_interval=0.01)
        def add(a, b=0):
            return a + b
        assert add(3, b=4) == 7

    def test_decorator_with_return_value(self):
        @rate_limit(min_interval=0.01)
        def get_data():
            return {"key": "value"}
        result = get_data()
        assert result == {"key": "value"}

    def test_decorator_throttles_rapid_calls(self):
        @rate_limit(min_interval=0.1)
        def noop():
            pass

        noop()  # First call
        start = time.time()
        noop()  # Should be throttled
        elapsed = time.time() - start
        assert elapsed >= 0.08
