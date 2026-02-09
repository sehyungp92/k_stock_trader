"""Tests for KIS rate budget management."""

import pytest
import time
import threading

from kis_core.rate_budget import TokenBucket, RateBudget, RateLimitedError


class TestTokenBucket:
    """Tests for TokenBucket class."""

    def test_initial_tokens(self):
        """Test bucket starts with full capacity."""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        assert bucket.tokens == 10

    def test_consume_single_token(self):
        """Test consuming single token."""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)

        result = bucket.try_consume(1)

        assert result is True
        assert bucket.tokens == 9

    def test_consume_multiple_tokens(self):
        """Test consuming multiple tokens."""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)

        result = bucket.try_consume(5)

        assert result is True
        assert bucket.tokens == 5

    def test_consume_fails_when_empty(self):
        """Test consume fails when not enough tokens."""
        bucket = TokenBucket(capacity=2, refill_rate=1.0)

        bucket.try_consume(2)  # Use all tokens
        result = bucket.try_consume(1)

        assert result is False

    def test_refill_over_time(self):
        """Test tokens refill over time."""
        bucket = TokenBucket(capacity=10, refill_rate=10.0)  # 10 tokens/sec

        bucket.try_consume(5)  # Use 5 tokens
        assert bucket.tokens == 5

        # Wait for refill
        time.sleep(0.5)  # Should add 5 tokens

        # Force refill check
        bucket.try_consume(0)
        assert bucket.tokens >= 9.0  # Should be close to full

    def test_capacity_cap(self):
        """Test tokens don't exceed capacity."""
        bucket = TokenBucket(capacity=10, refill_rate=100.0)

        # Wait for refill
        time.sleep(0.2)

        # Force refill check
        bucket.try_consume(0)
        assert bucket.tokens <= 10

    def test_thread_safety(self):
        """Test bucket is thread-safe."""
        bucket = TokenBucket(capacity=100, refill_rate=0.0)  # No refill
        results = []

        def consume():
            for _ in range(10):
                if bucket.try_consume(1):
                    results.append(True)

        threads = [threading.Thread(target=consume) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Exactly 100 successful consumes
        assert len(results) == 100


class TestRateBudget:
    """Tests for RateBudget class."""

    def test_default_budgets(self):
        """Test default budget configuration."""
        budget = RateBudget()

        assert "QUOTE" in budget.buckets
        assert "CHART" in budget.buckets
        assert "FLOW" in budget.buckets
        assert "ORDER" in budget.buckets
        assert "BALANCE" in budget.buckets
        assert "DEFAULT" in budget.buckets

    def test_custom_budgets(self):
        """Test custom budget configuration."""
        budget = RateBudget(budgets={"CUSTOM": (100, 2.0)})

        assert "CUSTOM" in budget.buckets
        assert budget.buckets["CUSTOM"].capacity == 100

    def test_try_consume_quote(self):
        """Test consuming from QUOTE bucket."""
        budget = RateBudget()

        result = budget.try_consume("QUOTE")

        assert result is True

    def test_try_consume_unknown_uses_default(self):
        """Test unknown endpoint uses DEFAULT bucket."""
        budget = RateBudget()

        result = budget.try_consume("UNKNOWN_ENDPOINT")

        assert result is True

    def test_separate_buckets(self):
        """Test each endpoint has separate bucket."""
        budget = RateBudget(budgets={
            "A": (2, 0.0),
            "B": (2, 0.0),
            "DEFAULT": (2, 0.0),
        })

        # Exhaust A
        budget.try_consume("A")
        budget.try_consume("A")
        a_result = budget.try_consume("A")

        # B should still work
        b_result = budget.try_consume("B")

        assert a_result is False
        assert b_result is True

    def test_cost_parameter(self):
        """Test cost parameter consumes multiple tokens."""
        budget = RateBudget(budgets={"TEST": (10, 0.0)})

        result = budget.try_consume("TEST", cost=5)

        assert result is True
        assert budget.buckets["TEST"].tokens == 5


class TestRateBudgetAsync:
    """Tests for async rate budget methods."""

    @pytest.mark.asyncio
    async def test_call_rest_success(self):
        """Test successful async call."""
        budget = RateBudget()

        async def mock_fn(x):
            return x * 2

        result = await budget.call_rest("QUOTE", mock_fn, 5)

        assert result == 10

    @pytest.mark.asyncio
    async def test_call_rest_rate_limited(self):
        """Test async call raises when rate limited."""
        budget = RateBudget(budgets={"TEST": (1, 0.0)})

        async def mock_fn():
            return "ok"

        # First call succeeds
        await budget.call_rest("TEST", mock_fn)

        # Second call should fail
        with pytest.raises(RateLimitedError):
            await budget.call_rest("TEST", mock_fn)


class TestRateLimitedError:
    """Tests for RateLimitedError."""

    def test_error_message(self):
        """Test error message is set."""
        error = RateLimitedError("QUOTE rate limited")
        assert "QUOTE" in str(error)

    def test_is_exception(self):
        """Test is proper exception."""
        assert issubclass(RateLimitedError, Exception)
