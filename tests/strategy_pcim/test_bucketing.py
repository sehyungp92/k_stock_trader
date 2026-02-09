"""Tests for PCIM premarket bucketing -- calls actual source functions."""

import pytest
from dataclasses import dataclass

from strategy_pcim.pipeline.candidate import Candidate
from strategy_pcim.premarket.bucketing import classify_bucket, apply_bucketing


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_candidate(**overrides) -> Candidate:
    """Create a Candidate with sensible defaults; override any field."""
    defaults = dict(
        influencer_id="inf_001",
        video_id="vid_001",
        symbol="005930",
        company_name="Samsung Electronics",
        conviction_score=0.85,
        close_prev=100_000.0,
        adtv_20d=20e9,
        market_cap=500e9,
    )
    defaults.update(overrides)
    return Candidate(**defaults)


@dataclass
class MockRegime:
    """Minimal regime stand-in with the two attributes bucketing reads."""
    name: str = "NORMAL"
    disable_bucket_a: bool = False


# ===========================================================================
# classify_bucket
# ===========================================================================

class TestClassifyBucket:
    """Tests for gap-percentage bucket classification (A / B / D)."""

    # --- Bucket A: 0.00 <= gap < 0.03 ---

    def test_bucket_a_at_zero(self):
        """Gap of 0.00 falls in bucket A (lower bound inclusive)."""
        assert classify_bucket(0.00) == "A"

    def test_bucket_a_mid_range(self):
        """Gap of 0.015 falls in bucket A."""
        assert classify_bucket(0.015) == "A"

    def test_bucket_a_just_below_upper(self):
        """Gap of 0.0299 still falls in bucket A (upper bound exclusive)."""
        assert classify_bucket(0.0299) == "A"

    # --- Bucket B: 0.03 <= gap < 0.07 ---

    def test_bucket_b_at_lower_bound(self):
        """Gap of 0.03 falls in bucket B (inclusive lower bound)."""
        assert classify_bucket(0.03) == "B"

    def test_bucket_b_mid_range(self):
        """Gap of 0.05 falls in bucket B."""
        assert classify_bucket(0.05) == "B"

    def test_bucket_b_just_below_upper(self):
        """Gap of 0.0699 still falls in bucket B."""
        assert classify_bucket(0.0699) == "B"

    # --- Bucket D: everything else ---

    def test_bucket_d_negative_gap(self):
        """Negative gap falls in bucket D."""
        assert classify_bucket(-0.02) == "D"

    def test_bucket_d_at_upper_b_boundary(self):
        """Gap of 0.07 (B upper bound) falls in bucket D."""
        assert classify_bucket(0.07) == "D"

    def test_bucket_d_large_gap(self):
        """Very large gap falls in bucket D."""
        assert classify_bucket(0.15) == "D"

    def test_bucket_d_slightly_negative(self):
        """Slightly negative gap falls in bucket D."""
        assert classify_bucket(-0.001) == "D"


# ===========================================================================
# apply_bucketing
# ===========================================================================

class TestApplyBucketing:
    """Tests for the full bucketing pipeline step."""

    def test_already_rejected_candidate_returned_early(self):
        """A candidate that already has a reject_reason is returned unchanged."""
        c = _make_candidate(reject_reason="ADTV_LT_5B")
        regime = MockRegime()

        result = apply_bucketing(c, expected_open=101_000.0, regime=regime)

        assert result.is_rejected()
        assert result.reject_reason == "ADTV_LT_5B"
        # bucket should NOT have been set
        assert result.bucket is None

    def test_bucket_d_rejected(self):
        """Gap outside A/B ranges results in NO_TRADE_BUCKET_D rejection."""
        c = _make_candidate(close_prev=100_000.0)
        regime = MockRegime()

        # expected_open 110_000 -> gap = 10% -> bucket D
        result = apply_bucketing(c, expected_open=110_000.0, regime=regime)

        assert result.is_rejected()
        assert result.reject_reason == "NO_TRADE_BUCKET_D"
        assert result.bucket == "D"

    def test_bucket_a_rejected_when_regime_disables(self):
        """Bucket A candidate is rejected when regime.disable_bucket_a is True."""
        c = _make_candidate(close_prev=100_000.0)
        regime = MockRegime(name="WEAK", disable_bucket_a=True)

        # expected_open 101_000 -> gap = 1% -> bucket A
        result = apply_bucketing(c, expected_open=101_000.0, regime=regime)

        assert result.is_rejected()
        assert result.reject_reason == "REGIME_DISALLOWS_BUCKET_A"
        assert result.bucket == "A"

    def test_bucket_a_successful(self):
        """Bucket A candidate passes when regime allows it."""
        c = _make_candidate(close_prev=100_000.0)
        regime = MockRegime(name="NORMAL", disable_bucket_a=False)

        # expected_open 101_500 -> gap = 1.5% -> bucket A
        result = apply_bucketing(c, expected_open=101_500.0, regime=regime)

        assert not result.is_rejected()
        assert result.bucket == "A"
        assert result.gap_pct == pytest.approx(0.015)
        assert result.expected_open == 101_500.0

    def test_bucket_b_successful(self):
        """Bucket B candidate passes regardless of regime.disable_bucket_a."""
        c = _make_candidate(close_prev=100_000.0)
        regime = MockRegime(name="WEAK", disable_bucket_a=True)

        # expected_open 105_000 -> gap = 5% -> bucket B
        result = apply_bucketing(c, expected_open=105_000.0, regime=regime)

        assert not result.is_rejected()
        assert result.bucket == "B"
        assert result.gap_pct == pytest.approx(0.05)

    def test_gap_pct_computation_accuracy(self):
        """Verify gap_pct = (expected_open - close_prev) / close_prev."""
        c = _make_candidate(close_prev=80_000.0)
        regime = MockRegime()

        # expected_open 82_000 -> gap = 2_000 / 80_000 = 0.025 -> bucket A
        result = apply_bucketing(c, expected_open=82_000.0, regime=regime)

        assert result.gap_pct == pytest.approx(0.025)
        assert result.bucket == "A"

    def test_negative_gap_rejected_as_bucket_d(self):
        """Negative gap (expected_open < close_prev) maps to bucket D."""
        c = _make_candidate(close_prev=100_000.0)
        regime = MockRegime()

        # expected_open 98_000 -> gap = -2% -> bucket D
        result = apply_bucketing(c, expected_open=98_000.0, regime=regime)

        assert result.is_rejected()
        assert result.reject_reason == "NO_TRADE_BUCKET_D"
        assert result.bucket == "D"
        assert result.gap_pct == pytest.approx(-0.02)

    def test_zero_gap_is_bucket_a(self):
        """Zero gap (expected_open == close_prev) falls in bucket A."""
        c = _make_candidate(close_prev=100_000.0)
        regime = MockRegime()

        result = apply_bucketing(c, expected_open=100_000.0, regime=regime)

        assert not result.is_rejected()
        assert result.bucket == "A"
        assert result.gap_pct == pytest.approx(0.0)

    def test_bucket_b_lower_boundary(self):
        """Gap of exactly 3% falls in bucket B."""
        c = _make_candidate(close_prev=100_000.0)
        regime = MockRegime()

        # expected_open 103_000 -> gap = 3% -> bucket B
        result = apply_bucketing(c, expected_open=103_000.0, regime=regime)

        assert not result.is_rejected()
        assert result.bucket == "B"
        assert result.gap_pct == pytest.approx(0.03)

    def test_close_prev_zero_handled(self):
        """If close_prev is 0 the gap defaults to 0 (no division error)."""
        c = _make_candidate(close_prev=0.0)
        regime = MockRegime()

        result = apply_bucketing(c, expected_open=100_000.0, regime=regime)

        # gap_pct should be 0 because of the guard, which maps to bucket A
        assert result.gap_pct == 0
        assert result.bucket == "A"
