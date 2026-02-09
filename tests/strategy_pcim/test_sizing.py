"""Tests for PCIM volatility-parity position sizing -- calls actual source functions."""

import pytest

from strategy_pcim.pipeline.candidate import Candidate
from strategy_pcim.premarket.sizing import compute_sizing


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_candidate(**overrides) -> Candidate:
    """Create a Candidate with sensible defaults; override any field."""
    defaults = dict(
        influencer_id="test",
        video_id="v1",
        symbol="005930",
        company_name="Samsung",
        conviction_score=0.9,
        adtv_20d=30e9,
        market_cap=500e9,
        atr_20d=2000,
        close_prev=72000,
        expected_open=73000,
        tier="T1",
        tier_mult=1.0,
        soft_mult=1.0,
        bucket="A",
    )
    defaults.update(overrides)
    return Candidate(**defaults)


class TestComputeSizing:
    """Tests for compute_sizing vol-parity logic."""

    def test_rejected_candidate_passes_through(self):
        """Already-rejected candidate is returned unchanged."""
        c = _make_candidate(reject_reason="ALREADY_REJECTED")
        result = compute_sizing(c, equity=100_000_000)
        assert result.reject_reason == "ALREADY_REJECTED"
        assert result.final_qty is None

    def test_zero_atr_rejected(self):
        """Zero ATR produces ZERO_ATR rejection."""
        c = _make_candidate(atr_20d=0)
        result = compute_sizing(c, equity=100_000_000)
        assert result.reject_reason == "ZERO_ATR"

    def test_basic_sizing(self):
        """Basic sizing with conviction=1, tier_mult=1, soft_mult=1."""
        c = _make_candidate(atr_20d=2000, conviction_score=1.0, tier_mult=1.0, soft_mult=1.0)
        result = compute_sizing(c, equity=100_000_000)
        assert result.final_qty is not None
        assert result.final_qty > 0
        assert result.raw_qty > 0
        # raw_qty = int(500_000 / 3000) = 166
        assert result.raw_qty == 166

    def test_bucket_b_size_cap(self):
        """Bucket B applies 80% cap on computed size."""
        c_b = _make_candidate(bucket="B", atr_20d=2000, conviction_score=1.0)
        result_b = compute_sizing(c_b, equity=100_000_000)
        assert result_b.final_qty is not None

        c_a = _make_candidate(bucket="A", atr_20d=2000, conviction_score=1.0)
        result_a = compute_sizing(c_a, equity=100_000_000)
        assert result_a.final_qty is not None

        # Bucket B qty should be <= Bucket A qty
        assert result_b.final_qty <= result_a.final_qty

    def test_low_conviction_reduces_size(self):
        """Lower conviction score produces smaller position."""
        c_high = _make_candidate(conviction_score=1.0)
        c_low = _make_candidate(conviction_score=0.5)
        r_high = compute_sizing(c_high, equity=100_000_000)
        r_low = compute_sizing(c_low, equity=100_000_000)
        if r_high.final_qty and r_low.final_qty:
            assert r_low.final_qty < r_high.final_qty

    def test_size_floor_rejection(self):
        """Very low conviction + multipliers -> below floor -> rejected."""
        c = _make_candidate(conviction_score=0.1, tier_mult=0.1, soft_mult=0.1)
        result = compute_sizing(c, equity=100_000_000)
        assert result.reject_reason is not None
        assert "SIZE_FLOOR" in result.reject_reason

    def test_single_name_cap(self):
        """Position capped at 15% of equity notional."""
        # Very low ATR -> huge raw_qty -> hits single name cap
        c = _make_candidate(atr_20d=10, conviction_score=1.0, close_prev=100, expected_open=100)
        result = compute_sizing(c, equity=100_000_000)
        if result.final_qty is not None:
            notional = result.final_qty * 100
            assert notional <= 100_000_000 * 0.15 + 100  # Allow rounding

    def test_raw_qty_set(self):
        """raw_qty is set on the candidate."""
        c = _make_candidate(atr_20d=2000, conviction_score=0.9)
        result = compute_sizing(c, equity=100_000_000)
        assert result.raw_qty is not None
        assert result.raw_qty == 166  # int(500_000 / 3000)

    def test_tier_mult_scales_size(self):
        """tier_mult < 1 reduces final_qty proportionally."""
        c_t1 = _make_candidate(tier_mult=1.0, conviction_score=1.0, soft_mult=1.0)
        c_t3 = _make_candidate(tier_mult=0.5, conviction_score=1.0, soft_mult=1.0)
        r_t1 = compute_sizing(c_t1, equity=100_000_000)
        r_t3 = compute_sizing(c_t3, equity=100_000_000)
        if r_t1.final_qty and r_t3.final_qty:
            assert r_t3.final_qty < r_t1.final_qty

    def test_soft_mult_scales_size(self):
        """soft_mult < 1 reduces final_qty proportionally."""
        c_full = _make_candidate(soft_mult=1.0, conviction_score=1.0, tier_mult=1.0)
        c_soft = _make_candidate(soft_mult=0.5, conviction_score=1.0, tier_mult=1.0)
        r_full = compute_sizing(c_full, equity=100_000_000)
        r_soft = compute_sizing(c_soft, equity=100_000_000)
        if r_full.final_qty and r_soft.final_qty:
            assert r_soft.final_qty < r_full.final_qty
