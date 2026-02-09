"""Tests for PCIM tradability tier classification -- calls actual source functions."""

import pytest

from strategy_pcim.pipeline.candidate import Candidate
from strategy_pcim.premarket.tier import classify_tier, apply_tier
from strategy_pcim.config.switches import PCIMSwitches


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
        bucket="A",
    )
    defaults.update(overrides)
    return Candidate(**defaults)


# ===========================================================================
# classify_tier
# ===========================================================================

class TestClassifyTier:
    """Tests for classify_tier based on ADTV thresholds."""

    def test_t1_high_adtv(self):
        """ADTV well above T1 threshold (30B) -> T1."""
        assert classify_tier(50e9) == "T1"

    def test_t2_medium_adtv(self):
        """ADTV between T2 (15B) and T1 (30B) -> T2."""
        assert classify_tier(20e9) == "T2"

    def test_t3_low_adtv(self):
        """ADTV below T2 threshold (15B) -> T3."""
        assert classify_tier(12e9) == "T3"

    def test_t1_boundary(self):
        """ADTV exactly at T1 threshold (30B) -> T1 (>= check)."""
        assert classify_tier(30e9) == "T1"

    def test_t2_boundary(self):
        """ADTV exactly at T2 threshold (15B) -> T2 (>= check)."""
        assert classify_tier(15e9) == "T2"

    def test_t2_just_below_t1(self):
        """ADTV just below T1 threshold -> T2."""
        assert classify_tier(29.9e9) == "T2"

    def test_t3_just_below_t2(self):
        """ADTV just below T2 threshold -> T3."""
        assert classify_tier(14.9e9) == "T3"


# ===========================================================================
# apply_tier
# ===========================================================================

class TestApplyTier:
    """Tests for apply_tier with T3/Bucket A switch handling."""

    def test_rejected_candidate_passes_through(self):
        """Already-rejected candidate is returned without tier assignment."""
        c = _make_candidate(reject_reason="ALREADY")
        result = apply_tier(c)
        assert result.tier is None  # Not applied

    def test_t1_applied(self):
        """T1 candidate gets tier='T1' and tier_mult=1.0."""
        c = _make_candidate(adtv_20d=50e9)
        result = apply_tier(c)
        assert result.tier == "T1"
        assert result.tier_mult == 1.0
        assert result.reject_reason is None

    def test_t2_applied(self):
        """T2 candidate gets tier='T2' and tier_mult=0.8."""
        c = _make_candidate(adtv_20d=20e9)
        result = apply_tier(c)
        assert result.tier == "T2"
        assert result.tier_mult == 0.8
        assert result.reject_reason is None

    def test_t3_applied(self):
        """T3 candidate with Bucket B gets tier='T3' and tier_mult=0.5."""
        c = _make_candidate(adtv_20d=12e9, bucket="B")
        result = apply_tier(c)
        assert result.tier == "T3"
        assert result.tier_mult == 0.5
        assert result.reject_reason is None

    def test_t3_bucket_a_allowed_permissive(self):
        """T3 Bucket A allowed with permissive switch."""
        switches = PCIMSwitches(t3_bucket_a_allowed=True)
        c = _make_candidate(adtv_20d=12e9, bucket="A")
        result = apply_tier(c, switches=switches)
        assert result.reject_reason is None
        assert result.tier == "T3"

    def test_t3_bucket_a_blocked_conservative(self):
        """T3 Bucket A blocked with conservative switch."""
        switches = PCIMSwitches(t3_bucket_a_allowed=False)
        c = _make_candidate(adtv_20d=12e9, bucket="A")
        result = apply_tier(c, switches=switches)
        assert result.reject_reason == "T3_NO_BUCKET_A"

    def test_t3_bucket_b_always_allowed(self):
        """T3 Bucket B is always allowed regardless of switch setting."""
        switches = PCIMSwitches(t3_bucket_a_allowed=False)
        c = _make_candidate(adtv_20d=12e9, bucket="B")
        result = apply_tier(c, switches=switches)
        assert result.reject_reason is None
        assert result.tier == "T3"

    def test_default_switches_allow_t3_bucket_a(self):
        """Default global switches (permissive) allow T3 Bucket A."""
        c = _make_candidate(adtv_20d=12e9, bucket="A")
        switches = PCIMSwitches()  # Defaults: t3_bucket_a_allowed=True
        result = apply_tier(c, switches=switches)
        assert result.reject_reason is None

    def test_conservative_switches_block_t3_bucket_a(self):
        """Conservative switches block T3 Bucket A."""
        switches = PCIMSwitches.conservative()
        c = _make_candidate(adtv_20d=12e9, bucket="A")
        result = apply_tier(c, switches=switches)
        assert result.reject_reason == "T3_NO_BUCKET_A"
