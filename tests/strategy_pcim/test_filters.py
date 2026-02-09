"""Tests for PCIM pipeline filters -- calls actual source functions."""

import pytest

from strategy_pcim.pipeline.candidate import Candidate
from strategy_pcim.pipeline.filters import (
    apply_hard_filters,
    apply_gap_reversal_filter,
    compute_soft_multiplier,
)
from strategy_pcim.config.switches import PCIMSwitches


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
        adtv_20d=20e9,       # 20 B  -- safely above ADTV_MIN (10 B)
        market_cap=500e9,     # 500 B -- safely within MCAP range
    )
    defaults.update(overrides)
    return Candidate(**defaults)


# ===========================================================================
# apply_hard_filters
# ===========================================================================

class TestApplyHardFilters:
    """Tests for hard-filter rejection logic."""

    def test_adtv_below_min_rejected(self):
        """ADTV < 5 B must return 'ADTV_LT_5B'."""
        c = _make_candidate(adtv_20d=3e9)
        result = apply_hard_filters(c, has_earnings_soon=False)
        assert result == "ADTV_LT_5B"

    def test_mcap_below_min_rejected(self):
        """Market cap < 30 B must return 'MCAP_LT_30B'."""
        c = _make_candidate(market_cap=20e9)
        result = apply_hard_filters(c, has_earnings_soon=False)
        assert result == "MCAP_LT_30B"

    def test_mcap_above_max_rejected(self):
        """Market cap > 15 T must return 'MCAP_GT_15T'."""
        c = _make_candidate(market_cap=16e12)
        result = apply_hard_filters(c, has_earnings_soon=False)
        assert result == "MCAP_GT_15T"

    def test_earnings_soon_rejected(self):
        """Earnings within window must return 'EARNINGS_WINDOW'."""
        c = _make_candidate()
        result = apply_hard_filters(c, has_earnings_soon=True)
        assert result == "EARNINGS_WINDOW"

    def test_all_passing_returns_none(self):
        """Candidate that satisfies every hard filter returns None."""
        c = _make_candidate(adtv_20d=20e9, market_cap=500e9)
        result = apply_hard_filters(c, has_earnings_soon=False)
        assert result is None

    def test_adtv_exactly_at_min_passes(self):
        """ADTV exactly at 5 B boundary passes (>=)."""
        c = _make_candidate(adtv_20d=5e9)
        result = apply_hard_filters(c, has_earnings_soon=False)
        assert result is None

    def test_mcap_exactly_at_min_passes(self):
        """Market cap exactly at 50 B boundary passes (>=)."""
        c = _make_candidate(market_cap=50e9)
        result = apply_hard_filters(c, has_earnings_soon=False)
        assert result is None

    def test_mcap_exactly_at_max_passes(self):
        """Market cap exactly at 5 T boundary passes (<=)."""
        c = _make_candidate(market_cap=5e12)
        result = apply_hard_filters(c, has_earnings_soon=False)
        assert result is None

    def test_rejection_priority_adtv_before_mcap(self):
        """When both ADTV and MCAP fail, ADTV is checked first."""
        c = _make_candidate(adtv_20d=1e9, market_cap=10e9)
        result = apply_hard_filters(c, has_earnings_soon=False)
        assert result == "ADTV_LT_5B"


# ===========================================================================
# apply_gap_reversal_filter
# ===========================================================================

class TestApplyGapReversalFilter:
    """Tests for gap-reversal rate filter."""

    def test_insufficient_data_returns_none(self):
        """Insufficient gap-reversal data skips the filter (returns None)."""
        c = _make_candidate(
            gap_rev_insufficient=True,
            gap_rev_rate=0.90,  # high, but should be ignored
            gap_rev_events=3,
        )
        result = apply_gap_reversal_filter(c)
        assert result is None

    def test_high_rate_above_default_threshold_rejected(self):
        """Rate above the default threshold (0.65) returns a rejection string."""
        c = _make_candidate(
            gap_rev_rate=0.75,
            gap_rev_events=15,
            gap_rev_insufficient=False,
        )
        result = apply_gap_reversal_filter(c)
        assert result is not None
        assert "GAP_REV_GT_" in result

    def test_low_rate_below_threshold_passes(self):
        """Rate below the threshold returns None."""
        c = _make_candidate(
            gap_rev_rate=0.40,
            gap_rev_events=20,
            gap_rev_insufficient=False,
        )
        result = apply_gap_reversal_filter(c)
        assert result is None

    def test_rate_exactly_at_threshold_passes(self):
        """Rate exactly at threshold (0.65) passes because check is strict >."""
        c = _make_candidate(
            gap_rev_rate=0.65,
            gap_rev_events=12,
            gap_rev_insufficient=False,
        )
        switches = PCIMSwitches(gap_reversal_threshold=0.65)
        result = apply_gap_reversal_filter(c, switches=switches)
        assert result is None

    def test_custom_switches_lower_threshold_rejects(self):
        """Conservative switches (0.60 threshold) reject a rate of 0.62."""
        c = _make_candidate(
            gap_rev_rate=0.62,
            gap_rev_events=11,
            gap_rev_insufficient=False,
        )
        switches = PCIMSwitches.conservative()  # threshold = 0.60
        result = apply_gap_reversal_filter(c, switches=switches)
        assert result is not None
        assert "GAP_REV_GT_60PCT" in result

    def test_custom_switches_higher_threshold_passes(self):
        """Permissive switches (0.70 threshold) allow a rate of 0.68."""
        c = _make_candidate(
            gap_rev_rate=0.68,
            gap_rev_events=14,
            gap_rev_insufficient=False,
        )
        switches = PCIMSwitches(gap_reversal_threshold=0.70)
        result = apply_gap_reversal_filter(c, switches=switches)
        assert result is None


# ===========================================================================
# compute_soft_multiplier
# ===========================================================================

class TestComputeSoftMultiplier:
    """Tests for soft-filter multiplier computation."""

    def test_no_penalties_returns_one(self):
        """High ADTV + low 5-day return yields multiplier 1.0."""
        c = _make_candidate(adtv_20d=50e9)
        mult = compute_soft_multiplier(c, five_day_return=0.05)
        assert mult == 1.0

    def test_adtv_soft_penalty_when_enabled(self):
        """ADTV in 10 B-15 B range with penalty enabled gives 0.5x."""
        c = _make_candidate(adtv_20d=12e9)
        switches = PCIMSwitches(enable_adtv_soft_penalty=True)
        mult = compute_soft_multiplier(c, five_day_return=0.05, switches=switches)
        assert mult == pytest.approx(0.5)

    def test_adtv_soft_penalty_disabled_by_default(self):
        """Default switches disable ADTV soft penalty -- no reduction."""
        c = _make_candidate(adtv_20d=12e9)
        switches = PCIMSwitches()  # enable_adtv_soft_penalty defaults to False
        mult = compute_soft_multiplier(c, five_day_return=0.05, switches=switches)
        assert mult == 1.0

    def test_five_day_return_penalty(self):
        """5-day return > 0.20 applies 0.5x multiplier."""
        c = _make_candidate(adtv_20d=50e9)
        mult = compute_soft_multiplier(c, five_day_return=0.25)
        assert mult == pytest.approx(0.5)

    def test_five_day_return_at_boundary_no_penalty(self):
        """5-day return exactly at 0.20 does NOT trigger penalty (strict >)."""
        c = _make_candidate(adtv_20d=50e9)
        mult = compute_soft_multiplier(c, five_day_return=0.20)
        assert mult == 1.0

    def test_combined_adtv_and_five_day_penalties(self):
        """Both penalties active: 0.5 * 0.5 = 0.25."""
        c = _make_candidate(adtv_20d=12e9)
        switches = PCIMSwitches(enable_adtv_soft_penalty=True)
        mult = compute_soft_multiplier(c, five_day_return=0.25, switches=switches)
        assert mult == pytest.approx(0.25)

    def test_adtv_penalty_disabled_via_switch(self):
        """Explicitly disabled ADTV penalty means only 5-day penalty applies."""
        c = _make_candidate(adtv_20d=12e9)
        switches = PCIMSwitches(enable_adtv_soft_penalty=False)
        mult = compute_soft_multiplier(c, five_day_return=0.25, switches=switches)
        assert mult == pytest.approx(0.5)

    def test_adtv_above_soft_range_no_penalty(self):
        """ADTV at 15 B (upper bound) is outside the soft range -- no penalty."""
        c = _make_candidate(adtv_20d=15e9)
        switches = PCIMSwitches(enable_adtv_soft_penalty=True)
        mult = compute_soft_multiplier(c, five_day_return=0.05, switches=switches)
        assert mult == 1.0

    def test_adtv_at_soft_range_lower_bound_applies_penalty(self):
        """ADTV exactly at 10 B is inside the soft range (>= low)."""
        c = _make_candidate(adtv_20d=10e9)
        switches = PCIMSwitches(enable_adtv_soft_penalty=True)
        mult = compute_soft_multiplier(c, five_day_return=0.05, switches=switches)
        assert mult == pytest.approx(0.5)
