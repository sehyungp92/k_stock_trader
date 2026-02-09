"""Tests for KPR FSM functions (in_lunch, after_entry_end, get_tod_multiplier, compute_confidence)."""

import pytest
from datetime import datetime, time

from strategy_kpr.core.fsm import in_lunch, after_entry_end, get_tod_multiplier, compute_confidence
from strategy_kpr.signals.investor import InvestorSignal
from strategy_kpr.signals.micro import MicroSignal
from strategy_kpr.signals.program import ProgramSignal
from strategy_kpr.config.switches import KPRSwitches


# ---------------------------------------------------------------------------
# in_lunch
# ---------------------------------------------------------------------------
class TestInLunch:
    def test_during_lunch_with_block_enabled(self):
        switches = KPRSwitches(enable_lunch_block=True)
        now = datetime(2025, 6, 1, 12, 0, 0)  # 12:00 is between 11:20 and 13:10
        assert in_lunch(now, switches=switches) is True

    def test_before_lunch_with_block_enabled(self):
        switches = KPRSwitches(enable_lunch_block=True)
        now = datetime(2025, 6, 1, 10, 0, 0)
        assert in_lunch(now, switches=switches) is False

    def test_after_lunch_with_block_enabled(self):
        switches = KPRSwitches(enable_lunch_block=True)
        now = datetime(2025, 6, 1, 13, 30, 0)
        assert in_lunch(now, switches=switches) is False

    def test_during_lunch_with_block_disabled(self):
        switches = KPRSwitches(enable_lunch_block=False)
        now = datetime(2025, 6, 1, 12, 0, 0)
        assert in_lunch(now, switches=switches) is False

    def test_lunch_start_boundary(self):
        switches = KPRSwitches(enable_lunch_block=True)
        now = datetime(2025, 6, 1, 11, 20, 0)  # exactly at LUNCH_START
        assert in_lunch(now, switches=switches) is True

    def test_lunch_end_boundary(self):
        switches = KPRSwitches(enable_lunch_block=True)
        now = datetime(2025, 6, 1, 13, 10, 0)  # exactly at LUNCH_END
        assert in_lunch(now, switches=switches) is True

    def test_just_after_lunch_end(self):
        switches = KPRSwitches(enable_lunch_block=True)
        now = datetime(2025, 6, 1, 13, 10, 1)  # 1 second after LUNCH_END
        assert in_lunch(now, switches=switches) is False


# ---------------------------------------------------------------------------
# after_entry_end
# ---------------------------------------------------------------------------
class TestAfterEntryEnd:
    def test_before_entry_end(self):
        now = datetime(2025, 6, 1, 13, 30, 0)
        assert after_entry_end(now) is False

    def test_exactly_at_entry_end(self):
        now = datetime(2025, 6, 1, 14, 0, 0)  # ENTRY_END = (14, 0)
        # Check is > not >=, so exactly at boundary should be False
        assert after_entry_end(now) is False

    def test_after_entry_end(self):
        now = datetime(2025, 6, 1, 14, 0, 1)
        assert after_entry_end(now) is True

    def test_well_after_entry_end(self):
        now = datetime(2025, 6, 1, 15, 0, 0)
        assert after_entry_end(now) is True


# ---------------------------------------------------------------------------
# get_tod_multiplier
# ---------------------------------------------------------------------------
class TestGetTodMultiplier:
    def test_morning_peak(self):
        """09:30-10:30 bracket returns 1.0."""
        t = time(9, 45)
        assert get_tod_multiplier(t) == 1.0

    def test_mid_morning(self):
        """10:30-11:20 bracket returns 0.8."""
        t = time(10, 45)
        assert get_tod_multiplier(t) == 0.8

    def test_early_afternoon(self):
        """13:10-14:00 bracket returns 0.9."""
        t = time(13, 30)
        assert get_tod_multiplier(t) == 0.9

    def test_late_session_default(self):
        """14:00-15:30 bracket returns 0.5 by default (conservative constant)."""
        t = time(14, 30)
        # Default KPRSwitches has tod_late_mult=0.65 (permissive)
        switches = KPRSwitches()
        assert get_tod_multiplier(t, switches=switches) == 0.65

    def test_late_session_conservative(self):
        """14:00-15:30 with conservative switches returns 0.5."""
        t = time(14, 30)
        switches = KPRSwitches(tod_late_mult=0.5)
        assert get_tod_multiplier(t, switches=switches) == 0.5

    def test_late_session_custom(self):
        """14:00-15:30 with custom tod_late_mult."""
        t = time(14, 30)
        switches = KPRSwitches(tod_late_mult=0.7)
        assert get_tod_multiplier(t, switches=switches) == 0.7

    def test_outside_all_brackets(self):
        """Time outside all brackets returns TOD_DEFAULT_MULT (0.8)."""
        t = time(8, 0)
        assert get_tod_multiplier(t) == 0.8

    def test_bracket_start_boundary(self):
        """Exactly at bracket start is inclusive."""
        t = time(9, 30)
        assert get_tod_multiplier(t) == 1.0

    def test_bracket_end_boundary(self):
        """Exactly at bracket end is exclusive (moves to next or default)."""
        t = time(10, 30)
        # 10:30 is start of next bracket (10:30-11:20) -> 0.8
        assert get_tod_multiplier(t) == 0.8


# ---------------------------------------------------------------------------
# compute_confidence
# ---------------------------------------------------------------------------
class TestComputeConfidence:
    # --- RED conditions ---
    def test_investor_distribute_is_red(self):
        result = compute_confidence(
            InvestorSignal.DISTRIBUTE, MicroSignal.ACCUMULATE,
            ProgramSignal.ACCUMULATE, prog_avail=True,
        )
        assert result == "RED"

    def test_micro_distribute_is_red(self):
        result = compute_confidence(
            InvestorSignal.STRONG, MicroSignal.DISTRIBUTE,
            ProgramSignal.ACCUMULATE, prog_avail=True,
        )
        assert result == "RED"

    def test_program_distribute_is_red(self):
        result = compute_confidence(
            InvestorSignal.STRONG, MicroSignal.ACCUMULATE,
            ProgramSignal.DISTRIBUTE, prog_avail=True,
        )
        assert result == "RED"

    def test_program_distribute_ignored_when_unavail(self):
        """Program DISTRIBUTE should not trigger RED when prog_avail=False."""
        result = compute_confidence(
            InvestorSignal.STRONG, MicroSignal.ACCUMULATE,
            ProgramSignal.DISTRIBUTE, prog_avail=False,
        )
        assert result != "RED"

    # --- CONFLICT handling ---
    def test_conflict_is_red_when_strict(self):
        switches = KPRSwitches(conflict_is_red=True)
        result = compute_confidence(
            InvestorSignal.CONFLICT, MicroSignal.ACCUMULATE,
            ProgramSignal.ACCUMULATE, prog_avail=True,
            switches=switches,
        )
        assert result == "RED"

    def test_conflict_is_yellow_when_permissive(self):
        switches = KPRSwitches(conflict_is_red=False)
        result = compute_confidence(
            InvestorSignal.CONFLICT, MicroSignal.ACCUMULATE,
            ProgramSignal.ACCUMULATE, prog_avail=True,
            switches=switches,
        )
        assert result == "YELLOW"

    # --- Two-pillar mode (prog_avail=False) ---
    def test_two_pillar_green(self):
        """STRONG + ACCUMULATE in two-pillar mode -> GREEN."""
        result = compute_confidence(
            InvestorSignal.STRONG, MicroSignal.ACCUMULATE,
            ProgramSignal.UNAVAILABLE, prog_avail=False,
        )
        assert result == "GREEN"

    def test_two_pillar_yellow_weak_investor(self):
        """NEUTRAL investor in two-pillar mode -> YELLOW."""
        result = compute_confidence(
            InvestorSignal.NEUTRAL, MicroSignal.ACCUMULATE,
            ProgramSignal.UNAVAILABLE, prog_avail=False,
        )
        assert result == "YELLOW"

    def test_two_pillar_yellow_weak_micro(self):
        """NEUTRAL micro in two-pillar mode -> YELLOW."""
        result = compute_confidence(
            InvestorSignal.STRONG, MicroSignal.NEUTRAL,
            ProgramSignal.UNAVAILABLE, prog_avail=False,
        )
        assert result == "YELLOW"

    def test_two_pillar_both_neutral(self):
        """Both neutral in two-pillar mode -> YELLOW."""
        result = compute_confidence(
            InvestorSignal.NEUTRAL, MicroSignal.NEUTRAL,
            ProgramSignal.UNAVAILABLE, prog_avail=False,
        )
        assert result == "YELLOW"

    # --- Three-pillar mode (prog_avail=True) ---
    def test_three_pillar_all_positive_green(self):
        """All three positive -> GREEN."""
        result = compute_confidence(
            InvestorSignal.STRONG, MicroSignal.ACCUMULATE,
            ProgramSignal.ACCUMULATE, prog_avail=True,
        )
        assert result == "GREEN"

    def test_three_pillar_two_positive_green(self):
        """Two of three positive -> GREEN."""
        result = compute_confidence(
            InvestorSignal.STRONG, MicroSignal.NEUTRAL,
            ProgramSignal.ACCUMULATE, prog_avail=True,
        )
        assert result == "GREEN"

    def test_three_pillar_investor_micro_positive_green(self):
        """Investor + micro positive, program neutral -> GREEN."""
        result = compute_confidence(
            InvestorSignal.STRONG, MicroSignal.ACCUMULATE,
            ProgramSignal.NEUTRAL, prog_avail=True,
        )
        assert result == "GREEN"

    def test_three_pillar_micro_program_positive_green(self):
        """Micro + program positive, investor neutral -> GREEN."""
        result = compute_confidence(
            InvestorSignal.NEUTRAL, MicroSignal.ACCUMULATE,
            ProgramSignal.ACCUMULATE, prog_avail=True,
        )
        assert result == "GREEN"

    def test_three_pillar_one_positive_yellow(self):
        """Only one positive -> YELLOW."""
        result = compute_confidence(
            InvestorSignal.STRONG, MicroSignal.NEUTRAL,
            ProgramSignal.NEUTRAL, prog_avail=True,
        )
        assert result == "YELLOW"

    def test_three_pillar_none_positive_yellow(self):
        """No positives -> YELLOW."""
        result = compute_confidence(
            InvestorSignal.NEUTRAL, MicroSignal.NEUTRAL,
            ProgramSignal.NEUTRAL, prog_avail=True,
        )
        assert result == "YELLOW"

    # --- Program UNAVAILABLE with prog_avail=True falls to two-pillar ---
    def test_program_unavailable_signal_falls_to_two_pillar(self):
        """ProgramSignal.UNAVAILABLE with prog_avail=True triggers two-pillar path."""
        result = compute_confidence(
            InvestorSignal.STRONG, MicroSignal.ACCUMULATE,
            ProgramSignal.UNAVAILABLE, prog_avail=True,
        )
        # prog_avail=True but signal is UNAVAILABLE -> two-pillar fallback
        assert result == "GREEN"
