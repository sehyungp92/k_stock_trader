"""Tests for _build_kpr_filter_decisions margin_pct computation."""

import pytest

from strategy_kpr.core.fsm import (
    _build_kpr_filter_decisions,
    _signal_margin,
    _INVESTOR_MARGIN,
    _MICRO_MARGIN,
    _PROGRAM_MARGIN,
    _CONFIDENCE_MARGIN,
)
from strategy_kpr.signals.investor import InvestorSignal
from strategy_kpr.signals.micro import MicroSignal
from strategy_kpr.signals.program import ProgramSignal


# ---------------------------------------------------------------------------
# _signal_margin helper
# ---------------------------------------------------------------------------

class TestSignalMargin:
    def test_enum_member(self):
        assert _signal_margin(MicroSignal.ACCUMULATE, _MICRO_MARGIN) == 100

    def test_string_fallback(self):
        """Non-enum strings should match by value."""
        assert _signal_margin("ACCUMULATE", _MICRO_MARGIN) == 100

    def test_unknown_signal_returns_zero(self):
        assert _signal_margin("SOMETHING_NEW", _MICRO_MARGIN) == 0


# ---------------------------------------------------------------------------
# Full function — ACCUMULATE / passing scenario
# ---------------------------------------------------------------------------

class TestFilterDecisionsAccumulate:
    """All signals at their strongest (passing) values."""

    def setup_method(self):
        self.decisions = _build_kpr_filter_decisions(
            investor=InvestorSignal.STRONG,
            micro=MicroSignal.ACCUMULATE,
            program=ProgramSignal.ACCUMULATE,
            prog_avail=True,
            confidence="GREEN",
        )
        self.by_filter = {d["filter"]: d for d in self.decisions}

    def test_all_pass(self):
        for d in self.decisions:
            assert d["passed"], f"{d['filter']} should pass"

    def test_investor_margin_positive(self):
        assert self.by_filter["investor_signal"]["margin_pct"] == 100

    def test_micro_margin_positive(self):
        assert self.by_filter["micro_signal"]["margin_pct"] == 100

    def test_program_margin_positive(self):
        assert self.by_filter["program_signal"]["margin_pct"] == 100

    def test_confidence_margin_positive(self):
        assert self.by_filter["confidence"]["margin_pct"] == 100


# ---------------------------------------------------------------------------
# CONFLICT / borderline scenario
# ---------------------------------------------------------------------------

class TestFilterDecisionsConflict:
    """Signals at their neutral / borderline values."""

    def setup_method(self):
        self.decisions = _build_kpr_filter_decisions(
            investor=InvestorSignal.NEUTRAL,
            micro=MicroSignal.NEUTRAL,
            program=ProgramSignal.NEUTRAL,
            prog_avail=True,
            confidence="YELLOW",
        )
        self.by_filter = {d["filter"]: d for d in self.decisions}

    def test_investor_margin_negative(self):
        # NEUTRAL investor does not pass (threshold is STRONG)
        assert self.by_filter["investor_signal"]["margin_pct"] == -50

    def test_micro_margin_borderline(self):
        assert self.by_filter["micro_signal"]["margin_pct"] == 0

    def test_program_margin_borderline(self):
        assert self.by_filter["program_signal"]["margin_pct"] == 0

    def test_confidence_margin_moderate(self):
        # YELLOW still passes (not RED) but margin is moderate
        assert self.by_filter["confidence"]["margin_pct"] == 50


# ---------------------------------------------------------------------------
# DISTRIBUTE / strong fail scenario
# ---------------------------------------------------------------------------

class TestFilterDecisionsDistribute:
    """Signals at their worst (failing) values."""

    def setup_method(self):
        self.decisions = _build_kpr_filter_decisions(
            investor=InvestorSignal.UNAVAILABLE,
            micro=MicroSignal.DISTRIBUTE,
            program=ProgramSignal.DISTRIBUTE,
            prog_avail=True,
            confidence="RED",
        )
        self.by_filter = {d["filter"]: d for d in self.decisions}

    def test_none_pass(self):
        for d in self.decisions:
            assert not d["passed"], f"{d['filter']} should fail"

    def test_investor_margin_strongly_negative(self):
        assert self.by_filter["investor_signal"]["margin_pct"] == -100

    def test_micro_margin_strongly_negative(self):
        assert self.by_filter["micro_signal"]["margin_pct"] == -100

    def test_program_margin_strongly_negative(self):
        assert self.by_filter["program_signal"]["margin_pct"] == -100

    def test_confidence_margin_strongly_negative(self):
        assert self.by_filter["confidence"]["margin_pct"] == -100


# ---------------------------------------------------------------------------
# prog_avail=False — program filter omitted
# ---------------------------------------------------------------------------

class TestFilterDecisionsNoProgramPillar:
    def test_no_program_filter_when_unavailable(self):
        decisions = _build_kpr_filter_decisions(
            investor=InvestorSignal.STRONG,
            micro=MicroSignal.ACCUMULATE,
            program=ProgramSignal.UNAVAILABLE,
            prog_avail=False,
            confidence="GREEN",
        )
        filters = [d["filter"] for d in decisions]
        assert "program_signal" not in filters

    def test_still_has_three_decisions(self):
        decisions = _build_kpr_filter_decisions(
            investor=InvestorSignal.STRONG,
            micro=MicroSignal.ACCUMULATE,
            program=ProgramSignal.UNAVAILABLE,
            prog_avail=False,
            confidence="GREEN",
        )
        assert len(decisions) == 3  # investor, micro, confidence


# ---------------------------------------------------------------------------
# Backward compatibility — structure unchanged
# ---------------------------------------------------------------------------

class TestFilterDecisionsStructure:
    """Verify the dict keys and types are unchanged."""

    def test_required_keys_present(self):
        decisions = _build_kpr_filter_decisions(
            investor=InvestorSignal.STRONG,
            micro=MicroSignal.ACCUMULATE,
            program=ProgramSignal.ACCUMULATE,
            prog_avail=True,
            confidence="GREEN",
        )
        required_keys = {"filter", "threshold", "actual", "passed", "margin_pct"}
        for d in decisions:
            assert set(d.keys()) == required_keys

    def test_margin_pct_is_numeric(self):
        decisions = _build_kpr_filter_decisions(
            investor=InvestorSignal.NEUTRAL,
            micro=MicroSignal.DISTRIBUTE,
            program=ProgramSignal.UNAVAILABLE,
            prog_avail=True,
            confidence="RED",
        )
        for d in decisions:
            assert isinstance(d["margin_pct"], (int, float))
