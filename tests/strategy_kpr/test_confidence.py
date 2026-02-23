"""Tests for KPR confidence calculation."""

import pytest
from unittest.mock import MagicMock

from strategy_kpr.core.fsm import compute_confidence
from strategy_kpr.signals.investor import InvestorSignal
from strategy_kpr.signals.micro import MicroSignal
from strategy_kpr.signals.program import ProgramSignal


class TestComputeConfidenceRED:
    """Tests for RED confidence conditions."""

    @pytest.fixture
    def mock_switches(self):
        """Create mock switches."""
        switches = MagicMock()
        switches.conflict_is_red = True
        switches.log_would_block = MagicMock()
        return switches

    def test_investor_distribute_is_red(self, mock_switches):
        """Test investor DISTRIBUTE returns RED."""
        result = compute_confidence(
            investor=InvestorSignal.DISTRIBUTE,
            micro=MicroSignal.ACCUMULATE,
            program=ProgramSignal.ACCUMULATE,
            prog_avail=True,
            switches=mock_switches,
        )
        assert result == "RED"

    def test_micro_distribute_is_red(self, mock_switches):
        """Test micro DISTRIBUTE returns RED."""
        result = compute_confidence(
            investor=InvestorSignal.STRONG,
            micro=MicroSignal.DISTRIBUTE,
            program=ProgramSignal.ACCUMULATE,
            prog_avail=True,
            switches=mock_switches,
        )
        assert result == "RED"

    def test_program_distribute_is_red(self, mock_switches):
        """Test program DISTRIBUTE returns RED when available."""
        result = compute_confidence(
            investor=InvestorSignal.STRONG,
            micro=MicroSignal.ACCUMULATE,
            program=ProgramSignal.DISTRIBUTE,
            prog_avail=True,
            switches=mock_switches,
        )
        assert result == "RED"

    def test_program_distribute_not_red_when_unavailable(self, mock_switches):
        """Test program DISTRIBUTE ignored when unavailable."""
        result = compute_confidence(
            investor=InvestorSignal.STRONG,
            micro=MicroSignal.ACCUMULATE,
            program=ProgramSignal.DISTRIBUTE,
            prog_avail=False,
            switches=mock_switches,
        )
        # Should be GREEN (both investor and micro positive in 2-pillar mode)
        assert result == "GREEN"

    def test_investor_conflict_is_red_with_switch(self, mock_switches):
        """Test investor CONFLICT returns RED when switch enabled."""
        mock_switches.conflict_is_red = True

        result = compute_confidence(
            investor=InvestorSignal.CONFLICT,
            micro=MicroSignal.ACCUMULATE,
            program=ProgramSignal.ACCUMULATE,
            prog_avail=True,
            switches=mock_switches,
        )
        assert result == "RED"

    def test_investor_conflict_is_yellow_without_switch(self, mock_switches):
        """Test investor CONFLICT returns YELLOW when switch disabled."""
        mock_switches.conflict_is_red = False

        result = compute_confidence(
            investor=InvestorSignal.CONFLICT,
            micro=MicroSignal.ACCUMULATE,
            program=ProgramSignal.ACCUMULATE,
            prog_avail=True,
            switches=mock_switches,
            symbol="005930",
        )
        assert result == "YELLOW"
        mock_switches.log_would_block.assert_called_once()


class TestComputeConfidenceTwoPillar:
    """Tests for two-pillar mode (program unavailable)."""

    @pytest.fixture
    def mock_switches(self):
        """Create mock switches."""
        switches = MagicMock()
        switches.conflict_is_red = True
        switches.log_would_block = MagicMock()
        return switches

    def test_both_positive_is_green(self, mock_switches):
        """Test both pillars positive returns GREEN."""
        result = compute_confidence(
            investor=InvestorSignal.STRONG,
            micro=MicroSignal.ACCUMULATE,
            program=ProgramSignal.UNAVAILABLE,
            prog_avail=False,
            switches=mock_switches,
        )
        assert result == "GREEN"

    def test_investor_strong_is_green_two_pillar(self, mock_switches):
        """Test investor STRONG in two-pillar mode returns GREEN (micro doesn't gate)."""
        result = compute_confidence(
            investor=InvestorSignal.STRONG,
            micro=MicroSignal.NEUTRAL,
            program=ProgramSignal.UNAVAILABLE,
            prog_avail=False,
            switches=mock_switches,
        )
        assert result == "GREEN"

    def test_micro_only_is_yellow(self, mock_switches):
        """Test only micro positive returns YELLOW."""
        result = compute_confidence(
            investor=InvestorSignal.NEUTRAL,
            micro=MicroSignal.ACCUMULATE,
            program=ProgramSignal.UNAVAILABLE,
            prog_avail=False,
            switches=mock_switches,
        )
        assert result == "YELLOW"

    def test_neither_positive_is_yellow(self, mock_switches):
        """Test neither positive returns YELLOW."""
        result = compute_confidence(
            investor=InvestorSignal.NEUTRAL,
            micro=MicroSignal.NEUTRAL,
            program=ProgramSignal.UNAVAILABLE,
            prog_avail=False,
            switches=mock_switches,
        )
        assert result == "YELLOW"


class TestComputeConfidenceThreePillar:
    """Tests for three-pillar mode (program available)."""

    @pytest.fixture
    def mock_switches(self):
        """Create mock switches."""
        switches = MagicMock()
        switches.conflict_is_red = True
        switches.log_would_block = MagicMock()
        return switches

    def test_all_three_positive_is_green(self, mock_switches):
        """Test all three pillars positive returns GREEN."""
        result = compute_confidence(
            investor=InvestorSignal.STRONG,
            micro=MicroSignal.ACCUMULATE,
            program=ProgramSignal.ACCUMULATE,
            prog_avail=True,
            switches=mock_switches,
        )
        assert result == "GREEN"

    def test_two_of_three_positive_is_green(self, mock_switches):
        """Test 2-of-3 pillars positive returns GREEN."""
        # investor + micro positive
        result = compute_confidence(
            investor=InvestorSignal.STRONG,
            micro=MicroSignal.ACCUMULATE,
            program=ProgramSignal.NEUTRAL,
            prog_avail=True,
            switches=mock_switches,
        )
        assert result == "GREEN"

        # investor + program positive
        result = compute_confidence(
            investor=InvestorSignal.STRONG,
            micro=MicroSignal.NEUTRAL,
            program=ProgramSignal.ACCUMULATE,
            prog_avail=True,
            switches=mock_switches,
        )
        assert result == "GREEN"

        # micro + program positive
        result = compute_confidence(
            investor=InvestorSignal.NEUTRAL,
            micro=MicroSignal.ACCUMULATE,
            program=ProgramSignal.ACCUMULATE,
            prog_avail=True,
            switches=mock_switches,
        )
        assert result == "GREEN"

    def test_one_of_three_positive_is_yellow(self, mock_switches):
        """Test 1-of-3 pillars positive returns YELLOW."""
        result = compute_confidence(
            investor=InvestorSignal.STRONG,
            micro=MicroSignal.NEUTRAL,
            program=ProgramSignal.NEUTRAL,
            prog_avail=True,
            switches=mock_switches,
        )
        assert result == "YELLOW"

    def test_none_positive_is_yellow(self, mock_switches):
        """Test no pillars positive returns YELLOW."""
        result = compute_confidence(
            investor=InvestorSignal.NEUTRAL,
            micro=MicroSignal.NEUTRAL,
            program=ProgramSignal.NEUTRAL,
            prog_avail=True,
            switches=mock_switches,
        )
        assert result == "YELLOW"


class TestComputeConfidenceEdgeCases:
    """Tests for edge cases in confidence calculation."""

    @pytest.fixture
    def mock_switches(self):
        """Create mock switches."""
        switches = MagicMock()
        switches.conflict_is_red = True
        switches.log_would_block = MagicMock()
        return switches

    def test_distribute_takes_priority_over_positive(self, mock_switches):
        """Test DISTRIBUTE takes priority even with other positives."""
        result = compute_confidence(
            investor=InvestorSignal.STRONG,
            micro=MicroSignal.DISTRIBUTE,  # This should make it RED
            program=ProgramSignal.ACCUMULATE,
            prog_avail=True,
            switches=mock_switches,
        )
        assert result == "RED"

    def test_program_unavailable_signal_treated_as_unavailable(self, mock_switches):
        """Test ProgramSignal.UNAVAILABLE treated same as prog_avail=False."""
        result = compute_confidence(
            investor=InvestorSignal.STRONG,
            micro=MicroSignal.ACCUMULATE,
            program=ProgramSignal.UNAVAILABLE,
            prog_avail=True,  # prog_avail True but signal UNAVAILABLE
            switches=mock_switches,
        )
        # Should use 2-pillar mode
        assert result == "GREEN"
