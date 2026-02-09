"""Tests for KMP exit conditions."""

import pytest
from unittest.mock import patch
import time

from strategy_kmp.core.state import SymbolState, State
from strategy_kmp.core.exits import (
    current_r,
    retracement_factor,
    update_trail,
    check_exit_conditions,
)


class TestCurrentR:
    """Tests for R-multiple calculation."""

    def test_positive_r(self):
        """Test positive R-multiple."""
        state = SymbolState(code="005930")
        state.entry_px = 72000
        state.structure_stop = 71000

        # Risk = 1000, gain = 2000, R = 2.0
        assert current_r(state, 74000) == 2.0

    def test_negative_r(self):
        """Test negative R-multiple."""
        state = SymbolState(code="005930")
        state.entry_px = 72000
        state.structure_stop = 71000

        # Risk = 1000, loss = -500, R = -0.5
        assert current_r(state, 71500) == -0.5

    def test_zero_r(self):
        """Test zero R-multiple at entry."""
        state = SymbolState(code="005930")
        state.entry_px = 72000
        state.structure_stop = 71000

        assert current_r(state, 72000) == 0.0

    def test_handles_zero_risk(self):
        """Test handles zero risk (shouldn't happen in practice)."""
        state = SymbolState(code="005930")
        state.entry_px = 72000
        state.structure_stop = 72000  # Same as entry

        # Should handle gracefully (uses 1e-9 as min risk)
        result = current_r(state, 73000)
        assert result > 0


class TestRetracementFactor:
    """Tests for adaptive retracement factor."""

    def test_early_factor(self):
        """Test factor in first 15 minutes."""
        # Minutes 0-15: factor = 0.5
        for minutes in [0, 5, 10, 15]:
            f = retracement_factor(minutes, "mixed", 0.1)
            assert f == 0.5

    def test_factor_ramps_after_15(self):
        """Test factor ramps after 15 minutes."""
        # At 30 minutes: 0.5 + min(0.25, 15 * 0.0167) = 0.5 + 0.25 = 0.75
        f = retracement_factor(30, "mixed", 0.1)
        assert f == pytest.approx(0.75, abs=0.01)

    def test_factor_caps_at_075(self):
        """Test factor caps at 0.75."""
        f = retracement_factor(60, "mixed", 0.1)
        assert f == pytest.approx(0.75, abs=0.01)

    def test_outflow_tightens(self):
        """Test outflow regime tightens factor."""
        f = retracement_factor(5, "outflow", 0.1)
        assert f == 0.7  # Max of base (0.5) and outflow (0.7)

    def test_negative_imbalance_tightens(self):
        """Test negative imbalance tightens factor."""
        f = retracement_factor(5, "mixed", -0.5)
        assert f == 0.7  # Max of base (0.5) and imbalance (0.7)

    def test_both_adverse_conditions(self):
        """Test both adverse conditions."""
        f = retracement_factor(5, "outflow", -0.5)
        assert f == 0.7


class TestUpdateTrail:
    """Tests for trailing stop update."""

    def test_updates_max_fav(self):
        """Test max favorable price is tracked."""
        state = SymbolState(code="005930")
        state.entry_px = 72000
        state.structure_stop = 71000
        state.entry_ts = time.time() - 600  # 10 minutes ago
        state.max_fav = 72000
        state.trail_px = 71000
        state.imb = 0.1

        update_trail(state, 73000, "mixed")

        assert state.max_fav == 73000

    def test_trail_moves_up(self):
        """Test trailing stop moves up with price."""
        state = SymbolState(code="005930")
        state.entry_px = 72000
        state.structure_stop = 71000
        state.entry_ts = time.time() - 600
        state.max_fav = 72000
        state.trail_px = 71000
        state.imb = 0.1

        update_trail(state, 74000, "mixed")

        # Gain = 2000, factor = 0.5, trail = 72000 + 2000*0.5 = 73000
        assert state.trail_px >= 73000

    def test_trail_never_drops(self):
        """Test trailing stop never moves down."""
        state = SymbolState(code="005930")
        state.entry_px = 72000
        state.structure_stop = 71000
        state.entry_ts = time.time() - 600
        state.max_fav = 74000
        state.trail_px = 73000
        state.imb = 0.1

        # Price dropped but trail should stay
        update_trail(state, 73500, "mixed")

        assert state.trail_px == 73000

    def test_trail_respects_structure_stop(self):
        """Test trail is at least structure stop."""
        state = SymbolState(code="005930")
        state.entry_px = 72000
        state.structure_stop = 71500
        state.entry_ts = time.time()
        state.max_fav = 72000
        state.trail_px = 0
        state.imb = 0.1

        update_trail(state, 72100, "mixed")

        assert state.trail_px >= state.structure_stop


class TestCheckExitConditions:
    """Tests for check_exit_conditions function."""

    @pytest.fixture
    def state(self):
        """Create symbol state for testing."""
        s = SymbolState(code="005930")
        s.entry_px = 72000
        s.entry_ts = time.time() - 1800  # 30 minutes ago
        s.structure_stop = 71000
        s.hard_stop = 70500
        s.or_high = 72000
        s.vwap = 71800
        s.max_fav = 73000
        s.trail_px = 72500
        s.imb = 0.1
        return s

    def test_risk_off_exits(self, state):
        """Test risk_off flag triggers exit."""
        should_exit, reason = check_exit_conditions(state, 73000, "mixed", risk_off=True)

        assert should_exit is True
        assert reason == "risk_off"

    def test_hard_stop_exits(self, state):
        """Test hard stop triggers exit."""
        should_exit, reason = check_exit_conditions(state, 70400, "mixed")

        assert should_exit is True
        assert reason == "hard_stop"

    def test_acceptance_failure_early(self):
        """Test acceptance failure in first 15 minutes."""
        state = SymbolState(code="005930")
        state.entry_px = 72000
        state.entry_ts = time.time() - 300  # 5 minutes ago
        state.structure_stop = 71000
        state.hard_stop = 70500
        state.or_high = 72000
        state.vwap = 71800
        state.max_fav = 72000
        state.trail_px = 71000
        state.imb = 0.1

        # Price below both OR high and VWAP
        should_exit, reason = check_exit_conditions(state, 71700, "mixed")

        assert should_exit is True
        assert reason == "acceptance_failure"

    def test_no_acceptance_failure_late(self, state):
        """Test no acceptance failure after 15 minutes."""
        # state.entry_ts is 30 minutes ago
        # Price below OR high and VWAP but past acceptance window
        should_exit, reason = check_exit_conditions(state, 71700, "mixed")

        # Should not exit for acceptance failure
        assert reason != "acceptance_failure"

    def test_stall_scratch(self, state):
        """Test stall scratch after 8+ minutes with low R."""
        state.entry_ts = time.time() - 600  # 10 minutes ago
        state.max_fav = 72000  # No progress
        state.trail_px = 71000

        # Price at entry = 0R (below 0.5R stall threshold)
        should_exit, reason = check_exit_conditions(state, 72000, "mixed")

        assert should_exit is True
        assert reason == "stall_scratch"

    def test_no_stall_early(self):
        """Test no stall scratch in first 8 minutes."""
        state = SymbolState(code="005930")
        state.entry_px = 72000
        state.entry_ts = time.time() - 300  # 5 minutes ago
        state.structure_stop = 71000
        state.hard_stop = 70500
        state.or_high = 72000
        state.vwap = 71800
        state.max_fav = 72000
        state.trail_px = 71000
        state.imb = 0.1

        # Even at 0R, should not stall early
        should_exit, reason = check_exit_conditions(state, 72000, "mixed")

        assert reason != "stall_scratch"

    def test_trailing_stop(self, state):
        """Test trailing stop triggers exit."""
        state.max_fav = 73000
        state.trail_px = 72800

        # Price below trail
        should_exit, reason = check_exit_conditions(state, 72700, "mixed")

        assert should_exit is True
        assert reason == "trailing_stop"

    def test_no_exit_profitable(self, state):
        """Test no exit when profitable and above trail."""
        state.max_fav = 73000
        state.trail_px = 72500

        should_exit, reason = check_exit_conditions(state, 73500, "mixed")

        assert should_exit is False
        assert reason == ""


class TestExitPriority:
    """Tests for exit condition priority."""

    def test_risk_off_highest_priority(self):
        """Test risk_off checked before other conditions."""
        state = SymbolState(code="005930")
        state.entry_px = 72000
        state.entry_ts = time.time()
        state.structure_stop = 71000
        state.hard_stop = 70500
        state.or_high = 72000
        state.vwap = 71800
        state.max_fav = 72000
        state.trail_px = 72500  # Would trigger trail stop
        state.imb = 0.1

        # Price below trail but risk_off set
        should_exit, reason = check_exit_conditions(state, 72400, "mixed", risk_off=True)

        assert reason == "risk_off"

    def test_hard_stop_before_acceptance(self):
        """Test hard stop before acceptance failure."""
        state = SymbolState(code="005930")
        state.entry_px = 72000
        state.entry_ts = time.time() - 300
        state.structure_stop = 71000
        state.hard_stop = 70500
        state.or_high = 72000
        state.vwap = 71800
        state.max_fav = 72000
        state.trail_px = 71000
        state.imb = 0.1

        # Price below hard stop
        should_exit, reason = check_exit_conditions(state, 70400, "mixed")

        assert reason == "hard_stop"
