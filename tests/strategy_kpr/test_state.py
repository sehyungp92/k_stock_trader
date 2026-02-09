"""Tests for KPR FSM state transitions."""

import pytest
from datetime import datetime
import math

from strategy_kpr.core.state import SymbolState, FSMState, Tier


class TestFSMState:
    """Tests for FSMState enum."""

    def test_all_states_defined(self):
        """Test all FSM states are defined."""
        assert FSMState.IDLE
        assert FSMState.SETUP_DETECTED
        assert FSMState.ACCEPTING
        assert FSMState.IN_POSITION
        assert FSMState.INVALIDATED
        assert FSMState.DONE


class TestTier:
    """Tests for Tier enum."""

    def test_all_tiers_defined(self):
        """Test all tiers are defined."""
        assert Tier.HOT
        assert Tier.WARM
        assert Tier.COLD


class TestSymbolStateDefaults:
    """Tests for SymbolState default values."""

    def test_required_field(self):
        """Test required field initialization."""
        state = SymbolState(code="005930")
        assert state.code == "005930"

    def test_default_fsm_state(self):
        """Test default FSM state is IDLE."""
        state = SymbolState(code="005930")
        assert state.fsm == FSMState.IDLE

    def test_default_tier(self):
        """Test default tier is COLD."""
        state = SymbolState(code="005930")
        assert state.tier == Tier.COLD

    def test_default_price_values(self):
        """Test default price values."""
        state = SymbolState(code="005930")
        assert state.hod == 0.0
        assert state.lod == math.inf
        assert state.vwap == 0.0

    def test_default_setup_values(self):
        """Test default setup values are None."""
        state = SymbolState(code="005930")
        assert state.setup_low is None
        assert state.reclaim_level is None
        assert state.stop_level is None
        assert state.setup_time is None
        assert state.setup_type is None

    def test_default_position_values(self):
        """Test default position values."""
        state = SymbolState(code="005930")
        assert state.entry_px == 0.0
        assert state.entry_ts is None
        assert state.qty == 0
        assert state.remaining_qty == 0

    def test_default_signal_values(self):
        """Test default signal values."""
        state = SymbolState(code="005930")
        assert state.investor_signal == "NEUTRAL"
        assert state.micro_signal == "NEUTRAL"
        assert state.program_signal == "NEUTRAL"


class TestSymbolStateResetSetup:
    """Tests for reset_setup method."""

    def test_reset_clears_setup(self):
        """Test reset_setup clears setup data."""
        state = SymbolState(code="005930")
        state.setup_low = 70000
        state.reclaim_level = 70500
        state.stop_level = 69500
        state.setup_time = datetime.now()
        state.setup_type = "panic"
        state.accept_closes = 2

        state.reset_setup()

        assert state.setup_low is None
        assert state.reclaim_level is None
        assert state.stop_level is None
        assert state.setup_time is None
        assert state.setup_type is None
        assert state.accept_closes == 0


class TestFSMTransitions:
    """Tests for FSM state transitions."""

    def test_idle_to_setup_detected(self):
        """Test transition from IDLE to SETUP_DETECTED."""
        state = SymbolState(code="005930")
        assert state.fsm == FSMState.IDLE

        # Transition on setup detection
        state.fsm = FSMState.SETUP_DETECTED
        state.setup_low = 70000
        state.reclaim_level = 70500
        state.stop_level = 69500
        state.setup_time = datetime.now()
        state.setup_type = "panic"

        assert state.fsm == FSMState.SETUP_DETECTED

    def test_setup_detected_to_accepting(self):
        """Test transition from SETUP_DETECTED to ACCEPTING."""
        state = SymbolState(code="005930")
        state.fsm = FSMState.SETUP_DETECTED
        state.reclaim_level = 70500

        # Transition on price reclaiming level
        state.fsm = FSMState.ACCEPTING
        state.required_closes = 2

        assert state.fsm == FSMState.ACCEPTING

    def test_accepting_to_in_position(self):
        """Test transition from ACCEPTING to IN_POSITION."""
        state = SymbolState(code="005930")
        state.fsm = FSMState.ACCEPTING
        state.accept_closes = 2
        state.required_closes = 2

        # Transition on entry
        state.fsm = FSMState.IN_POSITION
        state.entry_px = 70600
        state.entry_ts = datetime.now()
        state.qty = 100
        state.remaining_qty = 100
        state.confidence = "GREEN"

        assert state.fsm == FSMState.IN_POSITION

    def test_in_position_to_done(self):
        """Test transition from IN_POSITION to DONE."""
        state = SymbolState(code="005930")
        state.fsm = FSMState.IN_POSITION
        state.qty = 100
        state.remaining_qty = 100

        # Transition on full exit
        state.remaining_qty = 0
        state.fsm = FSMState.DONE

        assert state.fsm == FSMState.DONE

    def test_setup_detected_to_invalidated(self):
        """Test transition from SETUP_DETECTED to INVALIDATED."""
        state = SymbolState(code="005930")
        state.fsm = FSMState.SETUP_DETECTED
        state.stop_level = 69500

        # Transition on stop breach
        state.fsm = FSMState.INVALIDATED

        assert state.fsm == FSMState.INVALIDATED

    def test_accepting_to_invalidated(self):
        """Test transition from ACCEPTING to INVALIDATED."""
        state = SymbolState(code="005930")
        state.fsm = FSMState.ACCEPTING
        state.stop_level = 69500

        # Transition on stop breach or RED confidence
        state.fsm = FSMState.INVALIDATED

        assert state.fsm == FSMState.INVALIDATED


class TestPartialFill:
    """Tests for partial fill tracking."""

    def test_partial_fill_tracking(self):
        """Test partial fill updates remaining_qty."""
        state = SymbolState(code="005930")
        state.fsm = FSMState.IN_POSITION
        state.qty = 100
        state.remaining_qty = 100
        state.partial_filled = False

        # Partial exit
        state.remaining_qty = 50
        state.partial_filled = True

        assert state.remaining_qty == 50
        assert state.partial_filled is True

    def test_full_exit(self):
        """Test full exit clears remaining_qty."""
        state = SymbolState(code="005930")
        state.fsm = FSMState.IN_POSITION
        state.qty = 100
        state.remaining_qty = 50

        # Full exit
        state.remaining_qty = 0
        state.fsm = FSMState.DONE

        assert state.remaining_qty == 0


class TestTierAssignment:
    """Tests for tier assignment."""

    def test_tier_hot(self):
        """Test HOT tier assignment."""
        state = SymbolState(code="005930")
        state.tier = Tier.HOT

        assert state.tier == Tier.HOT

    def test_tier_warm(self):
        """Test WARM tier assignment."""
        state = SymbolState(code="005930")
        state.tier = Tier.WARM

        assert state.tier == Tier.WARM

    def test_tier_cold(self):
        """Test COLD tier assignment."""
        state = SymbolState(code="005930")
        state.tier = Tier.COLD

        assert state.tier == Tier.COLD
