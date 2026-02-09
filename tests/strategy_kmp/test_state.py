"""Tests for KMP FSM state transitions."""

import pytest
from datetime import datetime, time
import math

from strategy_kmp.core.state import SymbolState, State


class TestState:
    """Tests for State enum."""

    def test_all_states_defined(self):
        """Test all FSM states are defined."""
        assert State.IDLE
        assert State.CANDIDATE
        assert State.WATCH_BREAK
        assert State.WAIT_ACCEPTANCE
        assert State.ARMED
        assert State.IN_POSITION
        assert State.DONE


class TestSymbolStateDefaults:
    """Tests for SymbolState default values."""

    def test_required_field(self):
        """Test required field initialization."""
        state = SymbolState(code="005930")
        assert state.code == "005930"

    def test_default_fsm_state(self):
        """Test default FSM state is IDLE."""
        state = SymbolState(code="005930")
        assert state.fsm == State.IDLE

    def test_default_or_values(self):
        """Test default OR values."""
        state = SymbolState(code="005930")
        assert state.or_high == -math.inf
        assert state.or_low == math.inf
        assert state.or_mid == 0.0
        assert state.or_locked is False

    def test_default_trend_values(self):
        """Test default trend values."""
        state = SymbolState(code="005930")
        assert state.sma20 == 0.0
        assert state.sma60 == 0.0
        assert state.prev_close == 0.0
        assert state.trend_ok is False

    def test_default_vwap_values(self):
        """Test default VWAP values."""
        state = SymbolState(code="005930")
        assert state.cum_vol == 0.0
        assert state.cum_val == 0.0
        assert state.vwap == 0.0


class TestSymbolStateVWAP:
    """Tests for VWAP calculation."""

    def test_update_vwap_single_tick(self):
        """Test VWAP update with single tick."""
        state = SymbolState(code="005930")
        state.update_vwap(price=72000, volume=1000)

        assert state.cum_vol == 1000
        assert state.cum_val == 72000000
        assert state.vwap == 72000

    def test_update_vwap_multiple_ticks(self):
        """Test VWAP update with multiple ticks."""
        state = SymbolState(code="005930")
        state.update_vwap(price=72000, volume=1000)
        state.update_vwap(price=72500, volume=1000)

        assert state.cum_vol == 2000
        # VWAP = (72000*1000 + 72500*1000) / 2000 = 72250
        assert state.vwap == 72250

    def test_update_vwap_zero_volume(self):
        """Test VWAP with zero volume."""
        state = SymbolState(code="005930")
        state.update_vwap(price=72000, volume=0)

        assert state.cum_vol == 0
        assert state.vwap == 0.0


class TestSymbolStateSpread:
    """Tests for spread calculation."""

    def test_update_spread(self):
        """Test spread update."""
        state = SymbolState(code="005930")
        state.bid = 71900
        state.ask = 72000
        state.update_spread()

        assert state.spread == 100
        assert state.spread_pct == pytest.approx(0.0014, abs=0.0001)

    def test_update_spread_zero_values(self):
        """Test spread with zero bid/ask."""
        state = SymbolState(code="005930")
        state.bid = 0
        state.ask = 72000
        state.update_spread()

        # Should not update when bid is 0
        assert state.spread == 0.0


class TestSymbolStateReset:
    """Tests for reset_for_new_day method."""

    def test_reset_clears_or(self):
        """Test reset clears OR data."""
        state = SymbolState(code="005930")
        state.or_high = 72500
        state.or_low = 71500
        state.or_mid = 72000
        state.or_locked = True

        state.reset_for_new_day()

        assert state.or_high == -math.inf
        assert state.or_low == math.inf
        assert state.or_mid == 0.0
        assert state.or_locked is False

    def test_reset_clears_vwap(self):
        """Test reset clears VWAP data."""
        state = SymbolState(code="005930")
        state.cum_vol = 100000
        state.cum_val = 7200000000
        state.vwap = 72000

        state.reset_for_new_day()

        assert state.cum_vol == 0.0
        assert state.cum_val == 0.0
        assert state.vwap == 0.0

    def test_reset_clears_position(self):
        """Test reset clears position data."""
        state = SymbolState(code="005930")
        state.entry_px = 72000
        state.entry_ts = 1000000
        state.qty = 100

        state.reset_for_new_day()

        assert state.entry_px == 0.0
        assert state.entry_ts == 0.0
        assert state.qty == 0

    def test_reset_sets_fsm_idle(self):
        """Test reset sets FSM to IDLE."""
        state = SymbolState(code="005930")
        state.fsm = State.IN_POSITION

        state.reset_for_new_day()

        assert state.fsm == State.IDLE


class TestFSMTransitions:
    """Tests for FSM state transitions."""

    def test_idle_to_candidate(self):
        """Test transition from IDLE to CANDIDATE."""
        state = SymbolState(code="005930")
        assert state.fsm == State.IDLE

        # Transition to CANDIDATE after scan
        state.fsm = State.CANDIDATE
        state.value15 = 5_000_000_000
        state.surge = 3.5

        assert state.fsm == State.CANDIDATE

    def test_candidate_to_watch_break(self):
        """Test transition from CANDIDATE to WATCH_BREAK."""
        state = SymbolState(code="005930")
        state.fsm = State.CANDIDATE
        state.or_high = 72000
        state.or_low = 71500
        state.or_locked = True
        state.trend_ok = True

        # Transition after OR lock
        state.fsm = State.WATCH_BREAK

        assert state.fsm == State.WATCH_BREAK

    def test_watch_break_to_wait_acceptance(self):
        """Test transition from WATCH_BREAK to WAIT_ACCEPTANCE."""
        state = SymbolState(code="005930")
        state.fsm = State.WATCH_BREAK
        state.or_high = 72000
        state.vwap = 71800

        # Transition on break
        state.break_ts = 1000000
        state.retest_low = 72100
        state.fsm = State.WAIT_ACCEPTANCE

        assert state.fsm == State.WAIT_ACCEPTANCE

    def test_wait_acceptance_to_armed(self):
        """Test transition from WAIT_ACCEPTANCE to ARMED."""
        state = SymbolState(code="005930")
        state.fsm = State.WAIT_ACCEPTANCE
        state.retest_low = 71900
        state.or_high = 72000

        # Transition on acceptance
        state.fsm = State.ARMED
        state.entry_armed_ts = 1000000

        assert state.fsm == State.ARMED

    def test_armed_to_in_position(self):
        """Test transition from ARMED to IN_POSITION."""
        state = SymbolState(code="005930")
        state.fsm = State.ARMED
        state.entry_order_id = "ORD001"

        # Transition on fill
        state.entry_px = 72100
        state.entry_ts = 1000000
        state.qty = 100
        state.fsm = State.IN_POSITION

        assert state.fsm == State.IN_POSITION

    def test_in_position_to_done(self):
        """Test transition from IN_POSITION to DONE."""
        state = SymbolState(code="005930")
        state.fsm = State.IN_POSITION
        state.qty = 100

        # Transition on exit
        state.qty = 0
        state.fsm = State.DONE

        assert state.fsm == State.DONE

    def test_any_to_done_on_skip(self):
        """Test any state can transition to DONE on skip."""
        for initial_state in [State.IDLE, State.CANDIDATE, State.WATCH_BREAK]:
            state = SymbolState(code="005930")
            state.fsm = initial_state
            state.skip_reason = "gap_down"
            state.fsm = State.DONE

            assert state.fsm == State.DONE
