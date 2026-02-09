"""Tests for KMP FSM functions (is_accepted, acceptance_timed_out)."""

import pytest
import time
import math
from strategy_kmp.core.state import SymbolState, State
from strategy_kmp.core.fsm import is_accepted, acceptance_timed_out
from strategy_kmp.config.switches import KMPSwitches


class TestIsAccepted:
    def test_all_conditions_met_conservative(self):
        switches = KMPSwitches(require_held_support=True)
        s = SymbolState(code="005930")
        s.or_high = 72000
        s.vwap = 71800
        s.retest_low = 71850  # pulled back and held support
        assert is_accepted(s, 72100, switches=switches) is True

    def test_no_pullback_rejected(self):
        switches = KMPSwitches(require_held_support=True)
        s = SymbolState(code="005930")
        s.or_high = 72000
        s.retest_low = 72500  # no pullback (retest_low >= or_high)
        assert is_accepted(s, 72100, switches=switches) is False

    def test_no_reclaim_rejected(self):
        switches = KMPSwitches(require_held_support=True)
        s = SymbolState(code="005930")
        s.or_high = 72000
        s.vwap = 71800
        s.retest_low = 71850
        assert is_accepted(s, 71900, switches=switches) is False  # price <= or_high

    def test_held_support_fail_conservative(self):
        switches = KMPSwitches(require_held_support=True)
        s = SymbolState(code="005930")
        s.or_high = 72000
        s.vwap = 72000
        s.retest_low = 70000  # dropped too low
        assert is_accepted(s, 72100, switches=switches) is False

    def test_permissive_skips_held_support(self):
        switches = KMPSwitches(require_held_support=False)
        s = SymbolState(code="005930")
        s.or_high = 72000
        s.vwap = 72000
        s.retest_low = 70000  # dropped too low but permissive
        assert is_accepted(s, 72100, switches=switches) is True


class TestAcceptanceTimedOut:
    def test_not_timed_out(self):
        s = SymbolState(code="005930")
        s.break_ts = time.time()
        assert acceptance_timed_out(s) is False

    def test_timed_out(self):
        s = SymbolState(code="005930")
        s.break_ts = time.time() - 400  # more than 5 min
        assert acceptance_timed_out(s) is True

    def test_exactly_at_boundary(self):
        s = SymbolState(code="005930")
        s.break_ts = time.time() - 300  # exactly 5 min
        # Should NOT time out at exactly 300s since check is > not >=
        assert acceptance_timed_out(s) is False
