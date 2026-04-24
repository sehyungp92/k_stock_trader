"""Tests for KMP FSM functions (is_accepted, acceptance_timed_out, alpha_step entry handling)."""

import pytest
import time
import math
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock
from strategy_kmp.core.state import SymbolState, State
from strategy_kmp.core.fsm import is_accepted, acceptance_timed_out, alpha_step
from strategy_kmp.config.switches import KMPSwitches
from oms_client import IntentResult, IntentStatus


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
        s.break_ts = time.time() - 299.9  # just under 5 min (avoids timing flake)
        # Should NOT time out at exactly 300s since check is > not >=
        assert acceptance_timed_out(s) is False


def _make_submit_state():
    """Create a SymbolState ready for entry submission in WAIT_ACCEPTANCE."""
    s = SymbolState(code="005930")
    s.fsm = State.WAIT_ACCEPTANCE
    s.or_high = 70000
    s.or_low = 69000
    s.or_mid = 69500
    s.or_locked = True
    s.vwap = 69500
    s.retest_low = 69800
    s.break_ts = time.time()
    s.bid = 70100
    s.ask = 70200
    s.spread = 100
    s.spread_pct = 0.001
    s.rvol_1m = 5.0
    s.surge = 3.0
    s.vi_ref = 0
    s.prev_close = 69000
    s.structure_stop = 0
    s.hard_stop = 0
    return s


def _make_intent_result(status_name, message=""):
    """Create an IntentResult with the given status."""
    status = IntentStatus[status_name]
    return IntentResult(
        intent_id="test-intent-1",
        status=status,
        message=message,
        order_id="test-order-1" if status_name in ("EXECUTED", "APPROVED") else None,
        oms_received_at=time.time(),
        order_submitted_at=time.time(),
    )


class TestEntrySubmitHandling:
    """Tests for DEFERRED/REJECTED branching in alpha_step entry submission."""

    @pytest.mark.asyncio
    async def test_deferred_stays_retryable(self):
        """DEFERRED result should NOT transition to DONE."""
        s = _make_submit_state()
        oms = AsyncMock()
        oms.submit_intent = AsyncMock(return_value=_make_intent_result("DEFERRED", "equity=0"))
        exposure = MagicMock()
        exposure.can_enter.return_value = True

        now_kst = datetime(2026, 4, 24, 9, 20)  # within trading hours, before cutoff
        result = await alpha_step(
            s, price=70200, now_kst=now_kst, regime_ok=True,
            prog_regime="NORMAL", prog_mult=1.0, equity=100_000_000,
            atr_1m=500, last_5m_value=1e9, oms=oms, exposure=exposure,
        )
        assert result is None
        assert s.fsm == State.WAIT_ACCEPTANCE  # NOT DONE
        exposure.unreserve.assert_called_once()

    @pytest.mark.asyncio
    async def test_oms_unreachable_stays_retryable(self):
        """REJECTED with 'OMS unreachable' should NOT transition to DONE."""
        s = _make_submit_state()
        oms = AsyncMock()
        oms.submit_intent = AsyncMock(
            return_value=_make_intent_result("REJECTED", "OMS unreachable after 3 retries")
        )
        exposure = MagicMock()
        exposure.can_enter.return_value = True

        now_kst = datetime(2026, 4, 24, 9, 20)
        result = await alpha_step(
            s, price=70200, now_kst=now_kst, regime_ok=True,
            prog_regime="NORMAL", prog_mult=1.0, equity=100_000_000,
            atr_1m=500, last_5m_value=1e9, oms=oms, exposure=exposure,
        )
        assert result is None
        assert s.fsm == State.WAIT_ACCEPTANCE  # NOT DONE
        exposure.unreserve.assert_called_once()

    @pytest.mark.asyncio
    async def test_true_rejection_is_terminal(self):
        """REJECTED with other message should transition to DONE."""
        s = _make_submit_state()
        oms = AsyncMock()
        oms.submit_intent = AsyncMock(
            return_value=_make_intent_result("REJECTED", "max exposure reached")
        )
        exposure = MagicMock()
        exposure.can_enter.return_value = True

        now_kst = datetime(2026, 4, 24, 9, 20)
        result = await alpha_step(
            s, price=70200, now_kst=now_kst, regime_ok=True,
            prog_regime="NORMAL", prog_mult=1.0, equity=100_000_000,
            atr_1m=500, last_5m_value=1e9, oms=oms, exposure=exposure,
        )
        assert result is None
        assert s.fsm == State.DONE
        exposure.unreserve.assert_called_once()
