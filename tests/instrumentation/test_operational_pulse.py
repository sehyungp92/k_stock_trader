"""Tests for OperationalPulse counter and verdict logic."""

import time
from unittest.mock import patch

import pytest

from instrumentation.src.operational_pulse import (
    OperationalPulse,
    VERDICT_HEALTHY,
    VERDICT_DEGRADED,
    VERDICT_DATA_BLACKOUT,
    VERDICT_OMS_DOWN,
    VERDICT_NO_SIGNALS,
    VERDICT_IDLE,
    EMIT_INTERVAL_SEC,
)


class TestCounterBasics:
    def test_count_increments(self):
        p = OperationalPulse("ALPHA")
        p.count("cycle")
        p.count("cycle")
        p.count("md.ok", 5)
        assert p._counters["cycle"] == 2
        assert p._counters["md.ok"] == 5

    def test_count_with_tag(self):
        p = OperationalPulse("ALPHA")
        p.count("signal.blocked", tag="risk_off")
        p.count("signal.blocked", tag="risk_off")
        p.count("signal.blocked", tag="regime")
        assert p._counters["signal.blocked"] == 3
        assert p._tags["signal.blocked"]["risk_off"] == 2
        assert p._tags["signal.blocked"]["regime"] == 1

    def test_set_phase(self):
        p = OperationalPulse("PCIM")
        assert p._phase == "INIT"
        p.set_phase("NIGHT_PIPELINE")
        assert p._phase == "NIGHT_PIPELINE"


class TestVerdicts:
    def test_idle_zero_cycles(self):
        p = OperationalPulse("PCIM")
        assert p._compute_verdict() == VERDICT_IDLE

    def test_healthy(self):
        p = OperationalPulse("ALPHA")
        p.count("cycle", 100)
        p.count("md.attempt", 100)
        p.count("md.ok", 98)
        p.count("signal.eval", 50)
        p.count("signal.hit", 3)
        p.count("oms.call", 10)
        p.count("oms.ok", 10)
        assert p._compute_verdict() == VERDICT_HEALTHY

    def test_no_signals(self):
        p = OperationalPulse("ALPHA")
        p.count("cycle", 100)
        p.count("md.attempt", 100)
        p.count("md.ok", 100)
        p.count("signal.eval", 50)
        # zero signal.hit
        p.count("oms.call", 5)
        p.count("oms.ok", 5)
        assert p._compute_verdict() == VERDICT_NO_SIGNALS

    def test_oms_down(self):
        p = OperationalPulse("BETA")
        p.count("cycle", 10)
        p.count("oms.call", 10)
        p.count("oms.fail", 10)
        # zero oms.ok
        assert p._compute_verdict() == VERDICT_OMS_DOWN

    def test_data_blackout(self):
        p = OperationalPulse("ALPHA")
        p.count("cycle", 100)
        p.count("md.attempt", 100)
        p.count("md.ok", 40)  # 40% < 50%
        p.count("oms.call", 5)
        p.count("oms.ok", 5)
        assert p._compute_verdict() == VERDICT_DATA_BLACKOUT

    def test_degraded_md(self):
        p = OperationalPulse("ALPHA")
        p.count("cycle", 100)
        p.count("md.attempt", 100)
        p.count("md.ok", 80)  # 80% ??between 50% and 95%
        p.count("oms.call", 5)
        p.count("oms.ok", 5)
        assert p._compute_verdict() == VERDICT_DEGRADED

    def test_degraded_oms_fail_rate(self):
        p = OperationalPulse("BETA")
        p.count("cycle", 100)
        p.count("md.attempt", 100)
        p.count("md.ok", 100)
        p.count("oms.call", 10)
        p.count("oms.ok", 7)
        p.count("oms.fail", 3)  # 30% > 20%
        assert p._compute_verdict() == VERDICT_DEGRADED

    def test_healthy_no_md_attempts(self):
        """When no md.attempt, rate defaults to 100% ??should be HEALTHY."""
        p = OperationalPulse("GAMMA")
        p.count("cycle", 10)
        p.count("oms.call", 2)
        p.count("oms.ok", 2)
        assert p._compute_verdict() == VERDICT_HEALTHY


class TestSnapshot:
    def test_snapshot_returns_dict(self):
        p = OperationalPulse("ALPHA", version="2.3.4")
        p.count("cycle", 5)
        p.count("md.attempt", 100)
        p.count("md.ok", 95)
        p.count("signal.eval", 10)
        snap = p.snapshot()
        assert snap["verdict"] == VERDICT_NO_SIGNALS  # eval>0, hit=0
        assert snap["md_ok_pct"] == 95.0
        assert snap["signals_eval"] == 10
        assert snap["phase"] == "INIT"
        assert "counters" in snap

    def test_snapshot_does_not_reset(self):
        p = OperationalPulse("ALPHA")
        p.count("cycle", 5)
        p.snapshot()
        assert p._counters["cycle"] == 5


class TestEmission:
    def test_maybe_emit_before_interval(self):
        p = OperationalPulse("ALPHA")
        p.count("cycle")
        assert p.maybe_emit() is False

    def test_maybe_emit_after_interval(self):
        p = OperationalPulse("ALPHA")
        p.count("cycle", 100)
        p.count("md.attempt", 50)
        p.count("md.ok", 50)
        # Force elapsed time past interval
        p._last_emit_ts = time.time() - EMIT_INTERVAL_SEC - 1
        assert p.maybe_emit() is True
        # Counters should be reset
        assert p._counters.get("cycle", 0) == 0

    def test_reset_clears_all(self):
        p = OperationalPulse("ALPHA")
        p.count("cycle", 10)
        p.count("signal.blocked", tag="risk_off")
        p._reset()
        assert len(p._counters) == 0
        assert len(p._tags) == 0

    def test_phase_persists_after_reset(self):
        p = OperationalPulse("PCIM")
        p.set_phase("EXECUTION")
        p._reset()
        assert p._phase == "EXECUTION"

    def test_emit_logs_pulse_format(self, caplog):
        """Verify PULSE log line contains expected prefix and structure."""
        p = OperationalPulse("ALPHA", version="2.3.4")
        p.set_phase("MAIN_LOOP")
        p.count("cycle", 100)
        p.count("md.attempt", 50)
        p.count("md.ok", 48)
        p.count("signal.eval", 20)
        p.count("signal.hit", 2)
        p.count("signal.blocked", 1, tag="risk_off")
        p.count("oms.call", 5)
        p.count("oms.ok", 5)
        p._last_emit_ts = time.time() - 300

        with patch("instrumentation.src.operational_pulse.logger") as mock_logger:
            p.maybe_emit()
            call_args = mock_logger.info.call_args[0][0]
            assert "PULSE" in call_args
            assert "ALPHA" in call_args
            assert "MAIN_LOOP" in call_args
            assert "HEALTHY" in call_args
