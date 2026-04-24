"""OperationalPulse — lightweight in-memory counter for strategy health verdicts.

Strategies increment counters at key decision points; emits a structured
summary log every 5 minutes so operators can distinguish "no signals today"
from "something is silently broken."
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, Optional

from loguru import logger


# Verdict constants
VERDICT_HEALTHY = "HEALTHY"
VERDICT_DEGRADED = "DEGRADED"
VERDICT_DATA_BLACKOUT = "DATA_BLACKOUT"
VERDICT_OMS_DOWN = "OMS_DOWN"
VERDICT_NO_SIGNALS = "NO_SIGNALS"
VERDICT_IDLE = "IDLE"

# Thresholds
MD_OK_HEALTHY = 0.95
MD_OK_DEGRADED = 0.50
OMS_FAIL_DEGRADED = 0.20

EMIT_INTERVAL_SEC = 300  # 5 minutes


class OperationalPulse:
    """In-memory counter for strategy operational health."""

    def __init__(self, strategy_id: str, version: str = ""):
        self._strategy_id = strategy_id
        self._version = version
        self._counters: Dict[str, int] = defaultdict(int)
        self._tags: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._phase: str = "INIT"
        self._last_emit_ts: float = time.time()

    def count(self, key: str, n: int = 1, tag: Optional[str] = None) -> None:
        """Increment counter, optionally with a tag breakdown."""
        self._counters[key] += n
        if tag:
            self._tags[key][tag] += n

    def set_phase(self, phase: str) -> None:
        """Set current phase name (for phase-aware verdicts)."""
        self._phase = phase

    def maybe_emit(self) -> bool:
        """If emit interval has elapsed, log structured pulse and reset. Returns True if emitted."""
        now = time.time()
        elapsed = now - self._last_emit_ts
        if elapsed < EMIT_INTERVAL_SEC:
            return False
        self._emit(elapsed)
        self._reset()
        self._last_emit_ts = now
        return True

    def snapshot(self) -> dict:
        """Return current counters without resetting (for heartbeat enrichment)."""
        md_attempt = self._counters.get("md.attempt", 0)
        md_ok = self._counters.get("md.ok", 0)
        md_ok_pct = (md_ok / md_attempt * 100) if md_attempt > 0 else 100.0

        return {
            "verdict": self._compute_verdict(),
            "md_ok_pct": round(md_ok_pct, 1),
            "signals_eval": self._counters.get("signal.eval", 0),
            "phase": self._phase,
            "counters": dict(self._counters),
        }

    def _compute_verdict(self) -> str:
        """Compute health verdict from current counters."""
        cycles = self._counters.get("cycle", 0)
        md_attempt = self._counters.get("md.attempt", 0)
        md_ok = self._counters.get("md.ok", 0)
        signal_eval = self._counters.get("signal.eval", 0)
        signal_hit = self._counters.get("signal.hit", 0)
        oms_call = self._counters.get("oms.call", 0)
        oms_ok = self._counters.get("oms.ok", 0)
        oms_fail = self._counters.get("oms.fail", 0)

        # IDLE: zero cycles (strategy sleeping between phases)
        if cycles == 0:
            return VERDICT_IDLE

        # OMS_DOWN: all OMS calls failed
        if oms_call > 0 and oms_ok == 0:
            return VERDICT_OMS_DOWN

        # Market data rates
        md_ok_rate = (md_ok / md_attempt) if md_attempt > 0 else 1.0
        oms_fail_rate = (oms_fail / oms_call) if oms_call > 0 else 0.0

        # DATA_BLACKOUT: md ok rate < 50%
        if md_attempt > 0 and md_ok_rate < MD_OK_DEGRADED:
            return VERDICT_DATA_BLACKOUT

        # DEGRADED: md ok rate 50-95% OR OMS fail rate > 20%
        if (md_attempt > 0 and md_ok_rate < MD_OK_HEALTHY) or oms_fail_rate > OMS_FAIL_DEGRADED:
            return VERDICT_DEGRADED

        # NO_SIGNALS: healthy evaluation, zero hits
        if signal_eval > 0 and signal_hit == 0:
            return VERDICT_NO_SIGNALS

        return VERDICT_HEALTHY

    def _emit(self, elapsed: float) -> None:
        """Emit structured pulse log."""
        md_attempt = self._counters.get("md.attempt", 0)
        md_ok = self._counters.get("md.ok", 0)
        md_fail = self._counters.get("md.fail", 0)
        md_skip = self._counters.get("md.skip_budget", 0)
        md_ok_pct = (md_ok / md_attempt * 100) if md_attempt > 0 else 100.0

        signal_eval = self._counters.get("signal.eval", 0)
        signal_hit = self._counters.get("signal.hit", 0)
        signal_blocked = self._counters.get("signal.blocked", 0)

        oms_call = self._counters.get("oms.call", 0)
        oms_ok = self._counters.get("oms.ok", 0)

        verdict = self._compute_verdict()

        # Build blocked tag breakdown
        blocked_tags = self._tags.get("signal.blocked", {})
        blocked_str = ", ".join(f"{k}={v}" for k, v in sorted(blocked_tags.items()))
        blocked_detail = f" [{blocked_str}]" if blocked_str else ""

        version_str = f" {self._version}" if self._version else ""

        md_line = f"md: {md_ok}/{md_attempt} ok ({md_ok_pct:.1f}%)"
        if md_skip > 0:
            md_line += f", {md_skip} skip_budget"
        if md_fail > 0 and md_fail != (md_attempt - md_ok):
            md_line += f", {md_fail} fail"

        signals_line = f"signals: {signal_eval} eval, {signal_hit} hit, {signal_blocked} blocked{blocked_detail}"
        oms_line = f"oms: {oms_ok}/{oms_call} ok"

        logger.info(
            f"PULSE | {self._strategy_id}{version_str} | {elapsed:.0f}s | phase={self._phase}\n"
            f"  {md_line} | {signals_line}\n"
            f"  {oms_line} | verdict: {verdict}"
        )

    def _reset(self) -> None:
        """Reset counters after emission."""
        self._counters.clear()
        self._tags.clear()
