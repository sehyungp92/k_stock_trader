"""KPR Alpha Engine FSM."""

import time as time_module
from datetime import datetime, time
from typing import Optional
from loguru import logger

from .state import SymbolState, FSMState
from .setup_detection import detect_setup
from ..signals.investor import InvestorSignal
from ..signals.program import ProgramSignal
from ..signals.micro import MicroSignal
from ..config.constants import (
    STRATEGY_ID, ENTRY_START, ENTRY_END, LUNCH_START, LUNCH_END,
    BASE_RISK_PCT, GREEN_SIZE_MULT, YELLOW_SIZE_MULT,
    TOD_BRACKETS, TOD_DEFAULT_MULT, BASE_ACCEPT_CLOSES,
    STALE_SIZE_PENALTY, FLOW_STALE_DEFAULT,
)
from ..config.switches import kpr_switches
from oms_client import Intent, IntentType, Urgency, TimeHorizon, IntentConstraints, RiskPayload


def in_lunch(now: datetime, switches=None) -> bool:
    """
    Check if in lunch block period.

    Args:
        now: Current datetime
        switches: Optional KPRSwitches instance (defaults to global)

    Returns:
        True if in lunch block and lunch block is enabled
    """
    if switches is None:
        switches = kpr_switches

    t = now.time()
    is_lunch_time = time(LUNCH_START[0], LUNCH_START[1]) <= t <= time(LUNCH_END[0], LUNCH_END[1])

    if switches.enable_lunch_block:
        return is_lunch_time
    else:
        # Permissive: don't block during lunch, but log would-block
        if is_lunch_time:
            switches.log_would_block(
                "GLOBAL",
                "LUNCH_BLOCK",
                t.strftime("%H:%M"),
                f"{LUNCH_START[0]:02d}:{LUNCH_START[1]:02d}-{LUNCH_END[0]:02d}:{LUNCH_END[1]:02d}",
            )
        return False


def after_entry_end(now: datetime) -> bool:
    return now.time() > time(ENTRY_END[0], ENTRY_END[1])


def get_tod_multiplier(t: time, switches=None) -> float:
    """
    Get time-of-day sizing multiplier from TOD_BRACKETS.

    Args:
        t: Current time
        switches: Optional KPRSwitches instance (defaults to global)

    Returns:
        Size multiplier for current time period
    """
    if switches is None:
        switches = kpr_switches

    for (sh, sm), (eh, em), mult in TOD_BRACKETS:
        if time(sh, sm) <= t < time(eh, em):
            # Check if this is the late session bracket (14:00+)
            if (sh, sm) == (14, 0):
                # Use switch-configurable late session multiplier
                if switches.tod_late_mult != mult:
                    # Log would-block if using permissive (higher) mult
                    if switches.tod_late_mult > mult:
                        kpr_switches.log_would_block(
                            "GLOBAL",
                            "TOD_LATE_MULT",
                            switches.tod_late_mult,
                            mult,
                            {"time": t.strftime("%H:%M")},
                        )
                return switches.tod_late_mult
            return mult
    return TOD_DEFAULT_MULT


def compute_confidence(investor, micro, program, prog_avail: bool, switches=None, symbol: str = "") -> str:
    """
    3-pillar confidence with AUTO fallback.

    Args:
        investor: InvestorSignal
        micro: MicroSignal
        program: ProgramSignal
        prog_avail: Whether program signal is available
        switches: Optional KPRSwitches instance (defaults to global)
        symbol: Stock code for logging

    RED: any pillar is DISTRIBUTE, investor CONFLICT (if conflict_is_red=True)
    If program unavailable -> two-pillar mode:
        GREEN requires investor STRONG (micro still blocks via RED on DISTRIBUTE)
    If program available -> 2-of-3 positive -> GREEN
    Otherwise YELLOW.
    """
    if switches is None:
        switches = kpr_switches

    # RED: any distribute signal
    if investor == InvestorSignal.DISTRIBUTE:
        return "RED"
    if micro == MicroSignal.DISTRIBUTE:
        return "RED"
    if prog_avail and program == ProgramSignal.DISTRIBUTE:
        return "RED"

    # CONFLICT handling: configurable via switch
    if investor == InvestorSignal.CONFLICT:
        if switches.conflict_is_red:
            return "RED"
        else:
            # Permissive: CONFLICT -> YELLOW, log would-block
            switches.log_would_block(
                symbol or "UNKNOWN",
                "CONFLICT_SIGNAL",
                "YELLOW",
                "RED",
                {"investor_signal": investor.name},
            )
            return "YELLOW"

    if not prog_avail or program == ProgramSignal.UNAVAILABLE:
        # Two-pillar mode: need both investor + micro positive
        if investor == InvestorSignal.STRONG:
            return "GREEN"
        return "YELLOW"

    # Three-pillar mode: 2-of-3 positive -> GREEN
    positives = (
        (investor == InvestorSignal.STRONG)
        + (micro == MicroSignal.ACCUMULATE)
        + (program == ProgramSignal.ACCUMULATE)
    )
    return "GREEN" if positives >= 2 else "YELLOW"


async def alpha_step(s: SymbolState, bar: dict, vwap: float, now: datetime,
                     investor_sig, micro_sig, program_sig, prog_avail: bool,
                     regime_ok: bool, has_tick: bool, flow_stale: bool,
                     is_late: bool, atr: float, equity: float, oms,
                     drift_monitor=None, sector_exposure=None,
                     investor_age: float = 0.0) -> Optional[str]:

    # Drift gate: block all entries when drift detected
    if drift_monitor and drift_monitor.global_trade_block:
        return None

    high = float(bar.get('high', 0))
    low = float(bar.get('low', 0))
    close = float(bar.get('close', 0))
    bar_time = bar.get('timestamp', now)

    if high > s.hod:
        s.hod, s.hod_time = high, bar_time
    s.lod = min(s.lod, low)
    s.vwap = vwap

    if now.time() < time(ENTRY_START[0], ENTRY_START[1]) or in_lunch(now) or after_entry_end(now) or not regime_ok:
        return None

    # Invalidation: stop breach
    if s.fsm in (FSMState.SETUP_DETECTED, FSMState.ACCEPTING) and s.stop_level and low <= s.stop_level:
        s.fsm = FSMState.INVALIDATED
        return None

    # IN_POSITION is handled by exit engine, not here
    if s.fsm == FSMState.IN_POSITION:
        return None

    # IDLE -> SETUP_DETECTED
    if s.fsm == FSMState.IDLE:
        if detect_setup(s, bar, vwap, bar_time):
            s.fsm = FSMState.SETUP_DETECTED
            logger.info(f"{s.code}: Setup detected")
        return None

    # SETUP_DETECTED -> ACCEPTING (price reclaims setup_low)
    if s.fsm == FSMState.SETUP_DETECTED and s.reclaim_level and high >= s.reclaim_level:
        s.fsm = FSMState.ACCEPTING

        # Acceptance adders: proxy, program unavail, unfavorable regime
        # Note: flow_stale and is_late adders are optional (redundant with size penalties)
        adders = (not has_tick) + (not prog_avail) + (not regime_ok)

        # Optional stale flow acceptance adder (default: disabled, redundant with 0.85x size penalty)
        if kpr_switches.enable_stale_flow_acceptance_adder:
            adders += flow_stale
        elif flow_stale:
            # Log would-block: stale flow would have added +1 acceptance close
            kpr_switches.log_would_block(
                s.code,
                "STALE_FLOW_ACCEPT_ADDER",
                0,
                1,
                {"note": "Size penalty (0.85x) still applies"},
            )

        # Optional late acceptance adder (default: disabled, redundant with TOD sizing)
        if kpr_switches.enable_late_acceptance_adder:
            adders += is_late
        elif is_late:
            # Log would-block: is_late would have added +1 acceptance close
            kpr_switches.log_would_block(
                s.code,
                "LATE_ACCEPT_ADDER",
                0,
                1,
                {"note": "TOD sizing penalty still applies"},
            )

        s.required_closes = BASE_ACCEPT_CLOSES + adders
        return None

    # ACCEPTING -> entry
    if s.fsm == FSMState.ACCEPTING:
        if close >= s.reclaim_level:
            s.accept_closes += 1

        if s.accept_closes >= s.required_closes:
            # Sector cap check before entry
            if sector_exposure and not sector_exposure.can_enter(s.code, 1, close, equity):
                logger.debug(f"{s.code}: Sector cap reached for {sector_exposure.get_sector(s.code)}")
                return None

            confidence = compute_confidence(investor_sig, micro_sig, program_sig, prog_avail, symbol=s.code)
            if confidence == "RED":
                s.fsm = FSMState.INVALIDATED
                return None

            # Downgrade GREEN to YELLOW when investor flow is stale
            if investor_age > FLOW_STALE_DEFAULT and confidence == "GREEN":
                confidence = "YELLOW"
                logger.debug(f"{s.code}: Downgrade to YELLOW due to stale investor flow")

            stop = s.stop_level or close * 0.98
            risk = equity * BASE_RISK_PCT
            qty = int(risk / max(close - stop, 0.01))
            mult = GREEN_SIZE_MULT if confidence == "GREEN" else YELLOW_SIZE_MULT
            tod = get_tod_multiplier(now.time())

            # Stale investor flow size penalty
            stale_mult = STALE_SIZE_PENALTY if investor_age > FLOW_STALE_DEFAULT else 1.0
            qty = int(qty * mult * tod * stale_mult)

            if qty <= 0:
                return None

            # Unique signal_hash for idempotency (prevents dedup of re-entries same day)
            setup_ts = s.setup_time.strftime("%H%M") if s.setup_time else "0000"
            signal_hash = f"{setup_ts}_{int(s.setup_low or 0)}"
            rationale = "panic_reclaim" if s.setup_type == "panic" else "drift_reclaim"

            intent = Intent(
                intent_type=IntentType.ENTER, strategy_id=STRATEGY_ID, symbol=s.code,
                desired_qty=qty, urgency=Urgency.NORMAL, time_horizon=TimeHorizon.INTRADAY,
                signal_hash=signal_hash,
                constraints=IntentConstraints(limit_price=close),
                risk_payload=RiskPayload(entry_px=close, stop_px=stop, confidence=confidence, rationale_code=rationale),
            )
            result = await oms.submit_intent(intent)
            if result.status.name in ("EXECUTED", "APPROVED"):
                s.fsm = FSMState.IN_POSITION
                s.entry_px = close
                s.entry_ts = now
                s.qty = qty
                s.remaining_qty = qty
                s.confidence = confidence
                s.max_price = close
                s.trail_stop = 0.0
                s.partial_filled = False
                # Track order for timeout detection
                s.entry_order_id = intent.intent_id
                s.order_submit_ts = time_module.time()
                # Track sector exposure
                if sector_exposure:
                    sector_exposure.reserve(s.code, qty, close)
                return intent.intent_id
    return None
