"""
KMP Alpha Engine FSM.
"""

from __future__ import annotations
import time
from typing import Optional
from loguru import logger

from .state import SymbolState, State
from .gates import (
    lock_or_and_filter, spread_ok, rvol_ok, vi_blocked,
    min_surge_threshold, min_surge_threshold_strict, size_time_multiplier,
    minutes_since_0916, is_past_entry_cutoff,
)
from .sizing import compute_qty, apply_liquidity_cap, apply_nav_cap, build_sizing_context
from .tick_table import tick_size, round_to_tick
from ..config.constants import (
    ACCEPT_TIMEOUT_MIN, STRATEGY_ID, HARD_STOP_ATR_MULT, BASE_RISK_PCT,
)
from ..config.switches import kmp_switches
from oms_client import Intent, IntentType, Urgency, TimeHorizon, IntentConstraints, RiskPayload


def is_accepted(s: SymbolState, price: float, switches=None) -> bool:
    """
    Check if acceptance criteria met.

    Args:
        s: Symbol state
        price: Current price
        switches: Optional KMPSwitches instance (defaults to global)

    Returns:
        True if acceptance criteria met
    """
    if switches is None:
        switches = kmp_switches

    pulled_back = s.retest_low < s.or_high
    held_support = s.retest_low >= min(s.vwap, s.or_high) * 0.998
    reclaimed = price > s.or_high

    # Core acceptance logic with switch
    if switches.require_held_support:
        # Conservative: require all three conditions
        return pulled_back and held_support and reclaimed
    else:
        # Permissive: only require pullback and reclaim
        result = pulled_back and reclaimed

        # Log would-block if permissive allowed but strict would block
        if result and not held_support:
            support_level = min(s.vwap, s.or_high) * 0.998
            switches.log_would_block(
                s.code,
                "HELD_SUPPORT",
                s.retest_low,
                support_level,
                {"vwap": s.vwap, "or_high": s.or_high},
            )

        return result


def acceptance_timed_out(s: SymbolState) -> bool:
    """Check if acceptance window expired."""
    return (time.time() - s.break_ts) > ACCEPT_TIMEOUT_MIN * 60


async def alpha_step(
    s: SymbolState,
    price: float,
    now_kst,
    regime_ok: bool,
    prog_regime: str,
    prog_mult: float,
    equity: float,
    atr_1m: float,
    last_5m_value: float,
    oms,  # OMSClient
    exposure=None,  # SectorExposure for sector cap enforcement
    max_per_sector: int = 1,
    regime_breadth_ok: bool = True,
    not_chop: bool = True,
    instr=None,  # InstrumentationKit
) -> Optional[str]:
    """
    FSM step for a symbol.

    Returns intent_id if entry intent submitted, else None.
    """
    # Global regime gate
    if not regime_ok:
        if kmp_switches.regime_blocks_progression:
            # Conservative: block all FSM states
            if "regime" not in s._gate_logged:
                logger.debug(f"{s.code}: blocked by regime gate (fsm={s.fsm.name})")
                s._gate_logged.add("regime")
                if instr:
                    instr.on_signal_blocked(
                        symbol=s.code, signal="or_break", signal_id="kmp_breakout",
                        blocked_by="regime_gate", block_reason=f"maturity=early, fsm={s.fsm.name}",
                        signal_strength=s.surge,
                    )
            if s.fsm == State.ARMED and s.entry_order_id:
                result = await oms.submit_intent(Intent(
                    intent_type=IntentType.CANCEL_ORDERS,
                    strategy_id=STRATEGY_ID,
                    symbol=s.code,
                ))
                if result.status.name in ("EXECUTED",):
                    s.entry_order_id = None
                else:
                    logger.warning(f"{s.code}: Cancel {result.status.name} - keeping order tracking")
            return None
        else:
            # Permissive: only block ARMED (cancel working order), let others progress
            if s.fsm == State.ARMED:
                if "regime" not in s._gate_logged:
                    logger.debug(f"{s.code}: regime_off, blocking ARMED state")
                    s._gate_logged.add("regime")
                if s.entry_order_id:
                    result = await oms.submit_intent(Intent(
                        intent_type=IntentType.CANCEL_ORDERS,
                        strategy_id=STRATEGY_ID,
                        symbol=s.code,
                    ))
                    if result.status.name in ("EXECUTED",):
                        s.entry_order_id = None
                    else:
                        logger.warning(f"{s.code}: Cancel {result.status.name} - keeping order tracking")
                return None
            # Fall through — WATCH_BREAK and WAIT_ACCEPTANCE proceed
            if "regime_off_allow" not in s._gate_logged:
                logger.info(f"{s.code}: regime_off but allowing progression (fsm={s.fsm.name})")
                s._gate_logged.add("regime_off_allow")

    # Entry cutoff
    if is_past_entry_cutoff(now_kst):
        if s.fsm in (State.WATCH_BREAK, State.WAIT_ACCEPTANCE, State.ARMED):
            if instr:
                instr.on_signal_blocked(
                    symbol=s.code, signal="or_break", signal_id="kmp_breakout",
                    blocked_by="entry_cutoff", block_reason=f"maturity=early, fsm={s.fsm.name}",
                    signal_strength=s.surge,
                )
            logger.info(f"{s.code}: gate entry_cutoff, {s.fsm.name} -> DONE")
            s.fsm = State.DONE
        return None

    # Lock OR at 09:15
    if not s.or_locked and now_kst.hour == 9 and now_kst.minute >= 15:
        if not lock_or_and_filter(s):
            if instr:
                instr.on_signal_blocked(
                    symbol=s.code, signal="or_break", signal_id="kmp_breakout",
                    blocked_by="or_lock_fail", block_reason="maturity=early",
                    signal_strength=s.surge,
                )
            logger.info(
                f"{s.code}: gate OR_lock fail, DONE "
                f"(or_high={s.or_high:.0f}, or_low={s.or_low:.0f}, "
                f"or_pct={((s.or_high - s.or_low) / s.or_low * 100) if s.or_low > 0 else 0:.2f}%)"
            )
            s.fsm = State.DONE
            return None
        s.fsm = State.WATCH_BREAK

    # Need OR locked and past 09:16
    if not s.or_locked:
        if "or_not_locked" not in s._gate_logged:
            logger.debug(f"{s.code}: blocked, OR not locked yet")
            s._gate_logged.add("or_not_locked")
        return None
    if now_kst.hour == 9 and now_kst.minute < 16:
        if "too_early" not in s._gate_logged:
            logger.debug(f"{s.code}: blocked, waiting for 09:16")
            s._gate_logged.add("too_early")
        return None

    # Time decay checks
    m = minutes_since_0916(now_kst)
    surge_thresh = min_surge_threshold(m)
    if s.surge < surge_thresh:
        if "surge_decay" not in s._gate_logged:
            logger.debug(
                f"{s.code}: blocked by surge_decay "
                f"(surge={s.surge:.2f} < thresh={surge_thresh:.2f}, min_at_scan={s.value15:.2f})"
            )
            s._gate_logged.add("surge_decay")
        return None

    # Log would-block: passed permissive threshold but would fail strict
    strict_thresh = min_surge_threshold_strict(m)
    if s.surge < strict_thresh:
        kmp_switches.log_would_block(
            s.code,
            "MIN_SURGE",
            s.surge,
            strict_thresh,
            {"minutes": m, "permissive_thresh": surge_thresh},
        )

    # RVol gate (optional - redundant with quality score)
    if kmp_switches.enable_rvol_hard_gate:
        if not rvol_ok(s):
            return None
    else:
        # Log would-block if permissive but would fail strict RVOL gate
        if not rvol_ok(s):
            kmp_switches.log_would_block(
                s.code,
                "RVOL_HARD_GATE",
                s.rvol_1m,
                2.0,  # RVOL_MIN
                {"note": "Quality score still weights RVOL"},
            )

    # Spread gate
    if s.bid > 0 and s.ask > 0 and not spread_ok(s):
        if "spread" not in s._gate_logged:
            logger.debug(f"{s.code}: blocked by spread gate (spread_pct={s.spread_pct:.4f})")
            s._gate_logged.add("spread")
            if instr:
                instr.on_signal_blocked(
                    symbol=s.code, signal="or_break", signal_id="kmp_breakout",
                    blocked_by="spread_gate", block_reason=f"maturity=mid, spread_pct={s.spread_pct:.4f}",
                    signal_strength=s.surge,
                )
        return None

    tick = tick_size(price)

    # WATCH_BREAK -> WAIT_ACCEPTANCE
    if s.fsm == State.WATCH_BREAK:
        if price > (s.or_high + tick) and price > s.vwap:
            s.break_ts = time.time()
            s.retest_low = price
            s.fsm = State.WAIT_ACCEPTANCE
            logger.info(f"{s.code}: Break detected at {price:.0f}")
        return None

    # WAIT_ACCEPTANCE -> ARMED
    if s.fsm == State.WAIT_ACCEPTANCE:
        s.retest_low = min(s.retest_low, price)

        if acceptance_timed_out(s):
            if instr:
                instr.on_signal_blocked(
                    symbol=s.code, signal="or_break", signal_id="kmp_breakout",
                    blocked_by="acceptance_timeout", block_reason="maturity=mid",
                    signal_strength=s.surge,
                )
            s.fsm = State.DONE
            logger.info(f"{s.code}: Acceptance timeout")
            return None

        if not is_accepted(s, price):
            return None

        # Regime check before intent submission (permissive mode)
        if not regime_ok:
            if "regime_accepted" not in s._gate_logged:
                logger.info(f"{s.code}: accepted but regime_ok=False, holding")
                s._gate_logged.add("regime_accepted")
                if instr:
                    instr.on_signal_blocked(
                        symbol=s.code, signal="or_break", signal_id="kmp_breakout",
                        blocked_by="regime_post_accept",
                        block_reason="maturity=late, accepted_but_regime_off",
                        signal_strength=s.surge,
                    )
            return None

        # Compute entry parameters
        entry_trigger = round_to_tick(s.or_high + tick)

        # VI wall check
        if vi_blocked(s, entry_trigger, tick):
            if instr:
                instr.on_signal_blocked(
                    symbol=s.code, signal="or_break", signal_id="kmp_breakout",
                    blocked_by="vi_wall", block_reason="maturity=late",
                    signal_strength=s.surge,
                )
            s.fsm = State.DONE
            logger.info(f"{s.code}: VI blocked")
            return None

        # Structure stop
        s.structure_stop = round_to_tick(s.retest_low * 0.997)
        if s.structure_stop >= entry_trigger:
            return None

        # Hard stop
        s.hard_stop = round_to_tick(entry_trigger - HARD_STOP_ATR_MULT * atr_1m)

        # Size calculation
        time_mult = size_time_multiplier(m)
        raw_qty = compute_qty(s, equity, entry_trigger, s.structure_stop, prog_mult, time_mult, now_kst,
                              regime_breadth_ok=regime_breadth_ok, not_chop=not_chop)
        qty = apply_liquidity_cap(raw_qty, entry_trigger, last_5m_value)
        cap_reason = "liquidity_5m" if qty < raw_qty else ""
        nav_qty = apply_nav_cap(qty, entry_trigger, equity)
        if nav_qty < qty:
            cap_reason = "nav_cap"
        qty = nav_qty

        # Build sizing context for instrumentation
        from .sizing import quality_multiplier
        risk_per_share = max(entry_trigger - s.structure_stop, 0.0)
        qmult = quality_multiplier(s, now_kst, regime_breadth_ok=regime_breadth_ok, not_chop=not_chop)
        risk_krw = equity * BASE_RISK_PCT
        qty_base = int(risk_krw / risk_per_share) if risk_per_share > 0 else 0
        s.sizing_context = build_sizing_context(
            equity=equity, base_risk_pct=BASE_RISK_PCT,
            risk_per_share=risk_per_share, qty_base=qty_base,
            qmult=qmult, time_mult=time_mult, program_mult=prog_mult,
            final_qty=qty, cap_reason=cap_reason,
        )

        if qty <= 0:
            s.fsm = State.DONE
            return None

        # Sector cap check (before order to prevent races)
        if exposure is not None:
            if not exposure.can_enter(s.code, qty, entry_trigger, equity):
                if instr:
                    instr.on_signal_blocked(
                        symbol=s.code, signal="or_break", signal_id="kmp_breakout",
                        blocked_by="sector_cap",
                        block_reason=f"maturity=late, sector={exposure.get_sector(s.code)}",
                        signal_strength=s.surge,
                    )
                s.fsm = State.DONE
                s.skip_reason = "sector_cap"
                logger.info(f"{s.code}: Sector cap reached for {exposure.get_sector(s.code)}")
                return None

        # Submit entry intent
        s.pgm_regime_at_entry = prog_regime

        limit_px = round_to_tick(entry_trigger + max(3 * tick, 2.0 * s.spread))

        # Reserve sector slot BEFORE sending order (prevents races)
        if exposure is not None:
            exposure.reserve(s.code, qty, entry_trigger)

        intent = Intent(
            intent_type=IntentType.ENTER,
            strategy_id=STRATEGY_ID,
            symbol=s.code,
            desired_qty=qty,
            urgency=Urgency.HIGH,
            time_horizon=TimeHorizon.INTRADAY,
            constraints=IntentConstraints(
                stop_price=entry_trigger,
                limit_price=limit_px,
                expiry_ts=time.time() + 30,
            ),
            risk_payload=RiskPayload(
                entry_px=entry_trigger,
                stop_px=s.structure_stop,
                hard_stop_px=s.hard_stop,
                rationale_code="or_break_acceptance",
                confidence="GREEN" if qty > 0 else "YELLOW",
            ),
        )

        try:
            result = await oms.submit_intent(intent)
        except Exception as e:
            # Release reservation on error
            if exposure is not None:
                exposure.unreserve(s.code, qty, entry_trigger)
            logger.warning(f"{s.code}: Entry submission failed - {e}")
            s.fsm = State.DONE
            return None

        if result.status.name in ("EXECUTED", "APPROVED"):
            s.entry_order_id = result.order_id
            s.entry_armed_ts = time.time()
            s.fsm = State.ARMED
            logger.info(f"{s.code}: Armed entry at {entry_trigger:.0f}, qty={qty}")
            return intent.intent_id
        else:
            # Release reservation on rejection
            if exposure is not None:
                exposure.unreserve(s.code, qty, entry_trigger)
            if instr:
                instr.on_signal_blocked(
                    symbol=s.code, signal="or_break", signal_id="kmp_breakout",
                    blocked_by="entry_rejected",
                    block_reason=f"maturity=late, msg={result.message}",
                    signal_strength=s.surge,
                )
            logger.warning(f"{s.code}: Entry rejected - {result.message}")
            s.fsm = State.DONE

        return None

    return None
