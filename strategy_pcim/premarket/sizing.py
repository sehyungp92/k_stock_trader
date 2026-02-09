"""Vol-Parity Position Sizing."""

from loguru import logger

from ..pipeline.candidate import Candidate
from ..config.constants import SIZING, TIERS, TV5M_PROXY_DIVISOR, BUCKET_B


def compute_sizing(
    c: Candidate,
    equity: float,
    tv5m: float = None,
) -> Candidate:
    """
    Compute position size using volatility-parity.

    raw_qty = target_risk / stop_distance
    final_qty = raw_qty x conviction_score x soft_mult x tier_mult

    Caps: single name 15%, TV_5m participation, size floor 20%.
    """
    if c.is_rejected():
        return c

    target_risk = equity * SIZING["TARGET_RISK_PCT"]
    stop_distance = SIZING["STOP_ATR_MULT"] * c.atr_20d

    if stop_distance <= 0:
        c.reject_reason = "ZERO_ATR"
        return c

    raw_qty = int(target_risk / stop_distance)

    final_qty = int(
        raw_qty
        * c.conviction_score
        * c.soft_mult
        * c.tier_mult
    )

    c.raw_qty = raw_qty

    # Bucket B size cap
    if c.bucket == "B":
        final_qty = int(final_qty * BUCKET_B["MAX_SIZE_PCT_OF_COMPUTED"])

    c.final_qty = final_qty

    # Size floor check
    floor = int(SIZING["SIZE_FLOOR_PCT"] * raw_qty)
    if final_qty < floor:
        c.reject_reason = f"SIZE_FLOOR_REJECT_{final_qty}<{floor}"
        return c

    price = c.expected_open or c.close_prev
    notional = final_qty * price

    # Single name cap
    max_notional = SIZING["SINGLE_NAME_CAP_PCT"] * equity
    if notional > max_notional:
        capped_qty = int(max_notional / price)
        if capped_qty < floor:
            c.reject_reason = "SIZE_FLOOR_AFTER_SINGLE_NAME_CAP"
            return c
        c.final_qty = capped_qty
        notional = capped_qty * price

    # TV_5m participation cap
    if tv5m is None:
        tv5m = c.adtv_20d / TV5M_PROXY_DIVISOR

    tier_cap_pct = TIERS[c.tier]["tv5m_participation"]
    max_notional_tv5m = tier_cap_pct * tv5m

    if notional > max_notional_tv5m:
        capped_qty = int(max_notional_tv5m / price)
        if capped_qty < floor:
            c.reject_reason = "SIZE_FLOOR_AFTER_TV5M_CAP"
            return c
        c.final_qty = capped_qty
        notional = capped_qty * price

    c.final_notional = notional
    logger.debug(f"{c.symbol}: raw={raw_qty}, final={c.final_qty}, notional={notional/1e6:.1f}M")
    return c
