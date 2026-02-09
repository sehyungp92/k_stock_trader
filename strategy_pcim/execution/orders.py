"""Order Placement via OMS Intent API."""

from oms_client import Intent, IntentType, Urgency, TimeHorizon, IntentConstraints, RiskPayload

from ..pipeline.candidate import Candidate
from ..config.constants import STRATEGY_ID, TIERS, SIZING


def create_entry_intent(
    c: Candidate,
    current_price: float,
    urgency: Urgency = Urgency.NORMAL,
    expiry_ts: float = None,
) -> Intent:
    """Create entry Intent for OMS with tier-specific slip bands."""
    tier_config = TIERS[c.tier]
    slip_band = tier_config.get("slip_band")

    limit_price = current_price * (1 + slip_band) if slip_band else current_price
    stop_price = current_price - (SIZING["STOP_ATR_MULT"] * c.atr_20d)

    # Map conviction score to confidence enum
    confidence = "GREEN" if c.conviction_score >= 0.85 else "YELLOW"

    return Intent(
        intent_type=IntentType.ENTER,
        strategy_id=STRATEGY_ID,
        symbol=c.symbol,
        desired_qty=c.final_qty,
        urgency=urgency,
        time_horizon=TimeHorizon.SWING,
        constraints=IntentConstraints(
            limit_price=limit_price,
            max_slippage_bps=int((slip_band or 0.002) * 10000),
            expiry_ts=expiry_ts,
        ),
        risk_payload=RiskPayload(
            entry_px=current_price,
            stop_px=stop_price,
            rationale_code=f"pcim_{c.bucket}_{c.tier}",
            confidence=confidence,
        ),
    )


def create_exit_intent(
    symbol: str,
    qty: int,
    reason: str,
    urgency: Urgency = Urgency.NORMAL,
) -> Intent:
    """Create exit Intent for OMS."""
    return Intent(
        intent_type=IntentType.EXIT,
        strategy_id=STRATEGY_ID,
        symbol=symbol,
        desired_qty=qty,
        urgency=urgency,
        time_horizon=TimeHorizon.SWING,
        risk_payload=RiskPayload(rationale_code=reason),
    )


def create_partial_exit_intent(symbol: str, qty: int, reason: str) -> Intent:
    """Create partial exit (reduce) Intent. qty should be positive."""
    return Intent(
        intent_type=IntentType.REDUCE,
        strategy_id=STRATEGY_ID,
        symbol=symbol,
        desired_qty=abs(qty),  # Always positive for REDUCE
        urgency=Urgency.NORMAL,
        time_horizon=TimeHorizon.SWING,
        risk_payload=RiskPayload(rationale_code=reason),
    )
