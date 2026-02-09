"""Nulrimok Exit Management."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Optional
from loguru import logger

from oms_client import Intent, IntentType, Urgency, TimeHorizon, IntentConstraints, RiskPayload
from ..config.constants import (
    STRATEGY_ID, AVWAP_BREAKDOWN_PCT, BREAKDOWN_VOL_MULT,
    RECLAIM_BARS, TIME_STOP_MULTI_DAYS, TIME_STOP_MULTI_PNL,
    FLOW_EXIT_REPRICE_TIMEOUT_SEC,
)


class SetupType(Enum):
    UNKNOWN = auto()
    MOMENTUM = auto()
    MEAN_REVERSION = auto()
    FLOW_GRIND = auto()
    FAILED = auto()  # No progress after 2 sessions


FAILURE_RECLAIM_BARS = 5
MEAN_REV_PARTIAL_ATR_MULT = 1.5
MOMENTUM_SMA_PERIOD = 10
MEAN_REV_TRAIL_SMA_PERIOD = 5


@dataclass
class PositionState:
    ticker: str
    entry_time: datetime
    entry_price: float
    qty: int
    stop: float
    sessions_held: int = 0
    bars_since_breakdown: int = 0
    in_breakdown: bool = False
    max_price: float = 0.0
    setup: SetupType = SetupType.UNKNOWN
    bars_since_entry: int = 0
    entry_low: float = float('inf')
    close_history: list = field(default_factory=list)
    partial_taken: bool = False
    remaining_qty: int = 0
    atr30m: float = 0.0
    flow_grind_bars_below_avwap: int = 0  # For FLOW_GRIND 2-bar exit

    def __post_init__(self):
        if self.remaining_qty == 0:
            self.remaining_qty = self.qty


def classify_setup(pos: PositionState, bar: dict, avwap: float) -> SetupType:
    """Classify after a few bars: momentum, mean-rev, flow-grind, or failed."""
    if pos.setup != SetupType.UNKNOWN:
        return pos.setup
    if pos.bars_since_entry < 3:
        return SetupType.UNKNOWN
    high = float(bar.get('high', 0))
    close = float(bar.get('close', 0))
    # Failed: no progress after 2 sessions (spec §10.3)
    if pos.sessions_held >= 2:
        return SetupType.FAILED
    # Momentum: breaking higher
    if high > pos.max_price * 1.005:
        return SetupType.MOMENTUM
    # Flow grind: slow climb, holding above entry but not breaking out
    if close > pos.entry_price * 1.002 and close <= pos.max_price * 0.998:
        return SetupType.FLOW_GRIND
    # Mean reversion: stalling or bouncing without progress
    return SetupType.MEAN_REVERSION


def check_avwap_breakdown(pos: PositionState, bar: dict, avwap: float, vol_avg: float) -> bool:
    close = float(bar.get('close', 0))
    volume = float(bar.get('volume', 0))
    return (close < avwap * (1 - AVWAP_BREAKDOWN_PCT)
            and (volume / vol_avg if vol_avg > 0 else 1.0) > BREAKDOWN_VOL_MULT)


def _check_failure_to_reclaim(pos: PositionState, bar: dict) -> bool:
    """Exit if no reclaim within FAILURE_RECLAIM_BARS post-entry AND lower low prints."""
    if pos.bars_since_entry > FAILURE_RECLAIM_BARS:
        low = float(bar.get('low', 0))
        if low < pos.entry_low:
            return True
    return False


def _momentum_trail_stop(pos: PositionState) -> Optional[float]:
    """Trail at SMA10 of 30m closes for momentum setups."""
    if len(pos.close_history) >= MOMENTUM_SMA_PERIOD:
        return sum(pos.close_history[-MOMENTUM_SMA_PERIOD:]) / MOMENTUM_SMA_PERIOD
    return None


async def manage_nulrimok_position(pos: PositionState, bar: dict, avwap: float, vol_avg: float,
                                   is_market_close: bool, oms) -> Optional[str]:
    close = float(bar.get('close', 0))
    low = float(bar.get('low', 0))
    pos.max_price = max(pos.max_price, close)
    pos.bars_since_entry += 1
    pos.close_history.append(close)
    if pos.bars_since_entry == 1:
        pos.entry_low = low
    else:
        pos.entry_low = min(pos.entry_low, low)

    # Classify setup after enough bars
    pos.setup = classify_setup(pos, bar, avwap)

    exit_reason, urgency = None, Urgency.LOW

    # 1. AVWAP breakdown
    if check_avwap_breakdown(pos, bar, avwap, vol_avg):
        pos.in_breakdown = True
        exit_reason, urgency = "avwap_breakdown", Urgency.HIGH

    # 2. Failure to reclaim
    elif pos.in_breakdown:
        if close > avwap:
            pos.in_breakdown = False
            pos.bars_since_breakdown = 0
        else:
            pos.bars_since_breakdown += 1
            if pos.bars_since_breakdown >= RECLAIM_BARS:
                exit_reason = "failure_to_reclaim"

    # 3. Failure-to-reclaim post-entry (lower low within first N bars)
    elif _check_failure_to_reclaim(pos, bar):
        exit_reason = "failure_reclaim_lower_low"

    # 4. Setup-aware exits
    elif pos.setup == SetupType.MOMENTUM:
        trail = _momentum_trail_stop(pos)
        if trail and close < trail:
            exit_reason = "momentum_trail_stop"
    elif pos.setup == SetupType.MEAN_REVERSION:
        if not pos.partial_taken:
            # Use stored ATR30m if available, else fallback to peak-to-entry range
            atr_ref = pos.atr30m if pos.atr30m > 0 else (pos.max_price - pos.entry_price)
            if atr_ref > 0 and (close - pos.entry_price) >= MEAN_REV_PARTIAL_ATR_MULT * atr_ref:
                exit_reason = "mean_rev_partial"
                pos.partial_taken = True
        else:
            # Trail remaining 30%: higher of entry+0.5×ATR or 5SMA
            atr_ref = pos.atr30m if pos.atr30m > 0 else (pos.max_price - pos.entry_price) * 0.5
            trail_atr = pos.entry_price + 0.5 * atr_ref
            trail_sma = (sum(pos.close_history[-MEAN_REV_TRAIL_SMA_PERIOD:]) / MEAN_REV_TRAIL_SMA_PERIOD
                         if len(pos.close_history) >= MEAN_REV_TRAIL_SMA_PERIOD else 0)
            trail = max(trail_atr, trail_sma) if trail_sma > 0 else trail_atr
            if close < trail:
                exit_reason = "mean_rev_trail_remaining"
    elif pos.setup == SetupType.FLOW_GRIND:
        # Flow grind exit: AVWAP failure and no reclaim in 2 bars
        if close < avwap:
            pos.flow_grind_bars_below_avwap += 1
            if pos.flow_grind_bars_below_avwap >= 2:
                exit_reason = "flow_grind_avwap_failure"
        else:
            pos.flow_grind_bars_below_avwap = 0
    elif pos.setup == SetupType.FAILED and is_market_close:
        # Failed setup: exit 100% at close after 2 sessions with no progress
        exit_reason = "failed_setup"

    # 5. Time stops
    elif is_market_close and (close - pos.entry_price) / pos.entry_price <= 0:
        exit_reason = "time_stop_intraday"
    elif (pos.sessions_held >= TIME_STOP_MULTI_DAYS
          and (close - pos.entry_price) / pos.entry_price < TIME_STOP_MULTI_PNL):
        exit_reason = "time_stop_multiday"

    if exit_reason:
        # Partial exit for mean reversion: sell 70%, keep 30%
        if exit_reason == "mean_rev_partial":
            exit_qty = int(pos.remaining_qty * 0.70)
            if exit_qty <= 0:
                return None
            pos.remaining_qty -= exit_qty
        else:
            exit_qty = pos.remaining_qty

        intent = Intent(
            intent_type=IntentType.EXIT, strategy_id=STRATEGY_ID, symbol=pos.ticker,
            desired_qty=exit_qty, urgency=urgency, time_horizon=TimeHorizon.SWING,
            risk_payload=RiskPayload(rationale_code=exit_reason),
        )
        result = await oms.submit_intent(intent)
        if result.status.name in ("EXECUTED", "APPROVED"):
            logger.info(f"{pos.ticker}: Exit triggered - {exit_reason}, qty={exit_qty}")
            return intent.intent_id
    return None


async def handle_flow_reversal_exits(artifacts: list, oms, kis_api=None) -> None:
    """Execute flow reversal exits with marketable limit pricing."""
    import asyncio

    async def _reprice_if_stale(oms_client, ticker: str, timeout_sec: int):
        """Reprice exit order if not filled within timeout."""
        await asyncio.sleep(timeout_sec)
        try:
            pos = await oms_client.get_position(ticker)
            alloc_qty = pos.get_allocation(STRATEGY_ID) if pos else 0
            if alloc_qty > 0:
                logger.warning(f"{ticker}: Flow reversal exit not filled, repricing with HIGH urgency")
                await oms_client.submit_intent(Intent(
                    intent_type=IntentType.EXIT,
                    strategy_id=STRATEGY_ID,
                    symbol=ticker,
                    urgency=Urgency.HIGH,
                    time_horizon=TimeHorizon.SWING,
                    risk_payload=RiskPayload(rationale_code="flow_reversal_reprice"),
                ))
        except Exception as e:
            logger.warning(f"{ticker}: Reprice check failed: {e}")

    for artifact in artifacts:
        if not artifact.exit_at_open:
            continue

        # Get current price for marketable limit (cross spread, allow 0.5% slippage)
        limit_px = None
        if kis_api:
            try:
                quote = kis_api.get_current_price(artifact.ticker)
                bid = float(quote.get('bid', 0))
                if bid > 0:
                    limit_px = bid * 0.995
            except Exception:
                pass  # Fall back to market-like OMS behavior

        intent = Intent(
            intent_type=IntentType.EXIT,
            strategy_id=STRATEGY_ID,
            symbol=artifact.ticker,
            urgency=Urgency.HIGH,
            time_horizon=TimeHorizon.SWING,
            constraints=IntentConstraints(limit_price=limit_px) if limit_px else IntentConstraints(),
            risk_payload=RiskPayload(rationale_code="flow_reversal"),
        )
        result = await oms.submit_intent(intent)

        # Schedule reprice fallback if order is working
        if result.status.name == "EXECUTED":
            asyncio.create_task(_reprice_if_stale(oms, artifact.ticker, FLOW_EXIT_REPRICE_TIMEOUT_SEC))

        logger.info(f"{artifact.ticker}: Flow reversal exit submitted (limit={limit_px})")
