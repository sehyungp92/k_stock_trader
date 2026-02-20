"""Nulrimok Intraday Entry Logic."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import Optional
from loguru import logger

from oms_client import Intent, IntentType, Urgency, TimeHorizon, IntentConstraints, RiskPayload
from ..config.constants import STRATEGY_ID, ENTRY_VOL_DRYUP_PCT, INTRADAY_VOL_BONUS, MAX_RISK_PCT
from ..config.switches import nulrimok_switches
from ..dse.artifact import TickerArtifact


class EntryState(Enum):
    IDLE = auto()
    ARMED = auto()
    PENDING_FILL = auto()  # Order submitted, awaiting fill confirmation
    TRIGGERED = auto()
    DONE = auto()


@dataclass
class TickerEntryState:
    ticker: str
    state: EntryState = EntryState.IDLE
    arm_time: Optional[datetime] = None
    confirm_bars_remaining: int = 0
    last_30m_low: float = float('inf')
    pending_fill_cycles: int = 0  # Track cycles waiting for fill
    conf_type: str = ""  # Confirmation type for signal_hash
    anchor_date: str = ""  # For signal_hash

    def reset(self):
        self.state = EntryState.IDLE
        self.arm_time = None
        self.confirm_bars_remaining = 0
        self.last_30m_low = float('inf')
        self.pending_fill_cycles = 0
        self.conf_type = ""
        self.anchor_date = ""


def check_entry_conditions(artifact: TickerArtifact, bar: dict, sma5: float, vol_avg: float) -> bool:
    close = float(bar.get('close', 0))
    high, low = float(bar.get('high', 0)), float(bar.get('low', 0))
    volume = float(bar.get('volume', 0))
    in_band = (low <= artifact.band_upper) and (high >= artifact.band_lower)
    is_dip = close < sma5 if sma5 > 0 else False
    vol_ratio = volume / vol_avg if vol_avg > 0 else 1.0
    return in_band and is_dip and vol_ratio < ENTRY_VOL_DRYUP_PCT


def check_confirmation(entry_state: TickerEntryState, artifact: TickerArtifact, bar: dict) -> tuple:
    """Returns (confirmed, reason) or (False, "INVALIDATED") if entry should be disarmed."""
    close, low = float(bar.get('close', 0)), float(bar.get('low', 0))

    # Invalidation: close below band_lower - 0.2% disarms entry
    if close < artifact.band_lower * 0.998:
        return False, "INVALIDATED"

    if close > artifact.avwap_ref:
        return True, "RECLAIM"

    if (low > entry_state.last_30m_low
            and close <= artifact.band_upper * 1.003
            and close >= artifact.band_lower * 0.998):
        return True, "HIGHER_LOW"

    entry_state.last_30m_low = min(entry_state.last_30m_low, low)
    return False, ""


async def process_entry(entry_state: TickerEntryState, artifact: TickerArtifact, bar: dict,
                        sma5: float, vol_avg: float, now: datetime, equity: float, oms) -> Optional[str]:
    close = float(bar.get('close', 0))

    if entry_state.state == EntryState.IDLE:
        if check_entry_conditions(artifact, bar, sma5, vol_avg):
            entry_state.state = EntryState.ARMED
            entry_state.arm_time = now
            entry_state.confirm_bars_remaining = nulrimok_switches.confirm_bars
            entry_state.last_30m_low = float(bar.get('low', float('inf')))
            logger.info(f"{artifact.ticker}: Armed for entry")
        return None

    if entry_state.state == EntryState.ARMED:
        confirmed, conf_type = check_confirmation(entry_state, artifact, bar)

        # Invalidation: close below band_lower - 0.2% disarms immediately
        if conf_type == "INVALIDATED":
            logger.info(f"{artifact.ticker}: Entry invalidated (close below band)")
            entry_state.reset()
            return None

        if confirmed:
            risk_pct = artifact.recommended_risk or 0.005
            # Intraday volume bonus: +10% size if vol_ratio < 0.40 (very dry)
            volume = float(bar.get('volume', 0))
            vol_ratio = volume / vol_avg if vol_avg > 0 else 1.0
            if vol_ratio < 0.40:
                risk_pct *= INTRADAY_VOL_BONUS
            # Cap total multiplier to prevent overconcentration
            # Base is 0.005, max from TOP20 is 0.0075, with bonus = 0.00825
            risk_pct = min(risk_pct, MAX_RISK_PCT)
            # ATR-based stop: use max() to select tighter (higher) stop per spec
            atr30m = artifact.atr30m_est or 0.0
            if atr30m > 0:
                stop = max(artifact.avwap_ref - 1.2 * atr30m, artifact.band_lower * 0.993)
            else:
                stop = artifact.band_lower * 0.993
            qty = int((equity * risk_pct) / max(close - stop, 0.01))

            if qty <= 0:
                logger.warning(f"{artifact.ticker}: Entry confirmed ({conf_type}) but qty=0 (close={close:.0f} stop={stop:.0f}), resetting")
                entry_state.reset()
                return None

            # Generate signal_hash for idempotency
            bar_ts = bar.get('timestamp', now.strftime("%H%M"))
            signal_hash = f"{artifact.anchor_date or 'unk'}:{conf_type}:{bar_ts}"

            intent = Intent(
                intent_type=IntentType.ENTER, strategy_id=STRATEGY_ID, symbol=artifact.ticker,
                desired_qty=qty, urgency=Urgency.LOW, time_horizon=TimeHorizon.SWING,
                signal_hash=signal_hash,
                constraints=IntentConstraints(limit_price=artifact.avwap_ref),
                risk_payload=RiskPayload(entry_px=close, stop_px=stop,
                                         confidence="GREEN" if artifact.acceptance_pass else "YELLOW"),
            )

            result = await oms.submit_intent(intent)
            if result.status.name in ("EXECUTED", "APPROVED"):
                entry_state.state = EntryState.PENDING_FILL
                entry_state.conf_type = conf_type
                entry_state.anchor_date = artifact.anchor_date or ""
                logger.info(f"{artifact.ticker}: Entry submitted, awaiting fill ({conf_type})")
                return intent.intent_id

            # OMS rejected or unreachable — log and do NOT consume a confirmation bar
            logger.warning(f"{artifact.ticker}: Entry confirmed ({conf_type}) but OMS returned {result.status.name}: {result.message}")
            return None

        entry_state.confirm_bars_remaining -= 1
        if entry_state.confirm_bars_remaining <= 0:
            logger.info(f"{artifact.ticker}: Confirmation window expired, resetting entry state")
            entry_state.reset()
        return None

    return None
