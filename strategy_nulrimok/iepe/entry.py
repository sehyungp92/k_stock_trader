"""Nulrimok Intraday Entry Logic."""

import time as _time
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
    sizing_context: dict | None = None  # Cached for fill confirmation instrumentation
    signal_generated_at: float | None = None  # Execution timeline timestamps
    oms_received_at: float | None = None
    order_submitted_at: float | None = None

    def reset(self):
        self.state = EntryState.IDLE
        self.arm_time = None
        self.confirm_bars_remaining = 0
        self.last_30m_low = float('inf')
        self.pending_fill_cycles = 0
        self.conf_type = ""
        self.anchor_date = ""
        self.sizing_context = None
        self.signal_generated_at = None
        self.oms_received_at = None
        self.order_submitted_at = None


def check_entry_conditions(artifact: TickerArtifact, bar: dict, sma5: float, vol_avg: float) -> bool:
    close = float(bar.get('close', 0))
    high, low = float(bar.get('high', 0)), float(bar.get('low', 0))
    volume = float(bar.get('volume', 0))
    in_band = (low <= artifact.band_upper) and (high >= artifact.band_lower)
    is_dip = close < sma5 if sma5 > 0 else False
    vol_ratio = volume / vol_avg if vol_avg > 0 else 1.0
    passed = in_band and is_dip and vol_ratio < ENTRY_VOL_DRYUP_PCT
    if not passed:
        reasons = []
        if not in_band:
            reasons.append(f"not_in_band(low={low:.0f},hi={high:.0f},bL={artifact.band_lower:.0f},bU={artifact.band_upper:.0f})")
        if not is_dip:
            reasons.append(f"no_dip(close={close:.0f},sma5={sma5:.0f})")
        if vol_ratio >= ENTRY_VOL_DRYUP_PCT:
            reasons.append(f"vol_high({vol_ratio:.2f}>={ENTRY_VOL_DRYUP_PCT})")
        logger.debug(f"{artifact.ticker}: Entry conditions not met — {', '.join(reasons)}")
    return passed


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


def build_sizing_context(equity, recommended_risk, risk_per_share,
                         vol_bonus_applied, final_qty, cap_reason="", raw_qty=None):
    """Return sizing decision context for instrumentation."""
    return {
        "sizing_model": "risk_based_regime_adj",
        "target_risk_pct": recommended_risk,
        "account_equity": int(equity),
        "volatility_basis": round(float(risk_per_share), 2),
        "vol_bonus_applied": vol_bonus_applied,
        "raw_qty": int(raw_qty) if raw_qty is not None else int(final_qty),
        "final_qty": int(final_qty),
        "cap_reason": cap_reason,
    }


async def process_entry(entry_state: TickerEntryState, artifact: TickerArtifact, bar: dict,
                        sma5: float, vol_avg: float, now: datetime, equity: float, oms,
                        gross_exposure_pct: float = 0.0, regime_exposure_cap: float = 1.0,
                        instr=None) -> Optional[str]:
    close = float(bar.get('close', 0))

    if entry_state.state == EntryState.IDLE:
        conditions_met = check_entry_conditions(artifact, bar, sma5, vol_avg)
        # Emit indicator snapshot at signal evaluation
        if instr is not None:
            volume = float(bar.get('volume', 0))
            vol_ratio = volume / vol_avg if vol_avg > 0 else 1.0
            instr.on_indicator_snapshot(
                pair=artifact.ticker,
                indicators={
                    "avwap": artifact.avwap_ref if hasattr(artifact, 'avwap_ref') else 0.0,
                    "band_upper": artifact.band_upper,
                    "band_lower": artifact.band_lower,
                    "sma5": sma5,
                    "vol_ratio": round(vol_ratio, 3),
                    "flow_score": artifact.flow_score if hasattr(artifact, 'flow_score') else 0.0,
                },
                signal_name="nulrimok_avwap_dip",
                signal_strength=artifact.flow_score if hasattr(artifact, 'flow_score') else 0.0,
                decision="enter" if conditions_met else "skip",
                strategy_type="nulrimok",
            )
            # Emit filter decisions for entry conditions
            high, low = float(bar.get('high', 0)), float(bar.get('low', 0))
            in_band = (low <= artifact.band_upper) and (high >= artifact.band_lower)
            is_dip = close < sma5 if sma5 > 0 else False
            instr.on_filter_decision(
                pair=artifact.ticker, filter_name="avwap_band",
                passed=in_band, threshold=artifact.band_upper,
                actual_value=close,
                signal_name="nulrimok_avwap_dip",
                strategy_type="nulrimok",
            )
            instr.on_filter_decision(
                pair=artifact.ticker, filter_name="vol_dryup",
                passed=vol_ratio < ENTRY_VOL_DRYUP_PCT,
                threshold=ENTRY_VOL_DRYUP_PCT, actual_value=vol_ratio,
                signal_name="nulrimok_avwap_dip",
                strategy_type="nulrimok",
            )
        if conditions_met:
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
            if instr:
                instr.on_signal_blocked(
                    symbol=artifact.ticker, signal="avwap_dip_buy", signal_id="nulrimok_dip",
                    blocked_by="band_invalidation", block_reason="close below band_lower-0.2%",
                    signal_strength=0.0,
                )
            logger.info(f"{artifact.ticker}: Entry invalidated (close below band)")
            entry_state.reset()
            return None

        if confirmed:
            # Pre-check: skip if exposure headroom is exhausted
            exposure_cap = min(regime_exposure_cap, 0.90)  # Use tighter of regime cap and static 90%
            headroom_pct = max(exposure_cap - gross_exposure_pct, 0.0)
            if headroom_pct <= 0.005:  # Less than 0.5% headroom — no room for any entry
                if instr:
                    instr.on_signal_blocked(
                        symbol=artifact.ticker, signal="avwap_dip_buy", signal_id="nulrimok_dip",
                        blocked_by="exposure_headroom",
                        block_reason=f"gross={gross_exposure_pct:.1%}, cap={exposure_cap:.0%}",
                        signal_strength=0.0,
                        filter_decisions=[{
                            "filter": "exposure_headroom", "threshold": round(exposure_cap, 4),
                            "actual": round(gross_exposure_pct, 4), "passed": False,
                            "margin_pct": 0,
                        }],
                    )
                logger.warning(f"{artifact.ticker}: Entry confirmed ({conf_type}) but exposure headroom exhausted "
                               f"(gross={gross_exposure_pct:.1%}, cap={exposure_cap:.0%})")
                # Don't consume confirmation bar — will retry if exposure frees up
                return None

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

            # Cap qty to fit within exposure headroom
            if close > 0 and equity > 0:
                max_notional = headroom_pct * equity
                max_qty_by_exposure = int(max_notional / close)
                if qty > max_qty_by_exposure > 0:
                    logger.info(f"{artifact.ticker}: Scaling qty {qty}->{max_qty_by_exposure} to fit exposure headroom "
                                f"({headroom_pct:.1%} of {equity:.0f})")
                    qty = max_qty_by_exposure

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

            # Cache sizing context for fill confirmation instrumentation
            raw_qty = int((equity * risk_pct) / max(close - stop, 0.01))
            vol_bonus_applied = vol_ratio < 0.40
            cap_reason = "exposure_headroom" if qty < raw_qty else ""
            entry_state.sizing_context = build_sizing_context(
                equity=equity, recommended_risk=risk_pct,
                risk_per_share=close - stop, vol_bonus_applied=vol_bonus_applied,
                final_qty=qty, cap_reason=cap_reason, raw_qty=raw_qty,
            )

            _signal_ts = _time.time()
            result = await oms.submit_intent(intent)
            if result.status.name in ("EXECUTED", "APPROVED"):
                if instr:
                    instr.on_order_event(
                        order_id=getattr(result, 'order_id', '') or intent.intent_id,
                        pair=artifact.ticker,
                        order_type="LIMIT",
                        status="SUBMITTED",
                        requested_qty=qty,
                        requested_price=artifact.avwap_ref,
                        related_trade_id=intent.intent_id,
                    )
                entry_state.state = EntryState.PENDING_FILL
                entry_state.conf_type = conf_type
                entry_state.anchor_date = artifact.anchor_date or ""
                entry_state.signal_generated_at = _signal_ts
                entry_state.oms_received_at = getattr(result, 'oms_received_at', None)
                entry_state.order_submitted_at = getattr(result, 'order_submitted_at', None)
                logger.info(f"{artifact.ticker}: Entry submitted, awaiting fill ({conf_type})")
                return intent.intent_id

            # OMS rejected or unreachable — log and do NOT consume a confirmation bar
            if instr:
                instr.on_order_event(
                    order_id=getattr(result, 'order_id', '') or intent.intent_id,
                    pair=artifact.ticker,
                    order_type="LIMIT",
                    status="REJECTED",
                    requested_qty=qty,
                    requested_price=artifact.avwap_ref,
                    reject_reason=result.message or "",
                    related_trade_id=intent.intent_id,
                )
            if instr:
                instr.on_signal_blocked(
                    symbol=artifact.ticker, signal="avwap_dip_buy", signal_id="nulrimok_dip",
                    blocked_by="oms_rejected",
                    block_reason=f"{result.status.name}: {result.message}",
                    signal_strength=0.0,
                    blocking_positions=result.blocking_positions,
                    resource_conflict_type=result.resource_conflict_type or "",
                )
            logger.warning(
                f"{artifact.ticker}: Entry confirmed ({conf_type}) but OMS returned "
                f"{result.status.name}: {result.message} "
                f"[qty={qty}, limit={artifact.avwap_ref:.0f}, stop={stop:.0f}, entry_px={close:.0f}]"
            )
            return None

        entry_state.confirm_bars_remaining -= 1
        if entry_state.confirm_bars_remaining <= 0:
            if instr:
                instr.on_signal_blocked(
                    symbol=artifact.ticker, signal="avwap_dip_buy", signal_id="nulrimok_dip",
                    blocked_by="confirmation_expired",
                    block_reason=f"confirm_bars={nulrimok_switches.confirm_bars}",
                    signal_strength=0.0,
                )
            logger.info(f"{artifact.ticker}: Confirmation window expired, resetting entry state")
            entry_state.reset()
        return None

    return None
