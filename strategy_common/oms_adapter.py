from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from oms.intent import Intent, IntentConstraints, IntentType, RiskPayload, TimeHorizon, Urgency

from .actions import (
    CancelOrders,
    FlattenPosition,
    ReplaceProtectiveStop,
    StrategyAction,
    SubmitEntry,
    SubmitExit,
    SubmitPartialExit,
    SubmitProtectiveStop,
)


def action_to_intent(action: StrategyAction) -> Intent:
    metadata = _causal_metadata(action)
    if isinstance(action, SubmitEntry):
        execution_style = str(action.metadata.get("execution_style") or "") or ("CLOSE_AUCTION" if action.order_type == "CLOSE_AUCTION" else None)
        risk_stop = _float_or_none(action.metadata.get("protective_stop_price"))
        if risk_stop is None:
            risk_stop = _float_or_none(action.metadata.get("stop_price"))
        if risk_stop is None:
            risk_stop = action.stop_price
        entry_px = action.limit_price
        if entry_px is None and execution_style == "SYNTHETIC_STOP":
            entry_px = _float_or_none(action.metadata.get("entry_trigger_price")) or action.stop_price
        return Intent(
            intent_type=IntentType.ENTER,
            strategy_id=action.strategy_id,
            symbol=action.symbol,
            desired_qty=action.qty,
            urgency=Urgency.LOW if execution_style == "CLOSE_AUCTION" else Urgency.NORMAL,
            time_horizon=TimeHorizon.INTRADAY,
            constraints=IntentConstraints(
                limit_price=action.limit_price,
                stop_price=action.stop_price,
                expiry_ts=_expiry_epoch(action.metadata),
                execution_style=execution_style,
            ),
            risk_payload=RiskPayload(
                entry_px=entry_px,
                stop_px=risk_stop,
                hard_stop_px=risk_stop,
                rationale_code=action.reason,
                confidence=str(action.metadata.get("confidence", "YELLOW")),
            ),
            signal_hash=str(action.metadata.get("signal_hash") or action.metadata.get("candidate_hash") or ""),
            metadata=metadata,
        )
    if isinstance(action, SubmitPartialExit):
        execution_style = "CLOSE_AUCTION" if action.order_type == "CLOSE_AUCTION" else None
        return Intent(
            intent_type=IntentType.REDUCE,
            strategy_id=action.strategy_id,
            symbol=action.symbol,
            desired_qty=action.qty,
            urgency=Urgency.LOW if execution_style == "CLOSE_AUCTION" else Urgency.NORMAL,
            constraints=IntentConstraints(
                limit_price=action.limit_price,
                expiry_ts=_expiry_epoch(action.metadata),
                execution_style=execution_style,
            ),
            risk_payload=RiskPayload(rationale_code=action.reason),
            signal_hash=str(action.metadata.get("signal_hash") or action.metadata.get("candidate_hash") or ""),
            metadata=metadata,
        )
    if isinstance(action, SubmitExit):
        execution_style = "CLOSE_AUCTION" if action.order_type == "CLOSE_AUCTION" else None
        return Intent(
            intent_type=IntentType.EXIT,
            strategy_id=action.strategy_id,
            symbol=action.symbol,
            desired_qty=action.qty,
            urgency=Urgency.LOW if execution_style == "CLOSE_AUCTION" else Urgency.NORMAL,
            constraints=IntentConstraints(
                limit_price=action.limit_price,
                expiry_ts=_expiry_epoch(action.metadata),
                execution_style=execution_style,
            ),
            risk_payload=RiskPayload(rationale_code=action.reason),
            signal_hash=str(action.metadata.get("signal_hash") or action.metadata.get("candidate_hash") or ""),
            metadata=metadata,
        )
    if isinstance(action, SubmitProtectiveStop):
        return Intent(
            intent_type=IntentType.MODIFY_RISK,
            strategy_id=action.strategy_id,
            symbol=action.symbol,
            desired_qty=action.qty,
            constraints=IntentConstraints(stop_price=action.stop_price),
            risk_payload=RiskPayload(stop_px=action.stop_price, hard_stop_px=action.stop_price, rationale_code=action.reason),
            signal_hash=str(action.metadata.get("signal_hash") or action.metadata.get("candidate_hash") or ""),
            metadata=metadata,
        )
    if isinstance(action, ReplaceProtectiveStop):
        return Intent(
            intent_type=IntentType.MODIFY_RISK,
            strategy_id=action.strategy_id,
            symbol=action.symbol,
            desired_qty=action.qty,
            constraints=IntentConstraints(stop_price=action.stop_price),
            risk_payload=RiskPayload(stop_px=action.stop_price, hard_stop_px=action.stop_price, rationale_code=action.reason),
            signal_hash=str(action.metadata.get("signal_hash") or action.metadata.get("candidate_hash") or ""),
            metadata=metadata,
        )
    if isinstance(action, CancelOrders):
        return Intent(
            intent_type=IntentType.CANCEL_ORDERS,
            strategy_id=action.strategy_id,
            symbol=action.symbol,
            risk_payload=RiskPayload(rationale_code=action.reason),
            signal_hash=str(action.metadata.get("signal_hash") or action.metadata.get("candidate_hash") or ""),
            metadata=metadata,
        )
    if isinstance(action, FlattenPosition):
        return Intent(
            intent_type=IntentType.FLATTEN,
            strategy_id=action.strategy_id,
            symbol=action.symbol,
            risk_payload=RiskPayload(rationale_code=action.reason),
            signal_hash=str(action.metadata.get("signal_hash") or action.metadata.get("candidate_hash") or ""),
            metadata=metadata,
        )
    raise TypeError(f"Unsupported action: {type(action).__name__}")


def _float_or_none(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _expiry_epoch(metadata: dict) -> float | None:
    numeric = _float_or_none(metadata.get("expiry_ts"))
    if numeric is not None:
        return numeric
    raw = metadata.get("expiry_timestamp")
    if raw is None or raw == "":
        return None
    try:
        expiry = datetime.fromisoformat(str(raw))
    except (TypeError, ValueError):
        return None
    if expiry.tzinfo is None:
        expiry = expiry.replace(tzinfo=ZoneInfo("Asia/Seoul"))
    return expiry.timestamp()


def _causal_metadata(action: StrategyAction) -> dict:
    source = dict(getattr(action, "metadata", {}) or {})
    keys = (
        "action_ref",
        "source_artifact_hash",
        "source_fingerprint",
        "strategy_action_hash",
        "candidate_hash",
        "decision_ref",
        "sector",
        "candidate_rank",
        "route_family",
    )
    return {key: source[key] for key in keys if key in source and source[key] not in (None, "")}
