from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from oms.intent import Intent, IntentResult, IntentStatus
from oms_client.client import AccountState, PositionInfo

from .session_capture import PaperSessionRecorder


@dataclass(slots=True)
class RecordingOMSClient:
    """Record normalized OMS intents without submitting broker orders."""

    recorder: PaperSessionRecorder
    account_state: AccountState = field(default_factory=AccountState)
    positions: dict[str, PositionInfo] = field(default_factory=dict)
    record_only: bool = True

    async def submit_intent(self, intent: Intent) -> IntentResult:
        return IntentResult(
            intent_id=intent.intent_id,
            status=IntentStatus.ACCEPTED,
            message="dry_run_recorded_not_submitted",
            order_id=f"dry-run:{intent.strategy_id}:{intent.symbol}:{intent.intent_id}",
        )

    async def get_account_state(self) -> AccountState:
        return self.account_state

    async def get_all_positions(self) -> dict[str, PositionInfo]:
        return dict(self.positions)


def intent_to_json_dict(
    intent: Intent,
    *,
    dry_run: bool = True,
    submitted_to_broker: bool = False,
    intended_broker_submit: bool | None = None,
    actually_submitted_to_broker: bool | None = None,
    oms_status: str = "",
    broker_order_id: str = "",
    record_type: str | None = None,
) -> dict[str, Any]:
    metadata = dict(intent.metadata or {})
    intended_submit = bool((not dry_run) if intended_broker_submit is None else intended_broker_submit)
    actual_submit = bool(submitted_to_broker if actually_submitted_to_broker is None else actually_submitted_to_broker)
    return {
        "record_type": record_type or ("dry_run_oms_intent" if dry_run else "oms_intent"),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": bool(dry_run),
        "intended_broker_submit": intended_submit,
        "broker_submit_possible": intended_submit,
        "actually_submitted_to_broker": actual_submit,
        "submitted_to_broker": actual_submit,
        "oms_status": str(oms_status or ""),
        "broker_order_id": str(broker_order_id or ""),
        "intent_id": intent.intent_id,
        "idempotency_key": intent.idempotency_key,
        "intent_type": intent.intent_type.name,
        "strategy_id": intent.strategy_id,
        "symbol": str(intent.symbol).zfill(6),
        "desired_qty": intent.desired_qty,
        "target_qty": intent.target_qty,
        "urgency": intent.urgency.name,
        "time_horizon": intent.time_horizon.name,
        "constraints": asdict(intent.constraints),
        "risk_payload": asdict(intent.risk_payload),
        "signal_hash": intent.signal_hash,
        "metadata": metadata,
        "event_ref": metadata.get("event_ref", ""),
        "action_ref": metadata.get("action_ref", ""),
        "provisional_order_ref": metadata.get("provisional_order_ref", ""),
        "portfolio_decision_ref": metadata.get("portfolio_decision_ref", ""),
        "source_artifact_hash": metadata.get("source_artifact_hash", ""),
        "source_fingerprint": metadata.get("source_fingerprint", ""),
        "strategy_action_hash": metadata.get("strategy_action_hash", ""),
        "candidate_hash": metadata.get("candidate_hash", ""),
        "decision_ref": metadata.get("decision_ref", ""),
        "timestamp": intent.timestamp,
    }


def _intent_to_json_dict(intent: Intent) -> dict[str, Any]:
    return intent_to_json_dict(intent)
