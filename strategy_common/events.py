from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from .actions import StrategyAction, action_to_json_dict


def _freeze_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))


def _json_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class DecisionEvent:
    timestamp: datetime
    strategy_id: str
    symbol: str
    decision_code: str
    reason: str
    state_snapshot_ref: str = ""
    actions: tuple[StrategyAction, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "strategy_id", self.strategy_id.upper().strip())
        object.__setattr__(self, "actions", tuple(self.actions))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "strategy_id": self.strategy_id,
            "symbol": self.symbol,
            "decision_code": self.decision_code,
            "reason": self.reason,
            "state_snapshot_ref": self.state_snapshot_ref,
            "actions": [action_to_json_dict(action) for action in self.actions],
            "metadata": _json_value(dict(self.metadata)),
        }


@dataclass(frozen=True, slots=True)
class TradeOutcome:
    strategy_id: str
    symbol: str
    qty: int
    entry_decision_time: datetime
    entry_fill_time: datetime
    entry_price: float
    exit_fill_time: datetime | None = None
    exit_price: float | None = None
    gross_pnl: float = 0.0
    commission: float = 0.0
    net_pnl: float = 0.0
    realized: bool = False
    exit_reason: str = ""
    route_metadata: Mapping[str, Any] = field(default_factory=dict)
    cohort_metadata: Mapping[str, Any] = field(default_factory=dict)
    source_artifact_hash: str = ""
    mfe: float = 0.0
    mae: float = 0.0

    def __post_init__(self) -> None:
        if self.qty <= 0:
            raise ValueError("TradeOutcome.qty must be positive")
        if self.exit_fill_time and self.exit_fill_time < self.entry_fill_time:
            raise ValueError("TradeOutcome.exit_fill_time cannot be before entry_fill_time")
        object.__setattr__(self, "strategy_id", self.strategy_id.upper().strip())
        object.__setattr__(self, "route_metadata", _freeze_mapping(self.route_metadata))
        object.__setattr__(self, "cohort_metadata", _freeze_mapping(self.cohort_metadata))

    @property
    def r_multiple(self) -> float:
        risk = float(self.route_metadata.get("risk_per_share", 0.0) or 0.0)
        if risk <= 0:
            return 0.0
        return (float(self.exit_price or self.entry_price) - self.entry_price) / risk

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "symbol": self.symbol,
            "qty": self.qty,
            "entry_decision_time": self.entry_decision_time.isoformat(),
            "entry_fill_time": self.entry_fill_time.isoformat(),
            "entry_price": self.entry_price,
            "exit_fill_time": self.exit_fill_time.isoformat() if self.exit_fill_time else None,
            "exit_price": self.exit_price,
            "gross_pnl": self.gross_pnl,
            "commission": self.commission,
            "net_pnl": self.net_pnl,
            "realized": self.realized,
            "exit_reason": self.exit_reason,
            "route_metadata": _json_value(dict(self.route_metadata)),
            "cohort_metadata": _json_value(dict(self.cohort_metadata)),
            "source_artifact_hash": self.source_artifact_hash,
            "mfe": self.mfe,
            "mae": self.mae,
            "r_multiple": self.r_multiple,
        }


def decisions_to_json(events: Iterable[DecisionEvent]) -> list[dict[str, Any]]:
    return [event.to_json_dict() for event in events]

