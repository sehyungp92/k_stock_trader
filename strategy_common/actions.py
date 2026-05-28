from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Mapping, TypeAlias

OrderKind: TypeAlias = Literal["MARKET", "LIMIT", "STOP", "STOP_LIMIT", "CLOSE_AUCTION"]


def _freeze_metadata(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))


@dataclass(frozen=True, slots=True)
class SubmitEntry:
    strategy_id: str
    symbol: str
    qty: int
    order_type: OrderKind
    limit_price: float | None
    stop_price: float | None
    reason: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.qty <= 0:
            raise ValueError("SubmitEntry.qty must be positive")
        object.__setattr__(self, "strategy_id", self.strategy_id.upper().strip())
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class SubmitExit:
    strategy_id: str
    symbol: str
    qty: int | None
    order_type: OrderKind
    limit_price: float | None
    reason: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.qty is not None and self.qty <= 0:
            raise ValueError("SubmitExit.qty must be positive or None")
        object.__setattr__(self, "strategy_id", self.strategy_id.upper().strip())
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class SubmitPartialExit:
    strategy_id: str
    symbol: str
    qty: int
    order_type: OrderKind
    limit_price: float | None
    reason: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.qty <= 0:
            raise ValueError("SubmitPartialExit.qty must be positive")
        object.__setattr__(self, "strategy_id", self.strategy_id.upper().strip())
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class SubmitProtectiveStop:
    strategy_id: str
    symbol: str
    qty: int | None
    stop_price: float
    reason: str = "protective_stop"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.qty is not None and self.qty <= 0:
            raise ValueError("SubmitProtectiveStop.qty must be positive or None")
        if self.stop_price <= 0:
            raise ValueError("SubmitProtectiveStop.stop_price must be positive")
        object.__setattr__(self, "strategy_id", self.strategy_id.upper().strip())
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class ReplaceProtectiveStop:
    strategy_id: str
    symbol: str
    stop_price: float
    qty: int | None = None
    reason: str = "replace_protective_stop"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.stop_price <= 0:
            raise ValueError("ReplaceProtectiveStop.stop_price must be positive")
        object.__setattr__(self, "strategy_id", self.strategy_id.upper().strip())
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class CancelOrders:
    strategy_id: str
    symbol: str
    reason: str = "cancel_orders"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "strategy_id", self.strategy_id.upper().strip())
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class FlattenPosition:
    strategy_id: str
    symbol: str
    reason: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "strategy_id", self.strategy_id.upper().strip())
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


StrategyAction: TypeAlias = (
    SubmitEntry
    | SubmitExit
    | SubmitPartialExit
    | SubmitProtectiveStop
    | ReplaceProtectiveStop
    | CancelOrders
    | FlattenPosition
)


def action_to_json_dict(action: StrategyAction) -> dict[str, Any]:
    payload = {name: getattr(action, name) for name in action.__dataclass_fields__}
    payload["action_type"] = type(action).__name__
    if "metadata" in payload:
        payload["metadata"] = dict(payload["metadata"])
    return payload
