from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from strategy_common.market import MarketBar, require_completed_bar


@dataclass(frozen=True, slots=True)
class ReplayEvent:
    timestamp: datetime
    event_type: str
    symbol: str
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.event_type:
            raise ValueError("ReplayEvent.event_type is required")
        if not self.symbol:
            raise ValueError("ReplayEvent.symbol is required")
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload or {})))

    @classmethod
    def from_bar(cls, bar: MarketBar) -> "ReplayEvent":
        require_completed_bar(bar)
        return cls(
            timestamp=bar.timestamp,
            event_type="bar",
            symbol=bar.symbol,
            payload={"bar": bar},
        )

    @property
    def bar(self) -> MarketBar | None:
        value = self.payload.get("bar")
        return value if isinstance(value, MarketBar) else None

    def to_json_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key, value in self.payload.items():
            if isinstance(value, MarketBar):
                payload[key] = value.to_json_dict()
            else:
                isoformat = getattr(value, "isoformat", None)
                payload[key] = isoformat() if callable(isoformat) else value
        return {
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "symbol": self.symbol,
            "payload": payload,
        }


def require_time_ordered_events(events: Iterable[ReplayEvent]) -> list[ReplayEvent]:
    ordered = list(events)
    for previous, current in zip(ordered, ordered[1:]):
        if current.timestamp < previous.timestamp:
            raise ValueError("Replay events must be sorted by timestamp")
    return ordered

