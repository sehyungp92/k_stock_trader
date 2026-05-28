from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import Any, Iterable, Mapping


def _freeze_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))


@dataclass(frozen=True, slots=True)
class MarketBar:
    symbol: str
    timestamp: datetime
    timeframe: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    is_completed: bool = True
    source: str = ""
    source_fingerprint: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("MarketBar.symbol is required")
        if not isinstance(self.timestamp, datetime):
            raise TypeError("MarketBar.timestamp must be a datetime")
        if not self.timeframe:
            raise ValueError("MarketBar.timeframe is required")
        if self.high < self.low:
            raise ValueError("MarketBar.high must be >= low")
        if self.volume < 0:
            raise ValueError("MarketBar.volume must be non-negative")
        object.__setattr__(self, "open", float(self.open))
        object.__setattr__(self, "high", float(self.high))
        object.__setattr__(self, "low", float(self.low))
        object.__setattr__(self, "close", float(self.close))
        object.__setattr__(self, "volume", float(self.volume))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "timeframe": self.timeframe,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "is_completed": self.is_completed,
            "source": self.source,
            "source_fingerprint": self.source_fingerprint,
            "metadata": dict(self.metadata),
        }

    def __reduce__(self):
        return (
            self.__class__,
            (
                self.symbol,
                self.timestamp,
                self.timeframe,
                self.open,
                self.high,
                self.low,
                self.close,
                self.volume,
                self.is_completed,
                self.source,
                self.source_fingerprint,
                dict(self.metadata),
            ),
        )


@dataclass(frozen=True, slots=True)
class MarketFeatureBundle:
    symbol: str
    timestamp: datetime
    features: Mapping[str, Any] = field(default_factory=dict)
    source_fingerprint: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "features", _freeze_mapping(self.features))

    def __reduce__(self):
        return (self.__class__, (self.symbol, self.timestamp, dict(self.features), self.source_fingerprint))


def require_completed_bar(bar: MarketBar) -> MarketBar:
    if not bar.is_completed:
        raise ValueError(f"Incomplete bar cannot be replayed normally: {bar.symbol} {bar.timestamp}")
    return bar


def require_time_ordered(bars: Iterable[MarketBar]) -> list[MarketBar]:
    ordered = list(bars)
    for previous, current in zip(ordered, ordered[1:]):
        if current.timestamp < previous.timestamp:
            raise ValueError("Market bars must be sorted by timestamp")
    return ordered
