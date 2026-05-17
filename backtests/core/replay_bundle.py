from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Generic, TypeVar

from .replay_events import ReplayEvent, require_time_ordered_events

DataT = TypeVar("DataT")


@dataclass(frozen=True, slots=True)
class ReplayBundle(Generic[DataT]):
    data: DataT
    cache_key: str
    cache_source_fingerprint: str
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EventReplayBundle:
    events: tuple[ReplayEvent, ...]
    source_fingerprint: str
    data_root: Path | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        ordered = tuple(require_time_ordered_events(self.events))
        object.__setattr__(self, "events", ordered)

    @property
    def cache_source_fingerprint(self) -> str:
        return self.source_fingerprint

