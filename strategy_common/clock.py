from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

KST = ZoneInfo("Asia/Seoul")
UTC = timezone.utc


@dataclass(frozen=True, slots=True)
class ClockContext:
    """Deterministic exchange clock shared by live adapters and replay."""

    now_kst: datetime

    def __post_init__(self) -> None:
        value = self.now_kst
        if value.tzinfo is None:
            value = value.replace(tzinfo=KST)
        object.__setattr__(self, "now_kst", value.astimezone(KST))

    @property
    def now_utc(self) -> datetime:
        return self.now_kst.astimezone(UTC)

    @property
    def now_epoch(self) -> float:
        return self.now_utc.timestamp()

    @classmethod
    def fixed(cls, value: datetime) -> "ClockContext":
        return cls(value)

    @classmethod
    def real(cls) -> "ClockContext":
        return cls(datetime.now(KST))


def ensure_kst(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=KST)
    return value.astimezone(KST)


def clock_from(value: datetime | ClockContext | None = None) -> ClockContext:
    if isinstance(value, ClockContext):
        return value
    if isinstance(value, datetime):
        return ClockContext(value)
    return ClockContext.real()

