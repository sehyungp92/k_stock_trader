"""Past-only, session-grouped validation for learned KALCB candidate lists."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable, Sequence


OUTCOME_FEATURES = frozenset({"entered_shared_core", "selected_first30", "oracle_score",
    "positive_r", "positive_net_eod_r", "negative_mae_abs", "positive_mfe_r_proxy",
    "net_eod_r_proxy", "mae_r_proxy", "optimizer_positive_r_proxy", "optimizer_net_eod_r_proxy"})


def validate_predictor_columns(columns: Iterable[str]) -> None:
    leaked = sorted(set(columns) & OUTCOME_FEATURES)
    if leaked:
        raise ValueError(f"Outcome-derived predictors are forbidden: {leaked}")


def _day(value) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


@dataclass(frozen=True)
class PastOnlyFold:
    train_indices: tuple[int, ...]
    valid_indices: tuple[int, ...]
    train_through: str
    label_through: str
    valid_from: str
    valid_through: str


def past_only_folds(dates: Sequence, *, min_train_sessions: int = 60,
                    validation_sessions: int = 20, embargo_sessions: int = 1,
                    label_end_dates: Sequence | None = None) -> list[PastOnlyFold]:
    if min_train_sessions < 1 or validation_sessions < 1 or embargo_sessions < 0:
        raise ValueError("Invalid chronological window lengths")
    days = [_day(v) for v in dates]
    ends = days if label_end_dates is None else [_day(v) for v in label_end_dates]
    if len(days) != len(ends) or any(end < day for day, end in zip(days, ends)):
        raise ValueError("Invalid label availability dates")
    sessions = sorted(set(days))
    result = []
    for lo in range(min_train_sessions + embargo_sessions, len(sessions), validation_sessions):
        first = sessions[lo]
        train_days = set(sessions[:lo - embargo_sessions])
        valid_days = set(sessions[lo:lo + validation_sessions])
        train = tuple(i for i, d in enumerate(days) if d in train_days and ends[i] < first)
        valid = tuple(i for i, d in enumerate(days) if d in valid_days)
        if len({days[i] for i in train}) < min_train_sessions:
            continue
        result.append(PastOnlyFold(train, valid, max(days[i] for i in train).isoformat(),
            max(ends[i] for i in train).isoformat(), first.isoformat(), max(valid_days).isoformat()))
    if not result:
        raise ValueError("Insufficient sessions for a past-only validation window")
    return result
