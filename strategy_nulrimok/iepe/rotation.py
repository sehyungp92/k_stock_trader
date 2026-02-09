"""Nulrimok Active Set Rotation."""

from datetime import datetime
from typing import Dict, List, Tuple

from ..config.constants import ROTATION_INTERVAL_MIN
from .entry import TickerEntryState, EntryState


def rotate_active_set(active_set: List[str], overflow: List[str],
                      entry_states: Dict[str, TickerEntryState],
                      near_band_recently: Dict[str, bool],
                      daily_ranks: Dict[str, float],
                      now: datetime, last_rotation: datetime) -> Tuple[List[str], List[str], datetime]:

    if (now - last_rotation).total_seconds() / 60.0 < ROTATION_INTERVAL_MIN or not overflow:
        return active_set, overflow, last_rotation

    # Find demotion candidate
    candidates = [(t, daily_ranks.get(t, 0)) for t in active_set
                  if entry_states.get(t, TickerEntryState(t)).state not in (EntryState.ARMED, EntryState.TRIGGERED)
                  and not near_band_recently.get(t, False)]

    if not candidates:
        return active_set, overflow, now

    candidates.sort(key=lambda x: x[1])
    demote = candidates[0][0]
    promote = overflow[0]

    new_active = [t for t in active_set if t != demote] + [promote]
    new_overflow = overflow[1:] + [demote]

    return new_active, new_overflow, now
