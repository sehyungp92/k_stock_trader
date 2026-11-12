"""Availability-checked intraday candidate activation shared with runtime users."""
from datetime import timedelta, datetime
from .logic import _ensure_session, _snapshot_entry_context


def activate_candidate_snapshot(state, snapshot, available_at):
    if available_at.tzinfo is None or snapshot.trade_date != available_at.date():
        raise ValueError('Intraday candidate activation requires its dated information clock')
    if snapshot.generated_at is None or snapshot.generated_at > available_at:
        raise ValueError('Candidate update is not yet available')
    for c in snapshot.candidates:
        stamp=c.metadata.get('feature_available_at')
        if stamp and datetime.fromisoformat(stamp)>available_at:
            raise ValueError('Future candidate features')
    state.snapshot_hash=snapshot.artifact_hash
    state.source_fingerprint=snapshot.source_fingerprint
    state.meta['entry_context']=_snapshot_entry_context(snapshot)


def observe_candidate_bar(state, bar, opening_snapshot):
    """Warm a declared watch symbol without admission, ordering, or forecasts."""
    if not bar.is_completed or bar.timeframe.lower()!='5m':
        raise ValueError('Watch state requires completed five-minute observations')
    _ensure_session(state,bar,opening_snapshot)
    s=state.symbol_state(bar.symbol)
    if s.session_date!=bar.timestamp.date():s.reset_for_session(bar.timestamp.date(),None,0)
    if s.bars_today and s.bars_today[-1].timestamp>=bar.timestamp:
        raise ValueError('Duplicate watch observation')
    s.add_bar(bar)
