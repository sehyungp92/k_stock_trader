"""0039 private strategy module: one expiry-boundary native eligibility decision."""
from datetime import datetime, timedelta
from strategy_common.execution_session import session_cutoff
from strategy_olr.execution import (DECISION_CUTOFF, OLREntrySignal,
    find_olr_entry_signal, _common_entry_bar_passes, _running_vwap, _close_location)


def prefix(state, candidate, current, cutoff, count):
    if current is None or not current.is_completed or current.timeframe != '5m':
        return None
    old = state.symbol_state(candidate.symbol).session_bars
    bars = tuple(b for b in old if b.timestamp.date() == current.timestamp.date()
                 and b.timestamp < current.timestamp) + (current,)
    times = [b.timestamp for b in bars]
    if times != sorted(set(times)):
        raise ValueError('Unordered selection source prefix')
    start = datetime.combine(current.timestamp.date(), cutoff, current.timestamp.tzinfo)
    required = [start + timedelta(minutes=5*i) for i in range(-1, count)]
    if any(t not in times for t in required):
        return None
    return bars


def prepare(state, bars, config, snapshot):
    if not state.meta.get('expired_selection_enabled_v1') or snapshot is None:
        return
    from .logic import _entry_plan
    plan = _entry_plan(config)
    if (plan.mode != 'confirm_next_bar' or plan.max_signal_bars != 6
        or config.allocation_mode not in ('rank_decay', 'rank_weighted')
        or config.overnight_slot_count != 6):
        raise ValueError('0039 requires the registered native six-slot/six-bar policy')
    stamp = bars[0].timestamp
    cutoff = session_cutoff(stamp.date(), DECISION_CUTOFF)
    end = datetime.combine(stamp.date(), cutoff, stamp.tzinfo) + timedelta(minutes=25)
    if stamp != end or snapshot.trade_date != stamp.date():
        return
    history = state.meta.setdefault('expired_selection_cohorts_v1', {})
    key = (stamp+timedelta(minutes=5)).isoformat()
    if key in history:
        return  # Serialized decision is immutable, including an empty result.
    original = tuple(c for c in snapshot.candidates[:6] if c.tradable)
    current = {b.symbol:b for b in bars}
    reserved, expired, missing = [], [], []
    for c in original:
        ss = state.symbol_state(c.symbol)
        source = prefix(state,c,current.get(c.symbol),cutoff,6)
        committed = ss.position is not None or ss.pending_entry_order_id or ss.entry_attempted
        if source is None:
            missing.append(c.symbol)
        if committed or source is None or find_olr_entry_signal(source,c,plan,require_fill_bar=False):
            reserved.append(c.symbol)
        else:
            expired.append(c.symbol)
    allowed = []
    if len(original) >= config.min_selected:
        for c in snapshot.candidates[6:]:
            ss = state.symbol_state(c.symbol)
            source = prefix(state,c,current.get(c.symbol),cutoff,6)
            if not c.tradable or ss.position is not None or ss.pending_entry_order_id or ss.entry_attempted or source is None:
                continue
            previous = [b for b in source if b.timestamp.time().replace(tzinfo=None)<cutoff]
            if _common_entry_bar_passes(source[-1],_running_vwap(source),_close_location(source[-1]),previous[-1].close,plan):
                allowed.append(c)
    allowed = sorted(allowed,key=lambda c:(c.rank,c.symbol))[:len(expired)]
    denominator = sum(1/max(float(c.rank),1.)**max(config.rank_decay,0.) for c in original)
    history[key] = dict(available_at=key,source_label=stamp.isoformat(),snapshot_hash=snapshot.artifact_hash,
        reserved=reserved,expired=expired,missing=missing,
        candidates={c.symbol:dict(rank=c.rank,weight=min(config.max_position_pct,
            config.target_gross_exposure/max(float(c.rank),1.)**max(config.rank_decay,0.)/denominator)) for c in allowed})


def proposal(state, bar, snapshot, candidate):
    if not state.meta.get('expired_selection_enabled_v1'):
        return None
    key = (bar.timestamp+timedelta(minutes=5)).isoformat()
    cohort = state.meta.get('expired_selection_cohorts_v1',{}).get(key)
    if not cohort or candidate.symbol not in cohort['candidates']:
        return None
    if cohort['snapshot_hash'] != snapshot.artifact_hash or cohort['source_label'] != bar.timestamp.isoformat():
        raise ValueError('Selection decision source mismatch')
    index = len(state.symbol_state(candidate.symbol).session_bars)-1
    return cohort['candidates'][candidate.symbol]['weight'], OLREntrySignal(index+1,index,'expired_selection_entry'), cohort
