"""Persist causal signal observations independently of order admission.

Observation only: this ledger does not loosen eligibility or create an order.
The original confirmation, actual level crossing and reclaim are distinct.
"""
from datetime import timedelta
from strategy_olr.execution import find_olr_entry_signal, _running_vwap, DECISION_CUTOFF
from strategy_common.execution_session import session_cutoff

KEY='structural_signal_episodes_v1'


def observe(state,bar,candidate,snapshot,config,plan,portfolio):
    if candidate is None or not candidate.tradable:return
    ss=state.symbol_state(bar.symbol);bars=tuple(ss.session_bars)
    if not bars or bars[-1]!=bar:raise ValueError('Signal observer requires the exact completed prefix')
    if bar.timestamp.time().replace(tzinfo=None)<session_cutoff(bar.timestamp.date(),DECISION_CUTOFF):return
    number=sum(b.timestamp.time().replace(tzinfo=None)>=session_cutoff(bar.timestamp.date(),DECISION_CUTOFF) for b in bars)
    ledger=state.meta.setdefault(KEY,{})
    day=ledger.setdefault(str(bar.timestamp.date()),{});row=day.setdefault(bar.symbol,dict(episodes=[],observations=0))
    at=(bar.timestamp+timedelta(minutes=5)).isoformat()
    if row.get('last_available_at')==at:return
    if row.get('last_available_at','')>at:raise ValueError('Signal observation clock reversal')
    row['last_available_at']=at;row['observations']+=1
    blockers=[]
    if ss.position is not None:blockers.append('held')
    if ss.pending_entry_order_id:blockers.append('pending_entry')
    if ss.entry_attempted:blockers.append('attempt_consumed')
    if candidate.symbol not in {c.symbol for c in snapshot.candidates[:config.overnight_slot_count] if c.tradable}:blockers.append('outside_slots')
    if portfolio.cash<=0:blockers.append('no_cash')
    # A transient weak bar is recorded but never invalidates a confirmed native
    # acquisition. Structural episodes carry their own pre-existing price level.
    for ep in row['episodes']:
        if ep['kind']!='native_confirmation' and ep['status']=='accepted' and bar.close<ep['level']:
            ep.update(status='invalidated',invalidated_at=at,invalidation_close=bar.close)
    signal=find_olr_entry_signal(bars,candidate,plan,require_fill_bar=False)
    if signal and not any(e['kind']=='native_confirmation' for e in row['episodes']):
        origin=bars[signal.signal_index]
        row['episodes'].append(dict(id=f'{bar.symbol}:{origin.timestamp.isoformat()}:confirmation',kind='native_confirmation',
            status='observed',signal_label=origin.timestamp.isoformat(),known_at=(origin.timestamp+timedelta(minutes=5)).isoformat(),
            observed_at=at,blockers=blockers,admission_authority=False))
    if len(bars)<2 or number>plan.max_signal_bars:return
    prior=bars[:-1];level=max(float(candidate.prior_day_high or 0),max(b.high for b in prior))
    def add(kind,level,level_at):
        row['episodes'].append(dict(id=f'{bar.symbol}:{bar.timestamp.isoformat()}:{kind}',kind=kind,status='accepted',
            level=level,level_known_at=level_at,known_at=at,signal_label=bar.timestamp.isoformat(),
            blockers=blockers,admission_authority=False))
    if prior[-1].close<=level<bar.close:
        add('level_breakout',level,(prior[-1].timestamp+timedelta(minutes=5)).isoformat())
    prior_vwap=_running_vwap(prior);current_vwap=_running_vwap(bars)
    if prior[-1].close<prior_vwap and bar.close>current_vwap:
        add('vwap_reclaim',current_vwap,at)
