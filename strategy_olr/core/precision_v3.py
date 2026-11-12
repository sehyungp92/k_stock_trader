"""Causal structural confirmation and bounded late-entry helpers; research opt-in."""
from dataclasses import replace
from datetime import timedelta
from strategy_common.execution_session import session_cutoff
from strategy_olr.execution import DECISION_CUTOFF, _running_vwap


def entry_plan(plan, config):
    if not config.capacity_late_confirmation:
        return plan
    if plan.mode != 'confirm_next_bar' or plan.max_signal_bars != 6 or plan.after_bar != 0:
        raise ValueError('Late confirmation is registered only for the incumbent six-bar plan')
    return replace(plan, max_signal_bars=9)


def entry_evidence(bars, bar, config):
    if not (config.capacity_structural_partial or config.capacity_late_confirmation):
        return {}
    prefix=tuple(b for b in bars if b.timestamp.date()==bar.timestamp.date() and b.timestamp<=bar.timestamp)
    if not prefix:
        raise ValueError('Precision entry requires an observed current-session prefix')
    out={}
    if config.capacity_structural_partial:
        out['structural_entry_day_low']=min(float(b.low) for b in prefix)
    if config.capacity_late_confirmation:
        cutoff=session_cutoff(bar.timestamp.date(),DECISION_CUTOFF)
        number=sum(b.timestamp.time().replace(tzinfo=None)>=cutoff for b in prefix)
        out['late_confirmation_evidence']={'bar_number':number,'late':number>6,'original_max_bars':6,'extended_max_bars':9}
    return out


def observe_entry_day(position, bar):
    if bar.timestamp.date()==position.entry_time.date():
        old=position.metadata.get('structural_entry_day_low')
        if old is None:
            raise ValueError('Structural exit requires the original pre-entry session prefix')
        # Later buy fragments reapply the original order metadata. Keep the
        # observed post-entry minimum in a separate key absent from that order
        # so a fragment cannot erase an earlier completed entry-day low.
        observed=position.metadata.get('structural_observed_entry_day_low',old)
        low=min(float(old),float(observed),float(bar.low))
        position.metadata['structural_observed_entry_day_low']=low
        position.metadata['structural_entry_day_low']=low


def structural_evidence(symbol_state, position, bar, risk):
    if bar.timestamp.date()<=position.entry_time.date() or risk<=0:
        return None
    low=position.metadata.get('structural_entry_day_low')
    if low is None or low<=0:
        raise ValueError('Missing persisted entry-session low')
    # Deduplicate without allowing a conflicting revision or future observation.
    unique={}
    for b in symbol_state.session_bars:
        if b.timestamp.date()!=bar.timestamp.date() or b.timestamp>bar.timestamp:
            continue
        if b.timestamp in unique and unique[b.timestamp]!=b:
            raise ValueError('Revised completed bar')
        unique[b.timestamp]=b
    bars=tuple(unique[k] for k in sorted(unique))
    if len(bars)<3 or bars[-1].timestamp!=bar.timestamp or bars[-2].timestamp+timedelta(minutes=5)!=bar.timestamp:
        return None
    if (position.max_favorable_price-position.entry_price)/risk>=.5:
        return None
    checks=[]
    for index in (len(bars)-2,len(bars)-1):
        b=bars[index];vwap=_running_vwap(bars[:index+1]);close_r=(b.close-position.entry_price)/risk
        if close_r>-.4 or b.close>=low or b.close>=vwap:
            return None
        checks.append({'bar':b.timestamp.isoformat(),'close':b.close,'close_r':close_r,'vwap':vwap})
    return {'entry_session_low':low,'completed_bars':checks,'session_bar_count':len(bars),
            'decision_available_at':(bar.timestamp+timedelta(minutes=5)).isoformat()}
