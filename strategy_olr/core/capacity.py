"""Bounded, restartable liquidity forecasts using completed observations only.

Forecast quantities constrain requests, never broker fills. Sparse auction
history is explicit; entry capacity remains available as a separate bound.
"""
from datetime import datetime, timedelta, time
from math import floor, isfinite
from statistics import median

from kis_core.trading_calendar import get_trading_calendar
from strategy_common.clock import ensure_kst
from strategy_common.execution_session import (
    bar_execution_issue, bar_has_periodic_auction, bar_reopens_after_label)


def enabled(config):
    return bool(config.capacity_entry_bars or config.capacity_exit_quantile or config.capacity_exit_guard
                or config.capacity_broad_auction_history or config.capacity_sparse_exit_mode != 'native'
                or config.capacity_redistribute and config.capacity_redistribution_evidence=='entry')


def observe_routed(state, bar, config, snapshot):
    """The same explicit observation boundary in replay and the live core.

    Orderable history starts empty and persists only observed auctions for
    today's first N tradable names, held inventory or acknowledged working
    orders. Unrelated feed symbols cannot change its forecast.
    """
    conditional=config.capacity_history_scope in ('conditional_shrink','conditional_exit')
    if (conditional and snapshot
            and snapshot.trade_date==bar.timestamp.date() and phase(bar)=='auction'
            and (snapshot.generated_at.tzinfo is None or snapshot.generated_at>bar.timestamp)):
        raise ValueError('Conditional liquidity requires a selection known before the auction')
    if config.capacity_history_scope == 'orderable':
        symbol = str(bar.symbol).zfill(6)
        selected = {str(c.symbol).zfill(6) for c in snapshot.candidates[:config.overnight_slot_count]
                    if c.tradable} if snapshot and snapshot.trade_date == bar.timestamp.date() else set()
        current = state.symbols.get(symbol)
        committed = current and (current.position is not None or current.pending_entry_order_id
                                 or current.pending_exit_order_id)
        if symbol not in selected and not committed:
            return
    observe(state, bar, config)
    if config.capacity_history_scope=='conditional_exit':
        # Match the forecast target to the planned next-session liquidation.
        # The candidate snapshot may be loaded before its decision clock in a
        # replay, so only seal membership once its timestamp is observable.
        cohorts=state.meta.setdefault('capacity_exit_cohorts_v1',{})
        day=bar.timestamp.date().isoformat()
        for old in list(cohorts):
            if old<day:
                del cohorts[old]
        if (snapshot and snapshot.trade_date==bar.timestamp.date()
                and snapshot.generated_at.tzinfo is not None
                and snapshot.generated_at<=bar.timestamp+timedelta(minutes=5)):
            target=get_trading_calendar().next_trading_day(snapshot.trade_date).isoformat()
            cohort={'selection_day':snapshot.trade_date.isoformat(),
                'available_at':snapshot.generated_at.isoformat(),
                'symbols':sorted(str(c.symbol).zfill(6) for c in snapshot.candidates[:config.overnight_slot_count] if c.tradable)}
            if target in cohorts and cohorts[target]!=cohort:
                raise ValueError('Conditional exit membership changed after selection')
            cohorts[target]=cohort
        if str(bar.symbol).zfill(6) in cohorts.get(day,{}).get('symbols',[]) and phase(bar)=='auction':
            observe(state,bar,config,history_key='capacity_selected_history_v1')
    elif config.capacity_history_scope=='conditional_shrink' and snapshot and snapshot.trade_date==bar.timestamp.date():
        # Selection is known before that session's closing auction. Record
        # every eligible selected name, regardless of whether an order filled.
        selected={str(c.symbol).zfill(6) for c in snapshot.candidates[:config.overnight_slot_count] if c.tradable}
        if str(bar.symbol).zfill(6) in selected and phase(bar)=='auction':
            observe(state,bar,config,history_key='capacity_selected_history_v1')


def seed_history(state, bars, config, available_at):
    """Warm start from sourced past auction events, never future-day outcomes."""
    if config.capacity_history_scope == 'orderable':
        raise ValueError('Orderable history must restore its observed state; broad auction seeding is incompatible')
    if available_at.tzinfo is None:
        raise ValueError('Capacity seed requires an aware availability cutoff')
    bars = tuple(bars)
    if any(b.timestamp.tzinfo is None or b.timestamp+timedelta(minutes=5)>available_at
           or b.timestamp.date()>=available_at.date() or phase(b)!='auction' for b in bars):
        raise ValueError('Capacity seed must contain completed prior-session auctions only')
    if any(not b.source_fingerprint for b in bars):
        raise ValueError('Capacity seed requires source identity')
    # Daily refreshes overlap the previous warm start. Merge dated observations
    # idempotently, rejecting revisions rather than rewriting past evidence.
    import copy
    histories=copy.deepcopy(state.meta.get('capacity_history_v1',{}))
    for history in histories.values():
        if any(datetime.fromisoformat(t)+timedelta(minutes=5)>available_at
               for kind in ('auction','continuous') for t,_ in history[kind]):
            raise ValueError('Cannot reseed capacity state behind its observation clock')
    for bar in sorted(bars,key=lambda b:(b.timestamp,b.symbol)):
        history=histories.setdefault(str(bar.symbol).zfill(6),{'auction':[],'continuous':[]})
        values=dict(history['auction']);stamp=bar.timestamp.isoformat()
        if stamp in values and values[stamp]!=float(bar.volume):
            raise ValueError('Conflicting historical auction volume')
        values[stamp]=float(bar.volume)
        history['auction']=[[t,v] for t,v in sorted(values.items())][-20:]
    state.meta['capacity_history_v1']=histories
    state.meta['capacity_seed_receipt']={'available_at':available_at.isoformat(),'observations':len(bars),
        'sources':sorted({b.source_fingerprint for b in bars})}


def phase(bar):
    if bar.timeframe != '5m' or bar_execution_issue(bar) or bar_has_periodic_auction(bar):
        return None
    stamp = ensure_kst(bar.timestamp)
    calendar = get_trading_calendar()
    opening, closing = calendar.session_hours(stamp.date())
    if (stamp.time().replace(tzinfo=None) == calendar.closing_match_time(bar.symbol, stamp.date())
            and bar.open == bar.high == bar.low == bar.close):
        return 'auction'
    start = stamp.replace(hour=opening.hour, minute=opening.minute, second=0, microsecond=0)
    end = stamp.replace(hour=closing.hour, minute=closing.minute, second=0, microsecond=0)-timedelta(minutes=10)
    if start <= stamp and stamp+timedelta(minutes=5) <= end and not bar_reopens_after_label(bar):
        return 'continuous'
    return None


def observe(state, bar, config, *, history_key='capacity_history_v1'):
    if not enabled(config):
        return
    kind = phase(bar)
    if kind is None:
        return
    histories = state.meta.setdefault(history_key, {})
    history = histories.setdefault(str(bar.symbol).zfill(6), {'auction': [], 'continuous': []})
    rows = history[kind]
    stamp = bar.timestamp.isoformat()
    if rows and stamp == rows[-1][0]:
        if rows[-1][1] != float(bar.volume):
            raise ValueError('Conflicting capacity observation')
        return
    if rows and datetime.fromisoformat(rows[-1][0]) > bar.timestamp:
        raise ValueError('Nonchronological capacity observations')
    rows.append([stamp, float(bar.volume)])
    del rows[:-(20 if kind == 'auction' else 3)]


def forecast(state, bar, config):
    history = state.meta.get('capacity_history_v1', {}).get(str(bar.symbol).zfill(6), {})
    at = bar.timestamp + timedelta(minutes=5)
    evidence = {'asof': at.isoformat(), 'participation': config.capacity_participation}
    limits = []
    if (config.capacity_entry_bars or config.capacity_sparse_exit_mode == 'continuous'
            or config.capacity_redistribute and config.capacity_redistribution_evidence=='entry'):
        rows = [(datetime.fromisoformat(t), v) for t, v in history.get('continuous', [])]
        rows = [(t, v) for t, v in rows if t.date() == bar.timestamp.date() and t+timedelta(minutes=5) <= at
                and t >= bar.timestamp-timedelta(minutes=10)]
        closing = get_trading_calendar().session_hours(bar.timestamp.date())[1]
        # Last eligible continuous fill is at close minus 15 minutes.
        last = bar.timestamp.replace(hour=closing.hour, minute=closing.minute, second=0, microsecond=0)-timedelta(minutes=15)
        remaining = max(0, floor((last-at).total_seconds()/300)+1)
        horizon = min(config.capacity_entry_bars or remaining, remaining)
        quantity = floor(median([v for _, v in rows])*config.capacity_participation)*horizon if rows else 0
        if config.capacity_entry_bars:
            limits.append(quantity)
        evidence['entry'] = {'quantity': quantity, 'horizon': horizon,
            'samples': [[t.isoformat(), v] for t, v in rows]}
    auction = [(datetime.fromisoformat(t), v) for t, v in history.get('auction', [])]
    auction = [(t, v) for t, v in auction if t.date() < bar.timestamp.date()
               and t >= bar.timestamp-timedelta(days=60)]
    exit_qty = None
    base_exit_qty = None
    volume_evidence = {}
    conditioning_evidence = {}
    if len(auction) >= config.capacity_min_exit_observations:
        values = sorted(v for _, v in auction)
        q = config.capacity_exit_quantile or .5
        point = (len(values)-1)*q
        lo = floor(point)
        value = values[lo]+(values[min(lo+1,len(values)-1)]-values[lo])*(point-lo)
        if config.capacity_history_scope in ('conditional_shrink','conditional_exit'):
            broad_value=value
            conditional=state.meta.get('capacity_selected_history_v1',{}).get(str(bar.symbol).zfill(6),{})
            selected=[(datetime.fromisoformat(t),v) for t,v in conditional.get('auction',[])]
            selected=[(t,v) for t,v in selected if t.date()<bar.timestamp.date() and t>=bar.timestamp-timedelta(days=60)]
            n=len(selected);weight=n/(n+config.capacity_history_prior_strength)
            selected_value=None
            if n:
                points=sorted(v for _,v in selected);point=(n-1)*q;lo=floor(point)
                selected_value=points[lo]+(points[min(lo+1,n-1)]-points[lo])*(point-lo)
                value=weight*selected_value+(1-weight)*broad_value
            conditioning_evidence={'target':'next_session_exit' if config.capacity_history_scope=='conditional_exit' else 'signal_session',
                'broad_quantity':floor(broad_value*config.capacity_participation),
                'broad_quantile_volume':broad_value,'selected_quantile_volume':selected_value,
                'selected_observations':n,'prior_strength':config.capacity_history_prior_strength,
                'weight':weight,'samples':[[t.isoformat(),v] for t,v in selected]}
        base_exit_qty = floor(value*config.capacity_participation)
        scale = 1.0
        if config.capacity_volume_exponent:
            from strategy_common.execution_session import session_cutoff
            if ensure_kst(at).time().replace(tzinfo=None) < session_cutoff(at.date(),time(14,30)):
                raise ValueError('Afternoon volume feature is not available before its cutoff')
            s=state.symbols.get(str(bar.symbol).zfill(6))
            candidate=s.candidate if s else None
            # Only the sealed completed-prefix feature is read. Label fields
            # present in research snapshots are deliberately never consumed.
            features=(candidate.metadata.get('afternoon_features',{})
                if candidate and candidate.trade_date==bar.timestamp.date() else {})
            relative=features.get('rel_volume')
            if relative is not None:
                relative=float(relative)
                if not isfinite(relative) or relative<0:
                    raise ValueError('Invalid completed-prefix relative volume')
                scale=min(2.0,max(.5,relative**config.capacity_volume_exponent))
            volume_evidence={'relative_volume':relative,'exponent':config.capacity_volume_exponent,
                'scale':scale,'base_quantity':base_exit_qty,
                'feature_source':'sealed_afternoon_features.rel_volume',
                'status':'available' if relative is not None else 'unconditioned_missing_feature'}
        exit_qty = floor(value*config.capacity_participation*scale)
    evidence['exit'] = {'quantity': exit_qty, 'samples': [[t.isoformat(), v] for t, v in auction],
        'status': 'available' if exit_qty is not None else 'insufficient_history',
        'quantile': config.capacity_exit_quantile or .5, **({'volume_conditioning':volume_evidence} if volume_evidence else {}),
        **({'selection_conditioning':conditioning_evidence} if conditioning_evidence else {})}
    if config.capacity_exit_quantile and exit_qty is not None:
        limits.append(exit_qty)
    if exit_qty is None and config.capacity_sparse_exit_mode == 'continuous':
        # This is an entry feasibility bound, not invented exit evidence.
        limits.append(evidence['entry']['quantity'])
        evidence['sparse_fallback'] = 'remaining_continuous_entry_capacity'
    return min(limits) if limits else None, exit_qty, evidence
