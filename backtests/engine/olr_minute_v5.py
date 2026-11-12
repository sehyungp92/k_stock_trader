"""Research-only minute clock for unchanged five-minute OLR decisions.

This module is not activated by existing runners. Integration must also register
its capacity contract and independently qualify the fine and coarse sources.
The calibration control uses the same minute execution model with fast_exit=False.
"""
from dataclasses import asdict
from datetime import timedelta
from itertools import groupby
from math import floor

from backtests.engine.capacity import BarParticipationScenario, CapacityQuote, closing_auction_event
from kis_core.trading_calendar import get_trading_calendar
from strategy_common.clock import ensure_kst
from strategy_common.execution_session import bar_execution_issue, bar_has_periodic_auction, bar_reopens_after_label

MINUTE_CAPACITY_VERSION = 'causal-minute-participation-olr-v5'


def closing_observation(bar):
    """A sourced closing point can be observed even when it cannot execute.

    Periodic-auction calendars still prohibit capacity. This predicate is only
    for delivery of the existing coarse observation at label + five minutes.
    """
    stamp=ensure_kst(bar.timestamp)
    return (not bar_execution_issue(bar)
            and stamp.time().replace(tzinfo=None)==get_trading_calendar().closing_match_time(bar.symbol,stamp.date())
            and bar.open==bar.high==bar.low==bar.close)


def continuous_minute(bar):
    if bar is None or bar.timeframe != '1m' or bar_execution_issue(bar):
        return False
    at=ensure_kst(bar.timestamp)
    opening,closing=get_trading_calendar().session_hours(at.date())
    first=at.replace(hour=opening.hour,minute=opening.minute,second=0,microsecond=0)
    last=at.replace(hour=closing.hour,minute=closing.minute,second=0,microsecond=0)-timedelta(minutes=10)
    return (not at.second and not at.microsecond and first<=at and at+timedelta(minutes=1)<=last
            and not bar_has_periodic_auction(bar) and not bar_reopens_after_label(bar))


class MinuteParticipationScenario(BarParticipationScenario):
    def quote(self,bar,phase,previous_bar):
        if phase=='auction':
            if not closing_auction_event(bar):
                return CapacityQuote(0,0,None,'single_price_closing_auction_event','unsupported_auction_observation')
            source=bar;basis='single_price_closing_auction_event'
        elif phase=='open':
            basis='previous_contiguous_continuous_1m'
            if not continuous_minute(bar):
                return CapacityQuote(0,0,None,basis,'outside_continuous_interval')
            if (not continuous_minute(previous_bar) or previous_bar.symbol!=bar.symbol
                    or previous_bar.timestamp.date()!=bar.timestamp.date()
                    or previous_bar.timestamp+timedelta(minutes=1)!=bar.timestamp):
                return CapacityQuote(0,0,None,basis,'missing_prior_continuous_interval')
            source=previous_bar
        else:
            return CapacityQuote(0,0,None,'unsupported','unsupported_execution_phase')
        return CapacityQuote(max(0,floor(source.volume*self.participation)),float(source.volume),
            source.timestamp.isoformat(),basis,'available')

    def contract(self):
        value=super().contract()
        value.update(version=MINUTE_CAPACITY_VERSION,opening_basis='previous_contiguous_continuous_1m',
            decision_clock='unchanged_5m_alpha; optional_completed_1m_risk_exit',
            calibration='same_minute_execution_for_control_and_treatment')
        return value


def fixed_exit_plan(config):
    from strategy_olr.core.logic import _exit_plan
    plan=_exit_plan(config)
    if (plan.mode!='managed' or plan.failed_followthrough_bars!=1
            or plan.failed_followthrough_close_r!=-1 or plan.failed_followthrough_mfe_r!=4
            or plan.mfe_fade_start_r!=1 or plan.mfe_fade_gap_r!=1.25 or plan.mfe_fade_floor_r!=0
            or plan.vwap_fail_bars or plan.no_mfe_bars or plan.max_hold_bars or plan.hard_stop_enabled):
        raise ValueError('Minute study requires the unchanged registered two-rule exit plan')
    return plan


def minute_risk_step(adapter,bar,broker,*,fast_exit):
    """Consume only a completed minute; use the original native exit lifecycle."""
    from strategy_olr.core.logic import (_position_risk,_completed_bar_exit_reason,
        _managed_market_exit_at)
    from strategy_common.actions import CancelOrders
    at=bar.timestamp+timedelta(minutes=1)
    if broker._decision_available_at!=at:
        raise ValueError('Minute completion clock is not bound')
    symbol=bar.symbol;state=adapter.state;ss=state.symbols.get(symbol)
    if ss is None or ss.position is None or ss.position.qty_open<=0:
        return []
    position=ss.position
    if bar.timestamp<position.entry_time or (bar.timestamp==position.entry_time
            and position.metadata.get('execution_phase') not in ('open','interval')):
        return []
    if bar_execution_issue(bar):return []
    position.update_mark(bar.high,bar.low)
    if not fast_exit or bar.timestamp.date()<=position.entry_time.date() or not continuous_minute(bar):
        return []
    plan=fixed_exit_plan(adapter.config);risk=_position_risk(position)
    if risk<=0:return []
    pending=ss.pending_exit_order_id
    if pending:
        role=state.order_roles.get(pending,{})
        replaceable=(role.get('order_type')=='CLOSE_AUCTION' and role.get('reason')=='next_close_exit') or (
            role.get('order_type')=='MARKET' and role.get('reason')=='weak_followthrough_partial')
        if (not adapter.config.capacity_exit_priority or not replaceable
                or symbol in state.meta.get('priority_exit_pending',{})):
            return []
    reason=_completed_bar_exit_reason((bar,),0,position.entry_price,risk,
        position.max_favorable_price,plan,0)['reason']
    if reason not in getattr(broker,'minute_exit_reasons',('failed_followthrough','mfe_fade')):
        return []
    if not reason:return []
    if reason not in ('failed_followthrough','mfe_fade'):
        raise ValueError('Unregistered minute exit reason')
    evidence=dict(symbol=symbol,timestamp=bar.timestamp.isoformat(),available_at=at.isoformat(),
        bar=bar.to_json_dict(),entry_time=position.entry_time.isoformat(),entry_price=position.entry_price,
        risk=risk,high_water=position.max_favorable_price,quantity=position.qty_open,
        reason=reason,plan=asdict(plan),pending_exit=pending or None)
    broker.journal.append(dict(event='minute_exit_signal_v5',**evidence))
    if pending:
        state.meta.setdefault('priority_exit_pending',{})[symbol]=dict(cancel_order_id=pending,
            reason=reason,trigger_bar=bar.timestamp.isoformat())
        adapter._submit_actions([CancelOrders(adapter.strategy_id,symbol,'priority_risk_exit_cancel_auction')],broker,bar.timestamp)
        # Cancellation acknowledgement remains the native state-machine event;
        # replacement submission uses its existing remaining-quantity logic.
        return adapter._sync_order_updates(broker,bar.timestamp)
    result=_managed_market_exit_at(state,ss,bar.timestamp,position,plan,reason,risk)
    adapter._submit_actions(result.actions,broker,bar.timestamp)
    return list(result.decisions)+adapter._sync_order_updates(broker,bar.timestamp)


class MinuteOLRDriver:
    """Coarse decisions become observable only after their full five-minute bar.

    Fine bars reach the native broker first. Completed coarse observations are
    delivered by availability timers, including through gaps in the minute feed.
    The input schedules remain separate audit evidence; neither replaces the other.
    """
    def __init__(self,broker,adapter,*,fast_exit):
        from backtests.auto.shared.cache_keys import stable_signature
        if not isinstance(broker.capacity_policy,MinuteParticipationScenario):
            raise ValueError('Minute driver requires registered minute capacity')
        if adapter.strategy_id!='OLR':raise ValueError('Only standalone OLR is supported')
        fixed_exit_plan(adapter.config)
        self.broker,self.adapter,self.fast_exit=broker,adapter,bool(fast_exit)
        provider=getattr(adapter,'qualification_snapshot_sequence',None)
        snapshots=provider() if provider else [s for _,s in sorted(adapter.snapshots.items())]
        rows=[s.to_json_dict() for s in snapshots]
        broker.qualification_snapshots=rows
        broker.qualification_snapshot_scopes={'OLR':rows}
        broker.qualification_snapshot_hash=stable_signature(rows)
        broker.expected_batches=[];broker.coarse_expected_batches=[]
        broker.capacity_shadow_brokers=[];broker.expected_strategy_state_timestamps={'OLR':[]}
        broker.minute_clock_schedule=[];broker.minute_fast_exit=bool(fast_exit)
        broker.minute_exit_plan=asdict(fixed_exit_plan(adapter.config))
        broker.minute_exit_priority=bool(adapter.config.capacity_exit_priority)
        broker.minute_day_scopes=[];broker.minute_coarse_nav=[]

    def run_day(self,fine_bars,coarse_bars,*,symbols=None):
        from strategy_olr.core.logic import on_olr_timer
        broker,adapter=self.broker,self.adapter
        fine=tuple(fine_bars);coarse=tuple(coarse_bars)
        if not fine or not coarse:raise ValueError('Both independently bound source clocks are required')
        if any(b.timeframe!='1m' or not b.is_completed or not b.source_fingerprint
               or b.timestamp.tzinfo is None or b.timestamp.second or b.timestamp.microsecond for b in fine):
            raise ValueError('Fine input must contain sourced completed aligned minutes')
        if any(b.timeframe!='5m' or not b.is_completed or not b.source_fingerprint for b in coarse):
            raise ValueError('Coarse input must retain sourced native bars')
        if len({b.timestamp.date() for b in (*fine,*coarse)})!=1:
            raise ValueError('Driver requires exactly one session')
        key=lambda b:(b.timestamp,b.symbol)
        if list(fine)!=sorted(fine,key=key) or len({key(b) for b in fine})!=len(fine):
            raise ValueError('Fine source order or uniqueness violation')
        if list(coarse)!=sorted(coarse,key=key) or len({key(b) for b in coarse})!=len(coarse):
            raise ValueError('Coarse source order or uniqueness violation')
        scope=dict(event='minute_day_scope_v5',day=str(fine[0].timestamp.date()),
            symbols=sorted(symbols if symbols is not None else {b.symbol for b in fine}))
        broker.minute_day_scopes.append(scope);broker.journal.append(scope)
        batches=[tuple(items) for _,items in groupby(coarse,key=lambda b:b.timestamp)]
        cursor=0;decisions=[]
        def release(until):
            nonlocal cursor
            while cursor<len(batches) and batches[cursor][0].timestamp+timedelta(minutes=5)<=until:
                batch=batches[cursor];cursor+=1;available=batch[0].timestamp+timedelta(minutes=5)
                broker._decision_available_at=available
                broker.coarse_expected_batches.append(tuple(b.to_json_dict() for b in batch))
                broker.minute_clock_schedule.append(dict(kind='coarse',label=batch[0].timestamp.isoformat(),
                    available_at=available.isoformat(),delivered_by=until.isoformat(),symbols=[b.symbol for b in batch]))
                broker.journal.append(dict(event='coarse_observation_v5',**broker.minute_clock_schedule[-1]))
                decisions.extend(adapter.on_bars(batch,broker))
                decisions.extend(adapter._sync_new_fills(broker))
                decisions.extend(adapter._sync_order_updates(broker,batch[0].timestamp))
                broker.minute_coarse_nav.append(dict(timestamp=batch[0].timestamp.isoformat(),
                    available_at=available.isoformat(),equity=broker._portfolio_equity()))
        for stamp,items in groupby(fine,key=lambda b:b.timestamp):
            batch=tuple(items);release(stamp)
            broker.expected_batches.append(tuple(b.to_json_dict() for b in batch))
            broker.expected_strategy_state_timestamps['OLR'].append(stamp.isoformat())
            def fills():decisions.extend(adapter.on_fills(broker))
            broker.process_batch(batch,on_fills=fills)
            decisions.extend(adapter._sync_new_fills(broker))
            decisions.extend(adapter._sync_expired_orders(broker,stamp))
            decisions.extend(adapter._sync_order_updates(broker,stamp))
            available=stamp+timedelta(minutes=1);broker._decision_available_at=available
            broker.journal.append(dict(event='minute_observation_v5',timestamp=stamp.isoformat(),
                available_at=available.isoformat(),symbols=[b.symbol for b in batch]))
            for bar in batch:decisions.extend(minute_risk_step(adapter,bar,broker,fast_exit=self.fast_exit))
            observer=getattr(self,'structural_observer',None)
            if observer is not None:observer.observe(adapter,broker,available,batch)
            release(available)
            if stamp==fine[-1].timestamp:
                # Drain non-executable closing timers before the last native
                # state checkpoint. Never create a second checkpoint at a label.
                remaining=batches[cursor:]
                if any(not closing_observation(b) for pending in remaining for b in pending):
                    raise ValueError('Unobserved continuous coarse timer beyond final minute')
                for pending in remaining:
                    available=max(available,pending[0].timestamp+timedelta(minutes=5))
                    release(available)
            broker._decision_available_at=available
            decisions.extend(on_olr_timer(adapter.state,available,adapter.config).decisions)
            decisions.extend(adapter._sync_order_updates(broker,stamp))
            adapter._record_reconciliation(broker,stamp)
            broker._decision_available_at=None
        # A closing auction is a point print. Its native coarse callback may
        # occur after the final minute; do not invent executable timer bars.
        assert cursor==len(batches)
        broker._decision_available_at=None
        return decisions
