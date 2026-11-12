"""Read-only parent account/exposure census and separately copied reclaim orders.

The parent feed, snapshots and adapter transitions are never enlarged. At the
end of the actual minute90 decision batch, before its timer, a copy may consume
the missing *watch-only* source prefix and preview a last-ranked new issuer.
This is a conditional order preview, not an executable policy or future fill.
"""
from collections import Counter, defaultdict
from copy import copy, deepcopy
from dataclasses import asdict, replace, fields
from datetime import datetime, timedelta
from backtests.strategies.kalcb.mechanism_observer_v2 import clone_broker


def append_snapshot(base, row, value):
    from backtests.strategies.kalcb import executed_value as ev
    snap=ev.snapshot_for(row['trade_date'],[(row,value)],source='registered_reclaim_observer_v1')
    candidate=replace(snap.candidates[0],metadata={**snap.candidates[0].metadata,
        'candidate_rank':len(base.candidates)+1,'frontier_rank':len(base.candidates)+1,
        'reclaim_observer_only':True})
    candidates=base.candidates+(candidate,)
    meta={**base.metadata,'active_symbols':[c.symbol for c in candidates]}
    for k in ('active_symbol_count','candidate_pool_count','selection_count','frontier_symbol_count'):meta[k]=len(candidates)
    return replace(base,candidates=candidates,metadata=meta,generated_at=snap.generated_at)


def source_prefix(row, available):
    from strategy_common.market import MarketBar
    if datetime.fromisoformat(row['feature_available_at'])!=available:raise ValueError('Wrong preview clock')
    rows=row['opening_bars']
    if len(rows)!=18:raise ValueError('Incomplete preview prefix')
    opening=available-timedelta(minutes=90)
    result=[]
    for i,r in enumerate(rows):
        stamp=datetime.fromisoformat(r['timestamp'])
        if stamp!=opening+timedelta(minutes=5*i) or r['symbol']!=row['symbol']:raise ValueError('Invalid prefix time/symbol')
        result.append(MarketBar(row['symbol'],stamp,'5m',**{k:r[k] for k in ('open','high','low','close','volume')}))
    return result


def account_snapshot(adapter,broker):
    positions=[]
    for p in broker.positions.values():
        mark=broker.last_prices.get(p.symbol,p.avg_price)
        state=adapter.state.symbols.get(p.symbol);position=state.position if state else None
        candidate=state.candidate if state else None
        positions.append({'symbol':p.symbol,'qty':p.qty,'cost_basis':p.avg_price,'mark':mark,
            'marked_notional':p.qty*mark,'sector':position.sector if position else None,
            'daily_atr':candidate.daily_atr if candidate else None,
            'prior_day_close':candidate.prior_day_close if candidate else None,
            'entry_route':position.metadata.get('entry_route') if position else None})
    return {'portfolio':asdict(adapter._portfolio_view(broker)),'actual_cash':broker.cash,
        'actual_equity':broker._portfolio_equity(),'buying_power_available':broker._buying_power_available(),
        'buying_power_leverage':broker.buying_power_leverage,'positions':positions,
        'pending_orders':[asdict(o) for o in broker.orders],
        'session_meta':deepcopy(adapter.state.meta),
        'last_mark_timestamp':str(broker.timestamps[-1]) if broker.timestamps else None}


class ReclaimObserver:
    def __init__(self,opportunities):
        self.by_day=defaultdict(list);self.seen=set();self.previews=[];self.entries=[]
        self.depth=0;self.current=None
        for row,scores in opportunities:
            if not row['guard_at_rank1'] or not row['quarter_preview']['ready']:raise ValueError('Undeclared population')
            if row['opportunity_id'] in self.seen:raise ValueError('Duplicate opportunity')
            self.seen.add(row['opportunity_id']);self.by_day[row['trade_date']].append((row,scores))
        self.expected=set(self.seen);self.seen.clear()

    def preview(self,adapter,broker,row,scores,available):
        from strategy_kalcb.core.candidate_updates import observe_candidate_bar
        bars=source_prefix(row,available)
        snapshot=adapter.snapshots.get(available.date())
        updates=[s for s in adapter.timed_snapshots.get(available.date(),()) if s.generated_at<=available]
        if updates:snapshot=updates[-1]
        if snapshot is None:raise ValueError('Missing actual parent snapshot')
        record={'opportunity_id':row['opportunity_id'],'available_at':available.isoformat(),
            'forecasts':scores,'account_before':account_snapshot(adapter,broker),
            'already_candidate':row['symbol'] in snapshot.by_symbol(),
            'scope':'Separate copied-account last-ranked order preview after actual parent bars, before timer; never fills/PnL.'}
        state=adapter.state.symbols.get(row['symbol'])
        original_prefix=list(state.bars_today) if state else []
        original_prefix=[b for b in original_prefix if b.timestamp.date()==available.date()]
        if len(original_prefix)>18:raise ValueError('Future parent prefix')
        for b,expected in zip(original_prefix,bars):
            if b.timestamp!=expected.timestamp or any(getattr(b,k)!=getattr(expected,k) for k in ('open','high','low','close','volume')):
                raise ValueError('Parent/source prefix mismatch')
        record['native_overlap_bars']=len(original_prefix)
        if record['already_candidate']:
            record['reason']='already_parent_candidate';return record
        if (state and (state.position is not None or state.pending_entry_order_id)) or any(p.symbol==row['symbol'] and p.qty for p in broker.positions.values()):
            record['reason']='already_held_or_pending';return record
        if adapter.config.frontier_rotation_enabled or adapter.frontier_shadow is not None:raise ValueError('Undeclared frontier shadow')
        shadow=copy(adapter);shadow.state=deepcopy(adapter.state)
        for k in ('_terminal_cursor','_daily_active_symbols','_activated_intraday_hash','frontier_promotion_log'):
            setattr(shadow,k,deepcopy(getattr(adapter,k)))
        shadow.timed_snapshots=dict(adapter.timed_snapshots);shadow.frontier_shadow=None
        # Only the copy warms missing inactive watch observations. Already completed
        # native bars are checked above and never replayed into the real account.
        count=len(original_prefix)
        if count==18:
            raise ValueError('Noncandidate unexpectedly already observed final bar; explicit boundary required')
        for b in bars[count:-1]:observe_candidate_bar(shadow.state,b,adapter.snapshots.get(available.date()))
        raw=next((s for s in scores if s['recipe']=='raw48'),None)
        value=min(raw['equity_forecast'].values()) if raw else row['candidate']['selection_score']
        shadow.timed_snapshots[available.date()]=[*updates,append_snapshot(snapshot,row,value)]
        trial=clone_broker(broker);before={o.order_id for o in trial.orders}
        self.depth+=1
        try:decisions=self.original_on_bar(shadow,bars[-1],trial)
        finally:self.depth-=1
        orders=[asdict(o) for o in trial.orders if o.order_id not in before]
        record.update(new_orders=orders,decisions=[d.to_json_dict() for d in decisions],
            native_order_submitted=any(o['symbol']==row['symbol'] for o in orders),
            copied_watch_bars=17-count,preview_score=value)
        return record

    def install(self):
        from backtests.strategies.kalcb.runner import KALCBReplayAdapter as Adapter
        from backtests.engine.sim_broker import SimBroker
        from strategy_common.actions import SubmitEntry
        self.classes=(Adapter,SimBroker)
        self.original_on_bar=Adapter.on_bar;self.original_end=Adapter.on_timestamp_end
        self.original_actions=Adapter._submit_actions;self.original_submit=SimBroker.submit
        owner=self
        def actions(adapter,values,broker,stamp):
            old=owner.current;owner.current=adapter
            try:return owner.original_actions(adapter,values,broker,stamp)
            finally:owner.current=old
        def submitted(broker,action,stamp):
            row=None
            if not owner.depth and isinstance(action,SubmitEntry) and owner.current is not None:
                row={'submitted_at':stamp.isoformat(),'decision_available_at':str(getattr(broker,'_decision_available_at',None)),
                    'action':{f.name:(deepcopy(dict(action.metadata)) if f.name=='metadata' else deepcopy(getattr(action,f.name))) for f in fields(action)},
                    'account_before':account_snapshot(owner.current,broker)}
            result=owner.original_submit(broker,action,stamp)
            if row is not None:row['order_id']=result;owner.entries.append(row)
            return result
        def ended(adapter,stamp,bars,broker):
            if not owner.depth:
                available=stamp+timedelta(minutes=5)
                for row,scores in owner.by_day.get(str(stamp.date()),()):
                    key=row['opportunity_id'];clock=datetime.fromisoformat(row['feature_available_at'])
                    if key in owner.seen or available<clock:continue
                    if available==clock:record=owner.preview(adapter,broker,row,scores,available)
                    else:record={'opportunity_id':key,'reason':'native_decision_clock_absent','forecasts':scores}
                    owner.previews.append(record);owner.seen.add(key)
            return owner.original_end(adapter,stamp,bars,broker)
        Adapter._submit_actions=actions;SimBroker.submit=submitted;Adapter.on_timestamp_end=ended

    def restore(self):
        Adapter,Broker=self.classes
        Adapter._submit_actions=self.original_actions;Broker.submit=self.original_submit;Adapter.on_timestamp_end=self.original_end

    def summarize(self,fills):
        for missing in sorted(self.expected-self.seen):
            self.previews.append({'opportunity_id':missing,'reason':'no_native_clock_observed'})
        by_order={r['order_id']:r for r in self.entries if r['order_id']}
        matched=[f for f in fills if f['side']=='BUY' and f['order_id'] in by_order]
        if len(matched)!=sum(f['side']=='BUY' for f in fills):raise ValueError('Actual entry exposure census missed a fill')
        return {'preview_opportunities':len(self.previews),'copied_order_previews':sum(r.get('native_order_submitted',False) for r in self.previews),
            'preview_reasons':dict(Counter(r.get('reason','native_guard_preview') for r in self.previews)),
            'actual_entry_submissions':len(self.entries),'actual_filled_entry_orders':len({f['order_id'] for f in matched}),
            'actual_unique_entry_issuer_days':len({(f['symbol'],str(f['timestamp'])[:10]) for f in matched}),
            'scope':'Order previews are mutually exclusive conditional alternatives. Actual entry census is unchanged parent exposure, not incremental alpha.'}
