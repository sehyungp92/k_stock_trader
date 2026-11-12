"""Research interval service: commitment precedes observed executable volume.

OHLCV does not identify queue priority. Orders committed by the interval start
share a declared fraction of its observed volume, priced at its closing mark;
cash and inventory become available only at completion. The point auction keeps
its native clock. This is an explicit research proxy, never verified tick fills.
"""
from math import floor
from .capacity import CapacityQuote, closing_auction_event
from .olr_minute_v5 import MinuteParticipationScenario, continuous_minute

VERSION = 'olr-completed-interval-service-v1'


class IntervalParticipationScenario(MinuteParticipationScenario):
    def quote(self, bar, phase, previous_bar):
        if phase == 'interval' and continuous_minute(bar):
            return CapacityQuote(floor(bar.volume*self.participation), float(bar.volume),
                                 bar.timestamp.isoformat(), 'completed_interval_volume', 'available')
        if phase == 'auction' and closing_auction_event(bar):
            return CapacityQuote(floor(bar.volume*self.participation), float(bar.volume),
                                 bar.timestamp.isoformat(), 'single_price_closing_auction_event', 'available')
        return CapacityQuote(0, 0, None, 'unsupported', 'outside_executable_interval')

    def contract(self):
        out = super().contract()
        out.update(version=VERSION, supported_phases=['interval', 'auction'],
                   opening_basis='no_opening_execution',
                   continuous_price='completed_interval_close_plus_native_costs',
                   commitment='decision_available_at_lte_interval_start',
                   service_available_at='interval_start_plus_one_minute',
                   realized_volume='shared_service_bound_after_interval_completion',
                   phase_liquidity_verified=False)
        return out


def audit_interval_capacity(broker):
    """Independent source-volume, order-commitment and receipt verification."""
    from collections import defaultdict
    from datetime import datetime, timedelta
    from strategy_common.market import MarketBar
    from .olr_minute_audit_v5 import continuous_interval
    errors=[]; sources={}; offers={}; used=defaultdict(int); filled=defaultdict(int)
    cash=broker.initial_equity;positions=defaultdict(int);marks={};peak=cash;peak_gross=0
    dt=lambda x: x if isinstance(x,datetime) else datetime.fromisoformat(x)
    for batch in broker.expected_batches:
        for raw in batch:
            key=(raw['symbol'],raw['timestamp'])
            if key in sources:errors.append('duplicate_interval_source')
            sources[key]=MarketBar(**{**raw,'timestamp':dt(raw['timestamp'])})
    for row in broker.journal:
        if row['event'] in ('opening_marks','closing_marks','interval_marks'):marks.update(row['prices'])
        if row['event']=='capacity':
            o=row['order'];phase=row['phase'];key=(o['symbol'],row['timestamp'])
            b=sources.get(key);budget=0;source=None;basis='unsupported';reason='outside_executable_interval'
            if b is not None and ((phase=='interval' and continuous_interval(b)) or
                                  (phase=='auction' and closing_auction_event(b))):
                source=b;budget=floor(float(b.volume)*broker.capacity_policy.participation)
                basis='completed_interval_volume' if phase=='interval' else 'single_price_closing_auction_event'
                reason='available'
            expected=dict(budget=budget,predictor_volume=float(source.volume) if source else 0,
                predictor_timestamp=source.timestamp.isoformat() if source else None,
                volume_basis=basis,reason=reason)
            allowed=min(row['requested_qty'],max(0,budget-used[key]))
            if row['quote']!=expected or row['used_qty']!=used[key] or row['allowed_qty']!=allowed:
                errors.append('interval_capacity_quote_mismatch')
            ok=(o['order_id'],*key)
            if ok in offers:errors.append('duplicate_interval_offer')
            offers[ok]=allowed
        elif row['event']=='fill':
            f=row['fill'];o=row['order'];m=f['metadata'];raw=m['execution_bar'];key=(f['symbol'],raw['timestamp'])
            b=sources.get(key);phase=m['execution_phase'];available=dt(o['metadata']['decision_available_at'])
            if b is None:errors.append('missing_interval_source');continue
            expected_at=b.timestamp+timedelta(minutes=1 if phase=='interval' else 0)
            if dt(f['timestamp'])!=expected_at or available>b.timestamp or dt(o['submitted_at'])>=b.timestamp:
                errors.append('interval_service_before_commitment')
            if phase not in ('interval','auction'):errors.append('interval_unregistered_phase')
            q=f['qty'];offer=offers.pop((o['order_id'],*key),None)
            positions[f['symbol']]+=q if f['side']=='BUY' else -q
            cash=row['cash_after']
            if offer is None or q>offer:errors.append('interval_missing_or_exceeded_offer')
            used[key]+=q;filled[key]+=q
            if used[key]>floor(b.volume*broker.capacity_policy.participation):errors.append('interval_shared_capacity_exceeded')
        gross=sum(q*marks.get(s,0) for s,q in positions.items())
        peak=max(peak,cash+gross);peak_gross=max(peak_gross,gross)
    violations=[dict(symbol=k[0],timestamp=k[1],filled_shares=q,printed_shares=sources[k].volume)
                for k,q in filled.items() if q>sources[k].volume]
    if violations:errors.append('shared_bar_volume_exceeded')
    return dict(status='failed' if errors else 'model_consistent',errors=sorted(set(errors)),
                contract=broker.capacity_policy.contract(),phase_liquidity_verified=False,
                impossible_volume_events=violations,filled_intervals=len(filled),
                capital=dict(initial_equity=broker.initial_equity,peak_equity=peak,peak_gross_exposure=peak_gross,final_equity=cash+gross),
                terminal_open_shares=sum(p.qty for p in broker.positions.values()),
                terminal_pending_shares=sum(int(o.qty or 0) for o in broker.orders))
