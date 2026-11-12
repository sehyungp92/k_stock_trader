"""C15 research-only new-event reconsideration, using the native entry lifecycle."""
from datetime import timedelta

from .funded_horizon_v1 import HorizonReplay


class ReconsiderationReplay(HorizonReplay):
    def __init__(self, mode, signals):
        if mode not in ('disabled', 'reconsider'):
            raise ValueError('Unregistered reconsideration mode')
        super().__init__('veto', signals)
        self.reconsideration_mode = mode
        self.renewals = []
        self.considered = set()
        self.renewing = None

    def install(self):
        from strategy_kalcb.core import logic
        self.evaluate_original = logic._evaluate_entry
        super().install()
        logic._evaluate_entry = self.evaluate

    def restore(self):
        from strategy_kalcb.core import logic
        logic._evaluate_entry = self.evaluate_original
        super().restore()

    def quantity(self, symbol, qty, metadata):
        if self.renewing == self.day + '/' + symbol and metadata.get('entry_route') == 'first30_open_anchor':
            metadata['research_reconsideration'] = dict(
                version='c15_v1', opening_rejection_available_at=self.day+'T09:30:00+09:00',
                evidence_available_at=(self.stamp+timedelta(minutes=5)).isoformat(),
                evidence='completed_close_above_opening_high', opening_high=self.renewing_high)
            return self.allocate(symbol, qty, metadata)
        return super().quantity(symbol, qty, metadata)

    def evaluate(self, state, ss, bar, config, portfolio, *, first30_signal=False):
        key=str(bar.timestamp.date())+'/'+bar.symbol
        eligible=(self.reconsideration_mode=='reconsider' and key in self.attempts
                  and key not in self.considered and not first30_signal
                  and (bar.timestamp.hour,bar.timestamp.minute)==(9,30))
        if not eligible:
            return self.evaluate_original(state,ss,bar,config,portfolio,first30_signal=first30_signal)
        self.considered.add(key)
        prefix=ss.bars_today
        expected=[bar.timestamp.replace(hour=9,minute=0)+timedelta(minutes=5*i) for i in range(7)]
        valid=(len(prefix)==7 and [b.timestamp for b in prefix]==expected
               and all(b.is_completed for b in prefix))
        high=max(b.high for b in prefix[:6]) if valid else None
        record=dict(kind='renewal_considered',key=key,available_at=(bar.timestamp+timedelta(minutes=5)).isoformat(),
                    prefix_valid=valid,close=bar.close,opening_high=high,confirmed=bool(valid and bar.close>high),
                    cash=portfolio.cash,equity=portfolio.equity,open_positions=portfolio.open_positions,
                    prior_rejected_reason=ss.rejected_reason)
        self.renewals.append(record)
        if not record['confirmed']:
            return self.evaluate_original(state,ss,bar,config,portfolio,first30_signal=False)
        if ss.position is not None or ss.pending_entry_order_id:
            raise ValueError('Native flat/pending gate bypassed')
        self.renewing=key;self.renewing_high=high
        try:
            result=self.evaluate_original(state,ss,bar,config,portfolio,first30_signal=True)
        finally:
            self.renewing=None
        from strategy_common.actions import SubmitEntry
        entries=[a for a in result.actions if isinstance(a,SubmitEntry)]
        if any('research_reconsideration' not in a.metadata for a in entries):
            raise ValueError('Reconsideration did not use registered opening route')
        record.update(submitted_actions=len(entries),requested_quantity=sum(a.qty for a in entries),
                      native_reason=ss.rejected_reason,decision_reasons=[d.reason for d in result.decisions],
                      entry_metadata=[dict(a.metadata) for a in entries])
        return result

    def drive(self,broker,bars,adapters,*,accepts=None):
        bars=tuple(bars)
        result=super().drive(broker,bars,adapters,accepts=accepts)
        if self.reconsideration_mode!='reconsider':return result
        from strategy_common.actions import CancelOrders
        present={b.symbol:b for b in bars};adapter=adapters[0]
        # Fills through this completed interval precede its newly observable close.
        for order in tuple(broker.orders):
            tag=order.metadata.get('research_reconsideration')
            if not tag or order.side!='BUY':continue
            if order.strategy_id!='KALCB':raise ValueError('Foreign renewal order')
            if any(o.symbol==order.symbol and (o.strategy_id!='KALCB' or o.side=='SELL') for o in broker.orders):
                continue  # Existing native exit/protection has priority over broad CancelOrders.
            bar=present.get(order.symbol)
            if bar is None:continue
            available=bar.timestamp+timedelta(minutes=5)
            if available.isoformat()<=tag['evidence_available_at']:continue
            if bar.close>tag['opening_high']:continue
            clock=broker._decision_available_at;broker._decision_available_at=available
            try:
                adapter._submit_actions([CancelOrders('KALCB',order.symbol,
                    'research_reconsideration_breakout_lapsed_v1',{})],broker,bar.timestamp)
            finally:broker._decision_available_at=clock
            self.renewals.append(dict(kind='renewal_pending_cancel',key=self.day+'/'+order.symbol,
                available_at=available.isoformat(),order_id=order.order_id,close=bar.close,
                opening_high=tag['opening_high'],remaining_quantity=order.qty))
        return result
