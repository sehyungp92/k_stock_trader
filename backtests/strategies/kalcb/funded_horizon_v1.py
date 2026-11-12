"""Research-only causal opening veto versus scheduled first-interval liquidation."""
from datetime import datetime,timedelta


def enabled_signal(row, available):
    if row is None:return False
    if datetime.fromisoformat(row['available_at'])>available:
        raise ValueError('Future opening signal')
    if row['origin']>=row['day']:
        # Origins can equal prediction day: training is separately embargoed.
        if row['origin']>row['day']:raise ValueError('Future model origin')
    return row['predictions']['baseline'] < -23. and row['predictions']['combined'] < -33.


class HorizonReplay:
    def __init__(self,mode,signals):
        if mode not in ('disabled','veto','renew'):raise ValueError('Unregistered mode')
        self.mode=mode;self.signals=signals;self.day=None;self.stamp=None
        self.records=[];self.finished=set();self.attempts=set()

    def install(self):
        from backtests.engine import replay
        from strategy_kalcb.core import account_allocation_v1 as allocation
        self.kernel=replay.drive_timestamp_batch;self.allocate=allocation.allocate_quantity
        replay.drive_timestamp_batch=self.drive;allocation.allocate_quantity=self.quantity

    def restore(self):
        from backtests.engine import replay
        from strategy_kalcb.core import account_allocation_v1 as allocation
        replay.drive_timestamp_batch=self.kernel;allocation.allocate_quantity=self.allocate

    def quantity(self,symbol,qty,metadata):
        if self.mode=='disabled' or metadata.get('entry_route')!='first30_open_anchor' or qty<=0:
            return self.allocate(symbol,qty,metadata)
        key=self.day+'/'+symbol;signal=self.signals.get(key)
        if not enabled_signal(signal,self.stamp+timedelta(minutes=5)):
            return self.allocate(symbol,qty,metadata)
        self.attempts.add(key)
        if self.mode=='veto' or key in self.finished:
            self.records.append({'kind':'veto','key':key,'available_at':(self.stamp+timedelta(minutes=5)).isoformat(),
                                 'requested_qty':qty,'chosen_qty':0,'signal':signal})
            return 0
        return self.allocate(symbol,qty,metadata)

    def drive(self,broker,bars,adapters,*,accepts=None):
        bars=tuple(bars)
        if len(adapters)!=1 or accepts is not None:raise ValueError('One whole native account required')
        self.stamp=bars[0].timestamp;self.day=str(self.stamp.date())
        result=self.kernel(broker,bars,adapters)
        if self.mode!='renew' or (self.stamp.hour,self.stamp.minute)!=(9,30):return result
        from .holding_action_v1 import apply
        from strategy_common.actions import CancelOrders
        adapter=adapters[0];present={b.symbol:b for b in bars};available=self.stamp+timedelta(minutes=5)
        for key in sorted(k for k in self.attempts if k.startswith(self.day+'/') and k not in self.finished):
            symbol=key.split('/')[1];signal=self.signals[key]
            if not enabled_signal(signal,available):raise ValueError('Changed registered signal')
            if symbol not in present:
                self.records.append({'kind':'missing_0930_bar','key':key});continue
            ss=adapter.state.symbols[symbol];p=ss.position;own=[o for o in broker.orders if o.symbol==symbol]
            if any(o.side=='SELL' or o.strategy_id!='KALCB' for o in own) or (p and (p.exit_in_flight or p.partial_order_id)):
                self.records.append({'kind':'parent_exit_precedence','key':key});self.finished.add(key);continue
            if p and p.metadata.get('entry_route')!='first30_open_anchor':
                raise ValueError('Unexpected position ownership')
            if p:
                actual=broker.positions.get(('KALCB',symbol))
                if actual is None or actual.qty!=p.qty_open:raise ValueError('Core/broker quantity mismatch')
                row={'symbol':symbol,'day':self.day,'available_at':available.isoformat(),'signal_bar':self.stamp.isoformat(),
                     'entry_order_id':p.entry_order_id,'qty':p.qty_open,'management_reference':p.entry_price,
                     'management_risk_per_share':p.risk_per_share,'reason':'eligible'}
                action=apply(adapter,broker,row,'exit')
                self.records.append({'kind':'renew_exit','key':key,'row':row,'action':action,'signal':signal})
            elif own:
                if len(own)!=1 or own[0].side!='BUY' or own[0].metadata.get('entry_route')!='first30_open_anchor':
                    raise ValueError('Unowned pending cancellation')
                clock=broker._decision_available_at;broker._decision_available_at=available
                try:adapter._submit_actions([CancelOrders('KALCB',symbol,'research_horizon_pending_cancel_v1',{})],broker,self.stamp)
                finally:broker._decision_available_at=clock
                self.records.append({'kind':'pending_only_cancel','key':key,'available_at':available.isoformat(),'signal':signal})
            else:self.records.append({'kind':'already_flat','key':key})
            self.finished.add(key)
        return result
