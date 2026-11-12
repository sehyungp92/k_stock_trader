"""C11 pre-fill whole-request reservation on the unchanged VETO admission book."""
from copy import deepcopy
from datetime import timedelta
import math

from .funded_horizon_v1 import HorizonReplay


def native_key(order):
    return (order.submitted_at, order.strategy_id, order.symbol, order.order_id)


def action_scale(adapter,symbol,at):
    if not adapter.config.entry_action_quantity_scale_enabled:
        raise ValueError('Original learned quantity scaling is disabled')
    snapshot=adapter.snapshots.get(at.date())
    updates=[s for s in adapter.timed_snapshots.get(at.date(),()) if s.generated_at<=at]
    if updates:snapshot=updates[-1]
    candidates=[c for c in snapshot.candidates if c.symbol==symbol] if snapshot else []
    if len(candidates)!=1:raise ValueError('Missing original candidate quantity action')
    return float(candidates[0].metadata.get('action_quantity_scale',1.))


def reserve(intents, room, mode):
    """No quantity shrinkage, profitability gate, future quote or fitted parameter."""
    if mode not in ('queue', 'forecast') or not math.isfinite(room):
        raise ValueError('Invalid reservation contract')
    ordered=sorted(intents,key=lambda r:((-r['score'],) if mode=='forecast' else ())+tuple(r['queue_key']))
    left=max(0.,room);book={}
    for row in ordered:
        if not math.isfinite(row['cost']) or row['cost']<=0 or not math.isfinite(row['score']):
            raise ValueError('Invalid contemporaneous reservation')
        if row['symbol'] in book:raise ValueError('Duplicate opening symbol')
        take=row['cost']<=left
        book[row['symbol']]={'parent_qty':row['qty'],'qty':row['qty'] if take else 0,
                             'mechanism':'retained_whole_request_reservation_v1','arm':mode}
        if take:left-=row['cost']
    return book,left,[r['symbol'] for r in ordered]


class RetainedFundingReplay(HorizonReplay):
    def __init__(self,mode,signals):
        if mode not in ('disabled','queue','forecast'):raise ValueError('Unregistered reservation mode')
        super().__init__('veto',signals)
        self.reservation_mode=mode
        self.suppressed=set()
        self.reservation_records=[]

    def quantity(self,symbol,qty,metadata):
        key=self.day+'/'+symbol
        if metadata.get('entry_route')=='first30_open_anchor' and key in self.suppressed and qty>0:
            self.records.append(dict(kind='reservation_session_veto',key=key,requested_qty=qty,
                                     available_at=(self.stamp+timedelta(minutes=5)).isoformat()))
            return 0
        return super().quantity(symbol,qty,metadata)

    def preview(self,adapter,broker,bars):
        from .account_history_v1 import clone_complete_account
        aa,bb=clone_complete_account(adapter,broker)
        previous={o.order_id for o in bb.orders}
        # The preview uses the exact same VETO hook, but cannot pollute real actions.
        saved=(deepcopy(self.records),set(self.attempts),set(self.finished))
        try:self.kernel(bb,bars,(aa,))
        finally:self.records,self.attempts,self.finished=saved
        orders=[o for o in bb.orders if o.order_id not in previous and o.side=='BUY'
                and o.strategy_id=='KALCB' and o.metadata.get('entry_route')=='first30_open_anchor']
        return bb,orders

    def decision(self,adapter,broker,bars):
        bb,orders=self.preview(adapter,broker,bars)
        at=self.stamp+timedelta(minutes=5)
        record=dict(day=self.day,available_at=at.isoformat(),mode=self.reservation_mode,
                    changed=False,intents=[],room=None,reason='no_opening_intents')
        if not orders:return record,None
        ids={o.order_id for o in orders}
        costs=bb.costs;factor=(1+costs.slippage_bps/10000)*(1+costs.commission_bps/10000)
        pending=sum(o.qty*float(bb.last_prices[o.symbol])*factor for o in bb.orders
                    if o.side=='BUY' and o.order_id not in ids)
        room=max(0.,bb._buying_power_available()-pending)
        record.update(room=room,earlier_pending_reservation=pending,equity=bb._portfolio_equity(),
                      raw_buying_power=bb._buying_power_available(),buy_cost_factor=factor)
        for o in orders:
            m=o.metadata;signal=self.signals.get(self.day+'/'+o.symbol)
            if signal is None:
                record['reason']='unsupported_book_parent_preserved'
                return record,None
            if signal['available_at']!=at.isoformat() or m['decision_available_at']!=at.isoformat():
                raise ValueError('Opening forecast clock mismatch')
            q=signal['quantity_scale']
            if abs(action_scale(adapter,o.symbol,at)-q)>1e-12:
                raise ValueError('Original action quantity scale mismatch')
            price=float(m['entry_price_ref'])
            if price<=0 or o.qty<=0 or o.side!='BUY':raise ValueError('Invalid original intent')
            key=native_key(o)
            record['intents'].append(dict(symbol=o.symbol,order_id=o.order_id,qty=o.qty,
               reference_price=price,cost=o.qty*price*factor,score=min(signal['predictions'].values()),
               action_id=signal['action_id'],quantity_scale=q,queue_key=(key[0].isoformat(),*key[1:])))
        if sum(r['cost'] for r in record['intents'])<=room:
            record['reason']='affordable_book_parent_preserved'
            return record,None
        book,left,order=reserve(record['intents'],room,self.reservation_mode)
        for row in book.values():row['available_at']=at.isoformat()
        record.update(reason='supported_overbooked',book=book,remaining_room=left,priority=order,
                      changed=any(x['qty']!=x['parent_qty'] for x in book.values()))
        return record,book

    def drive(self,broker,bars,adapters,*,accepts=None):
        bars=tuple(bars)
        if self.reservation_mode=='disabled':return super().drive(broker,bars,adapters,accepts=accepts)
        if len(adapters)!=1 or accepts is not None:raise ValueError('One complete native account required')
        self.stamp=bars[0].timestamp;self.day=str(self.stamp.date())
        book=None
        if (self.stamp.hour,self.stamp.minute)==(9,25):
            record,book=self.decision(adapters[0],broker,bars)
            self.reservation_records.append(record)
        from strategy_kalcb.core.account_allocation_v1 import entry_allocation
        with entry_allocation(book):
            result=super().drive(broker,bars,adapters)
        if book:
            self.suppressed.update(self.day+'/'+s for s,row in book.items() if row['qty']==0)
        return result
