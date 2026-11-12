"""Observed native account path for a research decision-context provider.

Tracks opening marks, costs and closing marks instead of resetting peak NAV
at a trade or at a new session. It never receives future market observations.
The broker journal remains the independent execution authority.
"""
from datetime import datetime, timedelta
from math import isclose, isfinite


class ObservedAccount:
    def __init__(self, initial_equity, *, mark_interval_minutes=0):
        if not isfinite(initial_equity) or initial_equity<=0:raise ValueError('Invalid starting equity')
        self.initial=float(initial_equity)
        self.cash=float(initial_equity)
        self.high_water=float(initial_equity)
        self.marks={};self.holdings={};self.cursor=0;self.last_at=None
        if mark_interval_minutes not in (0,1):raise ValueError('Unregistered mark observation clock')
        self.mark_interval_minutes=mark_interval_minutes

    def consume(self, journal, at, *, actual_cash, actual_equity):
        if at.tzinfo is None:raise ValueError('Aware decision clock required')
        if len(journal)<self.cursor:raise ValueError('Native journal moved backwards')
        for event in journal[self.cursor:]:
            kind=event['event']
            if kind not in ('opening_marks','closing_marks','interval_marks','fill'):continue
            value=event['fill']['timestamp'] if kind=='fill' else event['timestamp']
            stamp=datetime.fromisoformat(value) if isinstance(value,str) else value
            if kind in ('closing_marks','interval_marks'):
                stamp+=timedelta(minutes=self.mark_interval_minutes)
            if stamp.tzinfo is None or stamp>at or self.last_at is not None and stamp<self.last_at:
                raise ValueError('Future or reversed account observation')
            self.last_at=stamp
            if kind=='fill':
                f=event['fill'];q=f['qty'];p=f['price'];fee=f['metadata']['fee']
                if type(q) is not int or q<=0 or not isfinite(p) or p<=0 or not isfinite(fee) or fee<0:
                    raise ValueError('Invalid native fill')
                if f['side'] not in ('BUY','SELL'):raise ValueError('Unknown fill side')
                if not isclose(self.cash,event['cash_before'],abs_tol=1e-5):raise ValueError('Cash-before mismatch')
                sign=1 if f['side']=='BUY' else -1
                self.cash-=sign*q*p+fee
                key=(f['strategy_id'],f['symbol'])
                self.holdings[key]=self.holdings.get(key,0)+sign*q
                if self.holdings[key]<0:raise ValueError('Negative observed holdings')
                if not isclose(self.cash,event['cash_after'],abs_tol=1e-5):raise ValueError('Cash-after mismatch')
            else:
                if any(not isfinite(p) or p<=0 for p in event['prices'].values()):raise ValueError('Invalid observed mark')
                self.marks.update(event['prices'])
            self.high_water=max(self.high_water,self.equity())
        self.cursor=len(journal)
        if not isclose(self.cash,actual_cash,rel_tol=1e-10,abs_tol=1e-5) or not isclose(self.equity(),actual_equity,rel_tol=1e-10,abs_tol=1e-5):
            raise ValueError('Research account disagrees with actual native account')
        return dict(equity=self.equity(),cash=self.cash,high_water=self.high_water,
                    drawdown_fraction=1-self.equity()/self.high_water,
                    journal_events_consumed=self.cursor,observed_at=at.isoformat())

    def equity(self):
        if any(q and symbol not in self.marks for (_,symbol),q in self.holdings.items()):
            raise ValueError('Unmarked observed holding')
        return self.cash+sum(q*self.marks.get(symbol,0) for (_,symbol),q in self.holdings.items())
