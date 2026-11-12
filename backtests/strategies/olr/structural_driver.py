"""Fixed mechanism clocks and complete-account engineering evidence.

The failed ridge model has no authority here. Marked-book/thesis observations
are evidence for subsequent supported actions, not hidden portfolio overlays.
"""
from dataclasses import dataclass
from pathlib import Path
from backtests.engine.olr_minute_v5 import MinuteOLRDriver
from .decision_account import ObservedAccount
from .funded_checkpoint import fork,save,load,economic_state

MODES={'control':(),'failure_clock':('failed_followthrough',),
       'fade_clock':('mfe_fade',),'both_clocks':('failed_followthrough','mfe_fade'),
       'structural_partial':()}


class BookObserver:
    def __init__(self,initial,interval_service):
        self.account=ObservedAccount(initial,mark_interval_minutes=1 if interval_service else 0)
        self.trace=[];self.last_states={}

    def observe(self,adapter,broker,at,bars):
        observed=self.account.consume(broker.journal,at,actual_cash=broker.cash,actual_equity=broker._portfolio_equity())
        positions=[]
        for symbol,ss in adapter.state.symbols.items():
            p=ss.position
            if p is None:continue
            risk=float(p.metadata.get('risk_per_share',0) or 0)
            mark=broker.last_prices[symbol];mfe=(p.max_favorable_price-p.entry_price)/risk if risk>0 else None
            status='liquidating' if ss.pending_exit_order_id else 'accepted' if mfe is not None and mfe>=1 else 'unproven'
            positions.append(dict(symbol=symbol,quantity=p.qty_open,mark=mark,entry=p.entry_price,
                current_notional=p.qty_open*mark,unrealized_pnl=p.qty_open*(mark-p.entry_price),
                observed_giveback=p.qty_open*max(0,p.max_favorable_price-mark),
                entry_atr=risk,mfe_r=mfe,state=status,pending_exit=ss.pending_exit_order_id))
        states={p['symbol']:p['state'] for p in positions}
        if positions and (at.minute%5==0 or states!=self.last_states):
            self.trace.append(dict(**observed,positions=positions,
                pending=[dict(id=o.order_id,symbol=o.symbol,side=o.side,quantity=o.qty) for o in broker.orders],
                scope='Observed marked account and thesis; no learned forecast or risk guarantee'))
        self.last_states=states


class StructuralDriver(MinuteOLRDriver):
    def __init__(self,broker,adapter,*,mode,output,interval_service=True,verify_checkpoint=True):
        if mode not in MODES:raise ValueError('Unregistered structural mechanism')
        super().__init__(broker,adapter,fast_exit=bool(MODES[mode]))
        broker.minute_exit_reasons=MODES[mode]
        adapter.state.meta['observe_structural_signals_v1']=True
        self.structural_observer=BookObserver(broker.initial_equity,interval_service)
        self.mode=mode;self.output=str(output);self.checkpoint_checked=not verify_checkpoint
        self.checkpoint_receipt=None

    def run_day(self,fine_bars,coarse_bars,*,symbols=None):
        fine_bars=tuple(fine_bars);coarse_bars=tuple(coarse_bars)
        twin=None
        if not self.checkpoint_checked and (self.broker.positions or self.broker.orders):
            self.checkpoint_checked=True
            path=Path(self.output)/'native_account_checkpoint.pkl'
            receipt=save(self,path)
            twin=load(path,receipt['sha256'])
            if economic_state(twin)!=economic_state(self):raise ValueError('Checkpoint changed native starting state')
            twin.run_day(fine_bars,coarse_bars,symbols=symbols)
        result=super().run_day(fine_bars,coarse_bars,symbols=symbols)
        if twin is not None:
            same=economic_state(twin)==economic_state(self)
            self.checkpoint_receipt=dict(**receipt,no_action_continuation_equal=same,
                through=str(fine_bars[-1].timestamp),
                original_outcome=economic_state(self),fork_outcome=economic_state(twin))
            if not same:raise ValueError('Complete funded continuation failed parity')
        return result
