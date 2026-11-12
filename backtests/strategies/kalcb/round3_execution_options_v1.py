"""Default-off replay probes. Native SimBroker remains the sole executor."""
from dataclasses import asdict
from datetime import timedelta
import json
from pathlib import Path
from .round3_integrity_v1 import LifecycleObserver,neutral


class ProbeComplete(Exception):
    pass


class ExecutionOptions(LifecycleObserver):
    def __init__(self,output,*,enabled=False,hydration=False,delay_minutes=0,smoke_through=None):
        super().__init__(output,enabled=enabled,hydration=hydration)
        if delay_minutes not in (0,5):raise ValueError('Unregistered execution delay')
        self.delay=delay_minutes;self.through=smoke_through;self.submit=None;self.broker=None
        self.delayed=0

    def install(self):
        if self.through and not self.enabled:raise ValueError('Smoke needs lifecycle journals')
        super().install()
        if self.delay:
            from backtests.engine.sim_broker import SimBroker
            from strategy_common.actions import SubmitEntry,SubmitExit,SubmitPartialExit,FlattenPosition
            self.submit=SimBroker.submit;owner=self
            def submit(broker,action,submitted_at):
                clock=broker._decision_available_at
                ordinary=isinstance(action,(SubmitEntry,SubmitExit,SubmitPartialExit,FlattenPosition))
                market=getattr(action,'order_type',action.metadata.get('order_type','MARKET'))=='MARKET'
                if ordinary and market and broker._activation_phase=='decision':
                    if clock is None:raise ValueError('Ordinary decision lacks completed-data clock')
                    broker._decision_available_at=clock+timedelta(minutes=owner.delay);owner.delayed+=1
                try:return owner.submit(broker,action,submitted_at)
                finally:broker._decision_available_at=clock
            SimBroker.submit=submit

    def restore(self):
        if self.submit is not None:
            from backtests.engine.sim_broker import SimBroker
            SimBroker.submit=self.submit
        super().restore()

    def drive(self,broker,bars,adapters,*,accepts=None):
        bars=tuple(bars);self.broker=broker
        if self.through and str(bars[0].timestamp.date())>self.through:
            if broker.orders or broker.positions:raise ValueError('Smoke boundary is not flat')
            raise ProbeComplete()
        return super().drive(broker,bars,adapters,accepts=accepts)

    def save_smoke(self):
        if self.broker is None:raise ValueError('No native smoke batches')
        from .retention_research_io_v1 import write
        b=self.broker
        write(self.output/'smoke_fills.json',[asdict(f) for f in b.fills])
        write(self.output/'smoke.json',dict(native_lifecycle_completed=True,
            eligible_for_scoring=False,through=self.through,batches=self.batches,fills=len(b.fills),
            inventory=len(b.positions),pending=len(b.orders),delay_minutes=self.delay,delayed_decisions=self.delayed,
            hydration=self.receipts))
