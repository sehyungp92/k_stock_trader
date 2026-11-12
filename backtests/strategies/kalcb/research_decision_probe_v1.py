"""C18 default-off selective timing probe; SimBroker is the sole executor."""
from datetime import timedelta
from .round3_execution_options_v1 import ExecutionOptions


class DecisionStudyOptions(ExecutionOptions):
    def __init__(self, output, *, target='off', **kwargs):
        if target not in ('off','entry','exit'): raise ValueError('Unregistered C18 timing target')
        if kwargs.get('delay_minutes',0): raise ValueError('Do not stack delay probes')
        super().__init__(output, **kwargs)
        self.target=target

    def install(self):
        super().install()
        if self.target=='off': return
        from backtests.engine.sim_broker import SimBroker
        from strategy_common.actions import SubmitEntry,SubmitExit,SubmitPartialExit,FlattenPosition
        self.submit=SimBroker.submit;owner=self
        def submit(broker, action, submitted_at):
            original_clock=broker._decision_available_at
            selected=isinstance(action,SubmitEntry) if owner.target=='entry' else isinstance(action,(SubmitExit,SubmitPartialExit,FlattenPosition))
            market=getattr(action,'order_type',action.metadata.get('order_type','MARKET'))=='MARKET'
            if selected and market and broker._activation_phase=='decision':
                if original_clock is None: raise ValueError('Missing completed-data activation clock')
                broker._decision_available_at=original_clock+timedelta(minutes=5)
                owner.delayed+=1
            try:return owner.submit(broker,action,submitted_at)
            finally:broker._decision_available_at=original_clock
        SimBroker.submit=submit
