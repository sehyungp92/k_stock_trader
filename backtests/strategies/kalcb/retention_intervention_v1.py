"""Observe same-state exit backouts without changing native decisions or fills."""
from copy import deepcopy
from dataclasses import replace


class RetentionObserver:
    def __init__(self, backout):
        self.backout = dict(backout)
        self.probes = []
        self.orders = {}
        self.pending = {}
        self.active_signals = {}
        self.depth = 0
        self.calls = 0

    def install(self):
        from strategy_kalcb.core import logic
        from backtests.engine.sim_broker import SimBroker
        from strategy_common.actions import SubmitEntry
        trigger, submit = logic._exit_reason, SimBroker.submit
        self.originals = logic, SimBroker, trigger, submit
        observer = self

        def observed(state, position, bar, config, *, current_r, mfe_r):
            if observer.depth:
                return trigger(state, position, bar, config, current_r=current_r, mfe_r=mfe_r)
            observer.calls += 1
            observer.depth += 1
            try:
                # Detached position prevents the hypothetical signal metadata
                # from entering the actual order or position lifecycle.
                prior = trigger(state, deepcopy(position), bar, replace(config, **observer.backout),
                                current_r=current_r, mfe_r=mfe_r)
                actual = trigger(state, position, bar, config, current_r=current_r, mfe_r=mfe_r)
            finally:
                observer.depth -= 1
            observer.pending.pop((bar.symbol, bar.timestamp), None)
            if actual == 'profit_retention' or prior == 'profit_retention':
                row = {'symbol': bar.symbol, 'bar_timestamp': bar.timestamp.isoformat(),
                       'entry_time': position.entry_time.isoformat(), 'entry_price': position.entry_price, 'qty_open': position.qty_open,
                       'actual_signal': actual == 'profit_retention',
                       'parent_signal_same_state': prior == 'profit_retention',
                       'actual_exit_reason': actual or None, 'parent_exit_reason': prior or None,
                       'current_r': current_r, 'mfe_r': mfe_r, 'hold_bars': position.hold_bars,
                       'signal': deepcopy(position.metadata.get('profit_retention_signal')),
                       'order_id': None}
                observer.probes.append(row)
                if actual == 'profit_retention':
                    observer.pending[(bar.symbol, bar.timestamp)] = row
                    observer.active_signals[bar.symbol] = row
            return actual

        def submitted(broker, action, submitted_at):
            order_id = submit(broker, action, submitted_at)
            if isinstance(action, SubmitEntry):
                observer.active_signals.pop(action.symbol, None)
            if getattr(action, 'reason', None) not in ('profit_retention', 'profit_retention_partial'):
                return order_id
            row = observer.pending.pop((action.symbol, submitted_at), None)
            if row is None:
                # A live liquidation can cancel and resubmit its remainder.
                # Retain the initial decision's evidence; a retry is not a new
                # independent signal and may not imply a new feasible fill.
                row = observer.active_signals.get(action.symbol)
            if row is None or action.metadata.get('entry_price') != row['entry_price']:
                raise ValueError('Profit-retention order lacks matching position signal')
            row = {**row, 'order_id':order_id, 'submitted_qty':action.qty,
                   'submitted_at':submitted_at.isoformat(), 'action_reason':action.reason}
            if order_id:
                observer.orders[order_id] = row
            return order_id

        logic._exit_reason = observed
        SimBroker.submit = submitted

    def restore(self):
        logic, broker, trigger, submit = self.originals
        logic._exit_reason, broker.submit = trigger, submit

    def summarize(self, fills):
        matched = []
        for fill in fills:
            probe = self.orders.get(fill['order_id'])
            if probe is None:
                continue
            if fill['side'] != 'SELL':
                raise ValueError('Exit observer matched a non-sell execution')
            matched.append({'order_id': fill['order_id'], 'symbol': fill['symbol'],
                            'timestamp': str(fill['timestamp']), 'qty': fill['qty'], 'price': fill['price'],
                            'execution_id': fill['metadata'].get('execution_id'),
                            'decision_available_at': fill['metadata'].get('decision_available_at'),
                            'parent_signal_same_state': probe['parent_signal_same_state'],
                            'parent_exit_reason': probe['parent_exit_reason']})
        direct = [r for r in matched if r['parent_exit_reason'] is None]
        ids = {(r['symbol'], r['timestamp'][:10]) for r in direct}
        days = {d for _, d in ids}
        return {'native_trigger_evaluations': self.calls, 'candidate_signal_differences': sum(
                    p['actual_signal'] != p['parent_signal_same_state'] for p in self.probes),
                'submitted_retention_orders': len(self.orders), 'filled_retention_orders': len({r['order_id'] for r in matched}),
                'direct_changed_exit_orders': len({r['order_id'] for r in direct}),
                'direct_changed_exit_issuer_days': len(ids), 'direct_changed_exit_sessions': len(days),
                'preferred_support_passed': len(ids) >= 10 and len(days) >= 8,
                'execution_checks': matched,
                'scope': 'Actual native sell fills following a retention exit when the complete parent exit-reason cascade returns no exit in the identical pre-decision position state. A parent exit for another reason at the same decision does not count. No hypothetical price or profit booked. Later account descendants and liquidation retries are not independent interventions. Counts are dependent observations; low support remains reported.'}
