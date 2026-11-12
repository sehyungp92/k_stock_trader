"""Exact native-state observations, without hypothetical profits or fills."""
from copy import copy, deepcopy
from dataclasses import replace
from datetime import timedelta


class MechanismObserver:
    def __init__(self, alternatives=()):
        self.alternatives = {}
        for row, forecasts in alternatives:
            self.alternatives[(row['trade_date'], row['symbol'])] = (row, forecasts)
        self.exit_probes = []; self.entry_probes = []; self.orders = {}; self.active = {}
        self.depth = 0; self.calls = 0

    def install(self):
        from strategy_kalcb.core import logic
        from backtests.strategies.kalcb import runner
        from backtests.engine.sim_broker import SimBroker
        from strategy_common.actions import SubmitEntry
        from strategy_kalcb.opening_failure_v1 import evidence
        trigger, submit, on_bar = logic._exit_reason, SimBroker.submit, runner.KALCBReplayAdapter.on_bar
        self.originals = logic, SimBroker, runner.KALCBReplayAdapter, trigger, submit, on_bar
        observer = self

        def observed(symbol_state, position, bar, config, *, current_r, mfe_r):
            if observer.depth:
                return trigger(symbol_state, position, bar, config, current_r=current_r, mfe_r=mfe_r)
            observer.calls += 1
            parent_config = replace(config, research_opening_failure_min_hold_bars=0)
            parent = trigger(symbol_state, deepcopy(position), bar, parent_config,
                             current_r=current_r, mfe_r=mfe_r)
            proofs = {str(h): evidence(symbol_state, position, bar, h) for h in (6, 8)}
            actual = trigger(symbol_state, position, bar, config, current_r=current_r, mfe_r=mfe_r)
            if any(proofs.values()) or actual == 'opening_thesis_failure':
                row = {'symbol': bar.symbol, 'bar_timestamp': bar.timestamp.isoformat(),
                       'entry_time': position.entry_time.isoformat(), 'entry_price': position.entry_price,
                       'qty_open': position.qty_open, 'parent_exit_reason': parent or None,
                       'actual_exit_reason': actual or None, 'proofs': proofs,
                       'minimum_hold': config.research_opening_failure_min_hold_bars}
                observer.exit_probes.append(row)
                if actual == 'opening_thesis_failure' and any(proofs.values()):
                    observer.active[bar.symbol] = row
            return actual

        def submitted(broker, action, submitted_at):
            order_id = submit(broker, action, submitted_at)
            if observer.depth:
                return order_id
            if isinstance(action, SubmitEntry):
                observer.active.pop(action.symbol, None)
            if getattr(action, 'reason', '') != 'opening_thesis_failure':
                return order_id
            row = observer.active.get(action.symbol)
            if not row or action.metadata.get('entry_price') != row['entry_price']:
                raise ValueError('Exit lacks exact native decision lineage')
            if order_id:
                observer.orders[order_id] = {**row, 'order_id': order_id,
                    'submitted_at': submitted_at.isoformat(), 'submitted_qty': action.qty}
            return order_id

        def on_actual_bar(adapter, bar, broker):
            key = (str(bar.timestamp.date()), bar.symbol)
            if not observer.depth and bar.timestamp.hour == 9 and bar.timestamp.minute == 55 and key in observer.alternatives:
                row, forecasts = observer.alternatives[key]
                available = bar.timestamp + timedelta(minutes=5)
                if row['feature_available_at'] != available.isoformat():
                    raise ValueError('Alternative information clock mismatch')
                if 'opening_bars' in row:
                    from datetime import datetime
                    symbol_state = adapter.state.symbols.get(bar.symbol)
                    prefix = list(symbol_state.bars_today) if symbol_state else []
                    prefix = [b for b in prefix if b.timestamp < bar.timestamp] + [bar]
                    expected = row['opening_bars']
                    if len(prefix) != len(expected): raise ValueError('Native alternative prefix length mismatch')
                    for actual_bar, saved in zip(prefix, expected):
                        if actual_bar.timestamp != datetime.fromisoformat(saved['timestamp']) or any(
                            getattr(actual_bar, k) != saved[k] for k in ('open','high','low','close','volume')):
                            raise ValueError('Native alternative prefix source mismatch')
                snapshot = adapter.snapshots.get(bar.timestamp.date())
                updates = [s for s in adapter.timed_snapshots.get(bar.timestamp.date(), ()) if s.generated_at <= available]
                if updates: snapshot = updates[-1]
                if snapshot is None: raise ValueError('Missing native parent snapshot')
                record = {'opportunity_id': row['opportunity_id'], 'available_at': available.isoformat(),
                          'forecasts': forecasts, 'already_candidate': bar.symbol in snapshot.by_symbol(),
                          'native_prefix_matched': 'opening_bars' in row,
                          'scope': 'Separate same-state native order preview. No future bars, fills or profit. Alternatives compete for one slot.'}
                if record['already_candidate']:
                    record['reason'] = 'already_parent_candidate'
                else:
                    from backtests.strategies.kalcb.late_entry_v1 import append_snapshot
                    value = max(min(f['baseline'], f['combined']) for f in forecasts)
                    updated = append_snapshot(snapshot, row, value, False)
                    # Copies retain actual cash, risk history, partial fills and pending
                    # orders. The real native adapter still runs untouched below.
                    shadow = copy(adapter); shadow.state = deepcopy(adapter.state)
                    shadow._daily_active_symbols = deepcopy(adapter._daily_active_symbols)
                    shadow._activated_intraday_hash = dict(adapter._activated_intraday_hash)
                    shadow.timed_snapshots = dict(adapter.timed_snapshots)
                    shadow.timed_snapshots[bar.timestamp.date()] = [*updates, updated]
                    if adapter.config.frontier_rotation_enabled:
                        raise ValueError('Observer requires unchanged nonrotating parent frontier')
                    shadow.frontier_shadow = None
                    memo = {id(broker.source_events): broker.source_events.copy()}
                    for dependent in getattr(broker, 'capacity_shadow_brokers', ()):
                        memo[id(dependent.source_events)] = dependent.source_events.copy()
                    trial_broker = deepcopy(broker, memo)
                    observer.depth += 1
                    try:
                        decisions = on_bar(shadow, bar, trial_broker)
                    finally:
                        observer.depth -= 1
                    record['decisions'] = [d.to_json_dict() for d in decisions]
                    before_ids = {order.order_id for order in broker.orders}
                    new_orders = [order for order in trial_broker.orders if order.order_id not in before_ids]
                    from dataclasses import asdict
                    record['new_orders'] = [asdict(order) for order in new_orders]
                    record['cash_before'] = broker.cash
                    record['portfolio_before'] = asdict(adapter._portfolio_view(broker))
                    record['native_order_submitted'] = bool(new_orders)
                observer.entry_probes.append(record)
            return on_bar(adapter, bar, broker)

        logic._exit_reason = observed
        SimBroker.submit = submitted
        runner.KALCBReplayAdapter.on_bar = on_actual_bar

    def restore(self):
        logic, broker, adapter, trigger, submit, on_bar = self.originals
        logic._exit_reason = trigger; broker.submit = submit; adapter.on_bar = on_bar

    def summarize(self, fills):
        matched = []
        for fill in fills:
            probe = self.orders.get(fill['order_id'])
            if probe:
                if fill['side'] != 'SELL': raise ValueError('Exit probe matched non-sell')
                matched.append({'order_id': fill['order_id'], 'execution_id': fill['metadata']['execution_id'],
                    'symbol': fill['symbol'], 'timestamp': str(fill['timestamp']), 'qty': fill['qty'],
                    'parent_exit_reason': probe['parent_exit_reason'], 'entry_time': probe['entry_time']})
        direct = [f for f in matched if f['parent_exit_reason'] is None]
        support = {}
        for hold in (6, 8):
            p = [r for r in self.exit_probes if r['proofs'][str(hold)] and r['parent_exit_reason'] is None]
            keys = {(r['symbol'], r['entry_time']) for r in p}
            support[str(hold)] = {'parent_no_exit_positions': len(keys),
                                 'sessions': len({r['bar_timestamp'][:10] for r in p})}
        return {'trigger_calls': self.calls, 'observed_opportunities': support,
                'submitted_exit_orders': len(self.orders), 'filled_exit_orders': len({r['order_id'] for r in matched}),
                'direct_filled_positions': len({(r['symbol'], r['entry_time']) for r in direct}),
                'direct_filled_sessions': len({r['timestamp'][:10] for r in direct}),
                'executions': matched, 'late_previews': len(self.entry_probes),
                'late_native_submits': sum(r.get('native_order_submitted', False) for r in self.entry_probes),
                'scope': 'Observer opportunities are not fills. Direct changed exits require actual fills and no parent exit in the same state. Retries and descendants are dependent.'}
