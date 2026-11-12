"""Opt-in, resumable native decision clock and explicit funded action probes.

No forecast or trading rule is installed. A research caller must explicitly
request an inventory target at a completed observation. The remainder of the
native account, including subsequent acquisitions, continues normally.
"""
from dataclasses import dataclass, field
from datetime import timedelta
from itertools import groupby
from pathlib import Path

from backtests.auto.shared.cache_keys import stable_signature
from backtests.engine.olr_minute_v5 import closing_observation, continuous_minute, minute_risk_step
from strategy_common.actions import CancelOrders, SubmitExit, SubmitPartialExit
from .funded_checkpoint import economic_state, save, load
from .structural_driver import StructuralDriver


@dataclass
class SessionCursor:
    fine: tuple
    coarse: tuple
    schedule_hash: str
    fine_index: int = 0
    coarse_index: int = 0
    decisions: list = field(default_factory=list)
    available_at: object = None
    last_batch: tuple = ()


def continuation_state(driver):
    """Bind the resumable clocks and marked account in addition to native state."""
    c = driver.session_cursor
    observer = driver.structural_observer
    account = observer.account
    return stable_signature(dict(native=economic_state(driver), cursor=None if c is None else dict(
        schedule_hash=c.schedule_hash, fine_index=c.fine_index, coarse_index=c.coarse_index,
        available_at=str(c.available_at), decisions=[d.to_json_dict() for d in c.decisions]),
        account=dict(initial=account.initial, cash=account.cash, high_water=account.high_water,
            marks=account.marks, holdings={str(k):v for k,v in account.holdings.items()},
            cursor=account.cursor, last_at=str(account.last_at)),
        observer_states=observer.last_states, observer_trace=observer.trace,
        pending_targets=driver.pending_targets, action_ids=sorted(driver.action_ids),
        action_trace=driver.action_trace))


class DecisionContinuationDriver(StructuralDriver):
    """Preserve the V3 callback order while making its cursor checkpointable."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.session_cursor = None
        self.pending_targets = {}
        self.action_ids = set()
        self.action_trace = []

    def start_day(self, fine_bars, coarse_bars, *, symbols=None):
        if self.session_cursor is not None:
            raise ValueError('Finish the current native session before starting another')
        fine, coarse = tuple(fine_bars), tuple(coarse_bars)
        if not fine or not coarse:
            raise ValueError('Both independently bound source clocks are required')
        if any(b.timeframe != '1m' or not b.is_completed or not b.source_fingerprint
               or b.timestamp.tzinfo is None or b.timestamp.second or b.timestamp.microsecond for b in fine):
            raise ValueError('Fine input must contain sourced completed aligned minutes')
        if any(b.timeframe != '5m' or not b.is_completed or not b.source_fingerprint for b in coarse):
            raise ValueError('Coarse input must retain sourced native bars')
        if len({b.timestamp.date() for b in (*fine, *coarse)}) != 1:
            raise ValueError('Driver requires exactly one session')
        key = lambda b: (b.timestamp, b.symbol)
        for name, rows in [('Fine', fine), ('Coarse', coarse)]:
            if list(rows) != sorted(rows, key=key) or len({key(b) for b in rows}) != len(rows):
                raise ValueError(name + ' source order or uniqueness violation')
        batches = lambda rows: tuple(tuple(items) for _, items in groupby(rows, key=lambda b:b.timestamp))
        digest = stable_signature(dict(fine=[b.to_json_dict() for b in fine],
                                       coarse=[b.to_json_dict() for b in coarse]))
        self.session_cursor = SessionCursor(batches(fine), batches(coarse), digest)
        scope = dict(event='minute_day_scope_v5', day=str(fine[0].timestamp.date()),
                     symbols=sorted(symbols if symbols is not None else {b.symbol for b in fine}))
        self.broker.minute_day_scopes.append(scope)
        self.broker.journal.append(scope)

    def _release(self, until):
        c, broker, adapter = self.session_cursor, self.broker, self.adapter
        while c.coarse_index < len(c.coarse) and c.coarse[c.coarse_index][0].timestamp + timedelta(minutes=5) <= until:
            batch = c.coarse[c.coarse_index]
            c.coarse_index += 1
            available = batch[0].timestamp + timedelta(minutes=5)
            broker._decision_available_at = available
            broker.coarse_expected_batches.append(tuple(b.to_json_dict() for b in batch))
            broker.minute_clock_schedule.append(dict(kind='coarse', label=batch[0].timestamp.isoformat(),
                available_at=available.isoformat(), delivered_by=until.isoformat(), symbols=[b.symbol for b in batch]))
            broker.journal.append(dict(event='coarse_observation_v5', **broker.minute_clock_schedule[-1]))
            c.decisions.extend(adapter.on_bars(batch, broker))
            c.decisions.extend(adapter._sync_new_fills(broker))
            c.decisions.extend(adapter._sync_order_updates(broker, batch[0].timestamp))
            broker.minute_coarse_nav.append(dict(timestamp=batch[0].timestamp.isoformat(),
                available_at=available.isoformat(), equity=broker._portfolio_equity()))

    def advance(self):
        """Finish one sourced interval and all native callbacks at that boundary."""
        from strategy_olr.core.logic import on_olr_timer
        c, broker, adapter = self.session_cursor, self.broker, self.adapter
        if c is None or c.fine_index >= len(c.fine):
            raise ValueError('No unconsumed interval in the active session')
        batch = c.fine[c.fine_index]
        stamp = batch[0].timestamp
        self._release(stamp)
        broker.expected_batches.append(tuple(b.to_json_dict() for b in batch))
        broker.expected_strategy_state_timestamps['OLR'].append(stamp.isoformat())
        def fills():
            c.decisions.extend(adapter.on_fills(broker))
        broker.process_batch(batch, on_fills=fills)
        c.decisions.extend(adapter._sync_new_fills(broker))
        c.decisions.extend(adapter._sync_expired_orders(broker, stamp))
        c.decisions.extend(adapter._sync_order_updates(broker, stamp))
        available = stamp + timedelta(minutes=1)
        broker._decision_available_at = available
        broker.journal.append(dict(event='minute_observation_v5', timestamp=stamp.isoformat(),
            available_at=available.isoformat(), symbols=[b.symbol for b in batch]))
        c.available_at, c.last_batch = available, batch
        self._service_targets()
        for bar in batch:
            c.decisions.extend(minute_risk_step(adapter, bar, broker, fast_exit=self.fast_exit))
        self.structural_observer.observe(adapter, broker, available, batch)
        self._release(available)
        if c.fine_index == len(c.fine) - 1:
            remaining = c.coarse[c.coarse_index:]
            if any(not closing_observation(b) for pending in remaining for b in pending):
                raise ValueError('Unobserved continuous coarse timer beyond final minute')
            for pending in remaining:
                available = max(available, pending[0].timestamp + timedelta(minutes=5))
                self._release(available)
        broker._decision_available_at = available
        c.decisions.extend(on_olr_timer(adapter.state, available, adapter.config).decisions)
        c.decisions.extend(adapter._sync_order_updates(broker, stamp))
        adapter._record_reconciliation(broker, stamp)
        broker._decision_available_at = None
        c.available_at = available
        c.fine_index += 1
        return available

    def finish_day(self):
        c = self.session_cursor
        if c is None:
            raise ValueError('No active native session')
        while c.fine_index < len(c.fine):
            self.advance()
        if c.coarse_index != len(c.coarse):
            raise ValueError('Unconsumed coarse callbacks')
        result = c.decisions
        self.session_cursor = None
        return result

    def run_day(self, fine_bars, coarse_bars, *, symbols=None):
        fine_bars, coarse_bars = tuple(fine_bars), tuple(coarse_bars)
        twin = None
        if not self.checkpoint_checked and (self.broker.positions or self.broker.orders):
            self.checkpoint_checked = True
            path = Path(self.output) / 'native_account_checkpoint.pkl'
            receipt = save(self, path)
            twin = load(path, receipt['sha256'])
            if continuation_state(twin) != continuation_state(self):
                raise ValueError('Checkpoint changed native starting state')
            twin.run_day(fine_bars, coarse_bars, symbols=symbols)
        self.start_day(fine_bars, coarse_bars, symbols=symbols)
        result = self.finish_day()
        if twin is not None:
            same = continuation_state(twin) == continuation_state(self)
            self.checkpoint_receipt = dict(**receipt, no_action_continuation_equal=same,
                through=str(fine_bars[-1].timestamp), original_outcome=continuation_state(self),
                fork_outcome=continuation_state(twin))
            if not same:
                raise ValueError('Complete funded continuation failed parity')
        return result

    def decision_view(self):
        """Expose only consumed account information, never the remaining tape."""
        c = self.session_cursor
        if c is None or not c.fine_index or c.available_at is None:
            raise ValueError('A completed interval is required')
        broker, adapter = self.broker, self.adapter
        account = self.structural_observer.account
        observed = account.consume(broker.journal, c.available_at,
            actual_cash=broker.cash, actual_equity=broker._portfolio_equity())
        return dict(**observed, source_prefix=broker.source_events.hexdigest(),
            source_events=broker.source_event_count,
            quantities={s:ss.position.qty_open for s,ss in adapter.state.symbols.items() if ss.position},
            marks=dict(broker.last_prices),
            pending_orders=[dict(id=o.order_id,symbol=o.symbol,side=o.side,quantity=o.qty) for o in broker.orders])

    def request_inventory_target(self, symbol, target, *, action_id):
        """Commit a one-shot sell target after acknowledged cancellation.

        Targets are residual current inventory, not original entry quantity.
        No entry is synthesized; released cash follows native later decisions.
        Pending acquisition or competing priority replacement is unsupported.
        """
        if not isinstance(action_id, str) or not action_id or action_id in self.action_ids:
            raise ValueError('An unused explicit action identity is required')
        c = self.session_cursor
        view = self.decision_view()
        rows = [b for b in c.last_batch if b.symbol == symbol and continuous_minute(b)]
        if len(rows) != 1 or c.available_at != rows[0].timestamp + timedelta(minutes=1):
            raise ValueError('Target requires a freshly completed own-symbol continuous interval')
        ss = self.adapter.state.symbols.get(symbol)
        quantity = ss.position.qty_open if ss and ss.position else 0
        if type(target) is not int or not 0 <= target <= quantity or quantity <= 0:
            raise ValueError('Invalid residual inventory target')
        if symbol in self.pending_targets or ss.pending_entry_order_id or symbol in self.adapter.state.meta.get('priority_exit_pending', {}):
            raise ValueError('Conflicting pending native lifecycle')
        resting = [o for o in self.broker.orders if o.symbol == symbol]
        if any(o.side != 'SELL' or o.order_id != ss.pending_exit_order_id for o in resting):
            raise ValueError('Untracked competing order is not a supported holding intervention')
        request = dict(action_id=action_id,symbol=symbol,target=target,starting_quantity=quantity,
            committed_at=c.available_at.isoformat(),source_prefix=view['source_prefix'],
            consumed_source_events=view['source_events'],decision_view_hash=stable_signature(view))
        self.action_ids.add(action_id)
        if target == quantity:
            self.action_trace.append(dict(**request,status='continue_native'))
            return
        self.pending_targets[symbol] = request
        self.action_trace.append(dict(**request,status='cancel_requested'))
        self.broker._decision_available_at = c.available_at
        try:
            self.adapter._submit_actions([CancelOrders('OLR',symbol,'funded_target_cancel')],self.broker,rows[0].timestamp)
            c.decisions.extend(self.adapter._sync_order_updates(self.broker,rows[0].timestamp))
            self._service_targets()
        finally:
            self.broker._decision_available_at = None

    def _service_targets(self):
        c, broker, adapter = self.session_cursor, self.broker, self.adapter
        for symbol, request in list(self.pending_targets.items()):
            # An acknowledgement and reconciled native pending state are both
            # required. A cancellation request alone never releases inventory.
            ss = adapter.state.symbols[symbol]
            if any(o.symbol == symbol for o in broker.orders) or ss.pending_exit_order_id or ss.pending_entry_order_id:
                continue
            bars = [b for b in c.last_batch if b.symbol == symbol and continuous_minute(b)]
            if len(bars) != 1:
                continue
            position = ss.position
            quantity = position.qty_open if position else 0
            sell = max(0, quantity-request['target'])
            self.pending_targets.pop(symbol)
            if not sell:
                self.action_trace.append(dict(**request,status='already_reduced',remaining_quantity=quantity))
                continue
            metadata = dict(source_artifact_hash=position.source_artifact_hash,
                candidate_rank=position.candidate_rank,candidate_score=position.candidate_score,
                risk_per_share=position.metadata.get('risk_per_share',0),
                trade_exit_plan=position.metadata.get('trade_exit_plan',{}),
                order_role='FUNDED_TARGET',funded_action_id=request['action_id'],
                target_inventory=request['target'],committed_at=request['committed_at'])
            action_type = SubmitExit if sell == quantity else SubmitPartialExit
            action = action_type('OLR',symbol,sell,'MARKET',None,'funded_inventory_target',metadata)
            adapter._submit_actions([action],broker,bars[0].timestamp)
            c.decisions.extend(adapter._sync_order_updates(broker,bars[0].timestamp))
            self.action_trace.append(dict(**request,status='submitted',submitted_at=c.available_at.isoformat(),
                                         submitted_quantity=sell,remaining_quantity=quantity))
