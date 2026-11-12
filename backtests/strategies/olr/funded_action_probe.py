"""One registered, causal full-account intervention; never a fitted policy.

The first eligible existing fade boundary is selected without an outcome table.
The native runner still selects each later session's scope from that arm's
actual book and orders. A checkpoint is evidence, not a fixed-size trade lane.
"""
import json
from datetime import timedelta
from pathlib import Path

from backtests.auto.shared.cache_keys import stable_signature
from backtests.engine.olr_minute_v5 import continuous_minute, fixed_exit_plan
from .decision_continuation import DecisionContinuationDriver, continuation_state
from .funded_checkpoint import save, load


VERSION = 'first-native-fade-funded-probe-v1'
ACTIONS = ('hold', 'half', 'exit')


class FundedActionProbeDriver(DecisionContinuationDriver):
    def __init__(self, *args, probe, **kwargs):
        if (set(probe) != {'version', 'action', 'control_reference'} or
                probe['version'] != VERSION or probe['action'] not in ACTIONS):
            raise ValueError('Unregistered funded probe')
        if kwargs.get('mode') != 'control' or not kwargs.get('interval_service', True):
            raise ValueError('Probe requires unchanged interval control')
        if (probe['action'] == 'hold') != (probe['control_reference'] is None):
            raise ValueError('Actions require an authenticated no-action reference')
        kwargs['verify_checkpoint'] = False
        super().__init__(*args, **kwargs)
        self.probe = probe
        self.broker.funded_probe_contract = dict(version=VERSION, action=probe['action'])
        self.probe_selection = None
        self.probe_receipt = None

    def eligible(self):
        """Existing predicate; current own-symbol state, no remaining tape."""
        from strategy_olr.core.logic import _position_risk, _completed_bar_exit_reason
        c = self.session_cursor
        if self.probe_selection is not None:
            return None
        for bar in sorted(c.last_batch, key=lambda b: b.symbol):
            ss = self.adapter.state.symbols.get(bar.symbol)
            p = ss.position if ss else None
            if (p is None or p.qty_open < 2 or ss.pending_entry_order_id or
                    bar.timestamp.date() <= p.entry_time.date() or
                    not continuous_minute(bar) or
                    c.available_at != bar.timestamp + timedelta(minutes=1) or
                    bar.symbol in self.adapter.state.meta.get('priority_exit_pending', {})):
                continue
            resting = [o for o in self.broker.orders if o.symbol == bar.symbol]
            if any(o.side != 'SELL' or o.order_id != ss.pending_exit_order_id or
                   o.order_type != 'CLOSE_AUCTION' or o.reason != 'next_close_exit' for o in resting):
                continue
            if bool(resting) != bool(ss.pending_exit_order_id):
                continue
            risk = _position_risk(p)
            reason = _completed_bar_exit_reason((bar,), 0, p.entry_price, risk,
                p.max_favorable_price, fixed_exit_plan(self.adapter.config), 0)['reason']
            if reason == 'mfe_fade':
                return dict(symbol=bar.symbol, quantity=p.qty_open, risk=risk,
                    entry_price=p.entry_price, entry_time=p.entry_time.isoformat(),
                    high_water=p.max_favorable_price, bar=bar.to_json_dict(),
                    available_at=c.available_at.isoformat(), pending_exit=ss.pending_exit_order_id or None)
        return None

    def advance(self):
        available = super().advance()
        candidate = self.eligible()
        if candidate is None:
            return available
        view = self.decision_view()
        candidate.update(source_prefix=view['source_prefix'], source_events=view['source_events'],
            decision_view_hash=stable_signature(view), ordinal=1)
        # The identical observation is in every arm's pre-intervention state.
        # Neither the remaining session nor any branch payoff selects the action.
        self.probe_selection = candidate
        self.broker.journal.append(dict(event='funded_probe_boundary_v1', **candidate))
        start = continuation_state(self)
        receipt = save(self, Path(self.output)/'native_account_checkpoint.pkl')
        shadow = load(receipt['path'], receipt['sha256'])
        if continuation_state(shadow) != start:
            raise ValueError('Intraday checkpoint changed the complete account')
        quantity = candidate['quantity']
        action = self.probe['action']
        target = quantity if action == 'hold' else quantity // 2 if action == 'half' else 0
        # Commit the pre-registered action before evaluating the future shadow.
        if action != 'hold':
            self.request_inventory_target(candidate['symbol'], target, action_id='first_native_fade_1')
        shadow.finish_day()
        shadow_signature = continuation_state(shadow)
        self.probe_receipt = dict(version=VERSION, action=action, target=target,
            decision=candidate, observed_account=view, start_signature=start,
            no_action_session_signature=shadow_signature, checkpoint=receipt,
            selected_without_future_outcome=True, current_day=candidate['available_at'][:10])
        return available

    def run_day(self, fine_bars, coarse_bars, *, symbols=None):
        self.start_day(fine_bars, coarse_bars, symbols=symbols)
        result = self.finish_day()
        r = self.probe_receipt
        if r is not None and not self.checkpoint_receipt:
            reference = self.probe['control_reference']
            if self.probe['action'] == 'hold':
                same = continuation_state(self) == r['no_action_session_signature']
            else:
                same = all(r[k] == reference[k] for k in (
                    'decision', 'observed_account', 'start_signature', 'no_action_session_signature'))
            if not same:
                raise ValueError('Real intraday no-action continuation does not match control')
            self.checkpoint_receipt = dict(**r['checkpoint'], no_action_continuation_equal=True,
                through=r['current_day'], original_outcome=r['no_action_session_signature'],
                fork_outcome=r['no_action_session_signature'],
                proof='Uninterrupted hold equals restored intraday suffix; action shadows match that control.')
        return result

    def export_probe(self):
        if self.probe_receipt is None or not self.checkpoint_receipt:
            raise ValueError('Registered causal boundary was never reached')
        if self.pending_targets:
            raise ValueError('Unresolved inventory commitment')
        value = dict(self.probe_receipt, action_trace=self.action_trace,
            no_action_continuation_equal=self.checkpoint_receipt['no_action_continuation_equal'],
            economic_objective_achieved=False, general_policy_qualification=False)
        path = Path(self.output)/'funded_probe_receipt.json'
        with path.open('x', encoding='utf8') as f:
            json.dump(value, f, indent=2, default=str)
        return value
