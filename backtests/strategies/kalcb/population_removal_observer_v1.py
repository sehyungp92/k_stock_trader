"""Observe a removal rule on an unchanged parent's actual order lifecycle.

Only the current issuer is removed from a detached pre-decision snapshot. The
original parent continues trading unchanged. Matched native BUY fills establish
that the gate could block that parent's executable entry in that identical state;
they do not establish an isolated PnL effect or the variant's later account path.
"""
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str,
                                     separators=(',', ':')).encode()).hexdigest()


class PopulationRemovalObserver:
    def __init__(self, removed_issuer_days):
        self.removed = frozenset(removed_issuer_days)
        self.probes = []
        self.orders = {}
        self.pending = {}
        self.installed = False
        self.actual_state_id = None

    def install(self):
        if self.installed:
            raise ValueError('Observer already installed')
        from backtests.strategies.kalcb import runner
        from backtests.engine.sim_broker import SimBroker
        from strategy_common.actions import SubmitEntry, action_to_json_dict
        from strategy_kalcb.core.serializers import snapshot_state
        step, submit = runner.step_kalcb_core, SimBroker.submit
        on_bar = runner.KALCBReplayAdapter.on_bar
        self.originals = runner, SimBroker, step, submit, on_bar
        observer = self

        def actual_bar(adapter, bar, broker):
            previous = observer.actual_state_id
            observer.actual_state_id = id(adapter.state)
            try:
                return on_bar(adapter, bar, broker)
            finally:
                observer.actual_state_id = previous

        def observed(state, bar, config, candidate_snapshot, portfolio):
            key = bar.timestamp.date().isoformat()+'/'+bar.symbol
            candidate = next((c for c in candidate_snapshot.candidates if c.symbol == bar.symbol), None) if candidate_snapshot else None
            # An opening removal does not suppress later independent checkpoints.
            eligible = (id(state) == observer.actual_state_id and key in observer.removed and candidate is not None and
                        not candidate.metadata.get('entry_checkpoint_at'))
            prior = deepcopy(state) if eligible else None
            account = deepcopy(portfolio) if eligible else None
            snapshot = deepcopy(candidate_snapshot) if eligible else None
            original_config = deepcopy(config) if eligible else None
            result = step(state, bar, config, candidate_snapshot, portfolio)
            entries = [a for a in result.actions if isinstance(a, SubmitEntry) and a.symbol == bar.symbol]
            if not eligible or not entries:
                return result
            if len(entries) != 1:
                raise ValueError('Ambiguous same-state parent entry')
            before = signature(snapshot_state(prior))
            alternative = replace(snapshot, candidates=tuple(c for c in snapshot.candidates if c.symbol != bar.symbol))
            # All other candidates, ranks, account resources and state are fixed.
            gated = step(prior, deepcopy(bar), original_config, alternative, account)
            suppressed = not any(isinstance(a, SubmitEntry) and a.symbol == bar.symbol for a in gated.actions)
            row = {'issuer_day': key, 'symbol': bar.symbol,
                   'decision_bar': bar.timestamp.isoformat(), 'pre_state_sha256': before,
                   'parent_snapshot_sha256': snapshot.artifact_hash,
                   'removed_candidate': candidate.to_json_dict(),
                   'portfolio': asdict(portfolio), 'parent_action': deepcopy(action_to_json_dict(entries[0])),
                   'same_state_gate_suppressed_entry': suppressed,
                   'gate_decisions': [d.to_json_dict() for d in gated.decisions]}
            observer.probes.append(row)
            observer.pending[(bar.symbol, bar.timestamp)] = row
            return result

        def submitted(broker, action, submitted_at):
            order_id = submit(broker, action, submitted_at)
            if isinstance(action, SubmitEntry):
                probe = observer.pending.pop((action.symbol, submitted_at), None)
                if probe is not None:
                    if action_to_json_dict(action) != probe['parent_action']:
                        raise ValueError('Native submitted entry differs from observed action')
                    if order_id:
                        observer.orders[order_id] = {**probe, 'order_id': order_id,
                                                    'submitted_at': submitted_at.isoformat()}
            return order_id

        runner.step_kalcb_core, SimBroker.submit = observed, submitted
        runner.KALCBReplayAdapter.on_bar = actual_bar
        self.installed = True

    def restore(self):
        if not self.installed:
            raise ValueError('Observer is not installed')
        runner, broker, step, submit, on_bar = self.originals
        runner.step_kalcb_core, broker.submit = step, submit
        runner.KALCBReplayAdapter.on_bar = on_bar
        self.installed = False

    def summarize(self, fills):
        matches = []
        seen = set()
        for fill in fills:
            order = self.orders.get(fill['order_id'])
            if order is None:
                continue
            if fill['side'] != 'BUY':
                raise ValueError('Removal entry observer matched an exit')
            metadata = fill['metadata']
            execution_id = metadata.get('execution_id')
            if not execution_id or execution_id in seen:
                raise ValueError('Missing or duplicate physical execution identity')
            seen.add(execution_id)
            if not order['same_state_gate_suppressed_entry']:
                continue
            matches.append({'issuer_day': order['issuer_day'], 'order_id': fill['order_id'],
                            'execution_id': execution_id, 'timestamp': str(fill['timestamp']),
                            'qty': fill['qty'], 'price': fill['price'],
                            'decision_available_at': metadata.get('decision_available_at'),
                            'submitted_at': order['submitted_at'],
                            'pre_state_sha256': order['pre_state_sha256']})
        issuer_days = {x['issuer_day'] for x in matches}
        sessions = {x.split('/')[0] for x in issuer_days}
        return {'declared_removed_issuer_days': len(self.removed),
                'same_state_parent_entry_probes': len(self.probes),
                'submitted_parent_entries': len(self.orders),
                'filled_parent_orders_blockable_by_gate': len({x['order_id'] for x in matches}),
                'direct_blockable_entry_issuer_days': len(issuer_days),
                'direct_blockable_entry_sessions': len(sessions),
                'months': {m: sum(x.startswith(m) for x in issuer_days) for m in sorted({x[:7] for x in sessions})},
                'execution_checks': matches,
                'scope': 'Actual filled parent entries whose identical pre-decision core state emits no entry when that issuer alone is removed from its opening snapshot. Parent replay is unchanged and requires exact economic bridge and independent source/account audits. Checkpoint entries are excluded; no hypothetical fill or PnL is credited. This proves direct gate support on the parent path, not isolated economic gain or descendant independence.'}
