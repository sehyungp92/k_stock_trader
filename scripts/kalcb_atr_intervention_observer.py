"""Observe native ATR sizing and same-state feasible shares without changing fills.

This diagnostic never books counterfactual PnL. Realized volume is an ex-post
feasibility check only. Economic output must exactly bridge the native reference.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import sys


def feasible_extra(actual, counterfactual, capacity_remaining, realized_remaining, affordable):
    """Conservative whole-order native feasibility, not hypothetical partial buying power."""
    alternative = min(counterfactual, capacity_remaining)
    if not affordable or alternative > realized_remaining:
        return 0
    return max(0, alternative-actual)


class Observer:
    def __init__(self):
        self.pending = {}; self.orders = {}; self.probes = []; self.fills = []; self.bars = {}

    def install(self):
        from strategy_kalcb.core import logic
        from strategy_common.actions import SubmitEntry
        from backtests.engine.sim_broker import SimBroker
        quantity, submit = logic.compute_entry_qty, SimBroker.submit
        policy, batch = SimBroker._policy_execution_quantity, SimBroker.process_batch
        self.originals = (logic, SimBroker, quantity, submit, policy, batch)
        observer = self

        def qty(**kwargs):
            actual = quantity(**kwargs); candidate = kwargs['candidate']
            observer.pending.pop(candidate.symbol, None)
            original = candidate.metadata.get('diagnostic_original_quantity')
            if original is None:
                return actual
            if kwargs['config'].risk_daily_vol_target:
                raise ValueError('Observer must explicitly account for additional volatility multiplier')
            changed = replace(candidate, metadata={**candidate.metadata, 'action_quantity_scale': original})
            restored = quantity(**{**kwargs, 'candidate': changed})
            if restored < actual:
                raise ValueError('ATR reduction increased native shares')
            row = {'symbol': candidate.symbol, 'actual_qty': actual, 'restored_qty': restored,
                   'inputs': {k: v for k, v in kwargs.items() if k not in ('candidate', 'config')},
                   'candidate': candidate.to_json_dict(), 'config': asdict(kwargs['config']),
                   'order_id': None}
            observer.probes.append(row); observer.pending[candidate.symbol] = row
            return actual

        def submitted(broker, action, submitted_at):
            order_id = submit(broker, action, submitted_at)
            row = observer.pending.pop(action.symbol, None) if isinstance(action, SubmitEntry) else None
            if row is not None:
                if action.qty != row['actual_qty']:
                    raise ValueError('Observed sizing differs from submitted shares')
                row.update(order_id=order_id, submitted_at=submitted_at.isoformat(), submitted_qty=action.qty)
                if order_id:
                    observer.orders[order_id] = row
            return order_id

        def execution(broker, order, price, quantity_allowed, timestamp):
            actual = policy(broker, order, price, quantity_allowed, timestamp)
            probe = observer.orders.get(order.order_id)
            if probe is None or order.side != 'BUY':
                return actual
            bar = observer.bars[order.symbol]
            quote = broker.capacity_policy.quote(bar, broker._phase, broker.previous_capacity_bars.get(bar.symbol))
            used = broker.capacity_used.get((bar.symbol, timestamp.isoformat(), broker._phase), 0)
            cap = max(0, quote.budget-used)
            previous = broker.order_filled.get(order.order_id, 0)
            restored_remaining = max(0, probe['restored_qty']-previous)
            restored = policy(broker, replace(order, qty=restored_remaining), price, min(restored_remaining, cap), timestamp)
            realized = max(0, math.floor(bar.volume*broker.capacity_policy.participation)-used)
            affordable = broker._can_afford(replace(order, qty=restored), price) if restored > 0 else False
            row = {'order_id': order.order_id, 'symbol': order.symbol, 'timestamp': timestamp.isoformat(),
                   'phase': broker._phase, 'actual_allowed': actual, 'restored_remaining': restored_remaining,
                   'restored_allowed': restored, 'capacity_remaining': cap, 'realized_capacity_remaining': realized,
                   'cash_before': broker.cash, 'open_notional_before': broker._open_notional(),
                   'execution_price': price, 'restored_affordable': affordable,
                   'entry_submission_at': order.submitted_at.isoformat(),
                   'quote': asdict(quote), 'bar': bar.to_json_dict(),
                   'feasible_extra_shares': feasible_extra(actual, restored, cap, realized, affordable),
                   'actual_fill_confirmed': False}
            observer.fills.append(row)
            return actual

        def processed(broker, bars, *, on_fills=None):
            bars = tuple(bars); observer.bars = {b.symbol: b for b in bars}
            first = len(observer.fills)
            actual = batch(broker, bars, on_fills=on_fills)
            for row in observer.fills[first:]:
                matching = [f for f in actual if f.order_id == row['order_id'] and f.timestamp.isoformat() == row['timestamp']
                            and f.metadata['execution_phase'] == row['phase']]
                if matching:
                    if len(matching) != 1 or matching[0].qty != row['actual_allowed']:
                        raise ValueError('Observed fill differs from native fill')
                    row['actual_fill_confirmed'] = True
                    row['execution_id'] = matching[0].metadata['execution_id']
                else:
                    row['feasible_extra_shares'] = 0
            return actual

        logic.compute_entry_qty = qty; SimBroker.submit = submitted
        SimBroker._policy_execution_quantity = execution; SimBroker.process_batch = processed

    def restore(self):
        logic, broker, qty, submit, policy, batch = self.originals
        logic.compute_entry_qty = qty; broker.submit = submit
        broker._policy_execution_quantity = policy; broker.process_batch = batch

    def summary(self):
        direct = [r for r in self.fills if r['actual_fill_confirmed'] and r['feasible_extra_shares'] > 0]
        issuer_days = {(r['symbol'], r['timestamp'][:10]) for r in direct}
        sessions = {day for _, day in issuer_days}
        submitted = [r for r in self.probes if r['order_id']]
        return {'quantity_evaluations': len(self.probes), 'submitted_atr_orders': len(submitted),
                'submitted_orders_with_changed_quantity': sum(r['restored_qty'] > r['actual_qty'] for r in submitted),
                'direct_feasible_fill_events': len(direct), 'direct_feasible_issuer_days': len(issuer_days),
                'direct_feasible_sessions': len(sessions), 'feasible_extra_shares': sum(r['feasible_extra_shares'] for r in direct),
                'preferred_support_passed': len(issuer_days) >= 10 and len(sessions) >= 8,
                'session_dates': sorted(sessions),
                'interpretation': 'Only ATR quantity restored in the identical native decision and pre-fill state; original order timing, affordability, prior-bar capacity and realized-volume audit limits retained. No hypothetical fills or PnL booked. Conditional mechanism support, not independent samples; later compounded trades cannot substitute for a direct feasible quantity effect.'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('code', 'plan', 'policies', 'reference', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--scenario', choices=['baseline', 'combined'], required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.code.resolve()))
    from scripts.kalcb_executed_value_replay import run, load_plan
    from backtests.auto.shared.cache_keys import source_generation_hash
    args.output.mkdir(parents=True, exist_ok=False)
    source = source_generation_hash(); observer = Observer(); observer.install()
    try:
        run(load_plan(args.plan), args.policies, 'interaction_cost_margin', args.scenario, args.output)
    finally:
        observer.restore()
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    result = read(args.output/'result.json'); prior = read(args.reference/'result.json')
    if not result['execution_validity_pass'] or source_generation_hash() != source:
        raise ValueError('Observer execution or source failure')
    for key in ('return_pct', 'drawdown_pct', 'trades', 'active_days'):
        if abs(result[key]-prior[key]) > 1e-8: raise ValueError('Observer economic bridge drift '+key)
    for name in ('daily.json', 'equity.json'):
        if read(args.output/name) != read(args.reference/name):
            raise ValueError('Observer full-account bridge drift '+name)
    # Forecast hashes can differ between historically equivalent policy exports.
    # Compare every physical execution and cash outcome, not those metadata labels.
    def physical_fill(row):
        return {**{k: row[k] for k in ('order_id','strategy_id','symbol','side','qty','price','timestamp','reason')},
                'execution': {k: row['metadata'].get(k) for k in ('execution_phase','base_price','fee','decision_available_at','submitted_at','execution_bar')}}
    if [physical_fill(r) for r in read(args.output/'fills.json')] != [physical_fill(r) for r in read(args.reference/'fills.json')]:
        raise ValueError('Observer physical execution bridge drift')
    def physical_trade(row):
        return {k: v for k, v in row.items() if k not in ('route_metadata','cohort_metadata','source_artifact_hash')}
    if [physical_trade(r) for r in read(args.output/'trades.json')] != [physical_trade(r) for r in read(args.reference/'trades.json')]:
        raise ValueError('Observer trade/account bridge drift')
    payload = {'complete': True, 'summary': observer.summary(), 'probes': observer.probes, 'execution_checks': observer.fills,
               'program': {'path': str(Path(__file__).resolve()), 'sha256': digest(__file__)},
               'source_generation_hash': source, 'native_reference': str(args.reference), 'full_account_bridge': 'passed',
               'scope': 'Observational screening diagnostic, not a separately source-qualified economic result.'}
    (args.output/'intervention_support.json').write_text(json.dumps(payload, indent=2, default=str, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(payload['summary']), flush=True)


if __name__ == '__main__': main()
