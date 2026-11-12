"""One unchanged frozen parent replay with native removal-gate observations.

This worker does not select finalists, fit policies, or schedule experiments.
The shared research lifecycle must bind every input before invoking it and
independently reconcile the output account before scoring any observation.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys


def removed_opening_entries(parent, gated):
    """Derive the gate from authenticated opening snapshots, never outcomes."""
    if parent.keys() != gated.keys():
        raise ValueError('Removal calendar differs from parent')
    removed = set()
    for day, snapshot in parent.items():
        before, after = snapshot.by_symbol(), gated[day].by_symbol()
        if not after.keys() <= before.keys():
            raise ValueError('Removal observer received additions')
        for symbol, candidate in after.items():
            original = before[symbol]
            # Repackaging may update documentary ranks/counts. The accepted
            # incumbent action itself must retain its value and quantity.
            if candidate.selection_score != original.selection_score or (
                candidate.metadata.get('action_quantity_scale') !=
                original.metadata.get('action_quantity_scale')
            ):
                raise ValueError('Removal observer received changed incumbent action')
        removed.update(str(day)+'/'+symbol for symbol in before.keys()-after.keys())
    return removed


def economic_bridge(actual, reference, read):
    """Require identical native parent accounts, marks and physical fills."""
    for name in ('daily.json', 'equity.json'):
        if read(actual/name) != read(reference/name):
            raise ValueError('Observer account bridge drift '+name)

    def physical(fill):
        return {**{k: fill[k] for k in (
            'order_id', 'strategy_id', 'symbol', 'side', 'qty', 'price',
            'timestamp', 'reason')},
            'execution': {k: fill['metadata'].get(k) for k in (
                'execution_phase', 'base_price', 'fee', 'decision_available_at',
                'submitted_at', 'execution_bar')}}

    if list(map(physical, read(actual/'fills.json'))) != list(map(physical, read(reference/'fills.json'))):
        raise ValueError('Observer physical fill bridge drift')
    return {'passed': True, 'reference': str(reference),
            'equity_marks': len(read(actual/'equity.json')),
            'scope': 'All parent daily balances, timestamped equity marks and physical fills reproduced exactly.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('code', 'plan', 'policies', 'gate-policies', 'audit-manifest',
                 'observer', 'registration', 'bridge', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--scenario', required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.code.resolve()))
    from scripts.kalcb_executed_value_replay import run, verify_policy, timed_policy
    from scripts.kalcb_executed_value_study import load_plan
    from backtests.auto.shared.cache_keys import source_generation_hash
    from backtests.strategies.kalcb import executed_value as ev

    read = lambda path: json.loads(Path(path).read_text(encoding='utf-8'))
    registration = read(args.registration)
    registration_hash = ev.digest(args.registration)
    if registration['scenario'] != args.scenario:
        raise ValueError('Unregistered observation scenario')
    bindings = registration['bindings']
    bound = {Path(b['path']).resolve(): b['sha256'] for b in bindings}
    if len(bound) != len(bindings):
        raise ValueError('Duplicate registration binding')

    def authenticate():
        for path, digest in bound.items():
            if ev.digest(path) != digest:
                raise ValueError('Changed registered removal-observer input '+str(path))

    required = [Path(__file__), args.observer, args.plan, args.audit_manifest,
                args.policies/'policies_manifest.json', args.gate_policies/'policies_manifest.json']
    required += [args.bridge/name for name in ('result.json', 'daily.json', 'equity.json', 'fills.json',
                                              'independent_account_verification.json')]
    if any(path.resolve() not in bound for path in required):
        raise ValueError('Unregistered removal-observer input')
    authenticate()
    if not read(args.bridge/'independent_account_verification.json')['passed']:
        raise ValueError('Parent reference has no independent account verification')
    if read(args.bridge/'result.json')['audit_status'] != 'passed':
        raise ValueError('Parent reference source audit failed')
    plan = load_plan(args.plan)
    source = source_generation_hash()
    if source != registration['source_generation_hash']:
        raise ValueError('Unregistered economic runtime')
    name = 'interaction_cost_margin'
    parent_manifest, parent = verify_policy(args.policies, name)
    gate_manifest, gated = verify_policy(args.gate_policies, name)
    timed_policy(args.policies, parent_manifest, parent)
    timed_policy(args.gate_policies, gate_manifest, gated)
    removed = removed_opening_entries(parent, gated)
    spec = importlib.util.spec_from_file_location('population_removal_observer_frozen', args.observer)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    observer = module.PopulationRemovalObserver(removed)
    args.output.mkdir(parents=True, exist_ok=False)
    arguments = {k: str(getattr(args, k)) for k in ('plan', 'policies', 'output', 'scenario', 'audit_manifest')}
    arguments['policy'] = name
    ev.write_json(args.output/'request.json', {
        'arguments': arguments, 'plan': plan, 'source_generation_hash': source,
        'observational_registration': {'path': str(args.registration), 'sha256': registration_hash},
        'observational_programs': {str(p): ev.digest(p) for p in (Path(__file__), args.observer)},
        'removed_opening_issuer_days': sorted(removed)})
    observer.install()
    try:
        run(plan, args.policies, name, args.scenario, args.output, args.audit_manifest)
    finally:
        observer.restore()
    if read(args.output/'result.json')['audit_status'] != 'passed':
        raise ValueError('Observed parent failed native source audit')
    bridge = economic_bridge(args.output, args.bridge, read)
    support = observer.summarize(read(args.output/'fills.json'))
    authenticate()
    if ev.digest(args.registration) != registration_hash:
        raise ValueError('Observation registration changed during replay')
    if source_generation_hash() != source:
        raise ValueError('Economic runtime changed during observation')
    load_plan(args.plan)
    ev.write_json(args.output/'observer_bridge.json', bridge)
    ev.write_json(args.output/'intervention_support.json', {
        'complete': True, 'summary': support, 'probes': observer.probes,
        'independent_account_validation_required': True})
    ev.write_json(args.output/'completion.json', {
        'complete': True, 'source_generation_hash': source,
        'native_parent_bridge_passed': True, 'independent_account_validation_required': True})
    print(json.dumps({'completed': str(args.output), 'blockable_filled_issuer_days':
                      support['direct_blockable_entry_issuer_days']}), flush=True)


if __name__ == '__main__':
    main()
