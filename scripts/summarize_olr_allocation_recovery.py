"""Summarize retained native OLR allocation trials without selecting new rules.

Completed screens remain authoritative. This report also shows finished cells
from running generations, but never labels an incomplete policy as qualified.
"""
from pathlib import Path
from collections import defaultdict
import argparse
import gzip
import hashlib
import json


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def economic_signature(case):
    fields=('strategy_id','symbol','qty','entry_decision_time','entry_fill_time',
            'entry_price','exit_fill_time','exit_price','gross_pnl','commission',
            'net_pnl','realized')
    # Forecast provenance and unused policy metadata do not create a new
    # economic path. Costs, quantities, times and the marked path still do.
    trades=[{key:row[key] for key in fields} for row in read(case/'trades.json')]
    payload=[trades,read(case/'equity.json'),read(case/'daily.json')]
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def interventions(path):
    mechanisms = defaultdict(set)
    with gzip.open(path / 'decisions.jsonl.gz', 'rt', encoding='utf-8') as stream:
        for line in stream:
            event = json.loads(line)
            meta = event.get('metadata') or {}
            key = (event['timestamp'][:10], event['symbol'])
            if (event['reason']=='scheduled_capital_release' and event['decision_code'] in ('EXIT_SUBMITTED','PARTIAL_EXIT_SUBMITTED')
                    or event['reason']=='opportunity_capital_rotation' and event['decision_code']=='CAPITAL_ROTATION_SUBMITTED'):
                mechanisms[event['reason']].add(key)
            # Fill/position records can carry old entry metadata. Count fresh
            # allocation decisions only, once per issuer and decision day.
            if event['decision_code'] not in ('ENTRY_SUBMITTED', 'ENTRY_SKIPPED'):
                continue
            for field in ('allocation_redistributed_qty', 'capacity_conditioned_qty_delta',
                          'capacity_budget_qty_delta', 'capacity_value_qty_delta',
                          'capacity_history_qty_delta'):
                if meta.get(field, 0):
                    mechanisms[field].add(key)
    return {'by_mechanism': {key: len(value) for key, value in mechanisms.items()},
            'distinct_issuer_days': len(set().union(*mechanisms.values()))}


def summarize(root, extra_generations=()):
    rows, generations = [], []
    economic_paths = defaultdict(list)
    paths={path.name:path for path in root.glob('generation_*')}
    for path in extra_generations:
        previous=paths.get(path.name)
        if previous and (previous/'registration.json').exists():
            if digest(previous/'registration.json')!=digest(path/'registration.json'):
                raise ValueError('A relocated generation changed its registration')
        paths[path.name]=path
    for generation in sorted(paths.values(), key=lambda p: int(p.name.split('_')[-1])):
        if int(generation.name.split('_')[-1]) < 4 or generation.name == 'generation_8':
            continue
        registration_path = generation / 'registration.json'
        if not registration_path.exists():
            continue
        registration = read(registration_path)
        verdict_path = generation / 'screen_verdict.json'
        verdict = read(verdict_path) if verdict_path.exists() else None
        generations.append({'generation': generation.name, 'path': str(generation),
            'registration_hash': digest(registration_path),
            'source_generation_hash': registration['source_generation_hash'],
            'screen_complete': bool(verdict and verdict['complete']),
            'eligible': verdict['eligible'] if verdict else None})
        for name, profile in registration['profiles'].items():
            record = {'generation': generation.name, 'recipe': name, 'profile': profile,
                      'cells': {}, 'screen_complete': bool(verdict and verdict['complete'])}
            for scenario in registration['scenarios']:
                case = generation / 'cases' / f'{name}_100000000_{scenario}'
                if not (case / 'completion.json').exists():
                    continue
                result = read(case / 'result.json')
                metrics = {key: result[key] for key in ('return_pct', 'drawdown_pct', 'trades',
                    'execution_validity_pass', 'audit_status', 'terminal_inventory', 'terminal_pending')}
                metrics['path'] = str(case)
                metrics['result_hash'] = digest(case / 'result.json')
                metrics['direct_support'] = interventions(case)
                signature = economic_signature(case)
                economic_paths[signature].append(f'{generation.name}/{name}/{scenario}')
                record['cells'][scenario] = metrics
            if record['cells']:
                rows.append(record)
    return {'scope': 'Reused development history; descriptive consolidation, not independent evidence',
        'reference': 'Selected median cap. Gates unchanged; no selection by this report.',
        'generations': generations, 'policies': rows,
        'completed_cells': sum(len(row['cells']) for row in rows),
        'distinct_economic_paths': len(economic_paths),
        'exact_duplicate_paths': [names for names in economic_paths.values() if len(names) > 1]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--generation', type=Path, action='append', default=[],
                        help='Additional frozen generation, e.g. on the local SSD')
    args = parser.parse_args()
    report = summarize(args.root, args.generation)
    (args.root / 'native_trial_summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('completed_cells', 'distinct_economic_paths', 'generations')}, indent=2))
