"""Register a stronger standalone challenge without changing frozen replay code."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backtests.strategies.kalcb.executed_value import digest, signature, write_json


def prepare(parent_path, output):
    parent = json.loads(parent_path.read_text(encoding='utf-8'))
    manifest = json.loads(Path(parent['manifest']).read_text(encoding='utf-8'))
    expected = manifest.pop('manifest_hash')
    if signature(manifest) != expected:
        raise ValueError('Changed parent manifest')
    output.mkdir(parents=True, exist_ok=False)
    root = output / 'inputs/files'
    for name in manifest['files']:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(Path(parent['code']) / name, target)
    spec = json.loads((root / parent['spec_path']).read_text(encoding='utf-8'))
    base = copy.deepcopy(spec['candidates']['kalcb_size_150'])
    jobs = []
    for name, fraction in [('kalcb_size_175', 1.75), ('kalcb_size_200', 2.)]:
        spec['candidates'][name] = {**base, 'reference_sizing': fraction}
        for scenario in ('baseline', 'combined'):
            jobs.append({'name': name + '_' + scenario, 'recipe': name,
                         'scenario': scenario, 'standalone': 'KALCB'})
    spec['reference_challenge'] = {
        'purpose': 'Challenge portfolio selection with additional executable standalone sizing, without increasing physical leverage.',
        'disclosure': 'Primary KALCB 1.25 baseline was observed at 86.24445835109975% return and 8.507294194572053% DD. Additional sizing levels registered before their outcomes; no primary risk gate is relaxed.',
        'selection': 'Retain original selection rule but include every feasible added standalone in both return floors.',
        'reused_development_only': True}
    spec_name = 'research/portfolio_current/standalone_challenge.json'
    write_json(root / spec_name, spec)
    manifest['files'][spec_name] = digest(root / spec_name)
    manifest['evidence']['portfolio_current_study'] = spec_name
    manifest['evidence']['standalone_challenge_parent_manifest_hash'] = expected
    write_json(output / 'inputs/manifest.json', {**manifest, 'manifest_hash': signature(manifest)})
    write_json(output / 'study.json', {**parent, **spec,
        'code': str(root.resolve()), 'manifest': str((output / 'inputs/manifest.json').resolve()), 'spec_path': spec_name})
    write_json(output / 'registration.json', {'parent_study': str(parent_path.resolve()),
        'parent_study_sha256': digest(parent_path), 'source_generation_hash': parent['source_generation_hash'],
        'jobs': jobs, 'selection': spec['allocation_selection'], 'disclosure': spec['reference_challenge'],
        'replay_code': 'Every Python file is unchanged from primary runtime_v3.'})
    print(json.dumps({'prepared': str(output), 'jobs': len(jobs), 'source': parent['source_generation_hash']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.study, args.output)
