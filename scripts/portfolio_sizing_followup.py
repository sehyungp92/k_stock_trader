"""Freeze a finite sizing interaction study and reuse identical parent replays."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from backtests.strategies.kalcb.executed_value import digest, signature, write_json
from scripts.portfolio_synergy_allocation_study import read, verify_case, run


def prepare(parent_path, output):
    parent = read(parent_path)
    manifest = read(parent['manifest'])
    expected = manifest.pop('manifest_hash')
    if signature(manifest) != expected:
        raise ValueError('Changed parent generation')
    output.mkdir(parents=True, exist_ok=False)
    root = output / 'inputs/files'
    for name in manifest['files']:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(Path(parent['code']) / name, target)
    spec = read(root / parent['spec_path'])
    base = copy.deepcopy(spec['candidates']['kalcb_size_150'])
    for name, fraction in [('kalcb_size_175', 1.75), ('kalcb_size_200', 2.)]:
        spec['candidates'][name] = {**copy.deepcopy(base), 'reference_sizing': fraction}
    base = copy.deepcopy(spec['candidates']['overnight_50'])
    for name, ksize, obudget, osize, symbol in (
        ('kalcb125_olr50', 1.25, .5, 1., .5),
        ('kalcb150_olr50', 1.5, .5, 1., .5),
        ('kalcb150_olr50_symbol75', 1.5, .5, 1., .75),
        ('kalcb150_olr35_symbol75', 1.5, .35, 1., .75),
        ('kalcb150_olr50_proportional', 1.5, .5, .5, .75),
        ('kalcb150_olr35_proportional', 1.5, .35, .35, .75),
        ('kalcb200_olr20_proportional', 2., .2, .2, 1.),
        ('kalcb200_olr10_proportional', 2., .1, .1, 1.)):
        recipe = copy.deepcopy(base)
        recipe['allocation'].update(kalcb_sizing_fraction=ksize, olr_budget_fraction=obudget,olr_sizing_fraction=osize)
        recipe['policy']['max_symbol_notional'] = symbol * spec['initial_equity']
        spec['candidates'][name] = recipe
    spec['sizing_followup'] = {
        'purpose': 'Test allocating more intraday sizing to KALCB while restricting OLR overnight admission; challenge against stronger standalone sizing.',
        'disclosure': 'Registered after primary KALCB 1.25 baseline and initial portfolio controls were observed, before any of the eight new portfolio or four new standalone-scenario outcomes. Supersedes unexecuted runtime_v4 and standalone_challenge. Proportional OLR sizing isolates the smaller-budget first-candidate concentration issue; 10% and 20% sleeves test the previously observed small-account capacity advantage. Reused development only.',
        'risk_gates': 'Unchanged: 11% baseline DD, 12.5% combined DD. No exception for near misses.',
        'reuse': 'Parent results may be reused only with identical replay Python, native inputs, recipe settings, account size and evaluation configuration.'}
    spec_name = 'research/portfolio_current/sizing_followup.json'
    write_json(root / spec_name, spec)
    manifest['files'][spec_name] = digest(root / spec_name)
    manifest['evidence']['portfolio_current_study'] = spec_name
    manifest['evidence']['sizing_followup_parent_manifest_hash'] = expected
    write_json(output / 'inputs/manifest.json', {**manifest, 'manifest_hash': signature(manifest)})
    write_json(output / 'study.json', {**parent, **spec, 'code': str(root.resolve()),
        'manifest': str((output / 'inputs/manifest.json').resolve()), 'spec_path': spec_name})
    jobs = read(parent_path.parent / 'registration.json')['jobs']
    for name in spec['candidates']:
        if name in parent['candidates']:
            continue
        for scenario in ('baseline', 'combined'):
            jobs.append({'name': name + '_' + scenario, 'recipe': name, 'scenario': scenario,
                         'standalone': 'KALCB' if name.startswith('kalcb_size_') else None})
    observed = {p.parent.name: digest(p) for p in (parent_path.parent / 'cases').glob('*/completion.json')}
    write_json(output / 'registration.json', {'parent_study': str(parent_path.resolve()),
        'parent_study_sha256': digest(parent_path), 'parent_manifest_hash': expected,
        'source_generation_hash': parent['source_generation_hash'], 'jobs': jobs,
        'selection': spec['allocation_selection'], 'observed_parent_completions_at_registration': observed,
        'new_cases': 20, 'reusable_parent_cases': 30, 'disclosure': spec['sizing_followup'],
        'selection_implementation_sha256': digest(REPO / 'scripts/portfolio_synergy_allocation_study.py')})
    print(json.dumps({'prepared': str(output), 'total_cases': len(jobs), 'new_cases': 20}), flush=True)


def reuse(study_path):
    study = read(study_path)
    registration = read(study_path.parent / 'registration.json')
    parent_path = Path(registration['parent_study'])
    if digest(parent_path) != registration['parent_study_sha256']:
        raise ValueError('Parent study changed')
    parent = read(parent_path)
    if (study['source_generation_hash'] != parent['source_generation_hash']
            or study['initial_equity'] != parent['initial_equity']
            or study['configuration'] != parent['configuration']):
        raise ValueError('Parent replays are not economically interchangeable')
    for job in read(parent_path.parent / 'registration.json')['jobs']:
        source = parent_path.parent / 'cases' / job['name']
        verify_case(source, parent)
        request = read(source / 'request.json')
        if (request['study_hash'] != digest(parent_path)
                or request['recipe'] != study['candidates'][job['recipe']]
                or request['arguments']['scenario'] != job['scenario']
                or request['arguments']['standalone'] != job['standalone']):
            raise ValueError('Changed replay settings prevent reuse')
        target = study_path.parent / 'cases' / job['name']
        if target.exists():
            verify_case(target, study)
            continue
        target.mkdir(parents=True)
        for path in source.iterdir():
            if not path.is_file():
                raise ValueError('Unexpected nested replay output')
            os.link(path, target / path.name)
        write_json(target / 'reuse_receipt.json', {'parent_output': str(source),
            'parent_study': str(parent_path), 'parent_study_sha256': digest(parent_path),
            'parent_completion_sha256': digest(source / 'completion.json'),
            'note': 'Unchanged completed artifacts reused; original request and source evidence retained.'})
    print('Verified and reused 30 identical parent cases.', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('prepare', 'run'))
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--output', type=Path)
    p.add_argument('--workers', type=int, default=2)
    args = p.parse_args()
    if args.command == 'prepare':
        prepare(args.study, args.output)
    else:
        reuse(args.study)
        run(args.study, args.workers)
