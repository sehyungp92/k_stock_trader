"""Verify or replay a qualified native portfolio research profile.

This entry point uses the profile's frozen source and market generation. It has
no live-order path. The legacy synthetic portfolio fixture is not used here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
DEFAULT_PROFILE = REPO / 'config/research/portfolio_synergy_selected.json'
ECONOMICS = ('return_pct', 'drawdown_pct', 'trades', 'active_days', 'strategy_counts')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def binding(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': digest(path)}


def bound(item):
    if digest(item['path']) != item['sha256']:
        raise ValueError('Changed research evidence: ' + item['path'])
    return read(item['path'])


def checked_case(path, source, recipe, scenario, *, full):
    completion = read(path / 'completion.json')
    result = read(path / 'result.json')
    audit = read(path / 'execution_audit.json')
    request = read(path / 'request.json')
    if (not completion.get('complete') or completion['source_generation_hash'] != source
            or completion['result_sha256'] != digest(path / 'result.json')
            or completion['audit_sha256'] != digest(path / 'execution_audit.json')):
        raise ValueError('Changed or incomplete replay: ' + str(path))
    for name, expected in completion.get('artifacts', {}).items():
        if digest(path / name) != expected:
            raise ValueError('Changed replay artifact: ' + str(path / name))
    if (result['source_generation_hash'] != source or result['recipe'] != recipe
            or result['scenario'] != scenario or result.get('standalone') is not None
            or not result['execution_validity_pass'] or not audit['execution_validity_pass']
            or result['terminal_inventory'] or result['terminal_pending']
            or result['intraday_residual_sessions']):
        raise ValueError('Replay execution gate failed: ' + str(path))
    allocation = result.get('allocation_audit')
    if not allocation or not allocation['pass'] or allocation['terminal_deferred_intents']:
        raise ValueError('Allocation evidence missing or failed')
    if read(path / 'allocation_audit.json') != allocation:
        raise ValueError('Allocation audit artifact does not match result')
    if request['recipe'].get('native_budget'):
        if (allocation.get('version') != 'native-budget-audit-v1'
                or audit.get('native_budget_audit') != allocation
                or not allocation.get('checked_admissions')):
            raise ValueError('Independent native funding reconstruction required')
    if any(request['arguments'].get(flag) for flag in
           ('absolute_caps', 'no_upward_compounding', 'canonical_costs', 'standalone')):
        raise ValueError('Qualification contains undeclared replay overrides')
    if full and (result['audit_status'] != 'passed' or audit['audit_status'] != 'passed'
                 or not request['arguments']['full_audit']):
        raise ValueError('Full native source qualification is required')
    return result, request


def native_comparison_evidence(root, study, finalist):
    """Bind actual matched controls and source bridges behind the selection gates."""
    from scripts.portfolio_native_budget_study import checked, verified_spec
    verified_spec(study)
    registration=read(root/'registration.json'); controls=root; lineage={}
    if registration.get('priority_protocol'):
        declaration=bound(registration['priority_protocol'])
        if declaration['source_generation_hash'] != study['source_generation_hash']:
            raise ValueError('Priority protocol source mismatch')
        lineage['priority_protocol']=registration['priority_protocol']
        support=finalist.get('priority_support') or {}
        if (support.get('minimum')!=declaration['minimum_distinct_conflict_sessions']
                or (finalist['winner']!=registration['retain_parent_on_exact_tie']
                    and support.get('distinct_potential_conflict_sessions',0)<support['minimum'])):
            raise ValueError('Unsupported priority change')
    if registration.get('standalone_reference_study'):
        parent=registration['standalone_reference_study']; parent_study=bound(parent)
        if (parent_study['source_generation_hash']!=study['source_generation_hash']
                or parent_study['initial_equity']!=study['initial_equity']):
            raise ValueError('Native reference account or source changed')
        controls=Path(parent['path']).parent
        lineage['reference_study']=parent
        validation_root=Path(registration['parent_study']).parent
        lineage['registration']=binding(validation_root/'registration.json')
        lineage['study']={'path':registration['parent_study'],'sha256':registration['parent_study_sha256']}
        validation_study=bound(lineage['study'])
        if validation_study['source_generation_hash']!=study['source_generation_hash']:
            raise ValueError('Parent behavior used a different source')
        lineage['selection']={'path':str(validation_root/'finalist.json'),'sha256':registration['parent_finalist_sha256']}
        parent_finalist=bound(lineage['selection'])
        # An unmodified parent winner must reproduce in every derived challenge.
        for scenario in ('baseline','combined'):
            old=validation_root/'cases'/(parent_finalist['winner']+'_'+scenario)
            new=root/'cases'/(parent_finalist['winner']+'_'+scenario)
            checked(old);checked(new)
            for name in ('trades.json','daily.json','equity.json'):
                if digest(old/name)!=digest(new/name):raise ValueError('Risk challenge changed parent behavior')
        # A priority study can inherit a risk study which itself inherits the
        # primary grid. Verify every earlier transition, not just the last one.
        cursor = validation_root
        seen = {root}
        lineage['earlier_ancestors'] = []
        while cursor != controls:
            if cursor in seen:
                raise ValueError('Cyclic native study lineage')
            seen.add(cursor)
            ancestor_registration = read(cursor / 'registration.json')
            parent_root = Path(ancestor_registration['parent_study']).parent
            ancestor = {'registration': binding(cursor / 'registration.json'),
                'study': {'path': ancestor_registration['parent_study'],
                          'sha256': ancestor_registration['parent_study_sha256']},
                'selection': {'path': str(parent_root / 'finalist.json'),
                              'sha256': ancestor_registration['parent_finalist_sha256']},
                'path_hashes': {}}
            parent_study = bound(ancestor['study'])
            parent_selection = bound(ancestor['selection'])
            if (parent_study['source_generation_hash'] != study['source_generation_hash']
                    or parent_study['initial_equity'] != study['initial_equity']):
                raise ValueError('Ancestor source or account changed')
            for scenario in ('baseline', 'combined'):
                name = parent_selection['winner'] + '_' + scenario
                old, new = parent_root / 'cases' / name, cursor / 'cases' / name
                checked(old); checked(new)
                hashes = {filename: digest(old / filename) for filename in ('trades.json', 'daily.json', 'equity.json')}
                if any(digest(new / filename) != sha for filename, sha in hashes.items()):
                    raise ValueError('Ancestor challenge changed parent behavior')
                ancestor['path_hashes'][scenario] = hashes
            lineage['earlier_ancestors'].append(ancestor)
            cursor = parent_root
    bridge_path=controls/'bridge.json'; bridges=read(bridge_path)
    for pair in bridges.values():
        for name,expected in pair['path_hashes'].items():
            for generation in ('current','parent'):
                if digest(Path(pair[generation])/name)!=expected:raise ValueError('Changed native source bridge')
    frequencies=[checked(Path(bridges['kalcb_native_'+s]['current']))['trades'] for s in ('baseline','combined')]
    if frequencies!=finalist['frequency_floors']:raise ValueError('Native frequency reference mismatch')
    control_registration=read(controls/'registration.json'); grouped={}; evidence={}
    for job in control_registration['jobs']:
        if job['group']!='standalone':continue
        path=controls/'cases'/(job['recipe']+'_'+job['scenario']);r=checked(path)
        request=read(path/'request.json')
        if (r['source_generation_hash']!=study['source_generation_hash'] or r['standalone']!=job['standalone']
                or request['study_hash']!=digest(controls/'study.json')):
            raise ValueError('Unmatched native standalone control')
        grouped.setdefault(job['recipe'],{})[job['scenario']]=r
        evidence[path.name]={name:binding(path/name) for name in ('result.json','request.json','execution_audit.json','completion.json')}
    feasible=[r for r in grouped.values() if r['baseline']['drawdown_pct']<=finalist['selection']['baseline_max_dd_pct']
              and r['combined']['drawdown_pct']<=finalist['selection']['combined_max_dd_pct']]
    floors=[max(r[s]['return_pct'] for r in feasible) for s in ('baseline','combined')]
    if floors!=finalist['return_floors']:raise ValueError('Native standalone return reference mismatch')
    return {'lineage':lineage,'bridge':binding(bridge_path),'standalone_controls':evidence,
            'reference_study':binding(controls/'study.json'),'reference_registration':binding(controls/'registration.json')}


def make_profile(study_path):
    study_path = Path(study_path).resolve()
    root = study_path.parent
    study = read(study_path)
    finalist = read(root / 'finalist.json')
    winner = finalist['winner']
    if not winner or finalist['source_generation_hash'] != study['source_generation_hash']:
        raise ValueError('No valid selected finalist; do not promote')
    if finalist['recipe'] != study['candidates'][winner]:
        raise ValueError('Finalist settings do not match frozen study')
    if finalist['recipe'].get('native_budget'):
        for scenario in ('baseline', 'combined', 'cost', 'participation'):
            if not (root / 'qualified' / (winner + '_' + scenario) / 'completion.json').exists():
                raise ValueError('Native profile requires all four full qualification scenarios')
    frequency_floors=finalist.get('frequency_floors')
    if frequency_floors is None and finalist.get('results',{}).get('kalcb_native'):
        frequency_floors=[r['trades'] for r in finalist['results']['kalcb_native']]
    if not frequency_floors or len(frequency_floors)!=2:
        raise ValueError('Missing native frequency benchmark')
    prior = finalist.get('prior_study')
    if prior:
        parent = Path(prior['path'])
        bound({'path':str(parent),'sha256':prior['sha256']})
        bound({'path':str(parent.parent/'finalist.json'),'sha256':prior['finalist_sha256']})
        registration = read(root/'registration.json')
        if (registration.get('parent_study_sha256')!=prior['sha256']
                or registration.get('parent_finalist_sha256')!=prior['finalist_sha256']
                or not finalist.get('source_bridge')):
            raise ValueError('Prior generation comparison lacks a registered source bridge')
        bridge=finalist.get('source_bridge_evidence')
        if not bridge:raise ValueError('Missing source bridge case evidence')
        for pair in bridge.values():
            summaries={}
            for generation,case in pair.items():
                for filename,sha in case['files'].items():
                    bound({'path':str(Path(case['path'])/filename),'sha256':sha})
                summaries[generation]=read(Path(case['path'])/'result.json')
            if any(summaries['current'][key]!=summaries['prior'][key] for key in ECONOMICS):
                raise ValueError('Source bridge no longer reproduces prior economics')
    challenge = finalist.get('reference_challenge')
    if challenge:
        extra_path = Path(challenge['study'])
        extra = bound({'path': str(extra_path), 'sha256': challenge['study_sha256']})
        registered = bound({'path': str(extra_path.parent / 'registration.json'),
                            'sha256': challenge['registration_sha256']})
        if extra['source_generation_hash'] != study['source_generation_hash']:
            raise ValueError('Standalone challenge used a different replay generation')
        if registered['parent_study_sha256'] != digest(study_path):
            raise ValueError('Standalone challenge used a different primary study')
        for name, files in challenge['case_evidence'].items():
            for filename, expected_hash in files.items():
                bound({'path': str(extra_path.parent / 'cases' / name / filename), 'sha256': expected_hash})
    cases = {}
    scenarios = ['baseline', 'combined']
    scenarios += [name for name in ('cost', 'participation')
                  if (root / 'qualified' / (winner + '_' + name) / 'completion.json').exists()]
    for scenario in scenarios:
        name = winner + '_' + scenario
        selected = None
        if scenario in ('baseline', 'combined'):
            selected, _ = checked_case(root / 'cases' / name, study['source_generation_hash'],
                                       winner, scenario, full=False)
        path = root / 'qualified' / name
        result, request = checked_case(path, study['source_generation_hash'], winner, scenario, full=True)
        if selected and any(result[key] != selected[key] for key in ECONOMICS):
            raise ValueError('Qualification changed finalist economics')
        if selected and finalist['recipe'].get('native_budget'):
            for filename in ('trades.json','daily.json','equity.json'):
                if digest(path/filename)!=digest(root/'cases'/name/filename):
                    raise ValueError('Native qualification changed the selected economic path')
        if request['recipe'] != finalist['recipe'] or request['study_hash'] != digest(study_path):
            raise ValueError('Qualification used different study or settings')
        if scenario in ('baseline', 'combined'):
            index = 0 if scenario == 'baseline' else 1
            dd_key = 'baseline_max_dd_pct' if scenario == 'baseline' else 'combined_max_dd_pct'
            if (result['drawdown_pct'] > finalist['selection'][dd_key]
                    or result['return_pct'] <= finalist['return_floors'][index]
                    or result['trades'] < frequency_floors[index]):
                raise ValueError('Qualified portfolio fails the selected risk or reference return floor')
        cases[scenario] = {'output': str(path),
            'evidence': {name: binding(path / name) for name in
                ('result.json', 'execution_audit.json', 'completion.json', 'request.json', 'allocation_audit.json')},
            'metrics': {key: result[key] for key in (*ECONOMICS, 'trades_per_21_sessions')}}
    native = 'native_budget' in finalist['recipe']
    profile = {'version': 'portfolio-native-budget-research-profile-v1' if native else 'portfolio-intent-research-profile-v1',
        'status': 'research_selected_full_native_audits_passed',
        'scope': 'reused_development_only', 'live_deployment_enabled': False,
        'independent_performance_established': False, 'completion_requires_future_sessions': False,
        'recipe': winner, 'settings': finalist['recipe'],
        'initial_equity_krw': study['initial_equity'],
        'evaluation_window': study['configuration']['date_range'],
        'source_generation_hash': study['source_generation_hash'],
        'study': binding(study_path), 'manifest': binding(study['manifest']),
        'registration': binding(root / 'registration.json'), 'selection': binding(root / 'finalist.json'),
        'reference_challenge': challenge,
        'prior_study': prior,
        'qualified_cases': cases,
        'selection_rule': finalist['selection'],
        'risk_interpretation': 'Entry exposure budgets and empirical development drawdown gates; not guaranteed loss limits.',
        'operational_scope': 'Native research replay. Broker/live parity has not been established for this allocator.'}
    if native:
        profile['native_comparison_evidence']=native_comparison_evidence(root,study,finalist)
    return profile


def verify_profile(path):
    path = Path(path).resolve()
    profile = read(path)
    if profile.get('version') == 'portfolio-research-pointer-v1':
        bound(profile['profile'])
        path = Path(profile['profile']['path'])
        profile = read(path)
    if profile.get('version') == 'portfolio-allocation-challenge-profile-v1':
        from scripts.portfolio_allocation_profile import verify_challenge_profile
        return verify_challenge_profile(profile)
    if profile.get('version') == 'portfolio-targeted-validation-profile-v1':
        from scripts.portfolio_targeted_result import verify_targeted_profile
        return verify_targeted_profile(profile)
    if profile.get('version') == 'portfolio-user-assigned-native-round-v1':
        from backtests.strategies.portfolio_synergy.native_round_diagnostics import verify_user_assigned_profile
        return verify_user_assigned_profile(profile)
    if (profile.get('version') not in {'portfolio-intent-research-profile-v1', 'portfolio-native-budget-research-profile-v1'}
            or profile.get('status') != 'research_selected_full_native_audits_passed'
            or profile.get('live_deployment_enabled') is not False):
        raise ValueError('A qualified research-only profile is required')
    for name in ('study', 'manifest', 'registration', 'selection'):
        bound(profile[name])
    for case in profile['qualified_cases'].values():
        for item in case['evidence'].values():
            bound(item)
    # Reconstruct rather than trusting mutable summary fields in the profile.
    expected = make_profile(profile['study']['path'])
    if profile != expected:
        raise ValueError('Profile summary does not match qualified frozen evidence')
    return profile


def save_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('publish', 'verify', 'replay'))
    parser.add_argument('--profile', type=Path, default=DEFAULT_PROFILE)
    parser.add_argument('--study', type=Path)
    parser.add_argument('--challenge-root', type=Path)
    parser.add_argument('--pointer', type=Path)
    parser.add_argument('--scenario', choices=('baseline', 'cost', 'participation', 'combined',
        'auction_price_baseline','auction_price_combined','auction_nonfill_baseline','auction_nonfill_combined'), default='baseline')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.command == 'publish':
        if bool(args.study) == bool(args.challenge_root):
            parser.error('publish requires exactly one of --study or --challenge-root and a new --profile path')
        if args.challenge_root:
            from scripts.portfolio_allocation_profile import make_challenge_profile
            profile = make_challenge_profile(args.challenge_root)
        else:
            profile = make_profile(args.study)
        save_new(args.profile, profile)
        verify_profile(args.profile)
        if args.pointer:
            save_new(args.pointer, {'version': 'portfolio-research-pointer-v1',
                                   'profile': binding(args.profile), 'live_deployment_enabled': False})
    else:
        profile = verify_profile(args.profile)
        if args.command == 'replay':
            if args.output is None or args.output.exists():
                parser.error('replay requires a new --output directory')
            if args.scenario not in profile['qualified_cases']:
                parser.error('scenario is not declared in this profile')
            case = profile['qualified_cases'][args.scenario]
            recipe, scenario = case.get('recipe',profile['recipe']),case.get('scenario',args.scenario)
            study = read(profile['study']['path'])
            command = [sys.executable, '-B', '-u', str(Path(study['code']) / 'scripts/portfolio_synergy_study.py'),
                       'replay', '--study', profile['study']['path'], '--recipe', recipe,
                       '--scenario', scenario, '--output', str(args.output.resolve()), '--full-audit']
            subprocess.run(command, cwd=study['code'], check=True)
            checked_case(args.output, profile['source_generation_hash'], recipe, scenario, full=True)
    print(json.dumps({'profile': str(args.profile.resolve()), 'recipe': profile['recipe'],
                      'initial_equity_krw': profile['initial_equity_krw'],
                      'verified': True, 'live_deployment_enabled': False,
                      'status': profile.get('status'), 'evaluation_window': profile.get('evaluation_window'),
                      'oos_acceptance_pass': profile.get('oos_acceptance_pass')}))


if __name__ == '__main__':
    main()
