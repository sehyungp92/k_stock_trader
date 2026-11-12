import copy
import json
import shutil

import pytest

from scripts.replay_selected_portfolio import (
    binding, digest, make_profile, verify_profile)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


@pytest.fixture
def qualified_study(tmp_path):
    recipe = {'allocation': {'olr_budget_fraction': .65}, 'quality': False}
    study = {'source_generation_hash': 'frozen-source', 'candidates': {'selected': recipe},
             'initial_equity': 100000000, 'configuration': {'date_range': {'start': '2025-08-07', 'end': '2026-03-31'}},
             'manifest': str(tmp_path / 'manifest.json')}
    write(tmp_path / 'study.json', study)
    write(tmp_path / 'manifest.json', {'test_manifest': True})
    write(tmp_path / 'registration.json', {'test_registration': True})
    write(tmp_path / 'finalist.json', {'winner': 'selected', 'recipe': recipe,
          'source_generation_hash': 'frozen-source', 'return_floors': [90., 90.],
          'frequency_floors': [369,371],
          'selection': {'baseline_max_dd_pct': 11., 'combined_max_dd_pct': 12.5}})
    for category in ('cases', 'qualified'):
        for scenario in ('baseline', 'combined'):
            path = tmp_path / category / ('selected_' + scenario)
            result = {'source_generation_hash': 'frozen-source', 'recipe': 'selected', 'scenario': scenario,
                'standalone': None, 'execution_validity_pass': True, 'terminal_inventory': {},
                'terminal_pending': 0, 'intraday_residual_sessions': 0, 'audit_status': 'passed',
                'allocation_audit': {'pass': True, 'terminal_deferred_intents': 0},
                'return_pct': 100., 'drawdown_pct': 9., 'trades': 540, 'active_days': 130,
                'strategy_counts': {'KALCB': 360, 'OLR': 180}, 'trades_per_21_sessions': 70.}
            write(path / 'result.json', result)
            write(path / 'allocation_audit.json', result['allocation_audit'])
            write(path / 'execution_audit.json', {'execution_validity_pass': True, 'audit_status': 'passed'})
            write(path / 'request.json', {'recipe': recipe, 'study_hash': digest(tmp_path / 'study.json'),
                  'arguments': {'full_audit': category == 'qualified'}})
            write(path / 'completion.json', {'complete': True, 'source_generation_hash': 'frozen-source',
                  'result_sha256': digest(path / 'result.json'), 'audit_sha256': digest(path / 'execution_audit.json')})
    return tmp_path / 'study.json'


def test_profile_and_pointer_bind_frozen_qualified_evidence(qualified_study):
    profile = make_profile(qualified_study)
    path = qualified_study.parent / 'profile.json'
    write(path, profile)
    pointer = path.with_name('pointer.json')
    write(pointer, {'version': 'portfolio-research-pointer-v1', 'profile': binding(path)})
    assert verify_profile(pointer) == profile
    assert profile['completion_requires_future_sessions'] is False


@pytest.mark.parametrize('change', ['settings', 'live', 'summary'])
def test_mutated_profile_cannot_override_qualified_evidence(qualified_study, change):
    profile = copy.deepcopy(make_profile(qualified_study))
    if change == 'settings':
        profile['settings']['allocation']['olr_budget_fraction'] = 1
    elif change == 'live':
        profile['live_deployment_enabled'] = True
    else:
        profile['qualified_cases']['baseline']['metrics']['return_pct'] += 1
    path = qualified_study.parent / 'profile.json'
    write(path, profile)
    with pytest.raises(ValueError):
        verify_profile(path)


def test_different_full_audit_economics_prevent_publication(qualified_study):
    path = qualified_study.parent / 'qualified/selected_baseline'
    result = json.loads((path / 'result.json').read_text())
    result['trades'] += 1
    write(path / 'result.json', result)
    completion = json.loads((path / 'completion.json').read_text())
    completion['result_sha256'] = digest(path / 'result.json')
    write(path / 'completion.json', completion)
    with pytest.raises(ValueError, match='changed finalist economics'):
        make_profile(qualified_study)


def test_no_feasible_finalist_cannot_be_published(qualified_study):
    path = qualified_study.parent / 'finalist.json'
    finalist = json.loads(path.read_text())
    finalist['winner'] = None
    write(path, finalist)
    with pytest.raises(ValueError, match='No valid selected finalist'):
        make_profile(qualified_study)


def test_failed_allocation_artifact_cannot_be_hidden_by_result_summary(qualified_study):
    path = qualified_study.parent / 'qualified/selected_baseline/allocation_audit.json'
    write(path, {'pass': False, 'terminal_deferred_intents': 0})
    with pytest.raises(ValueError, match='Allocation audit artifact'):
        make_profile(qualified_study)


def test_changed_parent_evidence_prevents_publication(qualified_study):
    root=qualified_study.parent;parent=root/'prior/study.json';write(parent,{'frozen':True})
    write(parent.parent/'finalist.json',{'winner':None})
    prior={'path':str(parent),'sha256':digest(parent),'finalist_sha256':digest(parent.parent/'finalist.json')}
    case=root/'cases/selected_baseline'
    evidence={'path':str(case),'files':{name:digest(case/name) for name in ('result.json','completion.json','execution_audit.json')}}
    path=root/'finalist.json';final=json.loads(path.read_text())
    final.update(prior_study=prior,source_bridge='exact',source_bridge_evidence={'case':{'current':evidence,'prior':evidence}})
    write(path,final)
    write(root/'registration.json',{'parent_study_sha256':prior['sha256'],'parent_finalist_sha256':prior['finalist_sha256']})
    assert make_profile(qualified_study)['prior_study']==prior
    write(parent,{'frozen':False})
    with pytest.raises(ValueError,match='Changed research evidence'):make_profile(qualified_study)


def test_frequency_gate_is_rechecked_before_publication(qualified_study):
    path=qualified_study.parent/'finalist.json';final=json.loads(path.read_text())
    final['frequency_floors']=[541,371];write(path,final)
    with pytest.raises(ValueError,match='Qualified portfolio fails'):make_profile(qualified_study)


@pytest.fixture
def native_qualified_study(qualified_study, monkeypatch):
    from scripts import replay_selected_portfolio as entrypoint
    root = qualified_study.parent
    study = json.loads(qualified_study.read_text())
    recipe = {'native_budget': {'kalcb_sizing_fraction': .5, 'olr_fraction': 1.}, 'quality': False}
    study['candidates']['selected'] = recipe
    study['initial_equity'] = 25_000_000
    write(qualified_study, study)
    finalist = json.loads((root / 'finalist.json').read_text())
    finalist['recipe'] = recipe
    write(root / 'finalist.json', finalist)
    allocation = {'version': 'native-budget-audit-v1', 'pass': True,
                  'terminal_deferred_intents': 0, 'checked_admissions': 10}
    for category in ('cases', 'qualified'):
        for scenario in ('baseline', 'combined'):
            path = root / category / ('selected_' + scenario)
            result = json.loads((path / 'result.json').read_text())
            result['allocation_audit'] = allocation
            write(path / 'result.json', result)
            write(path / 'allocation_audit.json', allocation)
            write(path / 'execution_audit.json', {'execution_validity_pass': True,
                  'audit_status': 'passed', 'native_budget_audit': allocation})
            request = json.loads((path / 'request.json').read_text())
            request.update(recipe=recipe, study_hash=digest(qualified_study))
            write(path / 'request.json', request)
            for filename in ('trades.json', 'daily.json', 'equity.json'):
                write(path / filename, [{'economic_path': 'original'}])
            completion = json.loads((path / 'completion.json').read_text())
            completion.update(result_sha256=digest(path / 'result.json'),
                              audit_sha256=digest(path / 'execution_audit.json'))
            write(path / 'completion.json', completion)
    for scenario in ('cost', 'participation'):
        path = root / 'qualified' / ('selected_' + scenario)
        shutil.copytree(root / 'qualified/selected_baseline', path)
        result = json.loads((path / 'result.json').read_text())
        result['scenario'] = scenario
        write(path / 'result.json', result)
        completion = json.loads((path / 'completion.json').read_text())
        completion['result_sha256'] = digest(path / 'result.json')
        write(path / 'completion.json', completion)
    # These tests isolate publication gates; actual frozen lineage is exercised
    # by publication/verification against the research replay artifacts.
    monkeypatch.setattr(entrypoint, 'native_comparison_evidence', lambda *args: {})
    return qualified_study


def test_native_profile_requires_all_four_full_audit_scenarios(native_qualified_study):
    assert len(make_profile(native_qualified_study)['qualified_cases']) == 4
    path = native_qualified_study.parent / 'qualified/selected_cost/completion.json'
    path.unlink()
    with pytest.raises(ValueError, match='all four full qualification'):
        make_profile(native_qualified_study)


def test_same_aggregate_metrics_cannot_hide_changed_native_trade_path(native_qualified_study):
    path = native_qualified_study.parent / 'qualified/selected_baseline/trades.json'
    write(path, [{'economic_path': 'different trades with identical summary metrics'}])
    with pytest.raises(ValueError, match='changed the selected economic path'):
        make_profile(native_qualified_study)


def test_native_profile_rejects_missing_independent_funding_audit(native_qualified_study):
    path = native_qualified_study.parent / 'qualified/selected_baseline'
    audit = json.loads((path / 'execution_audit.json').read_text())
    audit.pop('native_budget_audit')
    write(path / 'execution_audit.json', audit)
    completion = json.loads((path / 'completion.json').read_text())
    completion['audit_sha256'] = digest(path / 'execution_audit.json')
    write(path / 'completion.json', completion)
    with pytest.raises(ValueError, match='Independent native funding'):
        make_profile(native_qualified_study)
