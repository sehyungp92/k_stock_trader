import json

from scripts.portfolio_synergy_allocation_study import select
from scripts.replay_selected_portfolio import digest


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding='utf-8')


def case(root, name, ret, dd, trades):
    path = root / 'cases' / name
    write(path / 'result.json', {'return_pct': ret, 'drawdown_pct': dd, 'trades': trades,
          'active_days': 120, 'execution_validity_pass': True, 'terminal_inventory': {},
          'terminal_pending': 0, 'intraday_residual_sessions': 0})
    write(path / 'execution_audit.json', {'execution_validity_pass': True})
    write(path / 'completion.json', {'source_generation_hash': 'source',
          'result_sha256': digest(path / 'result.json'), 'audit_sha256': digest(path / 'execution_audit.json')})


def test_stronger_feasible_standalone_prevents_false_portfolio_promotion(tmp_path):
    primary = tmp_path / 'primary'
    extra = tmp_path / 'challenge'
    study = {'source_generation_hash': 'source', 'candidates': {'portfolio': {}, 'kalcb_size_125': {}, 'kalcb_size_150': {}},
             'initial_equity': 100000000, 'configuration': {'window': 'same'}}
    write(primary / 'study.json', study)
    selection = {'baseline_max_dd_pct': 11., 'combined_max_dd_pct': 12.5, 'near_return_band_pct_points': 1.}
    jobs = []
    for name, values in {'kalcb_native': (80, 8, 370), 'olr_native': (45, 10, 180),
                         'kalcb_size_125': (90, 9, 365), 'kalcb_size_150': (100, 10, 360),
                         'portfolio': (105, 10.5, 540)}.items():
        for scenario in ('baseline', 'combined'):
            case(primary, name + '_' + scenario, values[0], values[1], values[2])
            jobs.append({'name': name + '_' + scenario})
    write(primary / 'registration.json', {'selection': selection, 'jobs': jobs})
    write(extra / 'study.json', study)
    jobs = []
    for name, ret, dd in [('kalcb_size_175', 110, 10.8), ('kalcb_size_200', 120, 13)]:
        for scenario in ('baseline', 'combined'):
            case(extra, name + '_' + scenario, ret, dd, 355)
            jobs.append({'name': name + '_' + scenario, 'recipe': name,
                         'standalone': 'KALCB', 'scenario': scenario})
    write(extra / 'registration.json', {'parent_study_sha256': digest(primary / 'study.json'), 'jobs': jobs})
    select(primary / 'study.json', extra / 'study.json')
    result = json.loads((primary / 'finalist.json').read_text())
    assert result['winner'] is None
    assert result['return_floors'] == [110, 110]
    assert 'kalcb_size_175' in result['reference_frontier']
    assert 'kalcb_size_200' not in result['reference_frontier']
    assert result['rejected']['portfolio'] == ['standalone_return_frontier']
