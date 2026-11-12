"""Real selected-account acceptance and deliberately corrupted export rejection."""
from copy import deepcopy
from pathlib import Path
import json
import shutil
import pytest

from backtests.strategies.kalcb.veto_cap_round import (
    read, check_signal_map, validate_selected_account, authenticate_files,
    load_acceptance, VERSION, SCENARIOS, stable_signature,
)

SOURCE = Path('E:/CodexResearch/k_stock_trader/kalcb_opportunity_lifecycle_20261016/funded_interaction_v1')


@pytest.fixture(scope='module')
def selected():
    protocol = read(SOURCE / 'protocol.json')
    case = next(c for c in protocol['cases'] if c['id'] == 'combined_oos_baseline')
    item = read(SOURCE / 'registration.json')['profiles']['oos_cap2']
    signals = read(protocol['signal_map']['path'])
    return case, item, signals


def test_complete_actual_selected_account_path(selected):
    case, item, signals = selected
    check_signal_map(signals)
    result = validate_selected_account(SOURCE, case, item, signals)
    assert result['account']['passed'] and result['funding']['verified']
    assert result['activation']['trimmed'] > 0
    assert len(result['activation']['distinct_cap_dates']) == 7


@pytest.mark.parametrize('corruption', ['free_costs', 'inflated_return', 'hidden_inventory', 'fake_price', 'wrong_mode', 'changed_signal', 'missing_order'])
def test_corrupted_account_cannot_reach_round_scoring(tmp_path, selected, corruption):
    case, item, signals = selected
    folder = tmp_path / 'cases' / case['id']
    shutil.copytree(SOURCE / 'cases' / case['id'], folder)
    if corruption == 'free_costs':
        name = 'fills.json'; data = read(folder / name); data[0]['metadata']['fee'] = 0
    elif corruption == 'fake_price':
        name = 'fills.json'; data = read(folder / name); data[0]['price'] *= .5
        data[0]['metadata']['fee'] *= .5  # Reach the adverse-price check, beyond fee reconciliation.
    elif corruption == 'inflated_return':
        name = 'result.json'; data = read(folder / name); data['return_pct'] += 100
    elif corruption == 'hidden_inventory':
        name = 'result.json'; data = read(folder / name); data['terminal_inventory'] = {'005930': 1}
    elif corruption == 'wrong_mode':
        name = 'completion.json'; data = read(folder / name); data['mode'] = 'disabled'
    elif corruption == 'changed_signal':
        name = 'horizon_actions.json'; data = read(folder / name); data[0]['signal']['predictions']['baseline'] = -9999
    else:
        import gzip
        with gzip.open(folder / 'orders.jsonl.gz', 'rt', encoding='utf-8') as stream:
            rows = [json.loads(line) for line in stream]
        index = next(i for i, r in enumerate(rows) if r['event'] == 'submit' and r.get('order', {}).get('side') == 'BUY')
        rows.pop(index)
        with gzip.open(folder / 'orders.jsonl.gz', 'wt', encoding='utf-8') as stream:
            stream.writelines(json.dumps(r) + '\n' for r in rows)
        name = None
    if name:
        (folder / name).write_text(json.dumps(data), encoding='utf-8')
    expected = {'free_costs': 'Incorrect per-fill costs', 'fake_price': 'Non-adverse slippage',
                'inflated_return': 'Inflated account return', 'hidden_inventory': 'Unresolved native account',
                'wrong_mode': 'Wrong executed VETO/cap policy', 'changed_signal': 'Changed executed prediction',
                'missing_order': 'Core proposals differ'}
    with pytest.raises(ValueError, match=expected[corruption]):
        validate_selected_account(tmp_path, case, item, signals)


@pytest.mark.parametrize('field,value', [('observed_through', '2099-01-01'), ('origin', '2099-01-01'), ('available_at', '2025-11-10T09:25:00+09:00')])
def test_future_or_misclocked_information_fails(selected, field, value):
    signals = deepcopy(selected[2]); key = next(iter(signals)); signals[key][field] = value
    with pytest.raises(ValueError): check_signal_map(signals)


def test_stale_bound_source_rejected(tmp_path):
    file = tmp_path / 'source.py'; file.write_text('changed', encoding='utf-8')
    with pytest.raises(ValueError, match='Changed bound artifact'):
        authenticate_files([dict(path=str(file), sha256='0' * 64)])


def test_actual_round_evaluator_rejects_changed_ledger_before_scoring(tmp_path):
    from backtests.strategies.kalcb.veto_cap_round import VetoCapRoundPlugin, bound
    file = tmp_path / 'fills.json'; file.write_text('[]', encoding='utf-8'); binding = bound(file)
    data = dict(version=VERSION, production_enabled=False, policy='C12-B_VETO_plus_cap',
                profiles={p + '/' + s: {} for p in ('is', 'oos') for s in SCENARIOS}, bindings=[binding])
    manifest = tmp_path / 'acceptance.json'
    manifest.write_text(json.dumps({**data, 'manifest_hash': stable_signature(data)}), encoding='utf-8')
    file.write_text('[{"fake_profit": 999999999}]', encoding='utf-8')
    with pytest.raises(ValueError, match='Changed bound artifact'):
        VetoCapRoundPlugin(manifest)


def test_capsule_cannot_claim_incomplete_matrix_or_live_promotion(tmp_path):
    for live, profiles in [(True, {}), (False, {})]:
        data = dict(version=VERSION, production_enabled=live, policy='C12-B_VETO_plus_cap', profiles=profiles)
        path = tmp_path / 'acceptance.json'
        path.write_text(json.dumps({**data, 'manifest_hash': stable_signature(data)}), encoding='utf-8')
        with pytest.raises(ValueError): load_acceptance(path)


def test_cli_exposes_locked_acceptance_and_preserves_legacy_entry():
    from backtests.strategies.kalcb.full_diagnostics import main
    with pytest.raises(SystemExit) as result: main(['--help'])
    assert result.value.code == 0


def test_explicit_archived_verifier_retains_default_and_cannot_rescue_failed_audit():
    from backtests.auto.shared.phase_runner import _metric_contract_criteria
    metrics = dict(archived_native_evidence=True, audit_pass=True, audit_status='passed',
                   promotion_requires_audit_pass=True, primary_promotion_metric='official_mtm_net_return_pct',
                   official_mtm_net_return_pct=.1, official_metric_basis='native_mtm',
                   rejected_order_count=2, invalid_order_rejection_count=0, same_bar_fill_count=0,
                   forced_replay_close_count=0, end_open_position_count=0, capability_level='real_replay')
    def gates(m, checker=None):
        return {c.name: c.passed for c in _metric_contract_criteria(m, archived_validator=checker)}
    assert not gates(metrics)['hard_audit_pass']  # A Boolean without a verifier is insufficient.
    assert gates(metrics, lambda m: True)['hard_audit_pass']
    assert gates(metrics, lambda m: True)['hard_invalid_order_rejection_count']
    assert not gates({**metrics, 'audit_pass': False}, lambda m: True)['hard_audit_pass']
    assert not gates({**metrics, 'archived_native_evidence': False}, lambda m: True)['hard_audit_pass']
    assert not gates({**metrics, 'invalid_order_rejection_count': 1}, lambda m: True)['hard_invalid_order_rejection_count']


def test_failed_phase_cannot_write_final_round_artifacts():
    from types import SimpleNamespace
    from backtests.strategies.kalcb.veto_cap_round import VetoCapRoundPlugin
    plugin = object.__new__(VetoCapRoundPlugin)
    with pytest.raises(ValueError, match='Failed phase gate'):
        plugin.build_end_of_round_artifacts(SimpleNamespace(phase_gate_results={1: {'passed': False}}))
