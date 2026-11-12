"""The continuation never turns residual or unexpected defects into account passes."""
from pathlib import Path
import pytest
from scripts import kalcb_capital_sensitivity_v2 as runner

ROOT = runner.PREVIOUS
PROFILES = runner.read(ROOT/'registration.json')['profiles']


@pytest.mark.parametrize('name', runner.ATTEMPTED[:4])
def test_real_accepted_predecessors_still_pass(name):
    assert runner.classify_case(ROOT/'cases'/name, PROFILES[name])['qualified'] is True


def test_real_residual_stays_unqualified_without_pass_sidecar():
    name = 'compound_participation'; folder = ROOT/'cases'/name
    before = {p.name: p.stat().st_mtime_ns for p in folder.iterdir()}
    verdict = runner.classify_case(folder, PROFILES[name], save=True)
    assert verdict['qualified'] is False
    assert verdict['flags']['residual_sessions'] == {'2026-06-04': {'KALCB:240810': 36}}
    assert verdict['remaining_independent_checks_not_reached'] is True
    assert not (folder/'independent_account_verification.json').exists()
    assert before == {p.name: p.stat().st_mtime_ns for p in folder.iterdir()}


@pytest.mark.parametrize('message', ['Changed frozen manifest', 'A failed or unverified source replay cannot be promoted',
                                  'cash reconciliation', 'Unresolved native account extra'])
def test_unexpected_validation_errors_propagate(monkeypatch, message):
    def fail(*args, **kwargs): raise ValueError(message)
    monkeypatch.setattr(runner.diag, 'validate_capital_native', fail)
    name = 'compound_participation'
    with pytest.raises(ValueError, match=message):
        runner.classify_case(ROOT/'cases'/name, PROFILES[name], save=True)


def test_residual_error_without_raw_evidence_stops(monkeypatch):
    def fail(*args, **kwargs): raise ValueError('Unresolved native account')
    monkeypatch.setattr(runner.diag, 'validate_capital_native', fail)
    name = 'fixed_baseline'
    with pytest.raises(ValueError, match='lacks explicit account evidence'):
        runner.classify_case(ROOT/'cases'/name, PROFILES[name])


def test_residual_with_pass_sidecar_stops(monkeypatch, tmp_path):
    def fail(*args, **kwargs): raise ValueError('Unresolved native account')
    monkeypatch.setattr(runner.diag, 'validate_capital_native', fail)
    runner.write(tmp_path/'result.json', {'residual_sessions': {'date': {'symbol': 1}},
        'terminal_inventory': {}, 'terminal_pending': 0})
    runner.write(tmp_path/'independent_account_verification.json', {'passed': True})
    with pytest.raises(ValueError, match='has an independent pass sidecar'):
        runner.classify_case(tmp_path, {}, save=True)
