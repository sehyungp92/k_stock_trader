"""Boundary outcome completion must preserve prior records and maturity."""
from copy import deepcopy
import pytest
from scripts import kalcb_alignment_walk_forward_v3 as wf
from scripts.kalcb_full_oos_recovery import write, bound


def fixture(tmp_path, monkeypatch, premature=False):
    monkeypatch.setattr(wf, 'BOUNDARY', tmp_path)
    data = {'timed': {'combined': {}}}; risk = {'timed': {}}
    overlays = []
    for family in ('original', 'selected'):
        for i in range(4):
            key = f'2026-03-31/{i:06d}/q1'
            before = {'opportunity_id': key, 'trade_date': '2026-03-31', 'unresolved': True,
                      'label_valid': False, 'label_available_at': None, 'net_value_equity_bps': None}
            after = {**before, 'unresolved': False, 'label_valid': True,
                     'label_available_at': '2026-03-31T15:35:00+09:00' if premature else '2026-04-01T15:35:00+09:00',
                     'net_value_equity_bps': -.01, 'net_value_krw': -100., 'net_cash_flow_krw': -100.,
                     'risk_available_at': '2026-03-31T15:35:00+09:00', 'deadline_violation': True,
                     'deadline_inventory': {'2026-03-31': 1, '2026-04-01': 0}, 'physical_fill_failures': [],
                     'terminal_inventory': 0, 'terminal_pending': 0, 'buy_qty': 10, 'sell_qty': 10, 'status': 'filled'}
            overlays.append({'family': family, 'kind': 'timed', 'scenario': 'combined', 'prior_label': before, 'label': after})
            data['timed']['combined'][key] = {k: before[k] for k in ('label_valid', 'label_available_at', 'net_value_equity_bps')}
            risk['timed'][key] = {'unsafe': True, 'risk_available_at': after['risk_available_at']}
    write(tmp_path/'protocol.json', {'bound_inputs': []})
    write(tmp_path/'prefix_bridge.json', {'passed': True})
    write(tmp_path/'overlays.json', overlays)
    write(tmp_path/'completion.json', {'complete': True, **{k: bound(tmp_path/f'{k}.json') for k in ('protocol', 'overlays', 'prefix_bridge')}})
    return deepcopy(data), deepcopy(risk), deepcopy(data), deepcopy(risk), overlays


def test_completion_preserves_risk_failure_and_later_cash_availability(tmp_path, monkeypatch):
    original, risk, selected, selected_risk, evidence = fixture(tmp_path, monkeypatch)
    wf.apply_boundary_completion(original, risk, selected, selected_risk)
    for target in (original, selected):
        assert all(r['label_available_at'] == '2026-04-01T15:35:00+09:00' for r in target['timed']['combined'].values())
        assert all(r['net_value_equity_bps'] == -.01 for r in target['timed']['combined'].values())
    assert all(r['unsafe'] for r in risk['timed'].values())
    assert all(not r['prior_label']['label_valid'] for r in evidence)


def test_boundary_completion_cannot_become_known_on_march31(tmp_path, monkeypatch):
    *args, _ = fixture(tmp_path, monkeypatch, premature=True)
    with pytest.raises(ValueError, match='prematurely'): wf.apply_boundary_completion(*args)
