"""Distinguish late, real settlement losses from fabricated/unfinished returns."""
from copy import deepcopy
import pytest
from scripts.kalcb_alignment_walk_forward_v2 import economic_label_contract, paired_risk


def settled():
    return {'trade_date': '2026-04-01', 'deadline_violation': True, 'deadline_inventory': {'2026-04-01': 2, '2026-04-02': 0},
            'label_valid': True, 'physical_fill_failures': [], 'unresolved': False,
            'terminal_inventory': 0, 'terminal_pending': 0, 'status': 'filled', 'buy_qty': 10, 'sell_qty': 10,
            'net_value_krw': -100., 'net_cash_flow_krw': -100., 'net_value_equity_bps': -.01,
            'label_available_at': '2026-04-02T15:35:00+09:00', 'risk_available_at': '2026-04-01T15:35:00+09:00'}


def test_late_real_loss_keeps_both_cash_target_and_unsafe_risk():
    row = settled(); original = deepcopy(row)
    assert economic_label_contract(row) == 'settled_after_deadline'
    assert row == original
    assert paired_risk({'baseline': {'x': row}, 'combined': {'x': row}})['x']['unsafe']


@pytest.mark.parametrize('changes', [
    {'physical_fill_failures': ['shared_bar_volume_exceeded']},
    {'unresolved': True}, {'terminal_inventory': 1}, {'terminal_pending': 1},
    {'sell_qty': 11}, {'net_value_krw': float('nan')}, {'net_value_krw': 100.},
    {'label_available_at': '2026-03-31T15:35:00+09:00'}, {'deadline_violation': False},
])
def test_corrupted_valid_economic_label_fails_closed(changes):
    with pytest.raises(ValueError): economic_label_contract({**settled(), **changes})


def test_unresolved_cash_stays_unavailable_but_retains_failure_evidence():
    row = {**settled(), 'status': 'unresolved', 'unresolved': True, 'terminal_inventory': 2,
           'sell_qty': 8, 'label_valid': False, 'net_value_krw': None, 'net_value_equity_bps': None, 'label_available_at': None}
    assert economic_label_contract(row) == 'invalid_economic_target'
    assert paired_risk({'baseline': {'x': row}, 'combined': {'x': row}})['x']['unsafe']
    with pytest.raises(ValueError, match='exposes a mature return'):
        economic_label_contract({**row, 'net_value_equity_bps': 0.})


def test_cash_availability_and_physical_risk_have_separate_clocks():
    row = {**settled(), 'deadline_violation': False, 'deadline_inventory': {'2026-04-01': 0},
           'label_available_at': '2026-04-01T15:20:00+09:00'}
    assert economic_label_contract(row) == 'settled_without_deadline_violation'
