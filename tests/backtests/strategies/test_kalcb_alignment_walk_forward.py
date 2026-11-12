"""Causality and failure-population gates for target-alignment research."""
from copy import deepcopy
from datetime import date, timedelta

import pytest

from scripts.kalcb_alignment_walk_forward import calibrate, paired_risk, split_dates, verify_bindings
from scripts.kalcb_full_oos_recovery import bound


def test_reporting_split_preserves_cross_boundary_fold():
    sessions = ['2026-03-10', '2026-03-31', '2026-04-01', '2026-04-06', '2026-04-07']
    result = split_dates(sessions, '2026-03-10', '2026-04-06')
    assert result == {'is': sessions[:2], 'oos': sessions[2:4]}
    assert split_dates(sessions, '2026-04-07', '2026-04-07') == {'is': [], 'oos': sessions[4:]}


def test_first_fold_has_no_in_sample_calibration_residuals():
    sessions = [str(date(2025, 1, 1)+timedelta(days=i)) for i in range(85)]
    labels = {'baseline': {}, 'combined': {}}
    result = calibrate([], labels, sessions, sessions[61], sessions[80])
    assert all(c['latest_label_available_at'] is None for c in result.values())
    assert all(v['offset_bps'] == 0 for c in result.values() for v in c['cohorts'].values())
    assert all(c['valid_from'] == sessions[61] and c['valid_through'] == sessions[80] for c in result.values())


def test_current_fold_cannot_enter_calibration_even_if_its_label_looks_mature():
    sessions = [str(date(2025, 1, 1)+timedelta(days=i)) for i in range(85)]
    with pytest.raises(ValueError, match='current/future fold'):
        calibrate([{'trade_date': sessions[61]}], {}, sessions, sessions[61], sessions[80])


def test_failed_cash_episode_is_retained_as_paired_physical_evidence():
    row = {'status': 'filled', 'risk_available_at': '2026-03-31T15:35:00+09:00',
           'physical_fill_failures': [], 'deadline_violation': False, 'unresolved': False,
           'label_valid': True, 'net_value_equity_bps': 1.}
    failed = {**row, 'status': 'execution_invalid', 'physical_fill_failures': ['shared_bar_volume_exceeded'],
              'label_valid': False, 'net_value_equity_bps': None, 'risk_available_at': '2026-04-02T15:35:00+09:00'}
    result = paired_risk({'baseline': {'x': row}, 'combined': {'x': failed}})
    assert result == {'x': {'unsafe': True, 'risk_available_at': failed['risk_available_at']}}
    incomplete = deepcopy(failed); incomplete['status'] = 'source_incomplete'
    assert paired_risk({'baseline': {'x': row}, 'combined': {'x': incomplete}}) == {}
    with pytest.raises(ValueError, match='Unpaired'):
        paired_risk({'baseline': {'x': row}, 'combined': {}})


def test_registered_input_mutation_fails_closed(tmp_path):
    p = tmp_path/'source.py'; p.write_text('original')
    bindings = [bound(p)]; verify_bindings(bindings)
    p.write_text('changed')
    with pytest.raises(ValueError, match='Changed registered input'):
        verify_bindings(bindings)
