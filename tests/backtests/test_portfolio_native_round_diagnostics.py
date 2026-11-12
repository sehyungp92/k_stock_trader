import json
from pathlib import Path

import pytest

from backtests.strategies.portfolio_synergy.native_round_diagnostics import (
    WINDOW, verify_window, verify_user_assigned_profile, battery_case,
)
from scripts.register_portfolio_candidate_round import archive_round
from scripts.replay_selected_portfolio import verify_profile


def window_rows():
    daily = [{'day': WINDOW['start']}, {'day': WINDOW['end']}]
    equity = [[WINDOW['start'] + ' 09:00:00+09:00', 25_000_000],
              [WINDOW['end'] + ' 15:30:00+09:00', 26_000_000]]
    trade = {'entry_decision_time': '2026-03-30T09:25:00+09:00',
             'entry_fill_time': '2026-03-30T09:30:00+09:00',
             'exit_fill_time': '2026-03-31T15:30:00+09:00'}
    return daily, equity, [trade]


def test_march_31_boundary_is_inclusive():
    verify_window(*window_rows(), WINDOW)


@pytest.mark.parametrize('label,expected', [
    ('baseline', ('k_marginal', 'baseline')),
    ('combined', ('k_marginal', 'combined')),
    ('auction_price_baseline', ('k_marginal_price', 'baseline')),
    ('auction_nonfill_combined', ('k_marginal_nonfill', 'combined')),
])
def test_auction_stresses_keep_registered_recipe_and_scenario(label, expected):
    assert battery_case(label, 'k_marginal') == expected


def test_undeclared_stress_is_rejected():
    with pytest.raises(ValueError, match='Unregistered'):
        battery_case('auction_price_cost', 'k_marginal')


@pytest.mark.parametrize('field', ['entry_decision_time', 'entry_fill_time', 'exit_fill_time'])
def test_april_trade_is_never_in_sample(field):
    daily, equity, trades = window_rows()
    trades[0][field] = '2026-04-01T09:30:00+09:00'
    with pytest.raises(ValueError, match='Trade outside IS'):
        verify_window(daily, equity, trades, WINDOW)


def test_april_mark_is_never_in_sample():
    daily, equity, trades = window_rows()
    equity.append(['2026-04-01 09:00:00+09:00', 26_000_000])
    with pytest.raises(ValueError, match='Equity observations outside IS'):
        verify_window(daily, equity, trades, WINDOW)


def test_premature_end_is_rejected():
    daily, equity, trades = window_rows()
    daily[-1]['day'] = '2026-03-30'
    with pytest.raises(ValueError, match='IS boundary mismatch'):
        verify_window(daily, equity, trades, WINDOW)


def test_duplicate_session_is_rejected():
    daily, equity, trades = window_rows()
    daily.append(daily[-1])
    with pytest.raises(ValueError, match='Duplicate'):
        verify_window(daily, equity, trades, WINDOW)


def test_archival_preserves_exact_bytes_and_manifest(tmp_path):
    strategy = tmp_path / 'portfolio_synergy'
    original = strategy / 'round_1'
    (original / 'nested').mkdir(parents=True)
    content = b'original report\r\n\x00'
    (original / 'nested/report.bin').write_bytes(content)
    manifest = {'rounds': [{'round': 1, 'net_return_pct': 4.6}]}
    (strategy / 'rounds_manifest.json').write_text(json.dumps(manifest))
    archive = archive_round(strategy, 'test_archive')
    assert (archive / 'round_1/nested/report.bin').read_bytes() == content
    assert not original.exists()
    assert json.loads((archive / 'rounds_manifest.json').read_text()) == manifest
    assert json.loads((archive / 'archive_receipt.json').read_text())['verified_exact'] is True


def test_archive_refuses_escape_before_mutation(tmp_path):
    strategy = tmp_path / 'portfolio_synergy'
    (strategy / 'round_1').mkdir(parents=True)
    with pytest.raises(ValueError, match='escapes'):
        archive_round(strategy, '../../../escape')
    assert (strategy / 'round_1').exists()


def test_user_assignment_cannot_be_labelled_oos_passed():
    with pytest.raises(ValueError, match='Unsupported'):
        verify_user_assigned_profile({'version': 'portfolio-user-assigned-native-round-v1',
                                     'status': 'research_selected_full_native_audits_and_reserved_validation_passed'})


def test_selected_entrypoint_rejects_unbound_user_round(tmp_path):
    path = tmp_path / 'profile.json'
    path.write_text(json.dumps({'version': 'portfolio-user-assigned-native-round-v1',
        'status': 'user_selected_research_round_oos_failed',
        'registration': {'path': str(tmp_path / 'missing.json'), 'sha256': 'invented'}}))
    with pytest.raises(FileNotFoundError):
        verify_profile(path)
