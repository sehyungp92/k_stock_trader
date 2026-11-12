from datetime import datetime, timedelta
import importlib.util
from pathlib import Path

import pytest

p = Path(__file__).resolve().parents[1]/'backtests/strategies/kalcb/staged_support_v1.py'
s = importlib.util.spec_from_file_location('staged_support_under_test', p)
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)


def sample():
    start = datetime.fromisoformat('2025-08-08T09:00:00+09:00')
    bars = [((start+timedelta(minutes=5*i)).isoformat(), 100., 101., 99., 100., 10.) for i in range(8)]
    e = {'decision_available_at': '2025-08-08T09:30:00+09:00',
         'action': {'qty': 11, 'metadata': {'entry_price_ref': 100.}}}
    return e, bars


def test_future_bar_cannot_change_confirmation():
    e, bars = sample(); before = m.confirmation(e, bars)
    bars[7] = (bars[7][0], 1., 999999., 1., 1., 999999.)
    assert m.confirmation(e, bars) == before
    assert before['category'] == 'price_confirmed' and before['initial_qty'] == 5


def test_missing_and_zero_volume_abstain():
    e, b = sample()
    assert m.confirmation(e, b[1:])['category'] == 'invalid_prefix'
    b[3] = (*b[3][:-1], 0.)
    assert m.confirmation(e, b)['category'] == 'invalid_prefix'


def test_unknown_clock_does_not_move_window():
    e, b = sample(); e['decision_available_at'] = '2025-08-08T09:35:00+09:00'
    assert m.confirmation(e, b)['category'] == 'unsupported_opening_clock'


def test_original_reference_is_authenticated():
    e, b = sample(); e['action']['metadata']['entry_price_ref'] = 101.
    with pytest.raises(ValueError, match='reference differs'):
        m.confirmation(e, b)


def test_single_share_unchanged():
    e, b = sample(); e['action']['qty'] = 1
    assert m.confirmation(e, b)['category'] == 'single_share_unchanged'


def test_both_price_conditions_required():
    e, b = sample(); b[6] = (b[6][0], 100., 110., 99., 100., 1000.)
    assert m.confirmation(e, b)['category'] == 'price_not_confirmed'


def test_duplicate_lifetime_is_rejected():
    with pytest.raises(ValueError, match='duplicated'):
        m.summarize([{'position_id': 'same'}, {'position_id': 'same'}])
