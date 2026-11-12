import copy
import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('drift_under_test', Path(__file__).resolve().parents[1] / 'backtests/strategies/kalcb/entry_reference_drift_v1.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def fixture():
    row = dict(order_id='o', symbol='x', day='2026-01-01', position_id='x/t', planned_qty=10,
               decision_available_at='2026-01-01T09:30:00+09:00', original_reference=100.,
               parent_net_pnl_krw=7., parent_entry_cost_krw=1000.3)
    fill = dict(order_id='o', symbol='x', side='BUY', qty=10, price=100.03,
                timestamp='2026-01-01T09:30:00+09:00', metadata=dict(execution_id='f', entry_price_ref=100.,
                entry_route='first30_open_anchor', base_price=100., execution_phase='open', fee=.2,
                execution_bar=dict(open=100., volume=10000, timestamp='2026-01-01T09:30:00+09:00')))
    return row, fill


def test_flat_market_separates_slippage():
    r, f = fixture(); x = m.measure(r, [f], 3)
    assert x['cohort'] == 'flat_initial_gap' and x['initial_market_drift_krw'] == 0
    assert x['initial_slippage_krw'] == pytest.approx(.3) and x['candidate_pnl'] is None


def test_initial_and_later_offset_not_saved_profit():
    r, f = fixture(); g = copy.deepcopy(f); f['qty'] = 4; g['qty'] = 6
    f['metadata']['base_price'] = f['metadata']['execution_bar']['open'] = 101; f['price'] = 101.0303
    g['metadata']['base_price'] = g['metadata']['execution_bar']['open'] = 99; g['price'] = 99.0297
    g['timestamp'] = g['metadata']['execution_bar']['timestamp'] = '2026-01-01T09:35:00+09:00'; g['metadata']['execution_id'] = 'g'
    r['parent_entry_cost_krw'] = f['price']*4+g['price']*6
    x = m.measure(r, [g, f], 3)
    assert x['initial_market_drift_krw'] == 4 and x['later_market_drift_krw'] == -6
    assert x['parent_net_pnl_krw'] == 7 and x['candidate_pnl'] is None


def test_same_timestamp_slices_are_one_initial_block():
    r, f = fixture(); g = copy.deepcopy(f); f['qty'] = 3; g['qty'] = 7; g['metadata']['execution_id'] = 'g'
    assert m.measure(r, [g, f], 3)['first_qty'] == 10


def test_no_fill_retained():
    r, _ = fixture(); r.update(position_id=None, parent_net_pnl_krw=0., parent_entry_cost_krw=0.)
    assert m.measure(r, [], 3)['cohort'] == 'unfilled'


def test_duplicate_fill_rejected():
    r, f = fixture()
    with pytest.raises(ValueError): m.measure(r, [f, f], 3)


def test_predecision_fill_rejected():
    r, f = fixture(); r['decision_available_at'] = '2026-01-01T09:35:00+09:00'
    with pytest.raises(ValueError): m.measure(r, [f], 3)


def test_slippage_mismatch_rejected():
    r, f = fixture()
    with pytest.raises(ValueError): m.measure(r, [f], 13)


def test_wrong_reference_rejected():
    r, f = fixture(); r['original_reference'] = 99
    with pytest.raises(ValueError): m.measure(r, [f], 3)


def test_duplicate_lifetime_rejected():
    r, f = fixture(); x = m.measure(r, [f], 3)
    with pytest.raises(ValueError): m.summary([x, x])


def test_future_pnl_cannot_change_execution_measurements():
    r, f = fixture(); a = m.measure(r, [f], 3); r['parent_net_pnl_krw'] = -1e9; b = m.measure(r, [f], 3)
    for k in a:
        if k != 'parent_net_pnl_krw': assert a[k] == b[k]
