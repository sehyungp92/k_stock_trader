"""Semantic regressions plus real adapter/order/fill continuation equivalence."""
from copy import deepcopy
from dataclasses import replace
import json
import pytest

from backtests.strategies.kalcb.holding_account_observation_v2 import AccountObservation


def test_units_reset_restart_and_missing_baseline():
    a = AccountObservation('2026-01-05T10:00:00+09:00', '2026-01-05',
                           101., 100., 102., 40., 162.)
    assert a.session_return == pytest.approx(.01)
    assert replace(a, equity_krw=99.).session_return == pytest.approx(-.01)
    assert replace(a, session_start_equity_krw=101.).session_return == 0
    assert deepcopy(a) == a
    reset = replace(a, available_at='2026-01-06T09:00:00+09:00',
                    session_date='2026-01-06', session_start_equity_krw=101.)
    assert reset.session_return == 0
    with pytest.raises(ValueError):
        replace(a, session_date='2026-01-04')
    with pytest.raises(ValueError):
        replace(a, session_start_equity_krw=0)


def test_authoritative_core_session_reset_survives_state_restart():
    from datetime import date, datetime
    from types import SimpleNamespace
    from strategy_kalcb.core.state import KALCBState
    from strategy_kalcb.core.core_models import KALCBPortfolioView
    from strategy_kalcb.core.logic import _update_portfolio_context
    from backtests.strategies.kalcb.holding_account_observation_v2 import from_native
    state = KALCBState(); state.session_date = date(2026, 1, 5)
    _update_portfolio_context(state, KALCBPortfolioView(cash=100., equity=100.))
    _update_portfolio_context(state, KALCBPortfolioView(cash=102., equity=102.))
    adapter = SimpleNamespace(state=deepcopy(state))
    broker = SimpleNamespace(positions={}, last_prices={}, equity_curve=[100., 102.],
                             _portfolio_equity=lambda:102., _buying_power_available=lambda:204.)
    assert from_native(adapter, broker, datetime.fromisoformat('2026-01-05T10:00:00+09:00')).session_return == pytest.approx(.02)
    adapter.state.session_date = date(2026, 1, 6)
    with pytest.raises(ValueError):
        from_native(adapter, broker, datetime.fromisoformat('2026-01-06T09:00:00+09:00'))
    _update_portfolio_context(adapter.state, KALCBPortfolioView(cash=102., equity=102.))
    assert from_native(adapter, broker, datetime.fromisoformat('2026-01-06T09:00:00+09:00')).session_return == 0


def test_real_observation_path_and_unchanged_funded_branches(tmp_path, monkeypatch):
    from tests.test_kalcb_holding_account_v1 import fixture_run
    from backtests.strategies.kalcb import holding_account_observation_v2 as corrected
    from backtests.strategies.kalcb import holding_account_replay_v1 as replay
    before = tmp_path / 'before'; before.mkdir()
    _, base, detail = fixture_run(before, monkeypatch)
    after = tmp_path / 'after'; after.mkdir()
    seen = []
    def observe(adapter, broker, batch):
        rows = corrected.observations(adapter, broker, batch)
        if any(r['reason'] == 'eligible' for r in rows):
            meta = adapter.state.meta
            baseline = meta['portfolio_session_start_equity']
            try:
                meta['portfolio_session_start_equity'] = baseline * 1.01
                changed = corrected.observations(adapter, broker, batch)
                for a, b in zip(rows, changed):
                    if a['reason'] != 'eligible':
                        continue
                    assert a['features']['session_return'] != b['features']['session_return']
                    b['features']['session_return'] = a['features']['session_return']
                    assert a == b
                del meta['portfolio_session_start_equity']
                with pytest.raises(KeyError):
                    corrected.observations(adapter, broker, batch)
            finally:
                meta['portfolio_session_start_equity'] = baseline
            seen.extend(r for r in rows if r['reason'] == 'eligible')
        return rows
    monkeypatch.setattr(replay, 'observations', observe)
    _, treatment, treatment_detail = fixture_run(after, monkeypatch)
    assert (base, detail) == (treatment, treatment_detail)
    old = [json.loads(s) for s in (before / 'labels.jsonl').read_text().splitlines()]
    new = [json.loads(s) for s in (after / 'labels.jsonl').read_text().splitlines()]
    assert seen and any(r['features']['session_return'] != 0 for r in seen)
    for a, b in zip(old, new):
        assert a['boundary'] == b['boundary']
        assert a['hold_eod_state'] == b['hold_eod_state']
        for arm in ('half', 'exit'):
            for field in ('target', 'action', 'actual_action_qty', 'first_fill', 'last_fill'):
                assert a['contrasts'][arm][field] == b['contrasts'][arm][field]
        b['observation']['features']['session_return'] = a['observation']['features']['session_return']
        assert a['observation'] == b['observation']
