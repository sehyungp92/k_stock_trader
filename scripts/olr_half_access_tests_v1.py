"""0040 joint entry-to-management and allocation boundary checks."""
from dataclasses import replace
from datetime import timedelta
import json
import pytest
from strategy_common.actions import SubmitEntry, SubmitExit, SubmitPartialExit
from strategy_olr.core.allocation import step_olr_batch
from strategy_olr.core.core_models import OLRPortfolioView
from strategy_olr.core.logic import _maybe_submit_managed_exit
from strategy_olr.core.serializers import snapshot_state, restore_state
from tests.strategy_olr import test_expired_selection as access
from tests.strategy_olr.test_precision_v3 import structural_setup


def joint_config(cfg):
    return replace(cfg, capacity_exit_priority=True, capacity_structural_partial=True,
                   capacity_weak_partial_fraction=.5)


def extra_entry():
    cfg, state, snapshot, bars = access.setup()
    cfg = joint_config(cfg)
    result = step_olr_batch(state, bars, cfg, snapshot, OLRPortfolioView(1e8, 1e8))
    return cfg, next(a for a in result.actions if isinstance(a, SubmitEntry))


def test_extra_entry_preserves_original_r_and_captures_entry_session_low():
    cfg, action = extra_entry()
    plain, state, snapshot, bars = access.setup()
    expected = next(a for a in step_olr_batch(state, bars, plain, snapshot,
                    OLRPortfolioView(1e8, 1e8)).actions if isinstance(a, SubmitEntry))
    assert action.reason == 'expired_selection_entry'
    assert action.metadata['structural_entry_day_low'] == 98.
    for key in ('risk_per_share', 'allocation_weight', 'reserved_notional'):
        assert action.metadata[key] == expected.metadata[key]
    assert action.qty == expected.qty
    assert cfg.trade_entry_plan['max_signal_bars'] == 6


@pytest.mark.parametrize('full_risk', [False, True])
def test_filled_extra_has_native_half_lifetime_and_full_risk_priority(full_risk):
    _, action = extra_entry()
    cfg, state, current = structural_setup()
    state.meta['expired_selection_enabled_v1'] = True
    ss = state.symbols[current.symbol]
    position = ss.position
    # Carry real extra-entry evidence into a native filled-position fixture.
    # Keep its actual R: choose observed bars between -.4R and the -1R full guard.
    position.metadata.update(action.metadata)
    risk = action.metadata['risk_per_share']
    position.metadata['structural_entry_day_low'] = position.entry_price - .3 * risk
    position.max_favorable_price = position.entry_price
    close = position.entry_price - (1.2 if full_risk else .5) * risk
    ss.session_bars = [replace(current, open=position.entry_price, high=position.entry_price,
        low=close-.1*risk, close=close, timestamp=current.timestamp-timedelta(minutes=i)) for i in (10,5,0)]
    current = ss.session_bars[-1]
    result = _maybe_submit_managed_exit(state, current, cfg)
    if full_risk:
        sells = [a for a in result.actions if isinstance(a, SubmitExit)]
        assert len(sells) == 1 and sells[0].qty == position.qty_open
        assert sells[0].reason != 'weak_followthrough_partial'
    else:
        assert len(result.actions) == 1 and isinstance(result.actions[0], SubmitPartialExit)
        assert result.actions[0].qty == position.qty_open // 2
        assert result.actions[0].metadata['risk_per_share'] == risk
        assert result.actions[0].metadata['structural_partial_evidence']['entry_session_low'] == position.metadata['structural_entry_day_low']
        restored = restore_state(json.loads(json.dumps(snapshot_state(state))))
        assert not _maybe_submit_managed_exit(restored, current, cfg).actions


def test_joint_keeps_original_topups_before_extra_under_binding_cash(monkeypatch):
    original = access.setup
    def combined(*args, **kwargs):
        cfg, state, snapshot, bars = original(*args, **kwargs)
        return joint_config(cfg), state, snapshot, bars
    monkeypatch.setattr(access, 'setup', combined)
    access.test_original_native_topup_precedes_extra_allocation(monkeypatch)
