from dataclasses import replace
from datetime import date, datetime, timedelta

import pytest

from strategy_common.clock import KST
from strategy_common.market import MarketBar
from strategy_kalcb.config import KALCBConfig
from strategy_kalcb.core.core_models import KALCBFillEvent
from strategy_kalcb.core.logic import on_kalcb_fill, _manage_position
from strategy_kalcb.core.serializers import restore_state, snapshot_state
from strategy_kalcb.core.state import KALCBPositionState, KALCBState
from strategy_common.actions import SubmitExit
from strategy_kalcb.models import KALCBDailyCandidate
from strategy_kalcb.price_lock import price_lock_evidence

DAY = date(2025, 6, 2)
TS = datetime(2025, 6, 2, 9, 30, tzinfo=KST)
CFG = KALCBConfig.from_mapping(mutations={"kalcb.exit.hard_stop_enabled": False})


def buy(state, qty, price, minute, before, total=84, risk=300):
    return on_kalcb_fill(state, KALCBFillEvent("entry", "247540", "BUY", qty, price,
        TS + timedelta(minutes=minute), metadata={"execution_id": f"fill-{minute}",
        "order_qty": total, "filled_qty_before": before, "filled_qty_after": before + qty,
        "stop_price": 90600, "entry_price_ref": 90900, "risk_per_share": risk}), CFG)


def test_falling_partial_fills_do_not_create_past_mfe_and_survive_restart():
    state = KALCBState()
    buy(state, 36, 90827.24, 0, 0)
    p = state.symbol_state("247540").position
    p.update_mark(high=90800, low=90200)
    assert p.mfe_r() == 0
    state = restore_state(snapshot_state(state))
    buy(state, 32, 90227.06, 5, 36)
    p = state.symbol_state("247540").position
    p.update_mark(high=90200, low=89400)
    buy(state, 16, 89426.82, 10, 68)
    assert p.entry_price == pytest.approx(90331.85333333333)
    assert p.risk_per_share == 300
    assert p.mfe_r() == 0
    p.update_mark(high=89400, low=89200)
    assert p.mfe_r() == 0
    assert p.metadata["risk_reference_basis"] == "signal"
    # A genuinely new positive mark is allowed to establish progress.
    p.update_mark(high=90631.85333333333, low=90331.85333333333)
    assert p.mfe_r() == pytest.approx(1)


def test_gap_through_stop_keeps_the_signal_risk_unit():
    state = KALCBState()
    buy(state, 84, 87926.37, 0, 0)
    p = state.symbol_state("247540").position
    assert p.risk_per_share == 300  # Not the former KRW 1 floor.
    assert p.unrealized_r(p.entry_price + 70) == pytest.approx(70 / 300)


def test_actual_stress_failure_path_exits_at_six_bars():
    state = KALCBState()
    buy(state, 36, 90827.24, 0, 0)
    p = state.symbol_state("247540").position
    p.update_mark(high=90800, low=90200)
    buy(state, 32, 90227.06, 5, 36)
    p.update_mark(high=90200, low=89400)
    buy(state, 16, 89426.82, 10, 68)
    p.hold_bars = 5
    config = replace(CFG, exit_failed_followthrough_bars=6,
        exit_failed_followthrough_mfe_r=.75, exit_failed_followthrough_close_r=-.25)
    bar = MarketBar("247540", TS + timedelta(minutes=25), "5m", 88100, 88200, 87900, 88000, 1000)
    result = _manage_position(state, state.symbol_state("247540"), bar, config)
    assert [a.reason for a in result.actions] == ["failed_followthrough"]
    assert result.actions[0].qty == 84


def lock_fixture():
    c = KALCBDailyCandidate("377300", DAY, 39000, 37000, 38100, 1000, 1000, 6000)
    bars = [MarketBar("377300", TS + timedelta(minutes=i * 5), "5m", 49500, 49500, 49500, 49500, 1000) for i in range(2)]
    return c, bars, replace(CFG, exit_price_lock_bars=2)


def test_dated_sourced_limit_and_explicit_proxy_have_distinct_evidence():
    c, bars, config = lock_fixture()
    assert price_lock_evidence(c, bars, config) is None
    proxy = price_lock_evidence(c, bars, replace(config, exit_price_lock_allow_prior_close_proxy=True))
    assert proxy["reference_kind"] == "prior_close_proxy"
    c = replace(c, metadata={"upper_limit_price": 49500, "price_limit_trade_date": DAY.isoformat(), "price_limit_source": "exchange_fixture"})
    assert price_lock_evidence(c, bars, config)["reference_kind"] == "sourced_upper_limit"
    c = replace(c, metadata={**c.metadata, "price_limit_trade_date": "2025-05-30"})
    assert price_lock_evidence(c, bars, replace(config, exit_price_lock_allow_prior_close_proxy=True)) is None


@pytest.mark.parametrize("change", ["gap", "incomplete", "zero_volume", "different_price", "different_day"])
def test_price_lock_requires_adjacent_observed_completed_prints(change):
    c, bars, config = lock_fixture()
    config = replace(config, exit_price_lock_allow_prior_close_proxy=True)
    alterations = {"gap": {"timestamp": bars[1].timestamp + timedelta(minutes=5)},
        "incomplete": {"is_completed": False}, "zero_volume": {"volume": 0},
        "different_price": {"low": 49450}, "different_day": {"timestamp": bars[1].timestamp + timedelta(days=1)}}
    bars[1] = replace(bars[1], **alterations[change])
    assert price_lock_evidence(c, bars, config) is None


@pytest.mark.parametrize("limit", [0, -1, float("nan"), float("inf"), "bad"])
def test_invalid_supplied_limit_cannot_enable_proxy_fallback(limit):
    c, bars, config = lock_fixture()
    c = replace(c, metadata={"upper_limit_price": limit})
    assert price_lock_evidence(c, bars, replace(config, exit_price_lock_allow_prior_close_proxy=True)) is None


def test_price_lock_uses_normal_market_exit_with_observation_evidence():
    candidate, bars, config = lock_fixture()
    config = replace(config, exit_price_lock_allow_prior_close_proxy=True, exit_target_r=55)
    state = KALCBState()
    symbol = state.symbol_state(candidate.symbol)
    symbol.candidate = candidate
    for bar in bars:
        symbol.add_bar(bar)
    symbol.position = KALCBPositionState(candidate.symbol, 64, 64, 45563.665, TS,
        45400, 45400, 200, "first30_open", 0)
    result = _manage_position(state, symbol, bars[-1], config)
    assert len(result.actions) == 1
    action = result.actions[0]
    assert isinstance(action, SubmitExit)
    assert action.qty == 64 and action.order_type == "MARKET" and action.limit_price is None
    assert action.reason == "upper_price_lock"
    assert action.metadata["price_lock_evidence"]["last_bar"] == bars[-1].timestamp.isoformat()
    assert symbol.position.qty_open == 64  # An order is not a fill or liquidity guarantee.


def test_ordinary_target_exit_precedes_price_lock():
    from strategy_kalcb.core.logic import _exit_reason
    candidate, bars, config = lock_fixture()
    state = KALCBState()
    symbol = state.symbol_state(candidate.symbol)
    symbol.candidate = candidate
    for bar in bars:
        symbol.add_bar(bar)
    position = KALCBPositionState(candidate.symbol, 64, 64, 45563.665, TS,
        45400, 45400, 200, "first30_open", 0)
    reason = _exit_reason(symbol, position, bars[-1], replace(config,
        exit_price_lock_allow_prior_close_proxy=True, exit_target_r=5), current_r=19, mfe_r=19)
    assert reason == "target_r"
