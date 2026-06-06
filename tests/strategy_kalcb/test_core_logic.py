from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta

import pytest

from strategy_common.actions import FlattenPosition, ReplaceProtectiveStop, SubmitEntry, SubmitExit, SubmitPartialExit, SubmitProtectiveStop
from strategy_common.clock import KST
from strategy_common.events import TradeOutcome
from strategy_common.market import MarketBar
from backtests.strategies.kalcb.runner import _collapse_exit_legs, _trade_net_r
from strategy_kalcb.config import KALCBConfig
from strategy_kalcb.core.core_models import KALCBFillEvent, KALCBOrderUpdateEvent
from strategy_kalcb.core.core_models import KALCBPortfolioView
from strategy_kalcb.core.logic import on_kalcb_fill, on_kalcb_order_update, on_kalcb_timer, step_kalcb_core
from strategy_kalcb.core.serializers import restore_state, snapshot_state
from strategy_kalcb.core.state import KALCBPositionState, KALCBState, SymbolStage
from strategy_kalcb.models import EntryType
from strategy_kalcb.models import KALCBDailyCandidate, KALCBDailySnapshot
from strategy_kalcb.data import WebSocketRegistrationBudget
from strategy_kalcb.risk import compute_entry_qty


def test_opening_range_requires_six_completed_5m_bars_then_next_bar_entry():
    trade_date = date(2026, 1, 5)
    state = KALCBState()
    cfg = KALCBConfig.from_mapping({"kalcb.entry.entry_score_blocklist": []})
    snapshot = _snapshot(trade_date)
    portfolio = KALCBPortfolioView(cash=1_000_000.0)
    bars = _or_bars(trade_date)

    for bar in bars[:5]:
        result = step_kalcb_core(state, bar, cfg, snapshot, portfolio)
        assert not result.actions
        assert not state.symbol_state("005930").opening_range_built

    sixth = step_kalcb_core(state, bars[5], cfg, snapshot, portfolio)
    assert not sixth.actions
    assert state.symbol_state("005930").opening_range_built
    assert sixth.decisions[-1].decision_code == "opening_range_built"

    entry = step_kalcb_core(state, bars[6], cfg, snapshot, portfolio)
    assert entry.actions
    assert isinstance(entry.actions[0], SubmitEntry)
    assert entry.actions[0].reason.endswith("next_5m_open")
    assert entry.actions[0].metadata["fill_timing"] == "next_5m_open"


def test_failure_stop_tightening_is_action_based_and_does_not_close_state():
    trade_date = date(2026, 1, 5)
    state = KALCBState()
    symbol_state = state.symbol_state("005930")
    symbol_state.session_date = trade_date
    symbol_state.stage = SymbolStage.IN_POSITION
    symbol_state.position = KALCBPositionState(
        symbol="005930",
        qty_entry=10,
        qty_open=10,
        entry_price=100.0,
        entry_time=datetime(2026, 1, 5, 9, 35, tzinfo=KST),
        initial_stop=95.0,
        current_stop=95.0,
        risk_per_share=10.0,
        entry_type="KRX_OR_BREAKOUT",
        momentum_score=5,
        max_favorable_price=101.0,
        max_adverse_price=99.0,
        hold_bars=9,
    )
    bar = MarketBar("005930", datetime(2026, 1, 5, 10, 20, tzinfo=KST), "5m", 99.5, 100.5, 98.8, 99.0, 1000)

    result = step_kalcb_core(state, bar, KALCBConfig(), None, KALCBPortfolioView(cash=1_000_000.0))

    assert any(isinstance(action, ReplaceProtectiveStop) for action in result.actions)
    assert state.symbol_state("005930").position is not None


def test_stopless_mfe_giveback_exits_without_hard_stop():
    trade_date = date(2026, 1, 5)
    state = KALCBState()
    symbol_state = state.symbol_state("005930")
    symbol_state.session_date = trade_date
    symbol_state.stage = SymbolStage.IN_POSITION
    symbol_state.position = KALCBPositionState(
        symbol="005930",
        qty_entry=10,
        qty_open=10,
        entry_price=100.0,
        entry_time=datetime(2026, 1, 5, 9, 35, tzinfo=KST),
        initial_stop=90.0,
        current_stop=90.0,
        risk_per_share=10.0,
        entry_type="KRX_OR_BREAKOUT",
        momentum_score=5,
        max_favorable_price=170.0,
        max_adverse_price=99.0,
        hold_bars=15,
    )
    cfg = KALCBConfig.from_mapping(
        {
            "kalcb.exit.hard_stop_enabled": False,
            "kalcb.exit.mfe_giveback_enabled": True,
            "kalcb.exit.mfe_giveback_start_r": 6.0,
            "kalcb.exit.mfe_giveback_gap_r": 3.0,
            "kalcb.exit.mfe_giveback_min_hold_bars": 12,
            "kalcb.exit.quick_exit_enabled": False,
            "kalcb.exit.failure_stop_enabled": False,
            "kalcb.exit.adaptive_trail_enabled": False,
            "kalcb.exit.mfe_conviction_enabled": False,
            "kalcb.exit.flow_reversal_enabled": False,
        }
    )
    bar = MarketBar("005930", datetime(2026, 1, 5, 11, 0, tzinfo=KST), "5m", 132.0, 133.0, 129.0, 130.0, 1000)

    result = step_kalcb_core(state, bar, cfg, None, KALCBPortfolioView(cash=1_000_000.0))

    assert isinstance(result.actions[0], SubmitExit)
    assert result.actions[0].reason == "mfe_giveback"
    assert not any(isinstance(action, ReplaceProtectiveStop) for action in result.actions)
    assert state.symbol_state("005930").position.exit_in_flight is True


def test_mfe_floor_exits_only_after_proof_then_round_trip():
    trade_date = date(2026, 1, 5)
    state = KALCBState()
    symbol_state = state.symbol_state("005930")
    symbol_state.session_date = trade_date
    symbol_state.stage = SymbolStage.IN_POSITION
    symbol_state.position = KALCBPositionState(
        symbol="005930",
        qty_entry=10,
        qty_open=10,
        entry_price=100.0,
        entry_time=datetime(2026, 1, 5, 9, 35, tzinfo=KST),
        initial_stop=90.0,
        current_stop=90.0,
        risk_per_share=10.0,
        entry_type="KRX_OR_BREAKOUT",
        momentum_score=5,
        max_favorable_price=150.0,
        max_adverse_price=99.0,
        hold_bars=20,
    )
    cfg = KALCBConfig.from_mapping(
        {
            "kalcb.exit.hard_stop_enabled": False,
            "kalcb.exit.mfe_floor_enabled": True,
            "kalcb.exit.mfe_floor_start_r": 4.0,
            "kalcb.exit.mfe_floor_floor_r": 0.0,
            "kalcb.exit.mfe_floor_min_hold_bars": 12,
            "kalcb.exit.quick_exit_enabled": False,
            "kalcb.exit.failure_stop_enabled": False,
            "kalcb.exit.adaptive_trail_enabled": False,
            "kalcb.exit.mfe_conviction_enabled": False,
            "kalcb.exit.flow_reversal_enabled": False,
        }
    )
    bar = MarketBar("005930", datetime(2026, 1, 5, 11, 25, tzinfo=KST), "5m", 102.0, 103.0, 99.0, 100.0, 1000)

    result = step_kalcb_core(state, bar, cfg, None, KALCBPortfolioView(cash=1_000_000.0))

    assert isinstance(result.actions[0], SubmitExit)
    assert result.actions[0].reason == "mfe_floor"
    assert not any(isinstance(action, ReplaceProtectiveStop) for action in result.actions)
    assert state.symbol_state("005930").position.exit_in_flight is True


def test_mfe_floor_can_be_gated_by_entry_quality_metadata():
    def _state_with_metadata(*, cpr: float, frontier_rank: int) -> KALCBState:
        trade_date = date(2026, 1, 5)
        state = KALCBState()
        symbol_state = state.symbol_state("005930")
        symbol_state.session_date = trade_date
        symbol_state.stage = SymbolStage.IN_POSITION
        symbol_state.position = KALCBPositionState(
            symbol="005930",
            qty_entry=10,
            qty_open=10,
            entry_price=100.0,
            entry_time=datetime(2026, 1, 5, 9, 35, tzinfo=KST),
            initial_stop=90.0,
            current_stop=90.0,
            risk_per_share=10.0,
            entry_type="KRX_FIRST30_OPEN",
            momentum_score=5,
            max_favorable_price=150.0,
            max_adverse_price=99.0,
            hold_bars=20,
            metadata={
                "entry_route": "first30_open",
                "entry_route_mode": "first30_open",
                "entry_route_priority": 0,
                "entry_route_attempts": [{"name": "first30_open", "reason": "first30_open"}],
                "frontier_rank": frontier_rank,
                "first30_signal_bar_cpr": cpr,
                "first30_rel_volume": 1.8,
            },
        )
        return state

    cfg = KALCBConfig.from_mapping(
        {
            "kalcb.exit.hard_stop_enabled": False,
            "kalcb.exit.mfe_floor_enabled": True,
            "kalcb.exit.mfe_floor_start_r": 4.0,
            "kalcb.exit.mfe_floor_floor_r": 0.0,
            "kalcb.exit.mfe_floor_min_hold_bars": 12,
            "kalcb.exit.mfe_floor_min_frontier_rank": 6,
            "kalcb.exit.mfe_floor_max_first30_signal_cpr": 0.75,
            "kalcb.exit.mfe_floor_entry_routes": ["first30_open"],
            "kalcb.exit.quick_exit_enabled": False,
            "kalcb.exit.failure_stop_enabled": False,
            "kalcb.exit.adaptive_trail_enabled": False,
            "kalcb.exit.mfe_conviction_enabled": False,
            "kalcb.exit.flow_reversal_enabled": False,
        }
    )
    bar = MarketBar("005930", datetime(2026, 1, 5, 11, 25, tzinfo=KST), "5m", 102.0, 103.0, 99.0, 100.0, 1000)

    rejected = step_kalcb_core(_state_with_metadata(cpr=0.90, frontier_rank=8), bar, cfg, None, KALCBPortfolioView(cash=1_000_000.0))
    accepted = step_kalcb_core(_state_with_metadata(cpr=0.70, frontier_rank=8), bar, cfg, None, KALCBPortfolioView(cash=1_000_000.0))

    assert not rejected.actions
    assert isinstance(accepted.actions[0], SubmitExit)
    assert accepted.actions[0].reason == "mfe_floor"
    assert accepted.actions[0].metadata["entry_route"] == "first30_open"
    assert accepted.actions[0].metadata["entry_route_attempts"][0]["reason"] == "first30_open"


def test_conditional_target_exits_only_matching_entry_cohort():
    trade_date = date(2026, 1, 5)
    state = KALCBState()
    symbol_state = state.symbol_state("005930")
    symbol_state.session_date = trade_date
    symbol_state.stage = SymbolStage.IN_POSITION
    symbol_state.position = KALCBPositionState(
        symbol="005930",
        qty_entry=10,
        qty_open=10,
        entry_price=100.0,
        entry_time=datetime(2026, 1, 5, 9, 35, tzinfo=KST),
        initial_stop=99.0,
        current_stop=99.0,
        risk_per_share=1.0,
        entry_type="KRX_FIRST30_OPEN",
        momentum_score=5,
        max_favorable_price=170.0,
        max_adverse_price=99.0,
        hold_bars=36,
        metadata={
            "entry_route": "first30_open",
            "entry_route_mode": "first30_open",
            "first30_rel_volume": 2.5,
            "first30_signal_bar_cpr": 0.70,
        },
    )
    cfg = KALCBConfig.from_mapping(
        {
            "kalcb.exit.conditional_target_enabled": True,
            "kalcb.exit.conditional_target_r": 60.0,
            "kalcb.exit.conditional_target_min_hold_bars": 24,
            "kalcb.exit.conditional_target_max_first30_rel_volume": 3.0,
            "kalcb.exit.conditional_target_entry_routes": ["first30_open"],
            "kalcb.exit.target_r": 70.0,
            "kalcb.exit.quick_exit_enabled": False,
            "kalcb.exit.failure_stop_enabled": False,
            "kalcb.exit.adaptive_trail_enabled": False,
            "kalcb.exit.mfe_conviction_enabled": False,
            "kalcb.exit.flow_reversal_enabled": False,
        }
    )
    bar = MarketBar("005930", datetime(2026, 1, 5, 12, 30, tzinfo=KST), "5m", 160.0, 161.0, 159.0, 160.0, 1000)

    result = step_kalcb_core(state, bar, cfg, None, KALCBPortfolioView(cash=1_000_000.0))

    assert result.actions
    assert isinstance(result.actions[0], SubmitExit)
    assert result.actions[0].reason == "conditional_target_r"


def test_path_quality_exit_uses_causal_post_entry_context():
    trade_date = date(2026, 1, 5)
    state = KALCBState()
    symbol_state = state.symbol_state("005930")
    symbol_state.session_date = trade_date
    symbol_state.stage = SymbolStage.IN_POSITION
    symbol_state.or_high = 120.0
    symbol_state.or_low = 90.0
    for minute, close in ((35, 145.0), (40, 138.0), (45, 130.0), (50, 120.0)):
        symbol_state.add_bar(MarketBar("005930", datetime(2026, 1, 5, 9, minute, tzinfo=KST), "5m", close - 1.0, close + 3.0, close - 2.0, close, 10_000))
    symbol_state.position = KALCBPositionState(
        symbol="005930",
        qty_entry=10,
        qty_open=10,
        entry_price=100.0,
        entry_time=datetime(2026, 1, 5, 9, 35, tzinfo=KST),
        initial_stop=90.0,
        current_stop=90.0,
        risk_per_share=10.0,
        entry_type="KRX_FIRST30_OPEN",
        momentum_score=5,
        max_favorable_price=160.0,
        max_adverse_price=99.0,
        hold_bars=15,
        avwap_at_entry=110.0,
        or_high=120.0,
        or_low=90.0,
        metadata={
            "entry_route": "first30_open",
            "entry_route_mode": "first30_open",
            "frontier_rank": 4,
            "first30_rel_volume": 5.0,
        },
    )
    cfg = KALCBConfig.from_mapping(
        {
            "kalcb.exit.hard_stop_enabled": False,
            "kalcb.exit.path_quality_enabled": True,
            "kalcb.exit.path_quality_min_hold_bars": 12,
            "kalcb.exit.path_quality_min_mfe_r": 5.0,
            "kalcb.exit.path_quality_min_giveback_r": 5.0,
            "kalcb.exit.path_quality_min": {"bars_since_mfe": 3, "below_vwap_streak": 1},
            "kalcb.exit.path_quality_max": {"vwap_ret": -0.01, "or_position": 0.55},
            "kalcb.exit.path_quality_entry_routes": ["first30_open"],
            "kalcb.exit.quick_exit_enabled": False,
            "kalcb.exit.failure_stop_enabled": False,
            "kalcb.exit.adaptive_trail_enabled": False,
            "kalcb.exit.mfe_conviction_enabled": False,
            "kalcb.exit.flow_reversal_enabled": False,
        }
    )
    bar = MarketBar("005930", datetime(2026, 1, 5, 10, 0, tzinfo=KST), "5m", 108.0, 109.0, 103.0, 105.0, 1000)

    result = step_kalcb_core(state, bar, cfg, None, KALCBPortfolioView(cash=1_000_000.0))

    assert isinstance(result.actions[0], SubmitExit)
    assert result.actions[0].reason == "path_quality_exit"
    assert state.symbol_state("005930").position.metadata["exit_path_quality_context"]["bars_since_mfe"] >= 3


def test_failed_followthrough_is_exact_bar_by_default():
    trade_date = date(2026, 1, 5)
    state = KALCBState()
    symbol_state = state.symbol_state("005930")
    symbol_state.session_date = trade_date
    symbol_state.stage = SymbolStage.IN_POSITION
    symbol_state.position = KALCBPositionState(
        symbol="005930",
        qty_entry=10,
        qty_open=10,
        entry_price=100.0,
        entry_time=datetime(2026, 1, 5, 9, 35, tzinfo=KST),
        initial_stop=90.0,
        current_stop=90.0,
        risk_per_share=10.0,
        entry_type="KRX_FIRST30_OPEN",
        momentum_score=5,
        max_favorable_price=105.0,
        max_adverse_price=96.0,
        hold_bars=8,
    )
    cfg = KALCBConfig.from_mapping(
        {
            "kalcb.exit.hard_stop_enabled": False,
            "kalcb.exit.failed_followthrough_bars": 6,
            "kalcb.exit.failed_followthrough_mfe_r": 0.75,
            "kalcb.exit.failed_followthrough_close_r": -0.25,
            "kalcb.exit.failed_followthrough_persistent": False,
            "kalcb.exit.quick_exit_enabled": False,
            "kalcb.exit.failure_stop_enabled": False,
            "kalcb.exit.adaptive_trail_enabled": False,
            "kalcb.exit.mfe_conviction_enabled": False,
            "kalcb.exit.flow_reversal_enabled": False,
        }
    )
    bar = MarketBar("005930", datetime(2026, 1, 5, 10, 30, tzinfo=KST), "5m", 98.0, 98.5, 96.0, 97.0, 1000)

    result = step_kalcb_core(state, bar, cfg, None, KALCBPortfolioView(cash=1_000_000.0))

    assert not any(isinstance(action, SubmitExit) and action.reason == "failed_followthrough" for action in result.actions)
    assert state.symbol_state("005930").position.exit_in_flight is False


def test_persistent_failed_followthrough_exits_after_checkpoint():
    trade_date = date(2026, 1, 5)
    state = KALCBState()
    symbol_state = state.symbol_state("005930")
    symbol_state.session_date = trade_date
    symbol_state.stage = SymbolStage.IN_POSITION
    symbol_state.position = KALCBPositionState(
        symbol="005930",
        qty_entry=10,
        qty_open=10,
        entry_price=100.0,
        entry_time=datetime(2026, 1, 5, 9, 35, tzinfo=KST),
        initial_stop=90.0,
        current_stop=90.0,
        risk_per_share=10.0,
        entry_type="KRX_FIRST30_OPEN",
        momentum_score=5,
        max_favorable_price=105.0,
        max_adverse_price=96.0,
        hold_bars=8,
    )
    cfg = KALCBConfig.from_mapping(
        {
            "kalcb.exit.hard_stop_enabled": False,
            "kalcb.exit.failed_followthrough_bars": 6,
            "kalcb.exit.failed_followthrough_mfe_r": 0.75,
            "kalcb.exit.failed_followthrough_close_r": -0.25,
            "kalcb.exit.failed_followthrough_persistent": True,
            "kalcb.exit.quick_exit_enabled": False,
            "kalcb.exit.failure_stop_enabled": False,
            "kalcb.exit.adaptive_trail_enabled": False,
            "kalcb.exit.mfe_conviction_enabled": False,
            "kalcb.exit.flow_reversal_enabled": False,
        }
    )
    bar = MarketBar("005930", datetime(2026, 1, 5, 10, 30, tzinfo=KST), "5m", 98.0, 98.5, 96.0, 97.0, 1000)

    result = step_kalcb_core(state, bar, cfg, None, KALCBPortfolioView(cash=1_000_000.0))

    assert isinstance(result.actions[0], SubmitExit)
    assert result.actions[0].reason == "failed_followthrough"
    assert state.symbol_state("005930").position.exit_in_flight is True


def test_conditional_stop_activation_submits_protective_stop_after_mfe_proof():
    trade_date = date(2026, 1, 5)
    state = KALCBState()
    symbol_state = state.symbol_state("005930")
    symbol_state.session_date = trade_date
    symbol_state.stage = SymbolStage.IN_POSITION
    symbol_state.position = KALCBPositionState(
        symbol="005930",
        qty_entry=10,
        qty_open=10,
        entry_price=100.0,
        entry_time=datetime(2026, 1, 5, 9, 35, tzinfo=KST),
        initial_stop=90.0,
        current_stop=90.0,
        risk_per_share=10.0,
        entry_type="KRX_OR_BREAKOUT",
        momentum_score=5,
        max_favorable_price=150.0,
        max_adverse_price=99.0,
        hold_bars=15,
    )
    cfg = KALCBConfig.from_mapping(
        {
            "kalcb.exit.hard_stop_enabled": False,
            "kalcb.exit.conditional_stop_activate_r": 4.0,
            "kalcb.exit.conditional_stop_gap_r": 2.0,
            "kalcb.exit.conditional_stop_min_hold_bars": 12,
            "kalcb.exit.quick_exit_enabled": False,
            "kalcb.exit.failure_stop_enabled": False,
            "kalcb.exit.adaptive_trail_enabled": False,
            "kalcb.exit.mfe_conviction_enabled": False,
            "kalcb.exit.flow_reversal_enabled": False,
        }
    )
    bar = MarketBar("005930", datetime(2026, 1, 5, 11, 0, tzinfo=KST), "5m", 135.0, 136.0, 134.0, 135.0, 1000)

    result = step_kalcb_core(state, bar, cfg, None, KALCBPortfolioView(cash=1_000_000.0))

    assert isinstance(result.actions[0], SubmitProtectiveStop)
    assert result.actions[0].reason == "conditional_mfe_stop"
    assert result.actions[0].stop_price > 100.0
    assert state.symbol_state("005930").position.exit_in_flight is False


def test_partial_take_is_neutral_action_and_waits_for_fill():
    trade_date = date(2026, 1, 5)
    state = KALCBState()
    symbol_state = state.symbol_state("005930")
    symbol_state.session_date = trade_date
    symbol_state.stage = SymbolStage.IN_POSITION
    symbol_state.position = KALCBPositionState(
        symbol="005930",
        qty_entry=10,
        qty_open=10,
        entry_price=100.0,
        entry_time=datetime(2026, 1, 5, 9, 35, tzinfo=KST),
        initial_stop=95.0,
        current_stop=95.0,
        risk_per_share=10.0,
        entry_type="KRX_OR_BREAKOUT",
        momentum_score=5,
        hold_bars=4,
    )
    cfg = KALCBConfig.from_mapping(
        {
            "kalcb.exit.use_partial_takes": True,
            "kalcb.exit.partial_r_trigger": 0.6,
            "kalcb.exit.partial_fraction": 0.4,
            "kalcb.exit.quick_exit_enabled": False,
            "kalcb.exit.failure_stop_enabled": False,
            "kalcb.exit.adaptive_trail_enabled": False,
        }
    )
    bar = MarketBar("005930", datetime(2026, 1, 5, 10, 0, tzinfo=KST), "5m", 105.0, 107.0, 104.0, 106.5, 1000)

    result = step_kalcb_core(state, bar, cfg, None, KALCBPortfolioView(cash=1_000_000.0))
    second = step_kalcb_core(state, bar, cfg, None, KALCBPortfolioView(cash=1_000_000.0))

    assert isinstance(result.actions[0], SubmitPartialExit)
    assert result.actions[0].qty == 4
    assert state.symbol_state("005930").position is not None
    assert state.symbol_state("005930").position.partial_taken is False
    assert state.symbol_state("005930").position.partial_order_id == "__pending__"
    assert not second.actions


def test_partial_fill_replaces_remainder_stop_at_breakeven_buffer():
    trade_date = date(2026, 1, 5)
    state = KALCBState()
    symbol_state = state.symbol_state("005930")
    symbol_state.session_date = trade_date
    symbol_state.stage = SymbolStage.IN_POSITION
    symbol_state.position = KALCBPositionState(
        symbol="005930",
        qty_entry=10,
        qty_open=10,
        entry_price=100.0,
        entry_time=datetime(2026, 1, 5, 9, 35, tzinfo=KST),
        initial_stop=95.0,
        current_stop=95.0,
        risk_per_share=10.0,
        entry_type="KRX_OR_BREAKOUT",
        momentum_score=5,
    )
    state.order_roles["partial-1"] = {"order_role": "TP"}
    fill = KALCBFillEvent(
        order_id="partial-1",
        symbol="005930",
        side="SELL",
        qty=4,
        price=107.0,
        timestamp=datetime(2026, 1, 5, 10, 5, tzinfo=KST),
        reason="partial_profit",
    )

    result = on_kalcb_fill(state, fill, KALCBConfig())
    position = state.symbol_state("005930").position

    assert position is not None
    assert position.qty_open == 6
    assert position.partial_taken is True
    assert isinstance(result.actions[0], ReplaceProtectiveStop)
    assert result.actions[0].stop_price == pytest.approx(101.0)
    assert result.actions[0].qty == 6
    assert result.decisions[0].decision_code == "partial_filled"


def test_partial_trade_legs_are_collapsed_to_plan_level_r():
    entry_time = datetime(2026, 1, 5, 9, 35, tzinfo=KST)
    route = {"risk_per_share": 10.0, "entry_type": "KRX_OR_BREAKOUT"}
    partial = TradeOutcome(
        strategy_id="KALCB",
        symbol="005930",
        qty=4,
        entry_decision_time=entry_time - timedelta(minutes=5),
        entry_fill_time=entry_time,
        entry_price=100.0,
        exit_fill_time=entry_time + timedelta(minutes=10),
        exit_price=106.0,
        gross_pnl=24.0,
        net_pnl=24.0,
        realized=True,
        exit_reason="partial_profit",
        route_metadata=route,
        cohort_metadata={"exit_cohort": "partial"},
        mfe=10.0,
        mae=-2.0,
    )
    final = TradeOutcome(
        strategy_id="KALCB",
        symbol="005930",
        qty=6,
        entry_decision_time=entry_time - timedelta(minutes=5),
        entry_fill_time=entry_time,
        entry_price=100.0,
        exit_fill_time=entry_time + timedelta(minutes=30),
        exit_price=101.0,
        gross_pnl=6.0,
        net_pnl=6.0,
        realized=True,
        exit_reason="partial_breakeven",
        route_metadata=route,
        cohort_metadata={"exit_cohort": "protected_stop"},
        mfe=12.0,
        mae=-3.0,
    )

    collapsed = _collapse_exit_legs([partial, final])

    assert len(collapsed) == 1
    assert collapsed[0].qty == 10
    assert collapsed[0].exit_price == pytest.approx(103.0)
    assert _trade_net_r(collapsed[0]) == pytest.approx(0.3)
    assert collapsed[0].cohort_metadata["partial_taken"] is True


def test_state_serializer_roundtrip_preserves_opening_range_and_orders():
    trade_date = date(2026, 1, 5)
    state = KALCBState(snapshot_hash="hash-a", source_fingerprint="source-a")
    symbol_state = state.symbol_state("005930")
    symbol_state.reset_for_session(trade_date, _candidate(trade_date))
    symbol_state.or_high = 101.0
    symbol_state.or_low = 100.0
    symbol_state.opening_range_built = True
    symbol_state.touched_reclaim_levels["pullback_acceptance:campaign_box_high"] = True
    state.order_roles["order-1"] = {"order_role": "ENTRY"}

    restored = restore_state(snapshot_state(state))

    assert restored.snapshot_hash == "hash-a"
    assert restored.symbol_state("005930").or_high == pytest.approx(101.0)
    assert restored.symbol_state("005930").touched_reclaim_levels["pullback_acceptance:campaign_box_high"] is True
    assert restored.order_roles == state.order_roles


def test_symbol_state_caches_market_bars_without_persisting_duplicate_payload():
    trade_date = date(2026, 1, 5)
    state = KALCBState(snapshot_hash="hash-a", source_fingerprint="source-a")
    symbol_state = state.symbol_state("005930")
    symbol_state.reset_for_session(trade_date, _candidate(trade_date))
    for bar in _or_bars(trade_date)[:3]:
        symbol_state.add_bar(bar)

    assert symbol_state.bars_today is symbol_state.bars_today
    assert [bar.close for bar in symbol_state.bars_today] == [bar.close for bar in _or_bars(trade_date)[:3]]

    payload = snapshot_state(state)
    assert "_market_bars" not in payload["symbols"]["005930"]

    restored = restore_state(payload)
    assert [bar.close for bar in restored.symbol_state("005930").bars_today] == [bar.close for bar in _or_bars(trade_date)[:3]]


def test_ws_budget_enforces_strategy_slice_before_shared_reg_cap():
    budget = WebSocketRegistrationBudget(max_registrations=40, reserved_execution_regs=1, hot_regs_per_symbol=1, strategy_symbol_budget=2)
    assert budget.allocate_hot("005930")[0] is True
    assert budget.allocate_hot("000660")[0] is True
    ok, reason = budget.allocate_hot("035420")
    assert ok is False
    assert reason == "strategy_ws_slice_exhausted"


def test_config_accepts_plan_seed_top_level_sections():
    cfg = KALCBConfig.from_mapping(
        {
            "session": {"open": "09:00", "close": "15:30", "entry_window_end": "11:30", "ws_budget": 8},
            "timeframes": {"signal": "5m", "execution": "5m_next_open", "live_parity_fill_timing": "next_5m_open"},
            "entry": {"rvol_threshold": 2.2, "entry_score_size_mults": {"KRX_OR_BREAKOUT:7": 0.5}},
            "risk": {"regime_size_multipliers": {"B": 0.5}},
            "exits": {"quick_exit_bars": 8},
            "carry": {"mode": "off"},
            "live": {"rest_min_interval_paper_s": 0.5},
        }
    )

    assert cfg.ws_budget == 8
    assert cfg.entry_window_end.hour == 11
    assert cfg.entry_window_end.minute == 30
    assert cfg.rvol_threshold == pytest.approx(2.2)
    assert cfg.entry_score_size_mults == {"KRX_OR_BREAKOUT:7": 0.5}
    assert cfg.regime_mult_b == pytest.approx(0.5)
    assert cfg.quick_exit_bars == 8
    assert cfg.carry_mode.value == "off"


def test_frontier_config_preserves_ws_and_rest_limits():
    cfg = KALCBConfig.from_mapping({"kalcb.session.ws_budget": 8, "frontier": {"size": 24, "rotation_slots": 2}})

    assert cfg.frontier_size == 24
    assert cfg.frontier_rotation_slots == 2

    with pytest.raises(ValueError, match="frontier_size must be at least ws_budget"):
        KALCBConfig.from_mapping({"kalcb.session.ws_budget": 8, "kalcb.frontier.size": 4})


def test_mid_morning_or_caution_can_require_two_completed_closes():
    trade_date = date(2026, 1, 5)
    state = KALCBState()
    cfg = KALCBConfig.from_mapping(
        {
            "kalcb.entry.entry_score_blocklist": [],
            "kalcb.entry.or_caution_window_start": "09:30",
            "kalcb.entry.or_caution_window_end": "10:30",
            "kalcb.entry.or_caution_require_two_close": True,
        }
    )
    snapshot = _snapshot(trade_date)
    portfolio = KALCBPortfolioView(cash=1_000_000.0)

    for bar in _or_bars(trade_date)[:6]:
        step_kalcb_core(state, bar, cfg, snapshot, portfolio)
    result = step_kalcb_core(state, _or_bars(trade_date)[6], cfg, snapshot, portfolio)

    assert not result.actions
    assert result.decisions[-1].reason == "or_caution_two_close_missing"


def test_borderline_cpr_entry_requires_high_score_rescue_gate():
    trade_date = date(2026, 1, 5)
    state = KALCBState()
    cfg = KALCBConfig.from_mapping(
        {
            "kalcb.entry.entry_score_blocklist": [],
            "kalcb.entry.cpr_relax_threshold": 0.55,
            "kalcb.entry.cpr_relax_min_score": 5,
        }
    )
    snapshot = _snapshot(trade_date)
    portfolio = KALCBPortfolioView(cash=1_000_000.0)

    for bar in _or_bars(trade_date)[:6]:
        step_kalcb_core(state, bar, cfg, snapshot, portfolio)
    result = step_kalcb_core(state, _borderline_cpr_breakout(trade_date, adx=25.0), cfg, snapshot, portfolio)

    assert result.actions
    gates = result.actions[0].metadata["gate_decisions"]
    assert gates["cpr_relax"] is True


def test_borderline_cpr_rescue_can_use_non_cpr_score_four():
    trade_date = date(2026, 1, 5)
    state = KALCBState()
    cfg = KALCBConfig.from_mapping(
        {
            "kalcb.entry.entry_score_blocklist": [],
            "kalcb.entry.cpr_relax_threshold": 0.55,
            "kalcb.entry.cpr_relax_min_score": 4,
        }
    )
    snapshot = _snapshot(trade_date)
    portfolio = KALCBPortfolioView(cash=1_000_000.0)

    for bar in _or_bars(trade_date)[:6]:
        step_kalcb_core(state, bar, cfg, snapshot, portfolio)
    result = step_kalcb_core(state, _borderline_cpr_breakout(trade_date), cfg, snapshot, portfolio)

    assert result.actions
    assert result.actions[0].metadata["momentum_score"] == 4


def test_borderline_cpr_without_high_score_is_rejected():
    trade_date = date(2026, 1, 5)
    state = KALCBState()
    cfg = KALCBConfig.from_mapping(
        {
            "kalcb.entry.entry_score_blocklist": [],
            "kalcb.entry.cpr_relax_threshold": 0.55,
            "kalcb.entry.cpr_relax_min_score": 5,
        }
    )
    snapshot = _snapshot(trade_date)
    portfolio = KALCBPortfolioView(cash=1_000_000.0)

    for bar in _or_bars(trade_date)[:6]:
        step_kalcb_core(state, bar, cfg, snapshot, portfolio)
    result = step_kalcb_core(state, _borderline_cpr_breakout(trade_date), cfg, snapshot, portfolio)

    assert not result.actions
    assert result.decisions[-1].reason == "cpr_relax_score_too_low"


def test_raw_breakout_rejection_reports_rvol_gate_separately():
    trade_date = date(2026, 1, 5)
    state = KALCBState()
    cfg = KALCBConfig.from_mapping({"kalcb.entry.entry_score_blocklist": []})
    snapshot = _snapshot(trade_date)
    portfolio = KALCBPortfolioView(cash=1_000_000.0)

    for bar in _or_bars(trade_date)[:6]:
        step_kalcb_core(state, bar, cfg, snapshot, portfolio)
    result = step_kalcb_core(state, _borderline_cpr_breakout(trade_date, volume=1500, adx=25.0), cfg, snapshot, portfolio)

    assert not result.actions
    assert result.decisions[-1].reason == "rvol_below_min"


def test_secondary_rank_strict_gate_allows_only_high_quality_expansion():
    trade_date = date(2026, 1, 5)
    state = KALCBState()
    cfg = KALCBConfig.from_mapping(
        {
            "kalcb.entry.entry_score_blocklist": [],
            "kalcb.entry.secondary_rank_start": 2,
            "kalcb.entry.secondary_min_rvol": 6.0,
        }
    )
    snapshot = KALCBDailySnapshot(
        trade_date=trade_date,
        source_fingerprint="unit",
        generated_at=datetime(2026, 1, 5, tzinfo=KST),
        candidates=(_candidate_for_symbol(trade_date, "000660"), _candidate_for_symbol(trade_date, "005930")),
    )
    portfolio = KALCBPortfolioView(cash=1_000_000.0)

    for bar in _or_bars(trade_date)[:6]:
        step_kalcb_core(state, bar, cfg, snapshot, portfolio)
    result = step_kalcb_core(state, _or_bars(trade_date)[6], cfg, snapshot, portfolio)

    assert not result.actions
    assert state.symbol_state("005930").candidate_rank == 2
    assert result.decisions[-1].reason == "secondary_rvol_too_low"


def test_stale_live_snapshot_is_not_used_for_new_session():
    trade_date = date(2026, 1, 5)
    stale = _snapshot(trade_date - timedelta(days=1))
    state = KALCBState()

    result = step_kalcb_core(state, _or_bars(trade_date)[0], KALCBConfig(), stale, KALCBPortfolioView(cash=1_000_000.0))

    assert not result.actions
    assert state.snapshot_hash == ""
    assert state.symbol_state("005930").candidate is None
    assert state.symbol_state("005930").rejected_reason == "no_daily_candidate"


def test_opening_range_blocks_missing_regular_session_start():
    trade_date = date(2026, 1, 5)
    state = KALCBState()
    bars = [
        MarketBar(
            bar.symbol,
            bar.timestamp + timedelta(minutes=10),
            bar.timeframe,
            bar.open,
            bar.high,
            bar.low,
            bar.close,
            bar.volume,
        )
        for bar in _or_bars(trade_date)[:6]
    ]

    for bar in bars[:5]:
        step_kalcb_core(state, bar, KALCBConfig(), _snapshot(trade_date), KALCBPortfolioView(cash=1_000_000.0))
    result = step_kalcb_core(state, bars[5], KALCBConfig(), _snapshot(trade_date), KALCBPortfolioView(cash=1_000_000.0))

    assert result.decisions[-1].reason == "opening_range_incomplete"
    assert state.symbol_state("005930").stage == SymbolStage.BLOCKED


def test_ws_budget_ledger_counts_existing_strategy_slice(tmp_path):
    ledger = tmp_path / "ws_budget.json"
    first = WebSocketRegistrationBudget(strategy_symbol_budget=1, ledger_path=ledger)
    assert first.allocate_hot("005930")[0] is True

    restarted = WebSocketRegistrationBudget(strategy_symbol_budget=1, ledger_path=ledger)
    ok, reason = restarted.allocate_hot("000660")

    assert ok is False
    assert reason == "strategy_ws_slice_exhausted"


def test_entry_qty_uses_only_explicit_score_size_multipliers():
    cfg = KALCBConfig.from_mapping({"kalcb.risk.risk_per_trade_pct": 0.001})
    candidate = _candidate(date(2026, 1, 5))

    score4 = compute_entry_qty(
        cash=100_000_000,
        open_notional=0.0,
        entry_price=1000.0,
        stop_price=900.0,
        config=cfg,
        candidate=candidate,
        entry_type=EntryType.OR_BREAKOUT,
        momentum_score=4,
    )
    score7 = compute_entry_qty(
        cash=100_000_000,
        open_notional=0.0,
        entry_price=1000.0,
        stop_price=900.0,
        config=cfg,
        candidate=candidate,
        entry_type=EntryType.OR_BREAKOUT,
        momentum_score=7,
    )

    assert score4 == score7


def test_entry_qty_zero_when_participation_cap_fails():
    cfg = KALCBConfig.from_mapping({"kalcb.risk.risk_per_trade_pct": 0.001})
    thin = KALCBDailyCandidate(
        symbol="005930",
        trade_date=date(2026, 1, 5),
        prior_day_high=1000.0,
        prior_day_low=900.0,
        prior_day_close=950.0,
        daily_atr=30.0,
        expected_5m_volume=10.0,
        average_30m_volume=50.0,
        source_fingerprint="unit",
    )

    qty = compute_entry_qty(
        cash=100_000_000,
        open_notional=0.0,
        entry_price=1000.0,
        stop_price=900.0,
        config=cfg,
        candidate=thin,
        entry_type=EntryType.OR_BREAKOUT,
        momentum_score=5,
    )

    assert qty == 0


def test_first30_open_plan_emits_entry_after_completed_0925_bar():
    trade_date = date(2026, 1, 5)
    state = KALCBState()
    cfg = KALCBConfig.from_mapping(
        {
            "kalcb.entry.plan_mode": "first30_open",
            "kalcb.entry.rvol_threshold": 0.0,
            "kalcb.entry.cpr_threshold": 0.0,
            "kalcb.entry.momentum_score_min": 0,
            "kalcb.entry.entry_score_blocklist": [],
        }
    )
    snapshot = _snapshot(trade_date)
    portfolio = KALCBPortfolioView(cash=1_000_000.0)
    result = None

    for bar in _or_bars(trade_date)[:6]:
        result = step_kalcb_core(state, bar, cfg, snapshot, portfolio)

    assert result is not None
    assert result.actions
    assert isinstance(result.actions[0], SubmitEntry)
    assert result.decisions[0].decision_code == "opening_range_built"
    assert result.decisions[-1].decision_code == "entry"
    assert result.actions[0].metadata["entry_type"] == EntryType.FIRST30_OPEN.value
    assert result.actions[0].metadata["signal_bar"].endswith("09:25:00+09:00")
    assert result.actions[0].metadata["first30_bar_count"] == 6
    assert result.actions[0].metadata["first30_signal_bar"].endswith("09:25:00+09:00")
    assert result.actions[0].metadata["first30_open"] == pytest.approx(_or_bars(trade_date)[0].open)
    assert result.actions[0].metadata["first30_range_close_location"] == pytest.approx((100.8 - 100.0) / (101.0 - 100.0))
    assert "first30_signal_bar_cpr" in result.actions[0].metadata


def test_entry_routes_config_hydrates_from_nested_aliases():
    cfg = KALCBConfig.from_mapping(
        {
            "kalcb": {
                "entry": {
                    "routes": [
                        {"name": "anchor", "mode": "first30_open", "priority": 0},
                        {"name": "rank6_deferred", "mode": "deferred_continuation", "priority": 20, "after_bar": 2, "max_frontier_rank": 6},
                    ]
                }
            }
        }
    )

    assert len(cfg.entry_plan_routes) == 2
    assert cfg.entry_plan_routes[0]["name"] == "anchor"
    assert cfg.entry_plan_routes[1]["mode"] == "deferred_continuation"


def test_first30_route_emits_parity_metadata_on_submit_entry():
    trade_date = date(2026, 1, 5)
    state = KALCBState()
    cfg = KALCBConfig.from_mapping(
        {
            "kalcb.entry.routes": [{"name": "first30_primary", "mode": "first30_open", "priority": 0}],
            "kalcb.entry.rvol_threshold": 0.0,
            "kalcb.entry.cpr_threshold": 0.0,
            "kalcb.entry.momentum_score_min": 0,
            "kalcb.entry.entry_score_blocklist": [],
        }
    )
    snapshot = _snapshot(trade_date)
    portfolio = KALCBPortfolioView(cash=1_000_000.0)
    result = None

    for bar in _or_bars(trade_date)[:6]:
        result = step_kalcb_core(state, bar, cfg, snapshot, portfolio)

    assert result is not None
    assert result.actions
    metadata = result.actions[0].metadata
    assert metadata["entry_route"] == "first30_primary"
    assert metadata["entry_route_mode"] == "first30_open"
    assert metadata["entry_route_priority"] == 0
    assert metadata["entry_route_attempts"][0]["reason"] == "first30_open"
    assert "h3_current_r" not in metadata
    assert "h6_current_r" not in metadata


def test_entry_route_context_gate_uses_causal_session_snapshot_metadata():
    trade_date = date(2026, 1, 5)
    state = KALCBState()
    weak_context_candidate = replace(
        _candidate(trade_date),
        metadata={"first30_ret": -0.01, "first30_rel_volume": 2.0, "first30_close_location": 0.8},
    )
    snapshot = KALCBDailySnapshot(
        trade_date=trade_date,
        source_fingerprint="unit",
        generated_at=datetime(2026, 1, 5, tzinfo=KST),
        candidates=(weak_context_candidate,),
        metadata={"active_symbol_count": 1, "candidate_pool_count": 1, "selection_count": 1, "frontier_symbol_count": 1},
    )
    cfg = KALCBConfig.from_mapping(
        {
            "kalcb.entry.routes": [
                {
                    "name": "first30_primary",
                    "mode": "first30_open",
                    "priority": 0,
                    "context_min": {"session_first30_positive_share": 1.0},
                }
            ],
            "kalcb.entry.rvol_threshold": 0.0,
            "kalcb.entry.cpr_threshold": 0.0,
            "kalcb.entry.momentum_score_min": 0,
            "kalcb.entry.entry_score_blocklist": [],
        }
    )
    result = None

    for bar in _or_bars(trade_date)[:6]:
        result = step_kalcb_core(state, bar, cfg, snapshot, KALCBPortfolioView(cash=1_000_000.0))

    assert result is not None
    assert not result.actions
    assert result.decisions[-1].reason == "entry_context_min:session_first30_positive_share"
    assert result.decisions[-1].metadata["session_first30_positive_share"] == pytest.approx(0.0)


def test_entry_route_context_gate_falls_back_to_lower_priority_route():
    trade_date = date(2026, 1, 5)
    state = KALCBState()
    cfg = KALCBConfig.from_mapping(
        {
            "kalcb.entry.routes": [
                {
                    "name": "high_gap_boost",
                    "mode": "first30_open",
                    "priority": 0,
                    "context_min": {"first30_gap": 9.0},
                    "risk_mult": 1.2,
                },
                {"name": "first30_fallback", "mode": "first30_open", "priority": 10},
            ],
            "kalcb.entry.rvol_threshold": 0.0,
            "kalcb.entry.cpr_threshold": 0.0,
            "kalcb.entry.momentum_score_min": 0,
            "kalcb.entry.entry_score_blocklist": [],
        }
    )
    snapshot = _snapshot(trade_date)
    portfolio = KALCBPortfolioView(cash=1_000_000.0)
    result = None

    for bar in _or_bars(trade_date)[:6]:
        result = step_kalcb_core(state, bar, cfg, snapshot, portfolio)

    assert result is not None
    assert result.actions
    metadata = result.actions[0].metadata
    assert metadata["entry_route"] == "first30_fallback"
    assert metadata["entry_route_risk_mult"] == pytest.approx(1.0)
    assert metadata["entry_route_attempts"][0]["reason"] == "entry_context_min:first30_gap"


def test_entry_route_context_gate_can_use_session_sector_intraday_features():
    trade_date = date(2026, 1, 5)
    state = KALCBState()
    weak_sector_candidate = replace(
        _candidate(trade_date),
        metadata={
            "sector_intraday_score_pct": 40.0,
            "sector_intraday_ret": -0.01,
            "sector_intraday_effective_count": 2,
        },
    )
    snapshot = KALCBDailySnapshot(
        trade_date=trade_date,
        source_fingerprint="unit",
        generated_at=datetime(2026, 1, 5, tzinfo=KST),
        candidates=(weak_sector_candidate,),
        metadata={"active_symbol_count": 1, "candidate_pool_count": 1, "selection_count": 1, "frontier_symbol_count": 1},
    )
    cfg = KALCBConfig.from_mapping(
        {
            "kalcb.entry.routes": [
                {
                    "name": "sector_confirmed_first30",
                    "mode": "first30_open",
                    "priority": 0,
                    "context_min": {"session_sector_intraday_score_pct_mean": 50.0},
                }
            ],
            "kalcb.entry.rvol_threshold": 0.0,
            "kalcb.entry.cpr_threshold": 0.0,
            "kalcb.entry.momentum_score_min": 0,
            "kalcb.entry.entry_score_blocklist": [],
        }
    )
    result = None

    for bar in _or_bars(trade_date)[:6]:
        result = step_kalcb_core(state, bar, cfg, snapshot, KALCBPortfolioView(cash=1_000_000.0))

    assert result is not None
    assert not result.actions
    assert result.decisions[-1].reason == "entry_context_min:session_sector_intraday_score_pct_mean"
    assert result.decisions[-1].metadata["session_sector_intraday_score_pct_mean"] == pytest.approx(40.0)
    assert result.decisions[-1].metadata["session_sector_intraday_positive_share"] == pytest.approx(0.0)
    assert result.decisions[-1].metadata["session_sector_intraday_score_confirmed_share"] == pytest.approx(0.0)


def test_entry_route_context_exclude_can_block_candidate_sector():
    trade_date = date(2026, 1, 5)
    state = KALCBState()
    cfg = KALCBConfig.from_mapping(
        {
            "kalcb.entry.routes": [
                {
                    "name": "non_tech_first30",
                    "mode": "first30_open",
                    "priority": 0,
                    "context_exclude": {"sector": ["TECH"]},
                }
            ],
            "kalcb.entry.rvol_threshold": 0.0,
            "kalcb.entry.cpr_threshold": 0.0,
            "kalcb.entry.momentum_score_min": 0,
            "kalcb.entry.entry_score_blocklist": [],
        }
    )
    result = None

    for bar in _or_bars(trade_date)[:6]:
        result = step_kalcb_core(state, bar, cfg, _snapshot(trade_date), KALCBPortfolioView(cash=1_000_000.0))

    assert result is not None
    assert not result.actions
    assert result.decisions[-1].reason == "entry_context_exclude:sector"
    assert result.decisions[-1].metadata["sector"] == "TECH"


def test_dynamic_notional_cap_is_route_scoped_and_visible_in_entry_metadata():
    trade_date = date(2026, 1, 5)
    state = KALCBState()
    cfg = KALCBConfig.from_mapping(
        {
            "kalcb.entry.routes": [
                {
                    "name": "first30_primary",
                    "mode": "first30_open",
                    "priority": 0,
                    "dynamic_notional_enabled": True,
                    "dynamic_max_position_notional_pct": 0.20,
                    "dynamic_max_drawdown_pct": 0.05,
                    "dynamic_min_session_return_pct": 0.0,
                    "dynamic_max_open_positions": 4,
                }
            ],
            "kalcb.risk.max_position_notional_pct": 0.10,
            "kalcb.entry.rvol_threshold": 0.0,
            "kalcb.entry.cpr_threshold": 0.0,
            "kalcb.entry.momentum_score_min": 0,
            "kalcb.entry.entry_score_blocklist": [],
        }
    )
    snapshot = _snapshot(trade_date)
    portfolio = KALCBPortfolioView(cash=1_000_000.0, equity=1_000_000.0)
    result = None

    for bar in _or_bars(trade_date)[:6]:
        result = step_kalcb_core(state, bar, cfg, snapshot, portfolio)

    assert result is not None
    assert result.actions
    metadata = result.actions[0].metadata
    assert metadata["entry_route"] == "first30_primary"
    assert metadata["effective_max_position_notional_pct"] == pytest.approx(0.20)
    assert metadata["portfolio_drawdown_pct"] == pytest.approx(0.0)


def test_entry_route_sizing_multipliers_reduce_qty_and_emit_metadata():
    trade_date = date(2026, 1, 5)
    snapshot = _snapshot(trade_date)
    portfolio = KALCBPortfolioView(cash=1_000_000.0)
    common = {
        "kalcb.entry.rvol_threshold": 0.0,
        "kalcb.entry.cpr_threshold": 0.0,
        "kalcb.entry.momentum_score_min": 0,
        "kalcb.entry.entry_score_blocklist": [],
    }
    base_cfg = KALCBConfig.from_mapping(
        {
            **common,
            "kalcb.entry.routes": [{"name": "first30_primary", "mode": "first30_open", "priority": 0}],
        }
    )
    scaled_cfg = KALCBConfig.from_mapping(
        {
            **common,
            "kalcb.entry.routes": [
                {
                    "name": "first30_primary",
                    "mode": "first30_open",
                    "priority": 0,
                    "risk_mult": 0.5,
                    "notional_mult": 0.5,
                    "participation_mult": 0.5,
                }
            ],
        }
    )

    base_result = None
    base_state = KALCBState()
    for bar in _or_bars(trade_date)[:6]:
        base_result = step_kalcb_core(base_state, bar, base_cfg, snapshot, portfolio)

    scaled_result = None
    scaled_state = KALCBState()
    for bar in _or_bars(trade_date)[:6]:
        scaled_result = step_kalcb_core(scaled_state, bar, scaled_cfg, snapshot, portfolio)

    assert base_result is not None and scaled_result is not None
    assert base_result.actions and scaled_result.actions
    base_action = base_result.actions[0]
    scaled_action = scaled_result.actions[0]
    assert scaled_action.qty < base_action.qty
    assert scaled_action.metadata["entry_route_risk_mult"] == pytest.approx(0.5)
    assert scaled_action.metadata["entry_route_notional_mult"] == pytest.approx(0.5)
    assert scaled_action.metadata["entry_route_participation_mult"] == pytest.approx(0.5)


def test_entry_route_session_trade_cap_blocks_later_symbols():
    trade_date = date(2026, 1, 5)
    state = KALCBState()
    cfg = KALCBConfig.from_mapping(
        {
            "kalcb.entry.routes": [
                {"name": "first30_capped", "mode": "first30_open", "priority": 0, "max_session_trades": 1}
            ],
            "kalcb.entry.rvol_threshold": 0.0,
            "kalcb.entry.cpr_threshold": 0.0,
            "kalcb.entry.momentum_score_min": 0,
            "kalcb.entry.entry_score_blocklist": [],
        }
    )
    snapshot = KALCBDailySnapshot(
        trade_date=trade_date,
        source_fingerprint="unit",
        generated_at=datetime(2026, 1, 5, tzinfo=KST),
        candidates=(_candidate_for_symbol(trade_date, "005930"), _candidate_for_symbol(trade_date, "000660")),
    )
    portfolio = KALCBPortfolioView(cash=1_000_000.0)

    first_result = None
    for bar in _or_bars(trade_date)[:6]:
        first_result = step_kalcb_core(state, bar, cfg, snapshot, portfolio)

    second_result = None
    for bar in [replace(item, symbol="000660") for item in _or_bars(trade_date)[:6]]:
        second_result = step_kalcb_core(state, bar, cfg, snapshot, portfolio)

    assert first_result is not None and first_result.actions
    assert first_result.actions[0].metadata["entry_route_max_session_trades"] == 1
    assert first_result.actions[0].metadata["entry_route_session_count_before"] == 0
    assert second_result is not None
    assert not second_result.actions
    assert second_result.decisions[-1].reason == "entry_route_session_limit"


def test_entry_route_session_trade_cap_refunds_retryable_route_defer():
    trade_date = date(2026, 1, 5)
    state = KALCBState()
    cfg = KALCBConfig.from_mapping(
        {
            "kalcb.entry.routes": [
                {"name": "first30_capped", "mode": "first30_open", "priority": 0, "max_session_trades": 1}
            ],
            "kalcb.entry.rvol_threshold": 0.0,
            "kalcb.entry.cpr_threshold": 0.0,
            "kalcb.entry.momentum_score_min": 0,
            "kalcb.entry.entry_score_blocklist": [],
        }
    )
    snapshot = KALCBDailySnapshot(
        trade_date=trade_date,
        source_fingerprint="unit",
        generated_at=datetime(2026, 1, 5, tzinfo=KST),
        candidates=(_candidate_for_symbol(trade_date, "005930"), _candidate_for_symbol(trade_date, "000660")),
    )
    portfolio = KALCBPortfolioView(cash=1_000_000.0)

    first_result = None
    for bar in _or_bars(trade_date)[:6]:
        first_result = step_kalcb_core(state, bar, cfg, snapshot, portfolio)
    assert first_result is not None and first_result.actions
    first_action = first_result.actions[0]

    on_kalcb_order_update(
        state,
        KALCBOrderUpdateEvent(
            order_id="route-reject:1",
            symbol="005930",
            status="DEFERRED",
            timestamp=datetime(2026, 1, 5, 9, 31, tzinfo=KST),
            role="ENTRY",
            reason="OMS unreachable",
            metadata=dict(first_action.metadata),
        ),
    )

    second_result = None
    for bar in [replace(item, symbol="000660") for item in _or_bars(trade_date)[:6]]:
        second_result = step_kalcb_core(state, bar, cfg, snapshot, portfolio)

    assert second_result is not None
    assert second_result.actions
    assert second_result.actions[0].metadata["entry_route_session_count_before"] == 0


def test_secondary_entry_route_can_trigger_after_primary_anchor_rejects():
    trade_date = date(2026, 1, 5)
    state = KALCBState()
    cfg = KALCBConfig.from_mapping(
        {
            "kalcb.entry.routes": [
                {"name": "first30_primary", "mode": "first30_open", "priority": 0, "min_quality_votes": 99},
                {"name": "secondary_post_or", "mode": "post_or_momentum", "priority": 10, "after_bar": 0, "max_signal_bars": 2, "min_or_position": 0.45},
            ],
            "kalcb.entry.rvol_threshold": 0.0,
            "kalcb.entry.cpr_threshold": 0.0,
            "kalcb.entry.momentum_score_min": 0,
            "kalcb.entry.entry_score_blocklist": [],
        }
    )
    snapshot = _snapshot(trade_date)
    portfolio = KALCBPortfolioView(cash=1_000_000.0)
    first30_result = None

    for bar in _or_bars(trade_date)[:6]:
        first30_result = step_kalcb_core(state, bar, cfg, snapshot, portfolio)

    assert first30_result is not None
    assert not first30_result.actions
    assert first30_result.decisions[-1].reason == "entry_quality_votes"
    assert first30_result.decisions[-1].metadata["entry_route"] == "first30_primary"

    result = step_kalcb_core(state, _or_bars(trade_date)[6], cfg, snapshot, portfolio)

    assert result.actions
    assert isinstance(result.actions[0], SubmitEntry)
    assert result.actions[0].metadata["entry_route"] == "secondary_post_or"
    assert result.actions[0].metadata["entry_type"] == EntryType.POST_OR_MOMENTUM.value


def test_deferred_frontier_branch_requires_completed_h6_path_proof():
    trade_date = date(2026, 1, 5)
    state = KALCBState()
    candidate = replace(
        _candidate(trade_date),
        average_30m_volume=6_000.0,
        expected_5m_volume=1_000.0,
        metadata={
            "frontier_initial_active": False,
            "frontier_rank": 10,
            "frontier_role": "frontier_shadow",
            "daily_acceleration_5v20": 0.06,
        },
    )
    snapshot = KALCBDailySnapshot(
        trade_date=trade_date,
        source_fingerprint="unit",
        generated_at=datetime(2026, 1, 5, tzinfo=KST),
        candidates=(candidate,),
    )
    cfg = KALCBConfig.from_mapping(
        {
            "kalcb.entry.routes": [
                {"name": "first30_primary", "mode": "first30_open", "priority": 0, "min_quality_votes": 99},
                {
                    "name": "frontier_pathproof",
                    "mode": "deferred_continuation",
                    "priority": 20,
                    "after_bar": 4,
                    "max_signal_bars": 18,
                    "require_initial_active": False,
                    "max_frontier_rank": 12,
                    "risk_mult": 0.05,
                    "notional_mult": 0.05,
                    "context_min": {
                        "first30_gap_relvol": 0.01,
                        "first30_gap_retention_ratio": 0.50,
                        "daily_acceleration_5v20": 0.04,
                        "h3_current_r": 2.0,
                        "h6_current_r": 5.0,
                    },
                },
            ],
            "kalcb.exit.stop_mode": "fixed_pct",
            "kalcb.exit.stop_pct": 0.003,
            "kalcb.entry.rvol_threshold": 0.0,
            "kalcb.entry.cpr_threshold": 0.0,
            "kalcb.entry.momentum_score_min": 0,
            "kalcb.entry.entry_score_blocklist": [],
        }
    )
    portfolio = KALCBPortfolioView(cash=1_000_000.0)
    bars = _pathproof_bars(trade_date)

    first30_result = None
    for bar in bars[:6]:
        first30_result = step_kalcb_core(state, bar, cfg, snapshot, portfolio)

    assert first30_result is not None
    assert not first30_result.actions
    assert first30_result.decisions[-1].reason == "entry_quality_votes"

    pre_h6_result = None
    for bar in bars[6:11]:
        pre_h6_result = step_kalcb_core(state, bar, cfg, snapshot, portfolio)

    assert pre_h6_result is not None
    assert not pre_h6_result.actions
    assert pre_h6_result.decisions[-1].reason == "entry_context_min:h6_current_r"
    assert pre_h6_result.decisions[-1].metadata["entry_path_completed_bars"] == 5
    assert pre_h6_result.decisions[-1].metadata["h3_current_r"] == pytest.approx(3.0)
    assert "h6_current_r" not in pre_h6_result.decisions[-1].metadata

    accepted = step_kalcb_core(state, bars[11], cfg, snapshot, portfolio)

    assert accepted.actions
    metadata = accepted.actions[0].metadata
    assert metadata["entry_route"] == "frontier_pathproof"
    assert metadata["entry_route_mode"] == "deferred_continuation"
    assert metadata["entry_route_risk_mult"] == pytest.approx(0.05)
    assert metadata["frontier_initial_active"] is False
    assert metadata["entry_path_completed_bars"] == 6
    assert metadata["entry_path_reference"] == "first_post_first30_open_fixed_stop_pct"
    assert metadata["h3_current_r"] == pytest.approx(3.0)
    assert metadata["h6_current_r"] == pytest.approx(7.0)


def test_first30_source_gate_rejects_low_relative_volume():
    trade_date = date(2026, 1, 5)
    state = KALCBState()
    cfg = KALCBConfig.from_mapping(
        {
            "kalcb.entry.plan_mode": "first30_open",
            "kalcb.entry.min_first30_rel_volume": 1.0,
            "kalcb.entry.rvol_threshold": 0.0,
            "kalcb.entry.cpr_threshold": 0.0,
            "kalcb.entry.momentum_score_min": 0,
            "kalcb.entry.entry_score_blocklist": [],
        }
    )
    snapshot = _snapshot(trade_date)
    portfolio = KALCBPortfolioView(cash=1_000_000.0)
    result = None

    for bar in _or_bars(trade_date)[:6]:
        result = step_kalcb_core(state, bar, cfg, snapshot, portfolio)

    assert result is not None
    assert not result.actions
    assert result.decisions[-1].decision_code == "entry_rejected"
    assert result.decisions[-1].reason == "entry_first30_rel_volume"
    assert result.decisions[-1].metadata["first30_rel_volume"] == pytest.approx(0.0615)


def test_initial_active_gate_blocks_frontier_shadow_candidate():
    trade_date = date(2026, 1, 5)
    state = KALCBState()
    shadow = replace(
        _candidate(trade_date),
        metadata={"frontier_initial_active": False, "frontier_rank": 12, "frontier_selection_score": 0.0},
    )
    snapshot = KALCBDailySnapshot(
        trade_date=trade_date,
        source_fingerprint="unit",
        generated_at=datetime(2026, 1, 5, tzinfo=KST),
        candidates=(shadow,),
    )
    cfg = KALCBConfig.from_mapping(
        {
            "kalcb.entry.plan_mode": "first30_open",
            "kalcb.entry.require_initial_active": True,
            "kalcb.entry.rvol_threshold": 0.0,
            "kalcb.entry.cpr_threshold": 0.0,
            "kalcb.entry.momentum_score_min": 0,
            "kalcb.entry.entry_score_blocklist": [],
        }
    )
    result = None

    for bar in _or_bars(trade_date)[:6]:
        result = step_kalcb_core(state, bar, cfg, snapshot, KALCBPortfolioView(cash=1_000_000.0))

    assert result is not None
    assert not result.actions
    assert result.decisions[-1].decision_code == "entry_rejected"
    assert result.decisions[-1].reason == "entry_not_initial_active"


def test_entry_quality_vote_gate_rejects_weak_proof_stack():
    trade_date = date(2026, 1, 5)
    state = KALCBState()
    weak = replace(
        _candidate(trade_date),
        flow_score=-0.2,
        accumulation_score=-0.2,
        metadata={"frontier_rank": 20},
    )
    snapshot = KALCBDailySnapshot(
        trade_date=trade_date,
        source_fingerprint="unit",
        generated_at=datetime(2026, 1, 5, tzinfo=KST),
        candidates=(weak,),
    )
    cfg = KALCBConfig.from_mapping(
        {
            "kalcb.entry.plan_mode": "first30_open",
            "kalcb.entry.min_quality_votes": 2,
            "kalcb.entry.quality_min_first30_signal_cpr": 0.75,
            "kalcb.entry.quality_min_first30_rel_volume": 1.0,
            "kalcb.entry.quality_min_flow_score": 0.0,
            "kalcb.entry.quality_min_accumulation_score": 0.0,
            "kalcb.entry.quality_max_frontier_rank": 12,
            "kalcb.entry.rvol_threshold": 0.0,
            "kalcb.entry.cpr_threshold": 0.0,
            "kalcb.entry.momentum_score_min": 0,
            "kalcb.entry.entry_score_blocklist": [],
        }
    )
    result = None

    for bar in _or_bars(trade_date)[:6]:
        result = step_kalcb_core(state, bar, cfg, snapshot, KALCBPortfolioView(cash=1_000_000.0))

    assert result is not None
    assert not result.actions
    assert result.decisions[-1].decision_code == "entry_rejected"
    assert result.decisions[-1].reason == "entry_quality_votes"
    assert result.decisions[-1].metadata["entry_quality_votes"] == 1
    assert result.decisions[-1].metadata["entry_quality_required_votes"] == 2


def test_entry_quality_vote_gate_accepts_mixed_independent_proofs():
    trade_date = date(2026, 1, 5)
    state = KALCBState()
    strong = replace(
        _candidate(trade_date),
        flow_score=0.2,
        accumulation_score=0.1,
        metadata={"frontier_rank": 8},
    )
    snapshot = KALCBDailySnapshot(
        trade_date=trade_date,
        source_fingerprint="unit",
        generated_at=datetime(2026, 1, 5, tzinfo=KST),
        candidates=(strong,),
    )
    cfg = KALCBConfig.from_mapping(
        {
            "kalcb.entry.plan_mode": "first30_open",
            "kalcb.entry.min_quality_votes": 4,
            "kalcb.entry.quality_min_first30_signal_cpr": 0.75,
            "kalcb.entry.quality_min_first30_rel_volume": 1.0,
            "kalcb.entry.quality_min_flow_score": 0.0,
            "kalcb.entry.quality_min_accumulation_score": 0.0,
            "kalcb.entry.quality_max_frontier_rank": 12,
            "kalcb.entry.rvol_threshold": 0.0,
            "kalcb.entry.cpr_threshold": 0.0,
            "kalcb.entry.momentum_score_min": 0,
            "kalcb.entry.entry_score_blocklist": [],
        }
    )
    result = None

    for bar in _or_bars(trade_date)[:6]:
        result = step_kalcb_core(state, bar, cfg, snapshot, KALCBPortfolioView(cash=1_000_000.0))

    assert result is not None
    assert result.actions
    assert isinstance(result.actions[0], SubmitEntry)
    assert result.actions[0].metadata["entry_quality_votes"] == 4
    assert result.actions[0].metadata["entry_quality_required_votes"] == 4


def test_post_or_momentum_plan_uses_completed_0930_signal_bar():
    trade_date = date(2026, 1, 5)
    state = KALCBState()
    cfg = KALCBConfig.from_mapping(
        {
            "kalcb.entry.plan_mode": "post_or_momentum",
            "kalcb.entry.max_signal_bars": 1,
            "kalcb.entry.min_or_position": 0.45,
            "kalcb.entry.rvol_threshold": 0.0,
            "kalcb.entry.cpr_threshold": 0.0,
            "kalcb.entry.momentum_score_min": 0,
            "kalcb.entry.entry_score_blocklist": [],
        }
    )
    snapshot = _snapshot(trade_date)
    portfolio = KALCBPortfolioView(cash=1_000_000.0)

    for bar in _or_bars(trade_date)[:6]:
        step_kalcb_core(state, bar, cfg, snapshot, portfolio)
    result = step_kalcb_core(state, _or_bars(trade_date)[6], cfg, snapshot, portfolio)

    assert result.actions
    assert result.actions[0].metadata["entry_type"] == EntryType.POST_OR_MOMENTUM.value
    assert result.actions[0].metadata["signal_bar"].endswith("09:30:00+09:00")


def test_pullback_acceptance_plan_uses_shared_reclaim_state():
    trade_date = date(2026, 1, 5)
    state = KALCBState()
    cfg = KALCBConfig.from_mapping(
        {
            "kalcb.entry.plan_mode": "pullback_acceptance",
            "kalcb.entry.max_signal_bars": 2,
            "kalcb.entry.min_reclaim_ret": 0.001,
            "kalcb.entry.rvol_threshold": 0.0,
            "kalcb.entry.cpr_threshold": 0.0,
            "kalcb.entry.momentum_score_min": 0,
            "kalcb.entry.entry_score_blocklist": [],
        }
    )
    snapshot = _snapshot(trade_date)
    portfolio = KALCBPortfolioView(cash=1_000_000.0)

    for bar in _or_bars(trade_date)[:6]:
        step_kalcb_core(state, bar, cfg, snapshot, portfolio)

    pullback = MarketBar(
        "005930",
        datetime(2026, 1, 5, 9, 30, tzinfo=KST),
        "5m",
        100.7,
        100.9,
        99.9,
        100.2,
        2000,
    )
    reclaim = MarketBar(
        "005930",
        datetime(2026, 1, 5, 9, 35, tzinfo=KST),
        "5m",
        100.3,
        101.2,
        100.1,
        101.0,
        4000,
    )

    first = step_kalcb_core(state, pullback, cfg, snapshot, portfolio)
    result = step_kalcb_core(state, reclaim, cfg, snapshot, portfolio)

    assert not first.actions
    assert state.symbol_state("005930").touched_vwap is True
    assert result.actions
    assert result.actions[0].metadata["entry_type"] == EntryType.PULLBACK_ACCEPTANCE.value


def test_pullback_acceptance_can_use_campaign_box_high_level_source():
    trade_date = date(2026, 1, 5)
    state = KALCBState()
    cfg = KALCBConfig.from_mapping(
        {
            "kalcb.entry.plan_mode": "pullback_acceptance",
            "kalcb.entry.reclaim_level_source": "campaign_box_high",
            "kalcb.entry.max_signal_bars": 2,
            "kalcb.entry.min_reclaim_ret": 0.001,
            "kalcb.entry.rvol_threshold": 0.0,
            "kalcb.entry.cpr_threshold": 0.0,
            "kalcb.entry.momentum_score_min": 0,
            "kalcb.entry.entry_score_blocklist": [],
        }
    )
    candidate = replace(_candidate(trade_date), metadata={"campaign_box_high": 101.0, "structural_campaign_score": 7.0})
    snapshot = KALCBDailySnapshot(
        trade_date=trade_date,
        source_fingerprint="unit",
        generated_at=datetime(2026, 1, 5, tzinfo=KST),
        candidates=(candidate,),
    )
    portfolio = KALCBPortfolioView(cash=1_000_000.0)

    for bar in _or_bars(trade_date)[:6]:
        step_kalcb_core(state, bar, cfg, snapshot, portfolio)
    step_kalcb_core(
        state,
        MarketBar("005930", datetime(2026, 1, 5, 9, 30, tzinfo=KST), "5m", 100.9, 100.95, 100.8, 100.85, 2000),
        cfg,
        snapshot,
        portfolio,
    )
    result = step_kalcb_core(
        state,
        MarketBar("005930", datetime(2026, 1, 5, 9, 35, tzinfo=KST), "5m", 100.85, 101.5, 100.9, 101.3, 4000),
        cfg,
        snapshot,
        portfolio,
    )

    assert result.actions
    metadata = result.actions[0].metadata
    assert metadata["entry_reclaim_level_source"] == "campaign_box_high"
    assert metadata["entry_reclaim_level"] == pytest.approx(101.0)
    assert metadata["entry_reclaim_touch_key"] == "pullback_acceptance:campaign_box_high"
    assert state.symbol_state("005930").touched_reclaim_levels["pullback_acceptance:campaign_box_high"] is True
    assert metadata["entry_type"] == EntryType.PULLBACK_ACCEPTANCE.value


def test_campaign_breakout_level_can_require_two_reclaim_closes():
    trade_date = date(2026, 1, 5)
    state = KALCBState()
    cfg = KALCBConfig.from_mapping(
        {
            "kalcb.entry.plan_mode": "or_high_reclaim",
            "kalcb.entry.reclaim_level_source": "campaign_breakout_level",
            "kalcb.entry.min_reclaim_closes": 2,
            "kalcb.entry.max_signal_bars": 3,
            "kalcb.entry.min_reclaim_ret": 0.0,
            "kalcb.entry.rvol_threshold": 0.0,
            "kalcb.entry.cpr_threshold": 0.0,
            "kalcb.entry.momentum_score_min": 0,
            "kalcb.entry.entry_score_blocklist": [],
        }
    )
    candidate = replace(_candidate(trade_date), metadata={"campaign_breakout_level": 101.0, "structural_campaign_score": 7.0})
    snapshot = KALCBDailySnapshot(
        trade_date=trade_date,
        source_fingerprint="unit",
        generated_at=datetime(2026, 1, 5, tzinfo=KST),
        candidates=(candidate,),
    )
    portfolio = KALCBPortfolioView(cash=1_000_000.0)

    for bar in _or_bars(trade_date)[:6]:
        step_kalcb_core(state, bar, cfg, snapshot, portfolio)
    step_kalcb_core(state, MarketBar("005930", datetime(2026, 1, 5, 9, 30, tzinfo=KST), "5m", 100.9, 100.95, 100.8, 100.85, 2000), cfg, snapshot, portfolio)
    first_close = step_kalcb_core(state, MarketBar("005930", datetime(2026, 1, 5, 9, 35, tzinfo=KST), "5m", 100.9, 101.5, 100.9, 101.2, 4000), cfg, snapshot, portfolio)
    second_close = step_kalcb_core(state, MarketBar("005930", datetime(2026, 1, 5, 9, 40, tzinfo=KST), "5m", 101.2, 101.7, 101.1, 101.5, 4000), cfg, snapshot, portfolio)

    assert not first_close.actions
    assert second_close.actions
    metadata = second_close.actions[0].metadata
    assert metadata["entry_reclaim_level_source"] == "campaign_breakout_level"
    assert metadata["entry_reclaim_min_closes"] == 2
    assert metadata["entry_reclaim_close_count"] == 2


def test_timer_flatten_is_shared_neutral_action_at_flatten_time():
    trade_date = date(2026, 1, 5)
    state = KALCBState()
    symbol_state = state.symbol_state("005930")
    symbol_state.session_date = trade_date
    symbol_state.stage = SymbolStage.IN_POSITION
    symbol_state.position = KALCBPositionState(
        symbol="005930",
        qty_entry=10,
        qty_open=10,
        entry_price=100.0,
        entry_time=datetime(2026, 1, 5, 9, 35, tzinfo=KST),
        initial_stop=95.0,
        current_stop=95.0,
        risk_per_share=5.0,
        entry_type=EntryType.FIRST30_OPEN.value,
        momentum_score=3,
    )
    cfg = KALCBConfig.from_mapping({"kalcb.session.flatten_time": "15:20", "kalcb.carry.mode": "off"})

    early = on_kalcb_timer(state, datetime(2026, 1, 5, 15, 15, tzinfo=KST), cfg)
    result = on_kalcb_timer(state, datetime(2026, 1, 5, 15, 20, tzinfo=KST), cfg)

    assert not early.actions
    assert result.actions
    assert isinstance(result.actions[0], FlattenPosition)
    assert result.actions[0].reason == "eod_flatten"
    assert state.symbol_state("005930").position.exit_in_flight is True


def _snapshot(trade_date: date) -> KALCBDailySnapshot:
    return KALCBDailySnapshot(
        trade_date=trade_date,
        source_fingerprint="unit",
        generated_at=datetime(2026, 1, 5, tzinfo=KST),
        candidates=(_candidate(trade_date),),
    )


def _candidate(trade_date: date) -> KALCBDailyCandidate:
    return _candidate_for_symbol(trade_date, "005930")


def _candidate_for_symbol(trade_date: date, symbol: str) -> KALCBDailyCandidate:
    return KALCBDailyCandidate(
        symbol=symbol,
        trade_date=trade_date,
        prior_day_high=999.0,
        prior_day_low=95.0,
        prior_day_close=98.0,
        daily_atr=5.0,
        expected_5m_volume=1000.0,
        average_30m_volume=100000.0,
        sector="TECH",
        regime_tier="A",
        tradable=True,
        source_fingerprint="unit",
    )


def _or_bars(trade_date: date) -> list[MarketBar]:
    start = datetime.combine(trade_date, datetime.min.time(), tzinfo=KST).replace(hour=9)
    rows = [
        (100.0, 100.8, 100.0, 100.4, 1000),
        (100.4, 101.0, 100.1, 100.6, 1000),
        (100.6, 100.9, 100.2, 100.5, 900),
        (100.5, 100.7, 100.1, 100.3, 950),
        (100.3, 100.8, 100.0, 100.7, 1100),
        (100.7, 100.9, 100.2, 100.8, 1200),
        (101.0, 102.4, 100.8, 102.0, 5000),
    ]
    return [
        MarketBar("005930", start + timedelta(minutes=5 * index), "5m", open_, high, low, close, volume)
        for index, (open_, high, low, close, volume) in enumerate(rows)
    ]


def _pathproof_bars(trade_date: date) -> list[MarketBar]:
    bars = list(_or_bars(trade_date)[:6])
    start = datetime.combine(trade_date, datetime.min.time(), tzinfo=KST).replace(hour=9, minute=30)
    rows = [
        (101.0, 102.5, 100.8, 102.0, 5000),
        (102.0, 103.5, 101.8, 103.0, 5000),
        (103.0, 104.5, 102.8, 104.0, 5000),
        (104.0, 105.5, 103.8, 105.0, 5000),
        (105.0, 106.5, 104.8, 106.0, 5000),
        (106.0, 108.5, 105.8, 108.0, 5000),
    ]
    bars.extend(
        MarketBar("005930", start + timedelta(minutes=5 * index), "5m", open_, high, low, close, volume)
        for index, (open_, high, low, close, volume) in enumerate(rows)
    )
    return bars


def _borderline_cpr_breakout(trade_date: date, *, volume: float = 5000.0, adx: float = 0.0) -> MarketBar:
    start = datetime.combine(trade_date, datetime.min.time(), tzinfo=KST).replace(hour=9, minute=30)
    return MarketBar(
        "005930",
        start,
        "5m",
        100.9,
        102.0,
        100.0,
        101.14,
        volume,
        metadata={"adx": adx},
    )
