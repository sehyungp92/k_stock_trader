from __future__ import annotations

from datetime import date, datetime, timedelta

from strategy_common.clock import KST
from strategy_common.market import MarketBar
from strategy_olr.config import OLRConfig
from strategy_olr.core.core_models import OLROrderUpdateEvent, OLRPortfolioView
from strategy_olr.core.logic import on_olr_order_update, step_olr_core
from strategy_olr.core.state import OLRState
from strategy_olr.execution import OLREntryPlan, OLRExitPlan, simulate_olr_trade
from strategy_olr.models import OLRDailyCandidate, OLRDailySnapshot


def test_olr_completed_bar_entry_fills_next_bar_after_1430() -> None:
    trade_date = date(2026, 2, 2)
    candidate = _candidate("005930", trade_date)
    entry = OLREntryPlan(
        name="confirm",
        mode="confirm_next_bar",
        max_signal_bars=1,
        min_bar_ret=0.01,
        min_close_location=0.5,
    )
    outcome = simulate_olr_trade(
        trade_date,
        "005930",
        (
            _bar("005930", trade_date, 14, 25, 100.0, 101.0, 99.0, 100.0),
            _bar("005930", trade_date, 14, 30, 100.0, 103.0, 99.5, 102.0),
            _bar("005930", trade_date, 14, 35, 103.0, 105.0, 102.5, 104.0),
        ),
        (
            _bar("005930", trade_date + timedelta(days=1), 9, 0, 105.0, 108.0, 104.0, 107.0),
            _bar("005930", trade_date + timedelta(days=1), 15, 30, 107.0, 111.0, 106.0, 110.0),
        ),
        candidate,
        entry,
        OLRExitPlan(name="next_close", mode="next_close"),
        OLRConfig(auction_limit_offset_bps=500.0),
    )

    assert outcome is not None
    assert outcome.entry_reason == "confirm_next_bar"
    assert outcome.entry_time.astimezone(KST).time().isoformat(timespec="minutes") == "14:35"
    assert outcome.entry_price == 103.0
    assert outcome.exit_reason == "next_close"


def test_olr_close_auction_is_resting_entry_after_decision() -> None:
    trade_date = date(2026, 2, 2)
    outcome = simulate_olr_trade(
        trade_date,
        "005930",
        (
            _bar("005930", trade_date, 14, 25, 100.0, 101.0, 99.0, 100.0),
            _bar("005930", trade_date, 14, 30, 100.0, 103.0, 99.5, 102.0),
            _bar("005930", trade_date, 15, 15, 102.0, 104.0, 101.0, 103.0),
            _bar("005930", trade_date, 15, 30, 103.0, 106.0, 102.0, 105.0),
        ),
        (_bar("005930", trade_date + timedelta(days=1), 15, 30, 105.0, 110.0, 104.0, 109.0),),
        _candidate("005930", trade_date),
        OLREntryPlan(name="close", mode="close_auction"),
        OLRExitPlan(name="next_close", mode="next_close"),
        OLRConfig(auction_limit_offset_bps=500.0),
    )

    assert outcome is not None
    assert outcome.entry_reason == "close_auction"
    assert outcome.entry_time.astimezone(KST).time().isoformat(timespec="minutes") == "15:30"
    assert outcome.entry_price == 105.0
    assert outcome.exit_price == 109.0


def test_olr_close_auction_proxy_respects_bounded_limit_nonfill() -> None:
    trade_date = date(2026, 2, 2)
    outcome = simulate_olr_trade(
        trade_date,
        "005930",
        (
            _bar("005930", trade_date, 14, 25, 100.0, 101.0, 99.0, 100.0),
            _bar("005930", trade_date, 14, 30, 100.0, 103.0, 99.5, 102.0),
            _bar("005930", trade_date, 15, 30, 103.0, 106.0, 102.0, 105.0),
        ),
        (_bar("005930", trade_date + timedelta(days=1), 15, 30, 105.0, 110.0, 104.0, 109.0),),
        _candidate("005930", trade_date),
        OLREntryPlan(name="close", mode="close_auction"),
        OLRExitPlan(name="next_close", mode="next_close"),
        OLRConfig(auction_limit_offset_bps=0.0, slippage_bps=0.0),
    )

    assert outcome is None


def test_olr_retries_entry_after_deferred_route_update() -> None:
    trade_date = date(2026, 2, 2)
    state = OLRState()
    config = OLRConfig.from_mapping(
        {
            "olr.overnight.slot_count": 1,
            "olr.allocation.min_selected": 1,
            "olr.trade_plan.entry": {"name": "decision", "mode": "decision_next_open"},
        }
    )
    snapshot = OLRDailySnapshot(
        trade_date=trade_date,
        candidates=(_candidate("005930", trade_date),),
        source_fingerprint="retry-deferred",
        generated_at=datetime.combine(trade_date, datetime.min.time(), tzinfo=KST).replace(hour=8, minute=50),
    )
    portfolio = OLRPortfolioView(cash=1_000_000.0, equity=1_000_000.0)
    first = step_olr_core(
        state,
        _bar("005930", trade_date, 14, 30, 100.0, 101.0, 99.0, 100.0),
        config,
        snapshot,
        portfolio,
    )
    assert len(first.actions) == 1
    assert state.symbol_state("005930").entry_attempted is True

    on_olr_order_update(
        state,
        OLROrderUpdateEvent(
            order_id="prov-entry",
            symbol="005930",
            status="DEFERRED",
            timestamp=datetime.combine(trade_date, datetime.min.time(), tzinfo=KST).replace(hour=14, minute=31),
            side="BUY",
            reason="Equity not yet loaded - reconciliation pending",
        ),
        config,
    )
    assert state.symbol_state("005930").entry_attempted is False

    second = step_olr_core(
        state,
        _bar("005930", trade_date, 14, 35, 100.0, 101.0, 99.0, 100.5),
        config,
        snapshot,
        portfolio,
    )

    assert len(second.actions) == 1


def test_olr_mfe_fade_exit_waits_for_profit_then_exits_next_bar() -> None:
    trade_date = date(2026, 2, 2)
    next_date = trade_date + timedelta(days=1)
    outcome = simulate_olr_trade(
        trade_date,
        "005930",
        (
            _bar("005930", trade_date, 14, 25, 100.0, 101.0, 99.0, 100.0),
            _bar("005930", trade_date, 14, 30, 100.0, 103.0, 99.5, 102.0),
            _bar("005930", trade_date, 14, 35, 103.0, 105.0, 102.5, 104.0),
        ),
        (
            _bar("005930", next_date, 9, 0, 104.0, 110.0, 103.0, 109.0),
            _bar("005930", next_date, 9, 5, 109.0, 109.5, 106.0, 106.5),
            _bar("005930", next_date, 9, 10, 106.0, 106.5, 105.0, 105.5),
            _bar("005930", next_date, 15, 30, 105.0, 105.5, 103.0, 103.5),
        ),
        _candidate("005930", trade_date),
        OLREntryPlan(name="confirm", mode="confirm_next_bar", max_signal_bars=1, min_bar_ret=0.01, min_close_location=0.5),
        OLRExitPlan(name="mfe_fade", mode="managed", hard_stop_enabled=False, mfe_fade_start_r=1.5, mfe_fade_gap_r=1.0, mfe_fade_floor_r=0.0),
        OLRConfig(auction_limit_offset_bps=500.0, slippage_bps=0.0),
    )

    assert outcome is not None
    assert outcome.exit_reason == "mfe_fade"
    assert outcome.exit_time.astimezone(KST).time().isoformat(timespec="minutes") == "09:10"


def _candidate(symbol: str, trade_date: date) -> OLRDailyCandidate:
    return OLRDailyCandidate(
        symbol=symbol,
        trade_date=trade_date,
        prior_day_high=103.0,
        prior_day_low=97.0,
        prior_day_close=100.0,
        daily_atr=3.0,
        expected_5m_volume=100.0,
        average_30m_volume=600.0,
        rank=1,
        selection_score=100.0,
    )


def _bar(symbol: str, day: date, hour: int, minute: int, open_: float, high: float, low: float, close: float) -> MarketBar:
    return MarketBar(
        symbol=symbol,
        timestamp=datetime.combine(day, datetime.min.time(), tzinfo=KST).replace(hour=hour, minute=minute),
        timeframe="5m",
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=1000.0,
        is_completed=True,
        source="unit",
    )
