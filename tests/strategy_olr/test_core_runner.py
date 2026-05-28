from __future__ import annotations

from dataclasses import asdict
from datetime import date, datetime, time, timedelta

import pytest

from backtests.engine.sim_broker import BrokerCosts, SimBroker
from backtests.strategies.olr.runner import compile_olr_replay_bundle, run_olr_backtest
from strategy_common.actions import SubmitEntry
from strategy_common.clock import KST
from strategy_common.market import MarketBar
from strategy_olr.config import OLRConfig
from strategy_olr.execution import OLREntryPlan, OLRExitPlan
from strategy_olr.models import OLRDailyCandidate, OLRDailySnapshot
from strategy_olr.research import afternoon_selection_from_snapshot


def test_simbroker_close_auction_fills_only_on_close_bar():
    broker = SimBroker(initial_equity=1_000_000.0, costs=BrokerCosts(commission_bps=0.0, tax_bps_on_sell=0.0, slippage_bps=0.0))
    submitted = datetime(2026, 1, 5, 14, 30, tzinfo=KST)
    action = SubmitEntry(
        "OLR",
        "005930",
        10,
        "CLOSE_AUCTION",
        101.0,
        None,
        "close_auction_entry",
        {"auction_fill_time": "15:30"},
    )
    broker.submit(action, submitted)

    assert broker.process_bar(_bar("005930", date(2026, 1, 5), time(14, 35), 100.0, 100.5, 99.5, 100.2)) == []
    assert broker.process_bar(_bar("005930", date(2026, 1, 5), time(15, 20), 100.0, 100.5, 99.5, 100.2)) == []
    fills = broker.process_bar(_bar("005930", date(2026, 1, 5), time(15, 30), 100.0, 100.5, 99.5, 100.2))

    assert len(fills) == 1
    assert fills[0].timestamp.time().replace(second=0, microsecond=0) == time(15, 30)
    assert broker.same_bar_fill_violations == 0


def test_simbroker_close_auction_expires_on_deterministic_nonfill():
    broker = SimBroker(initial_equity=1_000_000.0, costs=BrokerCosts(commission_bps=0.0, tax_bps_on_sell=0.0, slippage_bps=0.0))
    broker.submit(
        SubmitEntry(
            "OLR",
            "005930",
            10,
            "CLOSE_AUCTION",
            101.0,
            None,
            "close_auction_entry",
            {"auction_fill_time": "15:30", "auction_nonfill_rate": 1.0, "auction_nonfill_key": "always"},
        ),
        datetime(2026, 1, 5, 14, 30, tzinfo=KST),
    )

    fills = broker.process_bar(_bar("005930", date(2026, 1, 5), time(15, 30), 100.0, 100.5, 99.5, 100.2))

    assert fills == []
    assert broker.auction_nonfill_count == 1
    assert len(broker.orders) == 0


def test_olr_synthetic_backtest_uses_core_simbroker_and_reports_mtm():
    result = run_olr_backtest({"capability_level": "synthetic", "initial_equity": 10_000_000.0}, {})

    assert result.metrics["total_trades"] == 1.0
    assert result.metrics["official_mtm_net_return_pct"] == pytest.approx(result.metrics["net_return_pct"])
    assert result.metrics["same_bar_fill_count"] == 0.0
    assert result.metrics["auction_order_count"] == 2.0
    assert result.metrics["auction_nonfill_count"] == 0.0
    assert result.metrics["forced_replay_close_count"] == 0.0
    assert result.metrics["final_equity"] == pytest.approx(result.replay_result.equity_curve[-1])
    assert result.metrics["end_open_position_count"] == result.metrics["open_position_count"] == 0.0
    assert result.metrics["net_return_pct_basis"] == "closed_trade_net_pnl_over_initial_equity"
    assert result.metrics["official_performance"] is False


def test_olr_runner_honors_custom_compiled_snapshot_bundle():
    bundle = compile_olr_replay_bundle()
    result = run_olr_backtest({"capability_level": "compiled", "olr.cost.slippage_bps": 0.0}, {}, replay_bundle=bundle)

    assert result.metrics["replay_mode"] == "olr_core_simbroker"
    assert result.metrics["official_metric_basis"] == "SimBroker.equity_curve_bar_level_mtm"


def test_olr_afternoon_selector_output_drives_only_post_1430_core_orders():
    trade_date = date(2026, 1, 5)
    next_date = trade_date + timedelta(days=1)
    snapshot = OLRDailySnapshot(
        trade_date=trade_date,
        source_fingerprint="selector-core-parity",
        generated_at=datetime.combine(trade_date, time(8, 50), tzinfo=KST),
        candidates=(
            _candidate("005930", trade_date, rank=1),
            _candidate("000660", trade_date, rank=2),
        ),
    )
    bars_by_key = {
        (trade_date, "005930"): (
            _bar("005930", trade_date, time(9, 0), 100.0, 103.0, 99.0, 102.0),
            _bar("005930", trade_date, time(14, 25), 102.0, 105.0, 101.0, 104.0),
            _bar("005930", trade_date, time(14, 30), 104.0, 104.2, 103.8, 104.0),
            _bar("005930", trade_date, time(15, 30), 104.0, 104.2, 103.8, 104.0),
            _bar("005930", next_date, time(14, 30), 106.0, 106.5, 105.5, 106.0),
            _bar("005930", next_date, time(15, 30), 106.0, 106.5, 105.5, 106.0),
        ),
        (trade_date, "000660"): (
            _bar("000660", trade_date, time(9, 0), 100.0, 101.0, 99.0, 100.5),
            _bar("000660", trade_date, time(14, 25), 100.5, 101.0, 100.0, 100.7),
            _bar("000660", trade_date, time(14, 30), 100.7, 160.0, 100.7, 160.0),
            _bar("000660", trade_date, time(15, 30), 160.0, 161.0, 159.0, 160.5),
            _bar("000660", next_date, time(14, 30), 161.0, 162.0, 160.0, 161.5),
            _bar("000660", next_date, time(15, 30), 161.5, 162.0, 161.0, 161.7),
        ),
    }
    cfg = OLRConfig.from_mapping(
        {
            "olr.afternoon.top_n": 1,
            "olr.afternoon.score_mode": "momentum",
            "olr.overnight.slot_count": 1,
            "olr.allocation.min_selected": 1,
            "olr.execution.auction_fill_time": "15:30",
            "olr.cost.slippage_bps": 0.0,
            "olr.execution.auction_adverse_bps": 0.0,
        }
    )
    selected = afternoon_selection_from_snapshot(snapshot, bars_by_key, cfg)
    bundle = compile_olr_replay_bundle(
        bars=[bar for bars in bars_by_key.values() for bar in bars],
        snapshots={trade_date: selected},
        source_fingerprint="selector-core-parity",
    )

    result = run_olr_backtest({"capability_level": "compiled"}, {}, replay_bundle=bundle)
    entries = [event for event in result.decisions if event.decision_code == "ENTRY_SUBMITTED"]
    buy_fills = [fill for fill in result.replay_result.broker.fills if fill.side == "BUY"]

    assert selected.metadata["intraday_selection_cutoff"] == "timestamp < 14:30 KST"
    assert [candidate.symbol for candidate in selected.candidates] == ["005930"]
    assert [event.symbol for event in entries] == ["005930"]
    assert all(event.timestamp.time() >= time(14, 30) for event in entries)
    assert [fill.symbol for fill in buy_fills] == ["005930"]
    assert buy_fills[0].timestamp.time().replace(second=0, microsecond=0) == time(15, 30)
    assert result.metrics["same_bar_fill_count"] == 0.0


def test_olr_core_replay_consumes_swept_non_auction_entry_plan():
    trade_date = date(2026, 1, 5)
    next_date = trade_date + timedelta(days=1)
    snapshot = OLRDailySnapshot(
        trade_date=trade_date,
        source_fingerprint="swept-plan",
        generated_at=datetime.combine(trade_date, time(8, 50), tzinfo=KST),
        candidates=(_candidate("005930", trade_date, rank=1),),
    )
    bars = [
        _bar("005930", trade_date, time(14, 25), 100.0, 100.5, 99.8, 100.0),
        _bar("005930", trade_date, time(14, 30), 100.0, 100.8, 99.9, 100.7),
        _bar("005930", trade_date, time(14, 35), 101.0, 101.5, 100.9, 101.2),
        _bar("005930", trade_date, time(15, 30), 101.2, 101.8, 101.0, 101.5),
        _bar("005930", next_date, time(14, 30), 103.0, 103.3, 102.8, 103.1),
        _bar("005930", next_date, time(15, 30), 103.1, 103.5, 103.0, 103.4),
    ]
    entry = OLREntryPlan(
        "confirm_next_bar_test",
        "confirm_next_bar",
        max_signal_bars=3,
        min_bar_ret=0.0,
        min_vwap_ret=-0.01,
        min_close_location=0.5,
    )
    exit_plan = OLRExitPlan("next_close_test", mode="next_close", hard_stop_enabled=False)
    bundle = compile_olr_replay_bundle(bars=bars, snapshots={trade_date: snapshot}, source_fingerprint="swept-plan")

    result = run_olr_backtest(
        {"capability_level": "compiled", "olr.cost.slippage_bps": 0.0, "olr.execution.auction_adverse_bps": 0.0},
        {"olr.trade_plan.entry": asdict(entry), "olr.trade_plan.exit": asdict(exit_plan)},
        replay_bundle=bundle,
    )
    entries = [event for event in result.decisions if event.decision_code == "ENTRY_SUBMITTED"]
    buy_fills = [fill for fill in result.replay_result.broker.fills if fill.side == "BUY"]

    assert entries[0].reason == "confirm_next_bar"
    assert buy_fills[0].timestamp.time().replace(second=0, microsecond=0) == time(14, 35)
    assert result.metrics["trade_entry_plan_name"] == "confirm_next_bar_test"
    assert result.metrics["trade_exit_plan_name"] == "next_close_test"
    assert result.metrics["auction_order_count"] == 1.0
    assert result.metrics["same_bar_fill_count"] == 0.0


def test_olr_core_replay_consumes_simple_managed_hard_stop_plan():
    trade_date = date(2026, 1, 5)
    next_date = trade_date + timedelta(days=1)
    snapshot = OLRDailySnapshot(
        trade_date=trade_date,
        source_fingerprint="managed-plan",
        generated_at=datetime.combine(trade_date, time(8, 50), tzinfo=KST),
        candidates=(_candidate("005930", trade_date, rank=1),),
    )
    bars = [
        _bar("005930", trade_date, time(14, 25), 100.0, 100.5, 99.8, 100.0),
        _bar("005930", trade_date, time(14, 30), 100.0, 100.8, 99.9, 100.7),
        _bar("005930", trade_date, time(14, 35), 101.0, 101.5, 100.9, 101.2),
        _bar("005930", trade_date, time(15, 30), 101.2, 101.8, 101.0, 101.5),
        _bar("005930", next_date, time(14, 30), 103.0, 103.3, 101.8, 103.1),
        _bar("005930", next_date, time(15, 30), 103.1, 103.5, 103.0, 103.4),
    ]
    entry = OLREntryPlan(
        "confirm_next_bar_test",
        "confirm_next_bar",
        max_signal_bars=3,
        min_bar_ret=0.0,
        min_vwap_ret=-0.01,
    )
    exit_plan = OLRExitPlan(
        "managed_decision_low_test",
        mode="managed",
        stop_mode="decision_low",
        hard_stop_enabled=True,
    )
    bundle = compile_olr_replay_bundle(bars=bars, snapshots={trade_date: snapshot}, source_fingerprint="managed-plan")

    result = run_olr_backtest(
        {"capability_level": "compiled", "olr.cost.slippage_bps": 0.0, "olr.execution.auction_adverse_bps": 0.0},
        {"olr.trade_plan.entry": asdict(entry), "olr.trade_plan.exit": asdict(exit_plan)},
        replay_bundle=bundle,
    )
    sell_fills = [fill for fill in result.replay_result.broker.fills if fill.side == "SELL"]

    assert result.metrics["trade_exit_plan_mode"] == "managed"
    assert result.metrics["official_trade_plan_supported"] == 1.0
    assert sell_fills[0].reason == "next_close_exit"
    assert result.metrics["open_order_count"] == 0.0
    assert result.metrics["end_open_position_count"] == 0.0
    assert result.metrics["same_bar_fill_count"] == 0.0


def test_olr_core_reserves_pending_entry_cash_before_same_timestamp_fills():
    trade_date = date(2026, 1, 5)
    next_date = trade_date + timedelta(days=1)
    snapshot = OLRDailySnapshot(
        trade_date=trade_date,
        source_fingerprint="pending-reservation",
        generated_at=datetime.combine(trade_date, time(8, 50), tzinfo=KST),
        candidates=(
            _candidate("000660", trade_date, rank=1),
            _candidate("005930", trade_date, rank=2),
        ),
    )
    bars = []
    for symbol in ("000660", "005930"):
        bars.extend(
            [
                _bar(symbol, trade_date, time(14, 25), 100.0, 100.2, 99.8, 100.0),
                _bar(symbol, trade_date, time(14, 30), 100.0, 101.0, 99.9, 100.5),
                _bar(symbol, trade_date, time(14, 35), 100.0, 101.0, 99.9, 100.5),
                _bar(symbol, next_date, time(14, 30), 101.0, 101.2, 100.8, 101.0),
                _bar(symbol, next_date, time(15, 30), 101.0, 101.2, 100.8, 101.0),
            ]
        )
    entry = OLREntryPlan("confirm_next_bar_test", "confirm_next_bar", max_signal_bars=2, min_bar_ret=0.0, min_vwap_ret=-0.01)
    exit_plan = OLRExitPlan("next_close_test", mode="next_close", hard_stop_enabled=False)
    bundle = compile_olr_replay_bundle(bars=bars, snapshots={trade_date: snapshot}, source_fingerprint="pending-reservation")

    result = run_olr_backtest(
        {
            "capability_level": "compiled",
            "initial_equity": 10_000.0,
            "olr.overnight.slot_count": 2,
            "olr.allocation.target_gross_exposure": 2.0,
            "olr.allocation.max_position_pct": 1.0,
            "olr.cost.slippage_bps": 0.0,
            "olr.cost.commission_bps": 0.0,
            "olr.cost.tax_bps_on_sell": 0.0,
        },
        {"olr.trade_plan.entry": asdict(entry), "olr.trade_plan.exit": asdict(exit_plan)},
        replay_bundle=bundle,
    )

    entries = [event for event in result.decisions if event.decision_code == "ENTRY_SUBMITTED"]
    skipped = [event for event in result.decisions if event.decision_code == "ENTRY_SKIPPED"]
    buy_fills = [fill for fill in result.replay_result.broker.fills if fill.side == "BUY"]

    assert len(entries) == 1
    assert len(skipped) == 1
    assert len(buy_fills) == 1
    assert result.metrics["rejected_order_count"] == 0.0


def test_olr_core_replay_market_exits_after_close_auction_nonfill():
    trade_date = date(2026, 1, 5)
    next_date = trade_date + timedelta(days=1)
    recovery_date = trade_date + timedelta(days=2)
    snapshot = OLRDailySnapshot(
        trade_date=trade_date,
        source_fingerprint="auction-nonfill",
        generated_at=datetime.combine(trade_date, time(8, 50), tzinfo=KST),
        candidates=(_candidate("005930", trade_date, rank=1),),
    )
    bars = [
        _bar("005930", trade_date, time(14, 25), 100.0, 100.5, 99.8, 100.0),
        _bar("005930", trade_date, time(14, 30), 100.0, 100.8, 99.9, 100.7),
        _bar("005930", trade_date, time(14, 35), 101.0, 101.5, 100.9, 101.2),
        _bar("005930", next_date, time(14, 30), 100.0, 100.2, 99.8, 100.0),
        _bar("005930", next_date, time(15, 30), 97.0, 97.2, 96.8, 97.0),
        _bar("005930", recovery_date, time(9, 0), 96.5, 96.8, 96.0, 96.3),
    ]
    entry = OLREntryPlan(
        "confirm_next_bar_test",
        "confirm_next_bar",
        max_signal_bars=3,
        min_bar_ret=0.0,
        min_vwap_ret=-0.01,
    )
    exit_plan = OLRExitPlan("next_close_test", mode="next_close", hard_stop_enabled=False)
    bundle = compile_olr_replay_bundle(bars=bars, snapshots={trade_date: snapshot}, source_fingerprint="auction-nonfill")

    result = run_olr_backtest(
        {
            "capability_level": "compiled",
            "olr.cost.slippage_bps": 0.0,
            "olr.execution.auction_adverse_bps": 0.0,
            "olr.execution.auction_limit_offset_bps": 0.0,
        },
        {"olr.trade_plan.entry": asdict(entry), "olr.trade_plan.exit": asdict(exit_plan)},
        replay_bundle=bundle,
    )
    sell_fills = [fill for fill in result.replay_result.broker.fills if fill.side == "SELL"]
    fallback_decisions = [event for event in result.decisions if event.decision_code == "EXIT_FALLBACK_SUBMITTED"]

    assert result.metrics["auction_nonfill_count"] == 1.0
    assert fallback_decisions
    assert sell_fills[0].reason == "auction_exit_nonfill_market_fallback"
    assert sell_fills[0].timestamp.time().replace(second=0, microsecond=0) == time(9, 0)
    assert result.metrics["open_order_count"] == 0.0
    assert result.metrics["end_open_position_count"] == 0.0
    assert result.metrics["entry_fill_count"] == result.metrics["exit_fill_count"]


def test_olr_core_replay_supports_managed_partial_target_and_cleans_resting_orders():
    trade_date = date(2026, 1, 5)
    next_date = trade_date + timedelta(days=1)
    snapshot = OLRDailySnapshot(
        trade_date=trade_date,
        source_fingerprint="managed-target",
        generated_at=datetime.combine(trade_date, time(8, 50), tzinfo=KST),
        candidates=(_candidate("005930", trade_date, rank=1),),
    )
    bars = [
        _bar("005930", trade_date, time(14, 25), 100.0, 100.5, 99.8, 100.0),
        _bar("005930", trade_date, time(14, 30), 100.0, 100.8, 99.9, 100.7),
        _bar("005930", trade_date, time(14, 35), 101.0, 101.5, 100.9, 101.2),
        _bar("005930", next_date, time(9, 0), 101.2, 103.3, 101.0, 103.0),
        _bar("005930", next_date, time(9, 5), 103.0, 105.0, 102.8, 104.5),
        _bar("005930", next_date, time(15, 30), 104.5, 104.8, 104.0, 104.4),
    ]
    entry = OLREntryPlan("confirm_next_bar_test", "confirm_next_bar", max_signal_bars=3, min_bar_ret=0.0, min_vwap_ret=-0.01)
    exit_plan = OLRExitPlan(
        "managed_target_test",
        mode="managed",
        stop_mode="fixed_pct",
        hard_stop_enabled=True,
        stop_pct=0.01,
        partial_trigger_r=1.0,
        partial_fraction=0.5,
        partial_stop_r=0.0,
        target_r=2.0,
    )
    bundle = compile_olr_replay_bundle(bars=bars, snapshots={trade_date: snapshot}, source_fingerprint="managed-target")

    result = run_olr_backtest(
        {
            "capability_level": "compiled",
            "olr.cost.slippage_bps": 0.0,
            "olr.cost.commission_bps": 0.0,
            "olr.cost.tax_bps_on_sell": 0.0,
        },
        {"olr.trade_plan.entry": asdict(entry), "olr.trade_plan.exit": asdict(exit_plan)},
        replay_bundle=bundle,
    )
    sell_reasons = [fill.reason for fill in result.replay_result.broker.fills if fill.side == "SELL"]

    assert "partial_target" in sell_reasons
    assert "target" in sell_reasons
    assert result.metrics["official_trade_plan_supported"] == 1.0
    assert result.metrics["open_order_count"] == 0.0
    assert result.metrics["end_open_position_count"] == 0.0
    assert result.metrics["same_bar_fill_count"] == 0.0


def _candidate(symbol: str, trade_date: date, *, rank: int) -> OLRDailyCandidate:
    return OLRDailyCandidate(
        symbol=symbol,
        trade_date=trade_date,
        prior_day_high=102.0,
        prior_day_low=98.0,
        prior_day_close=100.0,
        daily_atr=2.0,
        expected_5m_volume=1000.0,
        average_30m_volume=6000.0,
        rank=rank,
        rank_pct=float(rank),
        selection_score=100.0 - rank,
        metadata={"prior_close": 100.0, "atr20": 2.0},
    )


def _bar(symbol: str, day: date, clock: time, open_: float, high: float, low: float, close: float) -> MarketBar:
    return MarketBar(
        symbol=symbol,
        timestamp=datetime.combine(day, clock, tzinfo=KST),
        timeframe="5m",
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=1000,
    )
