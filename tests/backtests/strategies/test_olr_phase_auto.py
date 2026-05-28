from __future__ import annotations

from datetime import date, datetime, time
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from backtests.strategies.olr.phase_candidates import get_phase_candidates
from backtests.strategies.olr.phase_scoring import IMMUTABLE_SCORE_COMPONENTS, olr_reject_reason, score_olr_phase
from backtests.strategies.olr.plugin import OLROptimizationPlugin, _eligible_snapshot_dates, _filtered_training_bars_for_snapshots, _format_olr_diagnostics
from strategy_common.market import MarketBar
from strategy_olr.config import OLRConfig


def test_olr_phase_score_stays_compact_and_rewards_alpha_quality() -> None:
    assert len(IMMUTABLE_SCORE_COMPONENTS) <= 7

    base = {
        "official_mtm_net_return_pct": 0.35,
        "expected_total_r": 30.0,
        "total_trades": 180.0,
        "profit_factor": 1.15,
        "max_drawdown_pct": 0.13,
        "olr_alpha_capture": 0.12,
        "olr_discrimination_quality": 0.20,
        "same_bar_fill_count": 0.0,
        "rejected_order_count": 0.0,
        "forced_replay_close_count": 0.0,
        "end_open_position_count": 0.0,
        "olr_selected_negative_label_share": 0.55,
    }
    better = {
        **base,
        "official_mtm_net_return_pct": 0.55,
        "expected_total_r": 55.0,
        "profit_factor": 1.35,
        "olr_alpha_capture": 0.25,
        "olr_discrimination_quality": 0.42,
    }

    assert score_olr_phase(1, better) > score_olr_phase(1, base)
    assert olr_reject_reason(1, base) == ""


def test_olr_phase_score_uses_entry_economics_not_partial_exit_leg_count() -> None:
    strong_entry_baseline = {
        "official_mtm_net_return_pct": 0.8338,
        "entry_fill_count": 253.0,
        "total_trades": 253.0,
        "entry_level_expected_total_r": 57.8,
        "expected_total_r": 57.8,
        "entry_level_profit_factor": 2.06,
        "profit_factor": 2.06,
        "max_drawdown_pct": 0.083,
        "olr_alpha_capture": 0.256,
        "olr_discrimination_quality": 0.435,
    }
    leg_inflated_partial_exit = {
        **strong_entry_baseline,
        "official_mtm_net_return_pct": 0.4017,
        "entry_fill_count": 278.0,
        "total_trades": 427.0,
        "entry_level_expected_total_r": 55.0,
        "expected_total_r": 103.6,
        "entry_level_profit_factor": 1.47,
        "profit_factor": 1.47,
        "max_drawdown_pct": 0.102,
        "olr_alpha_capture": 0.435,
    }

    assert score_olr_phase(4, leg_inflated_partial_exit) < score_olr_phase(4, strong_entry_baseline)


def test_olr_phase_acceptance_rejects_material_mtm_regression() -> None:
    plugin = OLROptimizationPlugin({"capability_level": "synthetic"}, max_workers=1)
    criteria = plugin.phase_acceptance_criteria(
        phase=4,
        base_mutations={},
        base_metrics={"official_mtm_net_return_pct": 0.8338, "entry_fill_count": 253.0},
        final_metrics={"official_mtm_net_return_pct": 0.4017, "entry_fill_count": 278.0},
        greedy_result=SimpleNamespace(),
    )
    by_name = {criterion.name: criterion for criterion in criteria}

    assert by_name["hard_relative_mtm_non_regression"].passed is False
    assert by_name["hard_entry_frequency_retention"].passed is True


def test_olr_live_parity_timing_is_explicit_string() -> None:
    cfg = OLRConfig()

    assert cfg.live_parity_fill_timing == "completed_5m_signal_next_bar_or_resting_close_auction"
    assert OLRConfig.from_mapping({"olr": {"execution": {"live_parity_fill_timing": True}}}).live_parity_fill_timing == cfg.live_parity_fill_timing


def test_olr_final_diagnostics_surface_audit_cost_and_attribution() -> None:
    text = _format_olr_diagnostics(
        6,
        {
            "official_mtm_net_return_pct": 0.21,
            "net_return_pct": 0.213,
            "max_drawdown_pct": 0.059,
            "final_equity": 12_100_000.0,
            "entry_fill_count": 32.0,
            "entry_level_trade_count": 31.0,
            "exit_fill_count": 31.0,
            "end_open_position_count": 1.0,
            "entry_level_profit_factor": 2.92,
            "entry_level_expected_total_r": 14.0,
            "cost_policy": {"commission_bps": 2.0, "slippage_bps": 5.0, "tax_bps_on_sell": 18.0},
            "same_bar_fill_count": 0.0,
            "forced_replay_close_count": 0.0,
            "rejected_order_count": 0.0,
            "auction_order_count": 64.0,
            "auction_nonfill_count": 0.0,
            "open_order_count": 0.0,
            "expired_order_count": 0.0,
            "decision_hash": "d" * 64,
            "neutral_action_hash": "a" * 64,
            "fill_hash": "f" * 64,
            "trade_hash": "t" * 64,
            "source_snapshot_hash": "s" * 64,
            "final_state_hash": "z" * 64,
            "dynamic_overlay_selected_count": 4.0,
            "dynamic_overlay_trade_count": 2.0,
            "dynamic_overlay_realized_total_r": 1.5,
            "score_band_rule_trade_counts": {"mid_400_500_looser_breakout_dynamic_overlay": 2},
            "score_band_rule_realized_total_r": {"mid_400_500_looser_breakout_dynamic_overlay": 1.5},
        },
        SimpleNamespace(accepted_count=0, total_candidates=0, kept_features=[]),
        {
            "shared_decision_core": True,
            "live_parity_fill_timing": "completed_5m_signal_next_bar_or_resting_close_auction",
            "auction_mode": "resting_close_auction_after_14_30_decision",
        },
    )

    assert "Completed entry-level trades: 31" in text
    assert "End open positions: 1" in text
    assert "Cost policy:" in text
    assert "Neutral action hash:" in text
    assert "Final state hash:" in text
    assert "Dynamic overlay realized trades: 2" in text


def test_olr_phase_candidates_cover_signal_entry_exit_and_allocation() -> None:
    counts = {phase: len(get_phase_candidates(phase)) for phase in range(1, 7)}
    phase1 = get_phase_candidates(1)
    phase3 = get_phase_candidates(3)
    phase4 = get_phase_candidates(4)
    phase5 = get_phase_candidates(5)

    assert sum(counts.values()) >= 70
    assert max(counts.values()) <= 20
    assert any("olr.afternoon.score_calibration_mode" in item.mutations for item in phase1)
    assert any("olr.afternoon.max_score" in item.mutations for item in phase1)
    assert any("olr.afternoon.blocked_sectors" in item.mutations for item in phase1)
    assert any("olr.trade_plan.entry" in item.mutations for item in phase3)
    assert any("olr.trade_plan.exit" in item.mutations for item in phase4)
    assert any("olr.allocation.mode" in item.mutations for item in phase5)


def test_olr_official_replay_scope_uses_executable_slots_and_recovery_sessions() -> None:
    dates = (date(2026, 1, 2), date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7))
    symbols = ("000001", "000002", "000003")
    bars_by_key = {
        (day, symbol): (_bar(day, symbol),)
        for day in dates
        for symbol in symbols
    }
    dataset = SimpleNamespace(trading_dates=dates, bars_by_key=bars_by_key, config={})
    snapshot = SimpleNamespace(
        candidates=tuple(SimpleNamespace(symbol=symbol, tradable=True) for symbol in symbols),
    )

    bars, scope = _filtered_training_bars_for_snapshots(
        dataset,
        {dates[0]: snapshot},
        OLRConfig(overnight_slot_count=2, min_selected=1),
    )

    assert _eligible_snapshot_dates(dates) == dates[:2]
    assert {(bar.timestamp.date(), bar.symbol) for bar in bars} == {
        (dates[0], "000001"),
        (dates[0], "000002"),
        (dates[1], "000001"),
        (dates[1], "000002"),
        (dates[2], "000001"),
        (dates[2], "000002"),
    }
    assert scope["auction_exit_recovery_sessions"] == 2


def _bar(day: date, symbol: str) -> MarketBar:
    return MarketBar(
        symbol=symbol,
        timestamp=datetime.combine(day, time(15, 30), tzinfo=ZoneInfo("Asia/Seoul")),
        timeframe="5m",
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.0,
        volume=1000.0,
    )
