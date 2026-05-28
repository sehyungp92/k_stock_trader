from __future__ import annotations

import json
from dataclasses import replace
from datetime import date, datetime, time, timedelta
from pathlib import Path

import pytest

from backtests.strategies.kalcb.first30_signal_sweep import (
    DailyFeature,
    First30Context,
    First30Intraday,
    FlowFeature,
    KALCBFirst30Dataset,
    MarketFeature,
    Selection,
    _round_trip_cost_pct,
)
from backtests.strategies.kalcb import trade_plan_sweep as trade_plan_sweep_module
from backtests.strategies.kalcb.stage2_calibration import select_calibrated_stage2_rows
from backtests.strategies.kalcb.three_stage_pipeline import _stage3_sort_key
from backtests.strategies.kalcb.trade_plan_sweep import (
    EntrySpec,
    ExitSpec,
    PlanResult,
    TradePlanSpec,
    _evaluate_specs,
    _audit_replay_rows,
    _name_entry,
    _name_exit,
    _plan_sort_key,
    _core_config_for_spec,
    _prior_day_high,
    _resolve_folds,
    _selection_counts,
    _training_only_config,
    _refined_exit_specs,
    baseline_trade_plan_spec,
    build_exit_specs,
    compile_core_replay,
    eod_flatten_exit_spec,
    find_entry_signal,
    load_or_build_prepared_context,
    load_fixed_candidate_source,
    name_plan,
    promotion_pass,
    score_plan,
    simulate_trade,
)
from strategy_common.clock import KST
from strategy_common.market import MarketBar
from strategy_kalcb.config import KALCBConfig


TRADE_DATE = date(2026, 1, 5)


def test_fixed_optimized_candidate_source_loads_expected_top_row() -> None:
    source = load_fixed_candidate_source()

    assert source.source_sweep_hash.startswith("8ddb4881bd89")
    assert source.source_file_hash
    assert source.frontier.name.startswith("rs_trend_fx30")
    assert source.frontier.frontier_size == 30
    assert source.frontier.require_flow_available is True
    assert source.first30.name.startswith("hybrid_top1")
    assert source.first30.top_n == 1


def test_exit_sweep_generator_emits_only_core_valid_partial_stop_buffers() -> None:
    specs = build_exit_specs()

    assert specs
    assert all(spec.partial_stop_r >= 0.0 for spec in specs)
    assert all(refined.partial_stop_r >= 0.0 for spec in specs[:12] for refined in _refined_exit_specs(spec))


def test_exit_dedupe_preserves_sizing_relevant_nostop_variants() -> None:
    variants = [
        ExitSpec("a", hard_stop_enabled=False, stop_mode="atr", stop_atr_mult=0.5, stop_pct=0.004),
        ExitSpec("b", hard_stop_enabled=False, stop_mode="pct", stop_atr_mult=1.4, stop_pct=0.012),
    ]

    deduped = trade_plan_sweep_module._dedupe_exits(variants)

    assert len(deduped) == 2
    assert {spec.stop_mode for spec in deduped} == {"atr", "pct"}


def test_exit_dedupe_still_collapses_disabled_exit_management_fields() -> None:
    variants = [
        ExitSpec("a", hard_stop_enabled=False, target_r=0.0, partial_trigger_r=0.0, partial_fraction=0.0, trail_start_r=0.0),
        ExitSpec("b", hard_stop_enabled=False, target_r=-1.0, partial_trigger_r=0.0, partial_fraction=0.5, trail_start_r=0.0),
    ]

    deduped = trade_plan_sweep_module._dedupe_exits(variants)

    assert len(deduped) == 1
    assert deduped[0].target_r == pytest.approx(0.0)
    assert deduped[0].partial_trigger_r == pytest.approx(0.0)
    assert deduped[0].partial_fraction == pytest.approx(0.0)


def test_focused_seed_loader_keeps_distinct_outcomes(tmp_path: Path) -> None:
    entry = EntrySpec("", mode="first30_0930")
    exit_a = ExitSpec("", stop_mode="fixed_pct", hard_stop_enabled=False)
    exit_b = ExitSpec("", stop_mode="signal_low", hard_stop_enabled=False)

    def row(name: str, exit_spec: ExitSpec, broker: float) -> PlanResult:
        spec = name_plan(TradePlanSpec(name, entry, exit_spec))
        metrics = {
            "broker_net_return_pct": broker,
            "portfolio_equivalent_net_return_pct": broker,
            "exposure_normalized_slot_net_return_pct": broker,
            "equal_slot_net_return_pct": broker,
            "trade_count": 100.0,
            "broker_max_drawdown_pct": 0.04,
        }
        return PlanResult(spec, 1.0, False, "", metrics, tuple(), True)

    rows = [row("a", exit_a, 0.40), row("b", exit_a, 0.40), row("c", exit_b, 0.35)]
    path = tmp_path / "rows.jsonl"
    path.write_text("\n".join(json.dumps(trade_plan_sweep_module._row_payload(item), default=str) for item in rows), encoding="utf-8")

    loaded = trade_plan_sweep_module._load_focused_seed_rows(path, limit=2)

    assert len(loaded) == 2
    assert [item.train_metrics["broker_net_return_pct"] for item in loaded] == [0.40, 0.35]


def test_training_only_window_excludes_holdout_when_end_is_not_explicit() -> None:
    config = {"baseline": {"holdout_start": "2026-04-01"}}

    out = _training_only_config(config, train_only=True)

    assert out["use_full_available_window"] is False
    assert out["end"] == "2026-03-31"


def test_first30_0930_entry_is_allowed_after_completed_0925_bar() -> None:
    bars = _bars(
        [
            (100.0, 100.2, 99.8, 100.0),
            (100.0, 100.2, 99.8, 100.1),
            (100.1, 100.4, 99.9, 100.2),
            (100.2, 100.4, 100.0, 100.2),
            (100.2, 100.5, 100.1, 100.3),
            (100.3, 100.6, 100.2, 100.5),
            (100.6, 101.0, 100.5, 100.8),
        ]
    )
    signal = find_entry_signal(tuple(bars), _ctx(bars), _name_entry(EntrySpec("", "first30_0930")), KALCBConfig())

    assert signal is not None
    assert signal.signal_index == 5
    assert bars[signal.signal_index].timestamp.astimezone(KST).time() == time(9, 25)
    assert signal.fill_index == 6
    assert bars[signal.fill_index].timestamp.astimezone(KST).time() == time(9, 30)


def test_completed_confirmation_bar_fills_next_open_not_same_bar() -> None:
    bars = _bars(
        [
            (100.0, 100.2, 99.8, 100.0),
            (100.0, 100.2, 99.8, 100.1),
            (100.1, 100.4, 99.9, 100.2),
            (100.2, 100.4, 100.0, 100.2),
            (100.2, 100.5, 100.1, 100.3),
            (100.3, 100.6, 100.2, 100.5),
            (100.6, 102.0, 100.5, 101.8),
            (101.9, 103.0, 101.7, 102.5),
        ]
    )
    spec = _name_entry(EntrySpec("", "confirm_next_bar", max_signal_bars=1, min_bar_ret=0.001, min_vwap_ret=0.0))

    signal = find_entry_signal(tuple(bars), _ctx(bars), spec, KALCBConfig())

    assert signal is not None
    assert signal.signal_index == 6
    assert signal.fill_index == 7
    assert bars[signal.fill_index].timestamp.astimezone(KST).time() == time(9, 35)


def test_trade_plan_reclaim_level_source_matches_campaign_metadata() -> None:
    bars = _bars(
        [
            (100.0, 100.2, 99.8, 100.0),
            (100.0, 100.2, 99.8, 100.1),
            (100.1, 100.4, 99.9, 100.2),
            (100.2, 100.4, 100.0, 100.2),
            (100.2, 100.5, 100.1, 100.3),
            (100.3, 100.6, 100.2, 100.5),
            (100.9, 100.95, 100.8, 100.85),
            (100.85, 101.5, 100.9, 101.3),
            (101.3, 101.8, 101.2, 101.6),
        ]
    )
    spec = _name_entry(
        EntrySpec(
            "",
            "pullback_acceptance",
            max_signal_bars=2,
            min_reclaim_ret=0.001,
            reclaim_level_source="campaign_box_high",
        )
    )

    signal = find_entry_signal(tuple(bars), _ctx(bars), spec, KALCBConfig(), candidate_metadata={"campaign_box_high": 101.0})

    assert signal is not None
    assert signal.signal_index == 7
    assert signal.fill_index == 8
    assert signal.reason == "pullback_acceptance"


def test_trade_plan_campaign_breakout_can_require_two_reclaim_closes() -> None:
    bars = _bars(
        [
            (100.0, 100.2, 99.8, 100.0),
            (100.0, 100.2, 99.8, 100.1),
            (100.1, 100.4, 99.9, 100.2),
            (100.2, 100.4, 100.0, 100.2),
            (100.2, 100.5, 100.1, 100.3),
            (100.3, 100.6, 100.2, 100.5),
            (100.9, 100.95, 100.8, 100.85),
            (100.85, 101.5, 100.9, 101.2),
            (101.2, 101.8, 101.1, 101.5),
            (101.5, 102.0, 101.4, 101.8),
        ]
    )
    spec = _name_entry(
        EntrySpec(
            "",
            "or_high_reclaim",
            max_signal_bars=3,
            min_reclaim_ret=0.0,
            reclaim_level_source="campaign_breakout_level",
            min_reclaim_closes=2,
        )
    )

    signal = find_entry_signal(tuple(bars), _ctx(bars), spec, KALCBConfig(), candidate_metadata={"campaign_breakout_level": 101.0})

    assert signal is not None
    assert signal.signal_index == 8
    assert signal.fill_index == 9
    assert signal.reason == "or_high_reclaim"


def test_prior_day_high_is_causal_and_ignores_trade_date_row() -> None:
    dataset = _dataset(
        _bars(
            [
                (100.0, 100.2, 99.8, 100.0),
                (100.0, 100.2, 99.8, 100.1),
                (100.1, 100.4, 99.9, 100.2),
                (100.2, 100.4, 100.0, 100.2),
                (100.2, 100.5, 100.1, 100.3),
                (100.3, 100.6, 100.2, 100.5),
                (100.6, 101.0, 100.5, 100.8),
            ]
        )
    )
    rows = list(dataset.daily_by_symbol["005930"])
    rows.append({"ticker": "005930", "date": TRADE_DATE.isoformat(), "high": 9999.0, "close": 9999.0, "volume": 1.0})
    rows.insert(0, rows.pop(-2))
    dataset = replace(dataset, daily_by_symbol={"005930": rows})

    assert _prior_day_high(dataset, "005930", TRADE_DATE) == pytest.approx(112.0)


def test_same_bar_stop_target_conflict_resolves_stop_first() -> None:
    bars = _bars(
        [
            (100.0, 100.2, 99.8, 100.0),
            (100.0, 100.2, 99.8, 100.1),
            (100.1, 100.4, 99.9, 100.2),
            (100.2, 100.4, 100.0, 100.2),
            (100.2, 100.5, 100.1, 100.3),
            (100.3, 100.6, 100.2, 100.5),
            (100.0, 103.0, 98.0, 101.0),
        ]
    )
    outcome = simulate_trade(
        TRADE_DATE,
        "005930",
        tuple(bars),
        _ctx(bars),
        _name_entry(EntrySpec("", "first30_0930")),
        _name_exit(ExitSpec("", stop_mode="fixed_pct", stop_pct=0.01, hard_stop_enabled=True, target_r=0.5)),
        KALCBConfig(slippage_bps=0.0, commission_bps=0.0, tax_bps_on_sell=0.0),
    )

    assert outcome is not None
    assert outcome.stopped is True
    assert outcome.target_hit is False
    assert outcome.ambiguous_bar_count == 1
    assert outcome.gross_return_pct < 0.0


def test_partial_then_stop_applies_costs_once() -> None:
    bars = _bars(
        [
            (100.0, 100.2, 99.8, 100.0),
            (100.0, 100.2, 99.8, 100.1),
            (100.1, 100.4, 99.9, 100.2),
            (100.2, 100.4, 100.0, 100.2),
            (100.2, 100.5, 100.1, 100.3),
            (100.3, 100.6, 100.2, 100.5),
            (100.0, 100.8, 99.8, 100.6),
            (100.6, 100.7, 99.9, 100.1),
        ]
    )
    cfg = KALCBConfig(slippage_bps=5.0, commission_bps=1.0, tax_bps_on_sell=20.0)
    outcome = simulate_trade(
        TRADE_DATE,
        "005930",
        tuple(bars),
        _ctx(bars),
        _name_entry(EntrySpec("", "first30_0930")),
        _name_exit(
            ExitSpec(
                "",
                stop_mode="fixed_pct",
                stop_pct=0.01,
                hard_stop_enabled=True,
                partial_trigger_r=0.5,
                partial_fraction=0.5,
                partial_stop_r=0.0,
            )
        ),
        cfg,
    )

    assert outcome is not None
    assert outcome.partial_hit is True
    assert outcome.stopped is True
    assert outcome.gross_return_pct == pytest.approx(0.0025)
    assert outcome.net_return_pct == pytest.approx(outcome.gross_return_pct - _round_trip_cost_pct(cfg))


def test_trailing_stop_update_from_completed_bar_is_active_next_bar_only() -> None:
    bars = _bars(
        [
            (100.0, 100.2, 99.8, 100.0),
            (100.0, 100.2, 99.8, 100.1),
            (100.1, 100.4, 99.9, 100.2),
            (100.2, 100.4, 100.0, 100.2),
            (100.2, 100.5, 100.1, 100.3),
            (100.3, 100.6, 100.2, 100.5),
            (100.0, 100.8, 100.1, 100.6),
            (101.0, 101.1, 100.4, 100.5),
        ]
    )
    outcome = simulate_trade(
        TRADE_DATE,
        "005930",
        tuple(bars),
        _ctx(bars),
        _name_entry(EntrySpec("", "first30_0930")),
        _name_exit(ExitSpec("", stop_mode="fixed_pct", stop_pct=0.01, hard_stop_enabled=True, trail_start_r=0.5, trail_gap_r=0.2)),
        KALCBConfig(slippage_bps=0.0, commission_bps=0.0, tax_bps_on_sell=0.0),
    )

    assert outcome is not None
    assert outcome.exit_reason == "hard_stop"
    assert outcome.bars_held == 2
    assert outcome.gross_return_pct == pytest.approx(0.006)


def test_vwap_fail_exit_uses_next_bar_open() -> None:
    bars = _bars(
        [
            (100.0, 100.2, 99.8, 100.0),
            (100.0, 100.2, 99.8, 100.1),
            (100.1, 100.4, 99.9, 100.2),
            (100.2, 100.4, 100.0, 100.2),
            (100.2, 100.5, 100.1, 100.3),
            (100.3, 100.6, 100.2, 100.5),
            (100.0, 100.2, 99.3, 99.4),
            (99.7, 100.0, 99.5, 99.9),
        ]
    )
    outcome = simulate_trade(
        TRADE_DATE,
        "005930",
        tuple(bars),
        _ctx(bars),
        _name_entry(EntrySpec("", "first30_0930")),
        _name_exit(ExitSpec("", stop_mode="atr", hard_stop_enabled=False, vwap_fail_bars=1, vwap_fail_pct=0.0)),
        KALCBConfig(slippage_bps=0.0, commission_bps=0.0, tax_bps_on_sell=0.0),
    )

    assert outcome is not None
    assert outcome.exit_reason == "vwap_fail"
    assert outcome.gross_return_pct == pytest.approx(99.7 / 100.0 - 1.0)


def test_worker_parallel_and_single_worker_runs_have_identical_ordering(tmp_path: Path) -> None:
    bars = _bars(
        [
            (100.0, 100.2, 99.8, 100.0),
            (100.0, 100.2, 99.8, 100.1),
            (100.1, 100.4, 99.9, 100.2),
            (100.2, 100.4, 100.0, 100.2),
            (100.2, 100.5, 100.1, 100.3),
            (100.3, 100.6, 100.2, 100.5),
            (100.6, 101.0, 100.5, 100.8),
            (100.8, 101.4, 100.7, 101.2),
        ]
    )
    dataset = _dataset(bars)
    ctx = _ctx(bars)
    context_by_key = {(TRADE_DATE, "005930"): ctx}
    selections = [Selection(TRADE_DATE, "005930", 1.0, "unit")]
    counts = _selection_counts(selections, dataset.trading_dates)
    cfg = KALCBConfig(slippage_bps=0.0, commission_bps=0.0, tax_bps_on_sell=0.0)
    specs = [
        baseline_trade_plan_spec(),
        name_plan(TradePlanSpec("", _name_entry(EntrySpec("", "confirm_next_bar", max_signal_bars=1)), eod_flatten_exit_spec())),
    ]
    baseline = {"slot_cumulative_net_return_pct": 0.0, "active_day_net_pct": 0.0, "max_drawdown_net_pct": 0.0}
    folds = _resolve_folds(dataset.trading_dates, 1)

    single = _evaluate_specs(specs, selections, dataset, context_by_key, cfg, dataset.trading_dates, folds, counts, baseline, (), tmp_path, "single", max_workers=1)
    parallel = _evaluate_specs(specs, selections, dataset, context_by_key, cfg, dataset.trading_dates, folds, counts, baseline, (), tmp_path, "parallel", max_workers=2)

    assert [row.spec.name for row in parallel] == [row.spec.name for row in single]
    assert [row.train_metrics for row in parallel] == [row.train_metrics for row in single]
    assert "broker_net_return_pct" in single[0].train_metrics
    assert "official_mtm_net_return_pct" in single[0].train_metrics
    assert single[0].train_metrics["equal_slot_net_return_pct"] == single[0].train_metrics["slot_cumulative_net_return_pct"]
    assert single[0].train_metrics["exposure_normalized_slot_net_return_pct"] == single[0].train_metrics["portfolio_equivalent_net_return_pct"]
    if abs(single[0].train_metrics["broker_net_return_pct"]) > 1e-12:
        assert abs(single[0].train_metrics["exposure_normalized_slot_to_broker_ratio"] - 1.0) < 1e-12
    else:
        assert single[0].train_metrics["exposure_normalized_slot_to_broker_ratio"] == 0.0
    assert "final_equity" in single[0].train_metrics
    assert "end_open_position_count" in single[0].train_metrics
    assert single[0].train_metrics["same_bar_fill_count"] == 0.0


def test_full_audit_matches_fast_suppression_fills_trades_and_metrics(tmp_path: Path) -> None:
    bars = _bars(
        [
            (100.0, 100.2, 99.8, 100.0),
            (100.0, 100.2, 99.8, 100.1),
            (100.1, 100.4, 99.9, 100.2),
            (100.2, 100.4, 100.0, 100.2),
            (100.2, 100.5, 100.1, 100.3),
            (100.3, 100.6, 100.2, 100.5),
            (100.6, 101.2, 100.5, 101.0),
            (101.1, 101.6, 100.9, 101.4),
        ]
    )
    dataset = _dataset(bars)
    context_by_key = {(TRADE_DATE, "005930"): _ctx(bars)}
    selections = [Selection(TRADE_DATE, "005930", 1.0, "unit")]
    counts = _selection_counts(selections, dataset.trading_dates)
    cfg = KALCBConfig(slippage_bps=0.0, commission_bps=0.0, tax_bps_on_sell=0.0)
    compiled = compile_core_replay(selections, dataset, context_by_key, dataset.trading_dates, counts, cfg)
    spec = baseline_trade_plan_spec()
    baseline = {"slot_cumulative_net_return_pct": 0.0, "active_day_net_pct": 0.0, "max_drawdown_net_pct": 0.0}
    folds = _resolve_folds(dataset.trading_dates, 1)

    row = _evaluate_specs(
        [spec],
        selections,
        dataset,
        context_by_key,
        cfg,
        dataset.trading_dates,
        folds,
        counts,
        baseline,
        (),
        tmp_path,
        "audit",
        max_workers=1,
        compiled_replay=compiled,
    )[0]
    audit = _audit_replay_rows([row], compiled, cfg, dataset.trading_dates, counts, max_workers=1)[0]

    assert audit["audit_pass"] is True
    assert audit["max_abs_metric_delta"] == pytest.approx(0.0)
    assert audit["fill_hash_match"] is True
    assert audit["trade_hash_match"] is True
    assert audit["trading_decision_hash_match"] is True
    assert "official_mtm_net_return_pct" in audit["audit_metrics"]
    assert "final_equity" in audit["audit_metrics"]
    assert "end_open_position_count" in audit["audit_metrics"]
    assert audit["suppressed_entry_rejection_count"] >= 0


def test_three_stage_pipeline_ranks_audit_passed_promoted_rows_before_top_train() -> None:
    audit_passed = {
        "candidate_source": "audit_passed",
        "top_train": [{"name": "weak_fast", "train_metrics": {"broker_net_return_pct": 0.01}}],
        "top_promoted": [{"name": "promoted", "train_metrics": {"broker_net_return_pct": 0.02}}],
        "audit_replays": [
            {"name": "promoted", "audit_pass": True, "audit_metrics": {"broker_net_return_pct": 0.02}},
        ],
    }
    unaudited_fast = {
        "candidate_source": "unaudited_fast",
        "top_train": [{"name": "fast_only", "train_metrics": {"broker_net_return_pct": 0.50}}],
        "top_promoted": [],
        "audit_replays": [],
    }

    ranked = sorted([unaudited_fast, audit_passed], key=_stage3_sort_key)

    assert ranked[0]["candidate_source"] == "audit_passed"


def test_three_stage_pipeline_prefers_any_audit_pass_before_unaudited_promoted_fallback() -> None:
    audited_replay = {
        "candidate_source": "audited_replay",
        "top_train": [{"name": "fast", "train_metrics": {"broker_net_return_pct": 0.01}}],
        "top_promoted": [{"name": "promoted_without_audit", "train_metrics": {"broker_net_return_pct": 0.02}}],
        "audit_replays": [
            {"name": "diagnostic_audit", "audit_pass": True, "audit_metrics": {"broker_net_return_pct": 0.015}},
        ],
    }
    unaudited_promoted = {
        "candidate_source": "unaudited_promoted",
        "top_train": [{"name": "fast_only", "train_metrics": {"broker_net_return_pct": 0.10}}],
        "top_promoted": [{"name": "promoted_fast_only", "train_metrics": {"broker_net_return_pct": 0.10}}],
        "audit_replays": [],
    }

    ranked = sorted([unaudited_promoted, audited_replay], key=_stage3_sort_key)

    assert ranked[0]["candidate_source"] == "audited_replay"


def test_stage2_calibration_ranks_official_broker_net_over_proxy_net() -> None:
    high_proxy_bad_broker = _calibrated_stage2_row("high_proxy", rank=0, proxy=0.90, broker=0.01, mfe=1.2)
    lower_proxy_good_broker = _calibrated_stage2_row("better_broker", rank=1, proxy=0.20, broker=0.08, mfe=0.5)

    selected = select_calibrated_stage2_rows([high_proxy_bad_broker, lower_proxy_good_broker], finalist_count=2, require_audit_pass=True)

    assert [row["name"] for row in selected][:2] == ["better_broker", "high_proxy"]


def test_stage2_calibration_diversity_rows_must_be_broker_nonnegative_and_converting() -> None:
    rows = [
        _calibrated_stage2_row(f"primary_{index}", rank=index, proxy=0.10 + index * 0.01, broker=0.06 - index * 0.01, mfe=0.2)
        for index in range(3)
    ]
    rows.append(_calibrated_stage2_row("negative_broker_high_mfe", rank=3, proxy=0.80, broker=-0.01, mfe=3.0))
    rows.append(_calibrated_stage2_row("poor_conversion_high_mfe", rank=4, proxy=0.70, broker=0.02, mfe=2.5, filled_selected_rate=0.05))
    rows.append(_calibrated_stage2_row("eligible_diversity", rank=5, proxy=0.30, broker=0.01, mfe=2.0, filled_selected_rate=0.30))

    selected = select_calibrated_stage2_rows(rows, finalist_count=4, require_audit_pass=True)

    assert [row["name"] for row in selected] == ["primary_0", "primary_1", "primary_2", "eligible_diversity"]


def test_large_parallel_eval_uses_pickle_safe_core_payload(tmp_path: Path) -> None:
    bars = _bars(
        [
            (100.0, 100.2, 99.8, 100.0),
            (100.0, 100.2, 99.8, 100.1),
            (100.1, 100.4, 99.9, 100.2),
            (100.2, 100.4, 100.0, 100.2),
            (100.2, 100.5, 100.1, 100.3),
            (100.3, 100.6, 100.2, 100.5),
            (100.6, 101.0, 100.5, 100.8),
            (100.8, 101.4, 100.7, 101.2),
        ]
    )
    dataset = _dataset(bars)
    ctx = _ctx(bars)
    context_by_key = {(TRADE_DATE, "005930"): ctx}
    selections = [Selection(TRADE_DATE, "005930", 1.0, "unit")]
    counts = _selection_counts(selections, dataset.trading_dates)
    cfg = KALCBConfig(slippage_bps=0.0, commission_bps=0.0, tax_bps_on_sell=0.0)
    specs = [
        name_plan(TradePlanSpec("", _name_entry(EntrySpec("", "post_or_momentum", max_signal_bars=index + 1)), eod_flatten_exit_spec()))
        for index in range(32)
    ]
    baseline = {"slot_cumulative_net_return_pct": 0.0, "active_day_net_pct": 0.0, "max_drawdown_net_pct": 0.0}
    folds = _resolve_folds(dataset.trading_dates, 1)

    rows = _evaluate_specs(specs, selections, dataset, context_by_key, cfg, dataset.trading_dates, folds, counts, baseline, (), tmp_path, "process", max_workers=2)

    assert len(rows) == 32
    assert [row.spec.name for row in rows] == sorted(row.spec.name for row in rows)
    assert all(row.train_metrics["same_bar_fill_count"] == 0.0 for row in rows)


def test_prepared_context_cache_reuses_causal_replay_bundle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    bars = _bars(
        [
            (100.0, 100.2, 99.8, 100.0),
            (100.0, 100.2, 99.8, 100.1),
            (100.1, 100.4, 99.9, 100.2),
            (100.2, 100.4, 100.0, 100.2),
            (100.2, 100.5, 100.1, 100.3),
            (100.3, 100.6, 100.2, 100.5),
            (100.6, 101.0, 100.5, 100.8),
        ]
    )
    dataset = _dataset(bars)
    ctx = _ctx(bars)
    calls = {"prepare": 0}

    def fake_prepare(config: dict[str, object]) -> KALCBFirst30Dataset:
        calls["prepare"] += 1
        return dataset

    monkeypatch.setattr(trade_plan_sweep_module, "prepare_first30_dataset", fake_prepare)
    monkeypatch.setattr(trade_plan_sweep_module, "_build_contexts", lambda _dataset: {TRADE_DATE: (ctx,)})
    monkeypatch.setattr(
        trade_plan_sweep_module,
        "build_fixed_candidate_selections",
        lambda _source, _dataset, _contexts: (
            [Selection(TRADE_DATE, "005930", 1.0, "unit")],
            {TRADE_DATE: ("005930",)},
        ),
    )
    config = {
        "data_root": str(tmp_path / "intraday"),
        "daily_data_root": str(tmp_path / "daily"),
        "baseline": {"holdout_start": "2026-04-01"},
        "kalcb": {"session": {"flatten_time": "15:20"}},
    }

    first = load_or_build_prepared_context(config, output_dir=tmp_path, compiled_cache_dir=tmp_path / "cache", train_only=True)
    second = load_or_build_prepared_context(config, output_dir=tmp_path, compiled_cache_dir=tmp_path / "cache", train_only=True)
    rebuilt = load_or_build_prepared_context(
        config,
        output_dir=tmp_path,
        compiled_cache_dir=tmp_path / "cache",
        train_only=True,
        force_rebuild_cache=True,
    )

    assert calls["prepare"] == 2
    assert first.cache_metadata["cache_hit"] is False
    assert second.cache_metadata["cache_hit"] is True
    assert rebuilt.cache_metadata["cache_hit"] is False
    assert second.cache_key == first.cache_key
    assert second.compiled_replay.source_fingerprint == first.compiled_replay.source_fingerprint
    assert second.baseline_train == first.baseline_train


def test_official_trade_plan_objective_prefers_portfolio_net_over_slot_proxy() -> None:
    spec = baseline_trade_plan_spec()
    fold_rows = (
        {"fold": 1, "start": "2026-01-01", "end": "2026-01-02", "metrics": {"slot_cumulative_net_return_pct": 0.10}},
    )
    baseline = _objective_metrics(broker=0.05, slot=1.0, active=0.01, selected=0.01, dd=0.01)
    high_slot_low_portfolio = PlanResult(
        spec=spec,
        score=0.0,
        rejected=False,
        reject_reason="",
        train_metrics=_objective_metrics(broker=0.04, slot=9.0, active=0.03, selected=0.03, dd=0.01),
        fold_metrics=fold_rows,
    )
    lower_slot_higher_portfolio = PlanResult(
        spec=spec,
        score=0.0,
        rejected=False,
        reject_reason="",
        train_metrics=_objective_metrics(broker=0.08, slot=1.5, active=0.02, selected=0.02, dd=0.01),
        fold_metrics=fold_rows,
    )

    low_score, _ = score_plan(high_slot_low_portfolio.train_metrics, fold_rows, baseline)
    high_score, _ = score_plan(lower_slot_higher_portfolio.train_metrics, fold_rows, baseline)

    assert high_score > low_score
    assert _plan_sort_key(lower_slot_higher_portfolio) < _plan_sort_key(high_slot_low_portfolio)
    assert promotion_pass(high_slot_low_portfolio.train_metrics, fold_rows, baseline, fold_rows) is False
    assert promotion_pass(lower_slot_higher_portfolio.train_metrics, fold_rows, baseline, fold_rows) is True


def test_core_replay_uses_aggressive_but_capped_portfolio_risk_policy() -> None:
    cfg = KALCBConfig(
        ws_budget=10,
        risk_per_trade_pct=0.001,
        max_position_notional_pct=0.10,
        heat_cap_r=4.0,
        max_positions=6,
        max_per_sector=3,
    )

    plan_cfg = _core_config_for_spec(cfg, baseline_trade_plan_spec(), audit=False)

    assert plan_cfg.risk_per_trade_pct == pytest.approx(0.007)
    assert plan_cfg.max_position_notional_pct == pytest.approx(0.45)
    assert plan_cfg.max_positions == 8
    assert plan_cfg.max_per_sector == 8
    assert plan_cfg.heat_cap_r == pytest.approx(0.04 / 0.007)
    assert plan_cfg.intraday_leverage == pytest.approx(2.0)


def test_hard_drawdown_ceiling_rejects_portfolio_plans() -> None:
    metrics = _objective_metrics(broker=0.40, slot=10.0, active=0.03, selected=0.03, dd=0.081)
    baseline = _objective_metrics(broker=0.10, slot=1.0, active=0.01, selected=0.01, dd=0.02)

    score, reject = score_plan(metrics, (), baseline)

    assert score == 0.0
    assert reject.startswith("max_drawdown_ceiling")
    assert promotion_pass(metrics, (), baseline, ()) is False


def _objective_metrics(*, broker: float, slot: float, active: float, selected: float, dd: float) -> dict[str, float]:
    return {
        "selected_count": 100.0,
        "trade_count": 100.0,
        "signal_conversion": 1.0,
        "broker_net_return_pct": broker,
        "broker_expected_total_r": broker * 1000.0,
        "broker_max_drawdown_pct": dd,
        "slot_cumulative_net_return_pct": slot,
        "selected_day_net_pct": selected,
        "active_day_net_pct": active,
        "max_drawdown_net_pct": -dd,
        "avg_mfe_capture": 0.4,
        "net_win_share": 0.55,
        "mfe_ge_1_share": 0.45,
        "mae_le_neg_1_share": 0.1,
        "ambiguous_bar_count": 0.0,
    }


def _calibrated_stage2_row(
    name: str,
    *,
    rank: int,
    proxy: float,
    broker: float,
    mfe: float,
    filled_selected_rate: float = 0.5,
) -> dict[str, object]:
    selected = 100.0
    trades = selected * filled_selected_rate
    return {
        "name": name,
        "source_rank": rank,
        "source_section": "top_portfolio_proxy",
        "proxy_net_return_pct": proxy,
        "calibrated_broker_net_return_pct": broker,
        "calibrated_official_mtm_net_return_pct": broker,
        "calibrated_broker_max_drawdown_pct": 0.02,
        "trade_count": trades,
        "selected_count": selected,
        "filled_selected_rate": filled_selected_rate,
        "proxy_metrics": {"avg_mfe_r": mfe, "frontier_avg_size": 8.0},
        "audit_pass": True,
        "audit_status": "pass",
        "reject_reason": "",
    }


def _ctx(bars: list[MarketBar]) -> First30Context:
    pre = bars[:6]
    high = max(bar.high for bar in pre)
    low = min(bar.low for bar in pre)
    volume = sum(bar.volume for bar in pre)
    vwap = sum(((bar.high + bar.low + bar.close) / 3.0) * bar.volume for bar in pre) / volume
    daily = DailyFeature(
        symbol="005930",
        trade_date=TRADE_DATE,
        prev_close=99.0,
        atr14=2.0,
        return_5d=0.10,
        return_20d=0.0,
        return_60d=0.0,
        adv20_krw=5_000_000_000.0,
        volume_ratio_20d=1.0,
        close20_loc=0.8,
        close60_loc=0.8,
        above_sma20=True,
        above_sma60=True,
    )
    intraday = First30Intraday(
        open=pre[0].open,
        high=high,
        low=low,
        close=pre[-1].close,
        vwap=vwap,
        volume=volume,
        expected_30m_volume=volume,
    )
    return First30Context(
        day=TRADE_DATE,
        symbol="005930",
        sector="TECH",
        daily=daily,
        flow=FlowFeature(available=True),
        market=MarketFeature(score=1.0),
        intraday=intraday,
        bars=tuple(bars),
        post_bars=tuple(bar for bar in bars if bar.timestamp.astimezone(KST).time() >= time(9, 30)),
        first30_ret=intraday.close / intraday.open - 1.0,
        vwap_ret=intraday.close / intraday.vwap - 1.0,
        gap=intraday.open / daily.prev_close - 1.0,
        rel_volume=1.0,
        close_location=(intraday.close - intraday.low) / max(intraday.high - intraday.low, 1e-9),
        open_drawdown=intraday.low / intraday.open - 1.0,
        low_vs_prev_close=intraday.low / daily.prev_close - 1.0,
        range_atr=(intraday.high - intraday.low) / daily.atr14,
    )


def _dataset(bars: list[MarketBar]) -> KALCBFirst30Dataset:
    daily_rows = []
    for index in range(61):
        day = TRADE_DATE - timedelta(days=61 - index)
        daily_rows.append(
            {
                "ticker": "005930",
                "date": day.isoformat(),
                "open": 100.0,
                "high": 112.0 if index == 60 else 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 100_000.0,
            }
        )
    return KALCBFirst30Dataset(
        config={"kalcb": {"session": {"flatten_time": "15:20"}}},
        source_fingerprint="intraday-test",
        daily_source_fingerprint="daily-test",
        data_root=Path("unused"),
        daily_data_root=Path("unused"),
        timeframe="5m",
        symbols=("005930",),
        data_available_symbols=("005930",),
        daily_available_symbols=("005930",),
        unavailable_symbols=(),
        daily_by_symbol={"005930": daily_rows},
        flow_by_symbol={"005930": []},
        index_by_code={},
        trading_dates=(TRADE_DATE,),
        bars_by_key={(TRADE_DATE, "005930"): tuple(bars)},
        sector_map={"005930": "TECH"},
    )


def _bars(rows: list[tuple[float, float, float, float]]) -> list[MarketBar]:
    start = datetime(2026, 1, 5, 9, 0, tzinfo=KST)
    return [
        MarketBar(
            symbol="005930",
            timestamp=start + timedelta(minutes=5 * index),
            timeframe="5m",
            open=open_,
            high=high,
            low=low,
            close=close,
            volume=10_000.0,
        )
        for index, (open_, high, low, close) in enumerate(rows)
    ]
