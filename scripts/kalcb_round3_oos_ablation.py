from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backtests.analysis.metrics import compute_trade_metrics
from backtests.auto.shared.cache_keys import stable_signature
from backtests.config import load_yaml_config, normalize_runtime_config
from backtests.engine.replay import run_replay
from backtests.engine.sim_broker import BrokerCosts
from backtests.strategies.kalcb.fixed_trade_plan_phase import (
    HARD_REJECTS,
    SCORE_COMPONENTS,
    SOURCE_PATH_MUTATION,
    SOURCE_RANK_MUTATION,
    SOURCE_SECTION_MUTATION,
    _add_fold_metrics,
    _broker_trade_rows,
    _decision_summary,
    _normalize_mutations,
    _scaled_component,
    reject_reason,
    score_fixed,
)
from backtests.strategies.kalcb.runner import KALCBReplayAdapter, _collapse_exit_legs
from backtests.strategies.kalcb.trade_plan_sweep import (
    PRIMARY_OBJECTIVE_METRIC,
    _add_compiled_candidate_pool_metrics,
    _add_portfolio_equivalent_metrics,
    _add_return_divergence_metrics,
    _broker_trades_to_slot_outcomes,
    _clone_snapshots_for_replay,
    _fold_metrics_from_outcomes_for_dates,
    _replay_digest,
    load_or_build_prepared_context,
    summarize_outcomes,
)
from strategy_kalcb.config import KALCBConfig


OUT_DIR = REPO_ROOT / "tmp" / "kalcb_round3_oos_repair"
ROUND_ROOT = REPO_ROOT / "data" / "backtests" / "output" / "kalcb"
CONFIG_PATH = REPO_ROOT / "config" / "optimization" / "kalcb.yaml"

METRIC_KEYS = (
    "broker_net_return_pct",
    "official_mtm_net_return_pct",
    "trade_count",
    "active_days",
    "avg_trade_net_pct",
    "net_win_share",
    "broker_max_drawdown_pct",
    "avg_mfe_capture",
    "mae_le_neg_1_share",
    "avg_mfe_r",
    "avg_mae_r",
    "target_hit_share",
    "exit_reason_eod_flatten_share",
    "same_bar_fill_count",
    "end_open_position_count",
    "worst_fold_net",
    "median_fold_net",
)

MANDATORY_LABELS = {
    "r1_seed_rank4",
    "r1_plus_round2_failed_followthrough",
    "r2_final_rank4",
    "r2_final_switched_to_rank0",
    "r2_rank0_plus_round3_entry_gates",
    "r2_rank0_entry_plus_target36",
    "r2_rank0_entry_target_plus_round3_ff",
    "round3_final",
    "final_revert_source_rank4",
    "final_revert_round3_entry_gates",
    "final_remove_min_bar_ret",
    "final_remove_relvol_gate",
    "final_remove_target",
    "final_revert_failed_followthrough",
    "final_revert_round2_risk",
    "final_no_target_revert_source",
    "final_no_entry_no_target",
    "final_round2_exit_risk_source",
    "final_round2_exit_risk_rank0",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def _pct(value: Any) -> float:
    try:
        return float(value or 0.0) * 100.0
    except (TypeError, ValueError):
        return 0.0


def _num(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _clean_metric_row(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: metrics.get(key) for key in METRIC_KEYS if key in metrics}


def _source_from_diag(diag: dict[str, Any]) -> dict[str, Any]:
    source = dict(diag.get("source") or {})
    path = str(source.get("path") or "")
    if not path:
        raise ValueError("diagnostics_summary.json does not expose a source.path")
    return {
        SOURCE_PATH_MUTATION: path,
        SOURCE_SECTION_MUTATION: str(source.get("section") or "top_portfolio_proxy"),
        SOURCE_RANK_MUTATION: int(source.get("rank") or 0),
    }


def _source_from_mutations(mutations: dict[str, Any], default_source: dict[str, Any]) -> dict[str, Any]:
    return {
        SOURCE_PATH_MUTATION: str(mutations.get(SOURCE_PATH_MUTATION, default_source[SOURCE_PATH_MUTATION])),
        SOURCE_SECTION_MUTATION: str(mutations.get(SOURCE_SECTION_MUTATION, default_source[SOURCE_SECTION_MUTATION])),
        SOURCE_RANK_MUTATION: int(mutations.get(SOURCE_RANK_MUTATION, default_source[SOURCE_RANK_MUTATION])),
    }


def _remove_source_keys(mutations: dict[str, Any]) -> dict[str, Any]:
    out = dict(mutations)
    out.pop(SOURCE_PATH_MUTATION, None)
    out.pop(SOURCE_SECTION_MUTATION, None)
    out.pop(SOURCE_RANK_MUTATION, None)
    return out


def _merge(*parts: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for part in parts:
        out.update(dict(part or {}))
    return out


def _with_source(mutations: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    return _merge(mutations, source)


def _window_config(base: dict[str, Any], window: str) -> dict[str, Any]:
    cfg = deepcopy(base)
    cfg["use_full_available_window"] = False
    baseline = dict(cfg.get("baseline") or {})
    if window == "oos":
        cfg["start"] = baseline["holdout_start"]
        cfg["end"] = baseline["holdout_end"]
    else:
        cfg["start"] = base["start"]
        cfg["end"] = base["end"]
    return cfg


class ReplayHarness:
    def __init__(self, base_config: dict[str, Any], out_dir: Path, *, quick: bool = False):
        self.base_config = base_config
        self.out_dir = out_dir
        self.quick = quick
        self.contexts: dict[tuple[str, str, str, int], Any] = {}
        self.progress_path = out_dir / "progress.jsonl"
        if self.progress_path.exists():
            self.progress_path.unlink()

    def _status(self, stage: str, **extra: Any) -> None:
        payload = {"ts": datetime.utcnow().isoformat(timespec="seconds") + "Z", "stage": stage, **extra}
        print(json.dumps(payload, sort_keys=True, default=str), flush=True)
        _append_jsonl(self.progress_path, payload)

    def context(self, window: str, source: dict[str, Any]) -> Any:
        source_path = str(source[SOURCE_PATH_MUTATION])
        section = str(source[SOURCE_SECTION_MUTATION])
        rank = int(source[SOURCE_RANK_MUTATION])
        key = (window, source_path, section, rank)
        cached = self.contexts.get(key)
        if cached is not None:
            return cached
        cfg = _window_config(self.base_config, window)
        self._status("context_build_start", window=window, section=section, rank=rank)
        context = load_or_build_prepared_context(
            cfg,
            optimized_source=source_path,
            candidate_section=section,
            candidate_rank=rank,
            strict_candidate_source=False,
            output_dir=self.out_dir,
            train_only=False,
            fold_count=2,
            compiled_cache_dir=self.base_config.get("compiled_cache_dir"),
            force_rebuild_cache=False,
            status_callback=lambda stage, **extra: self._status(
                "context_" + stage,
                window=window,
                section=section,
                rank=rank,
                **extra,
            ),
        )
        self.contexts[key] = context
        return context

    def evaluate(self, label: str, mutations: dict[str, Any], window: str, default_source: dict[str, Any]) -> dict[str, Any]:
        started = time.monotonic()
        source = _source_from_mutations(mutations, default_source)
        context = self.context(window, source)
        initial_equity = float(context.compiled_replay.initial_equity)
        plan_cfg = KALCBConfig.from_mapping(context.training_config, {}).with_mutations(_normalize_mutations(_remove_source_keys(mutations)))
        costs = BrokerCosts(
            commission_bps=plan_cfg.commission_bps,
            tax_bps_on_sell=plan_cfg.tax_bps_on_sell,
            slippage_bps=plan_cfg.slippage_bps,
        )
        adapter = KALCBReplayAdapter(
            plan_cfg,
            _clone_snapshots_for_replay(context.compiled_replay.snapshots),
            initial_equity=initial_equity,
            costs=costs,
        )
        replay = run_replay(
            context.compiled_replay.bars,
            adapter,
            initial_equity=initial_equity,
            costs=costs,
            close_open_positions=False,
            bars_are_ordered=True,
            buying_power_leverage=max(float(plan_cfg.intraday_leverage), 1.0),
        )
        replay.decisions.extend(adapter._sync_new_fills(replay.broker))
        trades = _collapse_exit_legs(replay.trades)
        outcomes = _broker_trades_to_slot_outcomes(trades, plan_cfg)
        metrics = summarize_outcomes(outcomes, session_dates=context.train_dates, selection_counts=context.selection_counts)
        _add_compiled_candidate_pool_metrics(metrics, context.compiled_replay, context.train_dates, len(outcomes))
        broker_metrics = compute_trade_metrics(trades, replay.equity_curve, initial_equity=initial_equity)
        final_equity = float(replay.equity_curve[-1]) if replay.equity_curve else initial_equity
        metrics.update(
            {
                "broker_net_return_pct": float(broker_metrics.get("net_return_pct", 0.0)),
                "official_mtm_net_return_pct": final_equity / max(initial_equity, 1.0) - 1.0,
                "final_equity": final_equity,
                "end_open_position_count": float(len(replay.broker.positions)),
                "broker_net_profit": float(broker_metrics.get("net_profit", 0.0)),
                "broker_max_drawdown_pct": float(broker_metrics.get("max_drawdown_pct", 0.0)),
                "broker_expected_total_r": float(broker_metrics.get("expected_total_r", 0.0)),
                "broker_avg_r": float(broker_metrics.get("avg_r", 0.0)),
                "broker_mfe_capture": float(broker_metrics.get("mfe_capture", 0.0)),
                "broker_trade_count": float(broker_metrics.get("total_trades", 0.0)),
                "same_bar_fill_count": float(replay.broker.same_bar_fill_violations),
                "forced_replay_close_count": 0.0,
                "rejected_order_count": 0.0,
                "mark_to_market_equity_points": float(len(replay.equity_curve)),
                "broker_net_return_pct_basis": "closed_trade_net_pnl_over_initial_equity",
                "net_return_pct_basis": "closed_trade_net_pnl_over_initial_equity",
                "official_metric_basis": "SimBroker.equity_curve_bar_level_mtm",
                "primary_promotion_metric": PRIMARY_OBJECTIVE_METRIC,
                "total_trades": float(broker_metrics.get("total_trades", metrics.get("trade_count", 0.0))),
                "trades": float(broker_metrics.get("total_trades", metrics.get("trade_count", 0.0))),
                "win_rate": float(metrics.get("net_win_share", 0.0)),
                "source_fingerprint": context.compiled_replay.source_fingerprint,
                "feature_manifest_hash": context.cache_key,
                "candidate_snapshot_hash": context.compiled_replay.candidate_artifact_hash,
                "replay_mode": "fixed_candidate_shared_core_compiled_replay",
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "window": window,
            }
        )
        _add_portfolio_equivalent_metrics(metrics, outcomes, context.train_dates, initial_equity)
        _add_return_divergence_metrics(metrics)
        fold_rows = _fold_metrics_from_outcomes_for_dates(
            outcomes,
            context.train_dates,
            context.folds,
            context.selection_counts,
            initial_equity=initial_equity,
        )
        _add_fold_metrics(metrics, fold_rows)
        metrics["score_components"] = {name: _scaled_component(name, metrics) for name in SCORE_COMPONENTS}
        metrics["immutable_score"] = score_fixed(metrics)
        metrics["reject_reason_train_policy"] = reject_reason(metrics, HARD_REJECTS)
        metrics["training_window_start"] = context.train_dates[0].isoformat() if context.train_dates else ""
        metrics["training_window_end"] = context.train_dates[-1].isoformat() if context.train_dates else ""
        metrics["training_session_count"] = len(context.train_dates)
        trade_rows = tuple(_broker_trade_rows(trades))
        return {
            "label": label,
            "window": window,
            "source": source,
            "mutations": dict(mutations),
            "metrics": dict(metrics),
            "metric_row": _clean_metric_row(metrics),
            "fold_rows": fold_rows,
            "trade_rows": trade_rows,
            "decision_summary": _decision_summary(replay.decisions),
            "replay_digest": _replay_digest(replay, trades),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }


def _round_artifacts() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for round_name in ("round_1", "round_2", "round_3"):
        root = ROUND_ROOT / round_name
        optimized = _read_json(root / "optimized_config.json")
        diag_path = root / "diagnostics_summary.json"
        run_path = root / "run_summary.json"
        out[round_name] = {
            "root": str(root),
            "optimized": optimized,
            "diagnostics": _read_json(diag_path) if diag_path.exists() else {},
            "run_summary": _read_json(run_path) if run_path.exists() else {},
        }
    return out


def _candidate(label: str, kind: str, mutations: dict[str, Any], reason: str = "") -> dict[str, Any]:
    return {"label": label, "kind": kind, "mutations": dict(mutations), "reason": reason}


def _dedupe(candidates: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in candidates:
        key = stable_signature(item["mutations"])
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _build_candidate_set(artifacts: dict[str, dict[str, Any]], *, quick: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    r1 = dict(artifacts["round_1"]["optimized"]["mutations"])
    r2 = dict(artifacts["round_2"]["optimized"]["mutations"])
    r3 = dict(artifacts["round_3"]["optimized"]["mutations"])
    source_r2 = _source_from_diag(artifacts["round_2"]["diagnostics"])
    source_r3 = _source_from_diag(artifacts["round_3"]["diagnostics"])
    source_path = source_r3[SOURCE_PATH_MUTATION]

    base_r1 = _with_source(r1, source_r2)
    phase2_ff = {
        "kalcb.exit.failed_followthrough_bars": 8,
        "kalcb.exit.failed_followthrough_mfe_r": 1.0,
        "kalcb.exit.failed_followthrough_close_r": -0.5,
    }
    phase2_risk = {
        "kalcb.risk.max_position_notional_pct": 0.4,
        "kalcb.risk.risk_per_trade_pct": 0.0065,
    }
    phase3_source = source_r3
    phase3_entry = {
        "kalcb.entry.min_bar_ret": 0.005,
        "kalcb.entry.min_first30_rel_volume": 1.0,
    }
    phase3_target = {"kalcb.exit.target_r": 36.0}
    phase3_ff = {
        "kalcb.exit.failed_followthrough_bars": 10,
        "kalcb.exit.failed_followthrough_mfe_r": 1.25,
        "kalcb.exit.failed_followthrough_close_r": -0.25,
    }
    phase3_risk = {
        "kalcb.risk.max_position_notional_pct": 0.5,
        "kalcb.risk.risk_per_trade_pct": 0.0055,
    }

    sequence = [
        _candidate("r1_seed_rank4", "cumulative_sequence", base_r1, "round_1 fixed-pct seed evaluated on rank4 source"),
        _candidate("r1_plus_round2_failed_followthrough", "cumulative_sequence", _merge(base_r1, phase2_ff), "add round_2 failed-followthrough block"),
        _candidate("r2_final_rank4", "cumulative_sequence", _with_source(r2, source_r2), "round_2 accepted cumulative set"),
        _candidate("r2_final_switched_to_rank0", "cumulative_sequence", _with_source(r2, source_r3), "round_3 phase 1 source/rank switch only"),
        _candidate("r2_rank0_plus_round3_entry_gates", "cumulative_sequence", _merge(_with_source(r2, source_r3), phase3_entry), "add round_3 first30 return/relative-volume gates"),
        _candidate("r2_rank0_entry_plus_target36", "cumulative_sequence", _merge(_with_source(r2, source_r3), phase3_entry, phase3_target), "add round_3 36R target"),
        _candidate("r2_rank0_entry_target_plus_round3_ff", "cumulative_sequence", _merge(_with_source(r2, source_r3), phase3_entry, phase3_target, phase3_ff), "add round_3 target-aware failed-followthrough"),
        _candidate("round3_final", "cumulative_sequence", _with_source(r3, source_r3), "round_3 final cumulative set"),
    ]

    final = _with_source(r3, source_r3)
    ablations = [
        _candidate("final_revert_source_rank4", "ablation", _with_source(r3, source_r2), "remove round_3 source/rank switch"),
        _candidate("final_revert_round3_entry_gates", "ablation", _merge(final, {"kalcb.entry.min_bar_ret": 0.0, "kalcb.entry.min_first30_rel_volume": 0.0}), "remove both round_3 entry gates"),
        _candidate("final_remove_min_bar_ret", "ablation", _merge(final, {"kalcb.entry.min_bar_ret": 0.0}), "remove min first30 return gate"),
        _candidate("final_remove_relvol_gate", "ablation", _merge(final, {"kalcb.entry.min_first30_rel_volume": 0.0}), "remove first30 relative-volume gate"),
        _candidate("final_remove_target", "ablation", _merge(final, {"kalcb.exit.target_r": 0.0}), "remove 36R target"),
        _candidate("final_revert_failed_followthrough", "ablation", _merge(final, phase2_ff), "revert failed-followthrough to round_2 settings"),
        _candidate("final_revert_round2_risk", "ablation", _merge(final, phase2_risk), "revert risk sizing to round_2 settings"),
        _candidate("final_cap40_only", "ablation", _merge(final, {"kalcb.risk.max_position_notional_pct": 0.4}), "only undo round_3 notional cap"),
        _candidate("final_risk0065_only", "ablation", _merge(final, {"kalcb.risk.risk_per_trade_pct": 0.0065}), "only undo round_3 risk-per-trade"),
        _candidate("final_no_target_revert_source", "ablation_combo", _merge(final, source_r2, {"kalcb.exit.target_r": 0.0}), "remove target and source switch together"),
        _candidate("final_no_entry_no_target", "ablation_combo", _merge(final, {"kalcb.entry.min_bar_ret": 0.0, "kalcb.entry.min_first30_rel_volume": 0.0, "kalcb.exit.target_r": 0.0}), "remove entry gates and target"),
        _candidate("final_round2_exit_risk_source", "ablation_combo", _with_source(r2, source_r2), "complete revert to round_2 accepted set"),
        _candidate("final_round2_exit_risk_rank0", "ablation_combo", _with_source(r2, source_r3), "round_2 exits/risk with round_3 source"),
    ]

    perturbations: list[dict[str, Any]] = []
    sections = ("top_portfolio_proxy", "top_slot_return", "top_pareto", "top_combined")
    ranks = range(0, 8 if not quick else 3)
    for section in sections:
        for rank in ranks:
            source = {SOURCE_PATH_MUTATION: source_path, SOURCE_SECTION_MUTATION: section, SOURCE_RANK_MUTATION: rank}
            perturbations.append(_candidate(f"final_source_{section}_rank{rank}", "source_perturbation", _with_source(r3, source), "source section/rank perturbation"))
            if section == "top_portfolio_proxy":
                perturbations.append(_candidate(f"round2_plan_source_{section}_rank{rank}", "source_perturbation", _with_source(r2, source), "round_2 plan on alternate source rank"))

    min_bar_values = (0.0, 0.0025, 0.005, 0.0075, 0.01)
    relvol_values = (0.0, 0.75, 1.0, 1.25, 1.5)
    cpr_values = (0.0, 0.55, 0.65, 0.75)
    vwap_values = (0.0, 0.001, 0.002)
    close_loc_values = (0.0, 0.55, 0.65)
    if quick:
        min_bar_values = (0.0, 0.005, 0.0075)
        relvol_values = (0.0, 1.0, 1.25)
        cpr_values = (0.0, 0.65)
        vwap_values = (0.0,)
        close_loc_values = (0.0,)
    for min_bar in min_bar_values:
        for relvol in relvol_values:
            perturbations.append(
                _candidate(
                    f"entry_ret{min_bar:g}_relvol{relvol:g}",
                    "entry_perturbation",
                    _merge(final, {"kalcb.entry.min_bar_ret": min_bar, "kalcb.entry.min_first30_rel_volume": relvol}),
                    "first30 return and relative-volume grid",
                )
            )
    for cpr in cpr_values:
        for relvol in relvol_values:
            perturbations.append(
                _candidate(
                    f"entry_cpr{cpr:g}_relvol{relvol:g}",
                    "entry_perturbation",
                    _merge(final, {"kalcb.entry.min_first30_signal_cpr": cpr, "kalcb.entry.min_first30_rel_volume": relvol}),
                    "first30 CPR and relative-volume grid",
                )
            )
    for vwap in vwap_values:
        for close_loc in close_loc_values:
            perturbations.append(
                _candidate(
                    f"entry_vwap{vwap:g}_cl{close_loc:g}",
                    "entry_perturbation",
                    _merge(final, {"kalcb.entry.min_vwap_ret": vwap, "kalcb.entry.min_close_location": close_loc}),
                    "VWAP return and signal close-location grid",
                )
            )

    target_values = (0.0, 12.0, 16.0, 20.0, 24.0, 28.0, 30.0, 32.0, 36.0, 45.0, 60.0, 70.0)
    if quick:
        target_values = (0.0, 24.0, 36.0)
    for target in target_values:
        perturbations.append(
            _candidate(f"exit_target_{target:g}r", "exit_perturbation", _merge(final, {"kalcb.exit.target_r": target}), "target-r grid")
        )

    ff_variants = [
        ("off", {"kalcb.exit.failed_followthrough_bars": 0, "kalcb.exit.failed_followthrough_mfe_r": 0.0, "kalcb.exit.failed_followthrough_close_r": 0.0}),
        ("6_075_m025", {"kalcb.exit.failed_followthrough_bars": 6, "kalcb.exit.failed_followthrough_mfe_r": 0.75, "kalcb.exit.failed_followthrough_close_r": -0.25}),
        ("8_100_m050", phase2_ff),
        ("10_125_m025", phase3_ff),
        ("12_150_000", {"kalcb.exit.failed_followthrough_bars": 12, "kalcb.exit.failed_followthrough_mfe_r": 1.5, "kalcb.exit.failed_followthrough_close_r": 0.0}),
    ]
    if quick:
        ff_variants = ff_variants[:3]
    for name, muts in ff_variants:
        perturbations.append(_candidate(f"exit_ff_{name}", "exit_perturbation", _merge(final, muts), "failed-followthrough grid"))

    time_decay_variants = [
        ("36_2_0", {"kalcb.exit.time_decay_bars": 36, "kalcb.exit.time_decay_min_mfe_r": 2.0, "kalcb.exit.time_decay_max_current_r": 0.0}),
        ("48_3_05", {"kalcb.exit.time_decay_bars": 48, "kalcb.exit.time_decay_min_mfe_r": 3.0, "kalcb.exit.time_decay_max_current_r": 0.5}),
        ("72_5_1", {"kalcb.exit.time_decay_bars": 72, "kalcb.exit.time_decay_min_mfe_r": 5.0, "kalcb.exit.time_decay_max_current_r": 1.0}),
    ]
    max_hold_values = (48, 60, 72, 84)
    if quick:
        time_decay_variants = time_decay_variants[:1]
        max_hold_values = (60,)
    for name, muts in time_decay_variants:
        perturbations.append(_candidate(f"exit_time_decay_{name}", "exit_perturbation", _merge(final, muts), "targeted giveback/time decay"))
    for bars in max_hold_values:
        perturbations.append(_candidate(f"exit_max_hold_{bars}", "exit_perturbation", _merge(final, {"kalcb.exit.max_hold_bars": bars}), "max-hold cap"))

    hard_stop_variants = [
        ("fixed003", {"kalcb.exit.hard_stop_enabled": True, "kalcb.exit.stop_mode": "fixed_pct", "kalcb.exit.stop_pct": 0.003}),
        ("fixed005", {"kalcb.exit.hard_stop_enabled": True, "kalcb.exit.stop_mode": "fixed_pct", "kalcb.exit.stop_pct": 0.005}),
        ("fixed007", {"kalcb.exit.hard_stop_enabled": True, "kalcb.exit.stop_mode": "fixed_pct", "kalcb.exit.stop_pct": 0.007}),
        ("first30_low", {"kalcb.exit.hard_stop_enabled": True, "kalcb.exit.stop_mode": "first30_low", "kalcb.exit.stop_pct": 0.003}),
    ]
    if quick:
        hard_stop_variants = hard_stop_variants[:2]
    for name, muts in hard_stop_variants:
        perturbations.append(_candidate(f"exit_hard_stop_{name}", "exit_perturbation", _merge(final, muts), "hard-stop activation grid"))
        perturbations.append(_candidate(f"exit_no_target_hard_stop_{name}", "exit_perturbation", _merge(final, {"kalcb.exit.target_r": 0.0}, muts), "remove target plus hard stop"))

    risk_values = (0.003, 0.004, 0.005, 0.0055, 0.0065, 0.007)
    cap_values = (0.30, 0.35, 0.40, 0.45, 0.50)
    max_positions = (4, 6, 8)
    if quick:
        risk_values = (0.004, 0.0055, 0.0065)
        cap_values = (0.35, 0.5)
        max_positions = (6, 8)
    for risk in risk_values:
        for cap in cap_values:
            perturbations.append(
                _candidate(
                    f"risk_{risk:g}_cap{cap:g}",
                    "risk_perturbation",
                    _merge(final, {"kalcb.risk.risk_per_trade_pct": risk, "kalcb.risk.max_position_notional_pct": cap}),
                    "risk-per-trade and notional-cap grid",
                )
            )
    for pos in max_positions:
        perturbations.append(_candidate(f"risk_maxpos_{pos}", "risk_perturbation", _merge(final, {"kalcb.risk.max_positions": pos}), "max position count"))

    targeted = [
        _candidate("targeted_rank4_no_target_risk004_cap035", "targeted_repair", _merge(final, source_r2, {"kalcb.exit.target_r": 0.0, "kalcb.risk.risk_per_trade_pct": 0.004, "kalcb.risk.max_position_notional_pct": 0.35}), "reduce source/target/risk overfit together"),
        _candidate("targeted_rank4_no_target_relvol0_risk004_cap035", "targeted_repair", _merge(final, source_r2, {"kalcb.exit.target_r": 0.0, "kalcb.entry.min_first30_rel_volume": 0.0, "kalcb.risk.risk_per_trade_pct": 0.004, "kalcb.risk.max_position_notional_pct": 0.35}), "restore frequency while reducing sizing"),
        _candidate("targeted_rank4_no_target_hardstop003", "targeted_repair", _merge(final, source_r2, {"kalcb.exit.target_r": 0.0, "kalcb.exit.hard_stop_enabled": True, "kalcb.exit.stop_mode": "fixed_pct", "kalcb.exit.stop_pct": 0.003}), "rank4 plus loss containment"),
        _candidate("targeted_rank0_no_target_hardstop003_risk004", "targeted_repair", _merge(final, {"kalcb.exit.target_r": 0.0, "kalcb.exit.hard_stop_enabled": True, "kalcb.exit.stop_mode": "fixed_pct", "kalcb.exit.stop_pct": 0.003, "kalcb.risk.risk_per_trade_pct": 0.004, "kalcb.risk.max_position_notional_pct": 0.35}), "rank0 source with target removed and tighter loss containment"),
        _candidate("targeted_final_target24_time_decay48", "targeted_repair", _merge(final, {"kalcb.exit.target_r": 24.0, "kalcb.exit.time_decay_bars": 48, "kalcb.exit.time_decay_min_mfe_r": 3.0, "kalcb.exit.time_decay_max_current_r": 0.5}), "capture high MFE earlier and cut stale winners"),
        _candidate("targeted_final_target20_ff6_risk004", "targeted_repair", _merge(final, {"kalcb.exit.target_r": 20.0, "kalcb.exit.failed_followthrough_bars": 6, "kalcb.exit.failed_followthrough_mfe_r": 0.75, "kalcb.exit.failed_followthrough_close_r": -0.25, "kalcb.risk.risk_per_trade_pct": 0.004, "kalcb.risk.max_position_notional_pct": 0.35}), "faster failed-followthrough and lower sizing"),
        _candidate("targeted_rank4_target20_cpr55_relvol075", "targeted_repair", _merge(final, source_r2, {"kalcb.exit.target_r": 20.0, "kalcb.entry.min_first30_signal_cpr": 0.55, "kalcb.entry.min_first30_rel_volume": 0.75}), "quality gate with less aggressive rank/source"),
        _candidate("targeted_rank0_cpr65_relvol125_target24", "targeted_repair", _merge(final, {"kalcb.entry.min_first30_signal_cpr": 0.65, "kalcb.entry.min_first30_rel_volume": 1.25, "kalcb.exit.target_r": 24.0}), "tighter first30 confirmation plus lower target"),
        _candidate("targeted_round2_plan_rank0_cpr55_relvol075", "targeted_repair", _merge(_with_source(r2, source_r3), {"kalcb.entry.min_first30_signal_cpr": 0.55, "kalcb.entry.min_first30_rel_volume": 0.75}), "round_2 exit/risk with mild first30 confirmation"),
        _candidate("targeted_round2_plan_rank4_cpr55_relvol075", "targeted_repair", _merge(_with_source(r2, source_r2), {"kalcb.entry.min_first30_signal_cpr": 0.55, "kalcb.entry.min_first30_rel_volume": 0.75}), "round_2 plan plus mild first30 confirmation"),
    ]

    candidates = _dedupe([*sequence, *ablations, *perturbations, *targeted])
    metadata = {
        "source_r2": source_r2,
        "source_r3": source_r3,
        "round1_mutations": r1,
        "round2_mutations": r2,
        "round3_mutations": r3,
        "candidate_counts": Counter(item["kind"] for item in candidates),
        "phase_blocks": {
            "round2_failed_followthrough": phase2_ff,
            "round2_risk": phase2_risk,
            "round3_source": phase3_source,
            "round3_entry": phase3_entry,
            "round3_target": phase3_target,
            "round3_failed_followthrough": phase3_ff,
            "round3_risk": phase3_risk,
        },
    }
    return candidates, metadata


def _oos_repair_score(metrics: dict[str, Any]) -> float:
    net = _num(metrics.get("broker_net_return_pct"))
    trades = _num(metrics.get("trade_count"))
    avg_trade = _num(metrics.get("avg_trade_net_pct"))
    win = _num(metrics.get("net_win_share"))
    dd = abs(_num(metrics.get("broker_max_drawdown_pct")))
    same_bar = _num(metrics.get("same_bar_fill_count"))
    open_pos = _num(metrics.get("end_open_position_count"))
    return (
        1000.0 * net
        + 180.0 * math.tanh(avg_trade / 0.008)
        + 55.0 * math.tanh((trades - 12.0) / 12.0)
        + 55.0 * (win - 0.40)
        - 720.0 * dd
        - 80.0 * same_bar
        - 80.0 * open_pos
    )


def _combined_score(oos_metrics: dict[str, Any], train_metrics: dict[str, Any], final_train: dict[str, Any]) -> float:
    oos = _oos_repair_score(oos_metrics)
    train_net = _num(train_metrics.get("broker_net_return_pct"))
    final_train_net = max(_num(final_train.get("broker_net_return_pct")), 1e-9)
    train_ratio = train_net / final_train_net
    train_penalty = 0.0
    if train_ratio < 0.75:
        train_penalty += 150.0 * (0.75 - train_ratio)
    if _num(train_metrics.get("trade_count")) < 80:
        train_penalty += 0.75 * (80.0 - _num(train_metrics.get("trade_count")))
    if abs(_num(train_metrics.get("broker_max_drawdown_pct"))) > 0.08:
        train_penalty += 800.0 * (abs(_num(train_metrics.get("broker_max_drawdown_pct"))) - 0.08)
    return oos + 180.0 * math.tanh(train_net / 0.50) - train_penalty


def _impact_vs(results_by_label: dict[str, dict[str, Any]], label: str, baseline_label: str = "round3_final") -> dict[str, Any]:
    row = results_by_label[label]
    base = results_by_label[baseline_label]
    out: dict[str, Any] = {}
    for window in ("oos", "train"):
        if window not in row or window not in base:
            continue
        rm = row[window]["metrics"]
        bm = base[window]["metrics"]
        out[window] = {
            "net_delta_pct_points": 100.0 * (_num(rm.get("broker_net_return_pct")) - _num(bm.get("broker_net_return_pct"))),
            "trade_delta": _num(rm.get("trade_count")) - _num(bm.get("trade_count")),
            "win_rate_delta_pct_points": 100.0 * (_num(rm.get("net_win_share")) - _num(bm.get("net_win_share"))),
            "dd_delta_pct_points": 100.0 * (_num(rm.get("broker_max_drawdown_pct")) - _num(bm.get("broker_max_drawdown_pct"))),
        }
    return out


def _group_trade_rows(rows: Iterable[dict[str, Any]], key: str, *, top_n: int = 12) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key) or "UNKNOWN")].append(row)
    out: list[dict[str, Any]] = []
    for name, group in groups.items():
        pnl = sum(_num(row.get("net_pnl")) for row in group)
        net_r = sum(_num(row.get("r")) for row in group)
        out.append(
            {
                key: name,
                "trades": len(group),
                "net_pnl": pnl,
                "net_r": net_r,
                "win_rate": sum(1 for row in group if _num(row.get("net_pnl")) > 0) / max(len(group), 1),
                "avg_mfe_r": statistics.fmean(_num(row.get("mfe_r")) for row in group),
                "avg_mae_r": statistics.fmean(_num(row.get("mae_r")) for row in group),
            }
        )
    out.sort(key=lambda row: (row["net_pnl"], -row["trades"]))
    return out[:top_n]


def _bucket(value: Any, cuts: tuple[float, ...]) -> str:
    val = _num(value)
    for cut in cuts:
        if val < cut:
            return f"<{cut:g}"
    return f">={cuts[-1]:g}"


def _feature_buckets(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    specs = {
        "first30_ret": (-0.02, 0.0, 0.005, 0.01, 0.02),
        "first30_rel_volume": (0.5, 1.0, 1.5, 2.0, 3.0),
        "first30_signal_bar_cpr": (0.45, 0.55, 0.65, 0.75, 0.85),
        "first30_range_close_location": (0.25, 0.45, 0.55, 0.65, 0.75),
        "mfe_r": (0.5, 1.0, 2.0, 5.0, 10.0),
        "hold_bars": (12, 24, 36, 48, 60, 72),
    }
    report: dict[str, list[dict[str, Any]]] = {}
    for key, cuts in specs.items():
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[_bucket(row.get(key), cuts)].append(row)
        rows_out = []
        for bucket, group in groups.items():
            rows_out.append(
                {
                    "bucket": bucket,
                    "trades": len(group),
                    "net_pnl": sum(_num(row.get("net_pnl")) for row in group),
                    "avg_r": statistics.fmean(_num(row.get("r")) for row in group),
                    "win_rate": sum(1 for row in group if _num(row.get("net_pnl")) > 0) / max(len(group), 1),
                }
            )
        rows_out.sort(key=lambda row: row["net_pnl"])
        report[key] = rows_out
    return report


def _edge_case_diagnostics(final_oos: dict[str, Any]) -> dict[str, Any]:
    rows = list(final_oos.get("trade_rows") or [])
    initial_equity = 100_000_000.0
    metrics = dict(final_oos.get("metrics") or {})
    total_pnl = sum(_num(row.get("net_pnl")) for row in rows)
    worst = sorted(rows, key=lambda row: _num(row.get("net_pnl")))[:12]
    removals = []
    for k in (1, 2, 3, 5, 8, 10):
        removed = worst[: min(k, len(worst))]
        adjusted_pnl = total_pnl - sum(_num(row.get("net_pnl")) for row in removed)
        remaining = [row for row in rows if row not in removed]
        removals.append(
            {
                "remove_worst_k": k,
                "adjusted_net_return_pct": adjusted_pnl / initial_equity,
                "adjusted_win_rate": sum(1 for row in remaining if _num(row.get("net_pnl")) > 0) / max(len(remaining), 1),
                "remaining_trades": len(remaining),
            }
        )
    loss_rows = [row for row in rows if _num(row.get("net_pnl")) < 0]
    worst_loss_share = 0.0
    if loss_rows:
        total_losses = abs(sum(_num(row.get("net_pnl")) for row in loss_rows))
        worst_loss_share = abs(sum(_num(row.get("net_pnl")) for row in worst[:3])) / max(total_losses, 1.0)
    return {
        "final_oos_metrics": _clean_metric_row(metrics),
        "trade_count": len(rows),
        "total_closed_trade_pnl": total_pnl,
        "worst_trades": worst,
        "worst_3_loss_share_of_all_losses": worst_loss_share,
        "remove_worst_impacts": removals,
        "by_symbol_worst": _group_trade_rows(rows, "symbol"),
        "by_exit_reason_worst": _group_trade_rows(rows, "exit_reason"),
        "by_entry_date_worst": _group_trade_rows(rows, "entry_date"),
        "feature_buckets": _feature_buckets(rows),
    }


def _markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]], *, max_rows: int = 20) -> list[str]:
    lines = []
    lines.append("| " + " | ".join(label for label, _ in columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows[:max_rows]:
        vals = []
        for _, key in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                vals.append(f"{value:.4f}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return lines


def _summary_markdown(payload: dict[str, Any]) -> str:
    final_oos = payload["results_by_label"]["round3_final"]["oos"]["metrics"]
    final_train = payload["results_by_label"]["round3_final"]["train"]["metrics"]
    edge = payload["edge_case_diagnostics"]
    ablations = payload["ablation_impacts"]
    confirmed = payload["confirmed_train_ranked"]

    lines: list[str] = []
    lines.append("# KALCB Round 3 OOS Ablation And Repair")
    lines.append("")
    lines.append(f"- Train window: {payload['windows']['train']['start']} to {payload['windows']['train']['end']}")
    lines.append(f"- OOS window: {payload['windows']['oos']['start']} to {payload['windows']['oos']['end']}")
    lines.append(f"- Evaluated OOS candidates: {payload['counts']['oos_evaluated']}")
    lines.append(f"- Train-confirmed candidates: {payload['counts']['train_confirmed']}")
    lines.append("")
    lines.append("## Round 3 Final Baseline")
    lines.append(
        f"- Train: {_pct(final_train.get('broker_net_return_pct')):.2f}% net, "
        f"{_num(final_train.get('trade_count')):.0f} trades, "
        f"{_pct(final_train.get('net_win_share')):.1f}% win, "
        f"{_pct(final_train.get('broker_max_drawdown_pct')):.2f}% max DD."
    )
    lines.append(
        f"- OOS: {_pct(final_oos.get('broker_net_return_pct')):.2f}% net, "
        f"{_num(final_oos.get('trade_count')):.0f} trades, "
        f"{_pct(final_oos.get('net_win_share')):.1f}% win, "
        f"{_pct(final_oos.get('broker_max_drawdown_pct')):.2f}% max DD."
    )
    lines.append("")
    lines.append("## Edge-Case Check")
    lines.append(f"- Worst 3 trades explain {100.0 * _num(edge.get('worst_3_loss_share_of_all_losses')):.1f}% of OOS losses.")
    for row in edge["remove_worst_impacts"]:
        lines.append(
            f"- Remove worst {row['remove_worst_k']}: adjusted OOS net "
            f"{_pct(row['adjusted_net_return_pct']):.2f}% over {row['remaining_trades']} trades."
        )
    lines.append("")
    lines.append("## Ablation Impacts Vs Round 3")
    ablation_rows = []
    for label, impact in ablations.items():
        oos = impact.get("oos", {})
        train = impact.get("train", {})
        ablation_rows.append(
            {
                "label": label,
                "oos_net_delta_pp": oos.get("net_delta_pct_points", 0.0),
                "oos_trade_delta": oos.get("trade_delta", 0.0),
                "train_net_delta_pp": train.get("net_delta_pct_points", 0.0),
                "train_trade_delta": train.get("trade_delta", 0.0),
            }
        )
    ablation_rows.sort(key=lambda row: row["oos_net_delta_pp"], reverse=True)
    lines.extend(
        _markdown_table(
            ablation_rows,
            [
                ("Label", "label"),
                ("OOS Net Delta pp", "oos_net_delta_pp"),
                ("OOS Trade Delta", "oos_trade_delta"),
                ("Train Net Delta pp", "train_net_delta_pp"),
                ("Train Trade Delta", "train_trade_delta"),
            ],
            max_rows=20,
        )
    )
    lines.append("")
    lines.append("## Top Train-Confirmed Repairs")
    top_rows = []
    for row in confirmed[:20]:
        oos = row["oos"]["metrics"]
        train = row["train"]["metrics"]
        top_rows.append(
            {
                "label": row["label"],
                "combined_score": row["combined_score"],
                "oos_net_pct": _pct(oos.get("broker_net_return_pct")),
                "oos_trades": _num(oos.get("trade_count")),
                "oos_win_pct": _pct(oos.get("net_win_share")),
                "train_net_pct": _pct(train.get("broker_net_return_pct")),
                "train_trades": _num(train.get("trade_count")),
            }
        )
    lines.extend(
        _markdown_table(
            top_rows,
            [
                ("Label", "label"),
                ("Score", "combined_score"),
                ("OOS Net %", "oos_net_pct"),
                ("OOS Trades", "oos_trades"),
                ("OOS Win %", "oos_win_pct"),
                ("Train Net %", "train_net_pct"),
                ("Train Trades", "train_trades"),
            ],
            max_rows=20,
        )
    )
    lines.append("")
    lines.append("## Worst OOS Buckets")
    for key in ("by_symbol_worst", "by_exit_reason_worst", "by_entry_date_worst"):
        lines.append(f"### {key}")
        lines.extend(_markdown_table(edge[key], [(key.replace("_worst", ""), key.replace("by_", "").replace("_worst", "")), ("Trades", "trades"), ("Net PnL", "net_pnl"), ("Win", "win_rate")], max_rows=8))
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run a small smoke subset.")
    parser.add_argument("--top-train", type=int, default=45, help="Number of top OOS candidates to train-confirm.")
    parser.add_argument("--max-oos", type=int, default=0, help="Optional cap on OOS candidate count.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_config = normalize_runtime_config("kalcb", load_yaml_config(CONFIG_PATH))
    artifacts = _round_artifacts()
    candidates, metadata = _build_candidate_set(artifacts, quick=args.quick)
    if args.max_oos > 0:
        kept: dict[str, dict[str, Any]] = {item["label"]: item for item in candidates[: args.max_oos]}
        for item in candidates:
            if item["label"] in MANDATORY_LABELS:
                kept[item["label"]] = item
        candidates = list(kept.values())

    harness = ReplayHarness(base_config, OUT_DIR, quick=args.quick)
    default_source = metadata["source_r3"]
    windows = {
        "train": {"start": base_config["start"], "end": base_config["end"]},
        "oos": {"start": base_config["baseline"]["holdout_start"], "end": base_config["baseline"]["holdout_end"]},
    }

    harness._status("candidate_plan", total=len(candidates), counts=dict(Counter(item["kind"] for item in candidates)))
    oos_results: list[dict[str, Any]] = []
    results_by_label: dict[str, dict[str, Any]] = {}
    for index, candidate in enumerate(candidates, start=1):
        harness._status("evaluate_oos_start", index=index, total=len(candidates), label=candidate["label"], kind=candidate["kind"])
        try:
            result = harness.evaluate(candidate["label"], candidate["mutations"], "oos", default_source)
            result["kind"] = candidate["kind"]
            result["reason"] = candidate.get("reason", "")
            result["oos_repair_score"] = _oos_repair_score(result["metrics"])
            oos_results.append(result)
            results_by_label.setdefault(candidate["label"], {})["oos"] = result
            harness._status(
                "evaluate_oos_done",
                index=index,
                label=candidate["label"],
                net_return_pct=result["metrics"].get("broker_net_return_pct"),
                trades=result["metrics"].get("trade_count"),
                win_rate=result["metrics"].get("net_win_share"),
                elapsed=result["elapsed_seconds"],
            )
        except Exception as exc:
            harness._status("evaluate_oos_error", index=index, label=candidate["label"], error=repr(exc))

    top_oos = sorted(oos_results, key=lambda row: row["oos_repair_score"], reverse=True)
    train_labels = set(row["label"] for row in top_oos[: max(args.top_train, 0)])
    train_labels.update(MANDATORY_LABELS)
    label_to_candidate = {item["label"]: item for item in candidates}
    train_candidates = [label_to_candidate[label] for label in train_labels if label in label_to_candidate]
    harness._status("train_confirm_plan", total=len(train_candidates), labels=sorted(item["label"] for item in train_candidates))

    train_results: list[dict[str, Any]] = []
    for index, candidate in enumerate(train_candidates, start=1):
        harness._status("evaluate_train_start", index=index, total=len(train_candidates), label=candidate["label"], kind=candidate["kind"])
        try:
            result = harness.evaluate(candidate["label"], candidate["mutations"], "train", default_source)
            result["kind"] = candidate["kind"]
            result["reason"] = candidate.get("reason", "")
            train_results.append(result)
            results_by_label.setdefault(candidate["label"], {})["train"] = result
            harness._status(
                "evaluate_train_done",
                index=index,
                label=candidate["label"],
                net_return_pct=result["metrics"].get("broker_net_return_pct"),
                trades=result["metrics"].get("trade_count"),
                win_rate=result["metrics"].get("net_win_share"),
                elapsed=result["elapsed_seconds"],
            )
        except Exception as exc:
            harness._status("evaluate_train_error", index=index, label=candidate["label"], error=repr(exc))

    if "round3_final" not in results_by_label or "oos" not in results_by_label["round3_final"]:
        raise RuntimeError("round3_final OOS evaluation did not complete")
    if "train" not in results_by_label["round3_final"]:
        round3_candidate = label_to_candidate["round3_final"]
        results_by_label["round3_final"]["train"] = harness.evaluate(round3_candidate["label"], round3_candidate["mutations"], "train", default_source)

    final_train = results_by_label["round3_final"]["train"]["metrics"]
    confirmed: list[dict[str, Any]] = []
    for label, windows_result in results_by_label.items():
        if "oos" not in windows_result or "train" not in windows_result:
            continue
        row = {
            "label": label,
            "kind": windows_result["oos"].get("kind", ""),
            "reason": windows_result["oos"].get("reason", ""),
            "combined_score": _combined_score(windows_result["oos"]["metrics"], windows_result["train"]["metrics"], final_train),
            "oos": {
                "metrics": _clean_metric_row(windows_result["oos"]["metrics"]),
                "source": windows_result["oos"]["source"],
                "mutations": windows_result["oos"]["mutations"],
            },
            "train": {
                "metrics": _clean_metric_row(windows_result["train"]["metrics"]),
                "source": windows_result["train"]["source"],
                "mutations": windows_result["train"]["mutations"],
            },
        }
        confirmed.append(row)
    confirmed.sort(key=lambda row: row["combined_score"], reverse=True)

    ablation_labels = [item["label"] for item in candidates if item["kind"].startswith("ablation") or item["kind"] == "cumulative_sequence"]
    ablation_impacts = {
        label: _impact_vs(results_by_label, label)
        for label in ablation_labels
        if label in results_by_label and "oos" in results_by_label[label] and "train" in results_by_label[label]
    }
    edge = _edge_case_diagnostics(results_by_label["round3_final"]["oos"])

    compact_oos = [
        {
            "label": row["label"],
            "kind": row.get("kind", ""),
            "score": row.get("oos_repair_score", 0.0),
            "metrics": _clean_metric_row(row["metrics"]),
            "source": row["source"],
            "reason": row.get("reason", ""),
        }
        for row in sorted(oos_results, key=lambda item: item["oos_repair_score"], reverse=True)
    ]
    payload = {
        "generated_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "config": str(CONFIG_PATH),
        "windows": windows,
        "metadata": metadata,
        "counts": {
            "oos_evaluated": len(oos_results),
            "train_confirmed": len(train_results),
            "candidate_count": len(candidates),
        },
        "oos_ranked": compact_oos,
        "confirmed_train_ranked": confirmed,
        "ablation_impacts": ablation_impacts,
        "edge_case_diagnostics": edge,
        "results_by_label": {
            label: {
                window: {
                    "metrics": _clean_metric_row(result["metrics"]),
                    "source": result["source"],
                    "kind": result.get("kind", ""),
                    "reason": result.get("reason", ""),
                    "elapsed_seconds": result.get("elapsed_seconds", 0.0),
                }
                for window, result in windows_result.items()
            }
            for label, windows_result in results_by_label.items()
        },
        "top_recommendation": confirmed[0] if confirmed else {},
    }
    _write_json(OUT_DIR / "round3_oos_ablation_results.json", payload)
    _write_json(OUT_DIR / "round3_oos_trade_diagnostics.json", edge)
    (OUT_DIR / "round3_oos_ablation_summary.md").write_text(_summary_markdown(payload), encoding="utf-8")
    if confirmed:
        _write_json(OUT_DIR / "recommended_mutations.json", confirmed[0])
    harness._status(
        "complete",
        result_path=str(OUT_DIR / "round3_oos_ablation_results.json"),
        summary_path=str(OUT_DIR / "round3_oos_ablation_summary.md"),
    )


if __name__ == "__main__":
    main()
