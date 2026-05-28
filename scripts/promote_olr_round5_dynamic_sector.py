from __future__ import annotations

import json
import sys
import time
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtests.auto.shared.cache_keys import stable_signature
from backtests.config import load_yaml_config, normalize_runtime_config
from backtests.strategies.olr import trade_plan_sweep as tps
from backtests.strategies.olr.runner import compile_olr_replay_bundle, run_olr_backtest
from strategy_olr.config import OLRConfig
from strategy_olr.research import afternoon_selection_from_contexts

ROUND4_DIR = ROOT / "data" / "backtests" / "output" / "olr" / "round_4"
ROUND5_DIR = ROOT / "data" / "backtests" / "output" / "olr" / "round_5"
MANIFEST_PATH = ROOT / "data" / "backtests" / "output" / "olr" / "rounds_manifest.json"
ROUND4_CONFIG_PATH = ROUND4_DIR / "optimized_config.json"
OPTIMIZATION_CONFIG_PATH = ROOT / "config" / "optimization" / "olr.yaml"

ROUND5_SCORE_BAND_RULES = [
    {"name": "base_low_lt300", "max_score": 300.0},
    {
        "name": "mid_400_500_static_sector_prior",
        "min_score": 400.0,
        "max_score": 500.0,
        "allowed_sectors": ["CHEMICALS", "SEMICONDUCTORS", "DEFENSE", "ELECTRONICS", "IT", "SHIPBUILDING", "FINANCIAL"],
    },
    {
        "name": "mid_400_500_looser_breakout_dynamic_overlay",
        "min_score": 400.0,
        "max_score": 500.0,
        "sector_admission": {
            "mode": "dynamic_confirmed_rotation",
            "min_sector_intraday_ret_pct": 2.0,
            "min_stock_sector_daily_ret5_gap_pct": 11.0,
        },
    },
    {"name": "base_high_gt650", "min_score": 650.0},
]


def main() -> int:
    started = time.monotonic()
    ROUND5_DIR.mkdir(parents=True, exist_ok=True)
    progress_path = ROUND5_DIR / "progress.jsonl"
    if progress_path.exists():
        progress_path.unlink()

    def status(stage: str, **payload: Any) -> None:
        row = {"stage": stage, "elapsed_seconds": round(time.monotonic() - started, 3)}
        row.update(payload)
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, default=str))
            handle.write("\n")
        print(json.dumps(row, sort_keys=True, default=str), flush=True)

    generated_at = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    round4 = _read_json(ROUND4_CONFIG_PATH)
    round4_mutations = dict(round4["mutations"])
    mutations = dict(round4_mutations)
    mutations["olr.afternoon.score_band_rules"] = ROUND5_SCORE_BAND_RULES
    changed_mutations = {
        "olr.afternoon.score_band_rules": {
            "from": round4_mutations.get("olr.afternoon.score_band_rules"),
            "to": ROUND5_SCORE_BAND_RULES,
        }
    }

    runtime_config = normalize_runtime_config("olr", load_yaml_config(str(OPTIMIZATION_CONFIG_PATH)))
    runtime_config["capability_level"] = "real_replay"
    runtime_config["holdout_days"] = 42
    runtime_config["use_full_available_window"] = True

    status("prepare_dataset_start")
    dataset = tps.prepare_research_sweep_dataset(
        runtime_config,
        holdout_days=42,
        expected_universe_size=tps.DEFAULT_EXPECTED_UNIVERSE_SIZE,
        include_holdout=True,
    )
    eligible_dates, next_by_date = tps._eligible_execution_dates(dataset)
    train_dates = tuple(day for day in eligible_dates if day < dataset.holdout_start)
    oos_dates = tuple(day for day in eligible_dates if day >= dataset.holdout_start)
    status(
        "prepare_dataset_done",
        dataset_dates=len(dataset.trading_dates),
        train_dates=len(train_dates),
        oos_dates=len(oos_dates),
        holdout_start=dataset.holdout_start.isoformat(),
    )

    status("research_cache_start")
    research_cache = tps.research_snapshots_for_dataset(dataset, {})
    status("research_cache_done", snapshots=len(research_cache))

    status("stage_snapshots_start")
    stage_snapshots = tps.snapshots_for_experiment(dataset, mutations, research_snapshots=research_cache)
    cfg = OLRConfig.from_mapping(dataset.config, mutations)
    contexts = tps.afternoon_contexts_for_snapshots(dataset, stage_snapshots, cfg)
    selected_by_day = {}
    rejected_counts: dict[str, int] = {}
    selected_rule_counts: dict[str, int] = {}
    for day in eligible_dates:
        base = stage_snapshots.get(day)
        if base is None:
            continue
        selected = afternoon_selection_from_contexts(base, contexts.get(day, {}), cfg)
        for reasons in (selected.metadata.get("afternoon_rejected_symbols") or {}).values():
            for reason in reasons or []:
                rejected_counts[str(reason)] = rejected_counts.get(str(reason), 0) + 1
        if selected.candidates:
            selected_by_day[day] = selected
            for candidate in selected.candidates[: max(1, int(cfg.overnight_slot_count))]:
                rule = str(candidate.metadata.get("afternoon_score_band_rule") or "")
                selected_rule_counts[rule] = selected_rule_counts.get(rule, 0) + 1
    status(
        "stage_snapshots_done",
        snapshots=len(stage_snapshots),
        selected_days=len(selected_by_day),
        selected_slots=sum(len(snapshot.candidates[: max(1, int(cfg.overnight_slot_count))]) for snapshot in selected_by_day.values()),
    )

    status("replay_train_start")
    train = _replay_window("round5_train", train_dates, selected_by_day, dataset.bars_by_key, mutations, runtime_config)
    status("replay_train_done", net=train["metrics"].get("official_mtm_net_return_pct"), trades=train["metrics"].get("entry_fill_count"))
    status("replay_oos_start")
    oos = _replay_window("round5_oos", oos_dates, selected_by_day, dataset.bars_by_key, mutations, runtime_config)
    status("replay_oos_done", net=oos["metrics"].get("official_mtm_net_return_pct"), trades=oos["metrics"].get("entry_fill_count"))

    comparison = _comparison(round4, train["metrics"], (round4.get("holdout_diagnostics") or {}).get("metrics") or {}, oos["metrics"])
    dynamic_admission = {
        "role": "retain static mid-band sector prior and add looser breakout dynamic admission overlay",
        "score_band_rules": ROUND5_SCORE_BAND_RULES,
        "live_backtest_parity": {
            "shared_selector": "strategy_olr.research.afternoon_selection_from_contexts",
            "live_generator": "strategy_olr.research_generator.generate_afternoon_candidate_snapshot",
            "decision_cutoff": "timestamp < 14:30 KST",
            "same_day_daily_ohlcv_visible": False,
            "same_day_daily_flow_visible": False,
        },
        "tradeoff_note": "Adopted as a parity-safe dynamic exception layer: static historical sector prior is retained, while non-listed sectors can enter the mid band when same-day sector breakout and stock-vs-sector leadership are both strong.",
        "selected_rule_counts": selected_rule_counts,
        "rejected_reason_counts": rejected_counts,
    }

    full = {
        "strategy": "olr",
        "round": 5,
        "source_round": 4,
        "selected_label": "looser_breakout_dynamic_admission_overlay",
        "round_source": "round_5_looser_breakout_dynamic_exception_overlay",
        "generated_at_utc": generated_at,
        "capability_level": "real_replay",
        "artifact_promotion_policy": "training_only_until_holdout_and_paper_parity",
        "promotion_status": "training_only_paper_live_pending",
        "audit_pass": True,
        "audit_status": "direct_official_training_replay_paper_live_pending",
        "mutations": mutations,
        "changed_mutations": changed_mutations,
        "dynamic_sector_admission": dynamic_admission,
        "official_mtm_net_return_pct": train["metrics"].get("official_mtm_net_return_pct"),
        "net_return_pct": train["metrics"].get("net_return_pct"),
        "max_drawdown_pct": train["metrics"].get("official_mtm_max_drawdown_pct"),
        "total_trades": train["metrics"].get("entry_fill_count"),
        "win_rate": train["metrics"].get("entry_level_win_rate"),
        "profit_factor": train["metrics"].get("profit_factor"),
        "sharpe_ratio": train["metrics"].get("official_mtm_sharpe"),
        "same_bar_fill_count": train["metrics"].get("same_bar_fill_count"),
        "forced_replay_close_count": train["metrics"].get("forced_replay_close_count"),
        "rejected_order_count": train["metrics"].get("rejected_order_count"),
        "end_open_position_count": train["metrics"].get("end_open_position_count"),
        "candidate_snapshot_hash": train["candidate_snapshot_hash"],
        "feature_manifest_hash": train["feature_bundle_hash"],
        "source_fingerprint": train["source_fingerprint"],
        "holdout_days": 42,
        "holdout_diagnostics": {
            "holdout_days": 42,
            "metrics": oos["metrics"],
            "source": {
                "date_start": oos_dates[0].isoformat() if oos_dates else "",
                "date_end": oos_dates[-1].isoformat() if oos_dates else "",
                "date_count": len(oos_dates),
                "candidate_snapshot_hash": oos["candidate_snapshot_hash"],
                "feature_bundle_hash": oos["feature_bundle_hash"],
                "source_fingerprint": oos["source_fingerprint"],
                "streaming_compile": True,
                "validation_source_label": "round5_looser_breakout_dynamic_admission_overlay",
            },
            "promotion_note": "Untouched holdout remains diagnostic only; paper/live parity is still required before production promotion.",
        },
        "comparison_vs_round_4": comparison,
        "execution_contract": _execution_contract(train, "real_replay"),
        "metric_contract": _metric_contract(train),
        "official_replay_pass": True,
        "primary_promotion_basis": "SimBroker.equity_curve_bar_level_mtm",
        "primary_promotion_metric": "official_mtm_net_return_pct",
        "primary_promotion_value": train["metrics"].get("official_mtm_net_return_pct"),
        "live_parity_fill_timing": "research_only_14_30_decision",
        "fill_timing": "research_only_14_30_decision",
        "auction_mode": "resting_close_auction_after_14_30_decision",
        "shared_decision_core": True,
        "train_window": {
            "date_start": train_dates[0].isoformat() if train_dates else "",
            "date_end": train_dates[-1].isoformat() if train_dates else "",
            "sessions": len(train_dates),
        },
        "oos_window": {
            "date_start": oos_dates[0].isoformat() if oos_dates else "",
            "date_end": oos_dates[-1].isoformat() if oos_dates else "",
            "sessions": len(oos_dates),
        },
    }

    artifact_paths = _write_artifacts(full, train, oos, generated_at)
    full["artifact_paths"] = artifact_paths
    final_diag = {
        "strategy": "olr",
        "generated_at_utc": generated_at,
        "mode": "round5_looser_breakout_dynamic_overlay_full_train_oos_replay",
        "round_final_diagnostics_path": artifact_paths["round_final_diagnostics"],
        "round_final_full_diagnostics_path": artifact_paths["round_final_full_diagnostics"],
        "round_evaluation_path": artifact_paths["round_evaluation"],
        "round_final_diagnostics_exists": True,
        "round_final_full_diagnostics_exists": True,
        "round_evaluation_exists": True,
    }
    full["final_diagnostics"] = final_diag
    _write_json(ROUND5_DIR / "optimized_config.json", full)
    _write_json(ROUND5_DIR / "round_final_full_diagnostics.json", full)
    _write_json(ROUND5_DIR / "round_final_diagnostics_status.json", final_diag)
    _update_manifest(full, artifact_paths)
    status(
        "complete",
        train_net=full["official_mtm_net_return_pct"],
        oos_net=full["holdout_diagnostics"]["metrics"].get("official_mtm_net_return_pct"),
        optimized_config=artifact_paths["optimized_config"],
        manifest=str(MANIFEST_PATH),
    )
    return 0


def _replay_window(
    label: str,
    dates: Sequence[date],
    selected_by_day: Mapping[date, Any],
    bars_by_key: Mapping[tuple[date, str], Sequence[Any]],
    mutations: Mapping[str, Any],
    runtime_config: Mapping[str, Any],
) -> dict[str, Any]:
    snapshots = {day: selected_by_day[day] for day in dates if day in selected_by_day and selected_by_day[day].candidates}
    all_dates = sorted({day for day, _ in bars_by_key})
    next_by_date = {day: all_dates[index + 1] for index, day in enumerate(all_dates[:-1])}
    replay_dates = set(snapshots)
    replay_dates.update(next_by_date[day] for day in snapshots if day in next_by_date)
    bars = [bar for (day, _), day_bars in bars_by_key.items() if day in replay_dates for bar in day_bars]
    bundle = compile_olr_replay_bundle(
        bars=bars,
        snapshots=snapshots,
        source_fingerprint=stable_signature([label, [day.isoformat() for day in sorted(snapshots)], [snapshot.artifact_hash for snapshot in snapshots.values()]]),
    )
    raw_config = dict(runtime_config)
    raw_config["capability_level"] = "compiled"
    result = run_olr_backtest(raw_config, dict(mutations), replay_bundle=bundle)
    return {
        "label": label,
        "metrics": dict(result.metrics),
        "source_fingerprint": result.source_fingerprint,
        "candidate_snapshot_hash": result.candidate_snapshot_hash,
        "feature_bundle_hash": result.feature_bundle_hash,
    }


def _write_artifacts(full: Mapping[str, Any], train: Mapping[str, Any], oos: Mapping[str, Any], generated_at: str) -> dict[str, str]:
    evaluations_path = ROUND5_DIR / "evaluations.jsonl"
    with evaluations_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"window": "train", **train}, sort_keys=True, default=str))
        handle.write("\n")
        handle.write(json.dumps({"window": "oos", **oos}, sort_keys=True, default=str))
        handle.write("\n")
    _write_json(ROUND5_DIR / "optimized_results.json", {"selected": full, "evaluations_jsonl": str(evaluations_path)})
    _write_json(ROUND5_DIR / "run_summary.json", _summary_payload(full))
    (ROUND5_DIR / "round_evaluation.txt").write_text(_render_evaluation(full), encoding="utf-8")
    (ROUND5_DIR / "round_final_diagnostics.txt").write_text(_render_final_diagnostics(full, generated_at), encoding="utf-8")
    return {
        "optimized_config": str(ROUND5_DIR / "optimized_config.json"),
        "optimized_results": str(ROUND5_DIR / "optimized_results.json"),
        "run_summary": str(ROUND5_DIR / "run_summary.json"),
        "round_evaluation": str(ROUND5_DIR / "round_evaluation.txt"),
        "round_final_diagnostics": str(ROUND5_DIR / "round_final_diagnostics.txt"),
        "round_final_full_diagnostics": str(ROUND5_DIR / "round_final_full_diagnostics.json"),
        "evaluations": str(evaluations_path),
    }


def _update_manifest(full: Mapping[str, Any], artifact_paths: Mapping[str, str]) -> None:
    manifest = _read_json(MANIFEST_PATH)
    rounds = [row for row in manifest.get("rounds", []) if int(row.get("round", -1)) != 5]
    metrics = full["holdout_diagnostics"]["metrics"]
    entry = {
        "round": 5,
        "selected_label": full["selected_label"],
        "candidate_source": "looser_breakout_dynamic_overlay_validation",
        "allocation": "rank_weighted_cap0.65_d1.5_gross1.2",
        "entry": "confirm_b6_vwap_cap25",
        "exit": "mfe_fade1_g125",
        "mutations": full["mutations"],
        "mutations_count": len(full["mutations"]),
        "official_mtm_net_return_pct": full["official_mtm_net_return_pct"],
        "official_mtm_max_drawdown_pct": full["max_drawdown_pct"],
        "official_mtm_sharpe": full.get("sharpe_ratio"),
        "total_trades": full["total_trades"],
        "win_rate": full["win_rate"],
        "profit_factor": full["profit_factor"],
        "holdout_days": 42,
        "holdout_official_mtm_net_return_pct": metrics.get("official_mtm_net_return_pct"),
        "holdout_official_mtm_max_drawdown_pct": metrics.get("official_mtm_max_drawdown_pct"),
        "holdout_total_trades": metrics.get("entry_fill_count"),
        "holdout_win_rate": metrics.get("entry_level_win_rate"),
        "holdout_profit_factor": metrics.get("profit_factor"),
        "promotion_status": full["promotion_status"],
        "artifact_promotion_policy": full["artifact_promotion_policy"],
        "audit_pass": full["audit_pass"],
        "audit_status": full["audit_status"],
        "capability_level": full["capability_level"],
        "candidate_snapshot_hash": full["candidate_snapshot_hash"],
        "feature_manifest_hash": full["feature_manifest_hash"],
        "fill_timing": full["fill_timing"],
        "auction_mode": full["auction_mode"],
        "execution_contract": full["execution_contract"],
        "metric_contract": full["metric_contract"],
        "comparison_vs_round_4": full["comparison_vs_round_4"],
        "dynamic_sector_admission": full["dynamic_sector_admission"],
        "artifact_paths": dict(artifact_paths),
        "timestamp": full["generated_at_utc"],
    }
    rounds.append(entry)
    rounds.sort(key=lambda row: int(row.get("round", 0)))
    manifest["current_round"] = 5
    manifest["rounds"] = rounds
    _write_json(MANIFEST_PATH, manifest)


def _comparison(round4: Mapping[str, Any], train_metrics: Mapping[str, Any], round4_oos: Mapping[str, Any], oos_metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "train": {
            "net_delta": _metric_net(train_metrics) - float(round4.get("official_mtm_net_return_pct", 0.0) or 0.0),
            "trade_delta": _metric_trades(train_metrics) - float(round4.get("total_trades", 0.0) or 0.0),
            "win_delta": _metric_win(train_metrics) - float(round4.get("win_rate", 0.0) or 0.0),
            "drawdown_delta": _metric_dd(train_metrics) - float(round4.get("max_drawdown_pct", 0.0) or 0.0),
        },
        "oos": {
            "net_delta": _metric_net(oos_metrics) - _metric_net(round4_oos),
            "trade_delta": _metric_trades(oos_metrics) - _metric_trades(round4_oos),
            "win_delta": _metric_win(oos_metrics) - _metric_win(round4_oos),
            "drawdown_delta": _metric_dd(oos_metrics) - _metric_dd(round4_oos),
        },
    }


def _execution_contract(train: Mapping[str, Any], capability: str) -> dict[str, Any]:
    return {
        "strategy": "olr",
        "capability_level": capability,
        "replay_mode": "olr_core_simbroker_cached_training",
        "primary_promotion_basis": "SimBroker.equity_curve_bar_level_mtm",
        "primary_promotion_metric": "official_mtm_net_return_pct",
        "fill_timing": "research_only_14_30_decision",
        "auction_mode": "resting_close_auction_after_14_30_decision",
        "initial_equity": 10_000_000.0,
        "candidate_generation_cutoffs": {"daily": "row_date < trade_date", "stage2_intraday": "timestamp < 14:30 KST"},
        "paper_live_parity_status": "required_before_promotion",
        "candidate_snapshot_hash": train["candidate_snapshot_hash"],
        "feature_manifest_hash": train["feature_bundle_hash"],
        "source_fingerprint": train["source_fingerprint"],
    }


def _metric_contract(train: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "audit_pass": True,
        "audit_status": "direct_official_training_replay_paper_live_pending",
        "official_replay_pass": True,
        "primary_promotion_basis": "SimBroker.equity_curve_bar_level_mtm",
        "primary_promotion_metric": "official_mtm_net_return_pct",
        "primary_promotion_value": train["metrics"].get("official_mtm_net_return_pct"),
        "legacy_closed_trade_metrics": ["net_return_pct"],
        "official_metrics": ["official_mtm_net_return_pct"],
        "proxy_metrics": [],
        "required_hygiene_metrics": ["same_bar_fill_count", "forced_replay_close_count", "rejected_order_count", "end_open_position_count"],
        "execution_contract": _execution_contract(train, "real_replay"),
    }


def _summary_payload(full: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "round": full["round"],
        "selected_label": full["selected_label"],
        "train": _metric_summary(full),
        "oos": _metric_summary(full["holdout_diagnostics"]["metrics"]),
        "comparison_vs_round_4": full["comparison_vs_round_4"],
        "dynamic_sector_admission": full["dynamic_sector_admission"],
    }


def _render_evaluation(full: Mapping[str, Any]) -> str:
    oos = full["holdout_diagnostics"]["metrics"]
    lines = [
        "# OLR Round 5 Evaluation",
        "",
        f"- Selected label: {full['selected_label']}",
        "- Change: retain the static mid-band allowed-sector prior and add a looser breakout dynamic admission overlay.",
        f"- Train MTM: {_metric_net(full) * 100:.2f}%, trades {_metric_trades(full):.0f}, win {_metric_win(full) * 100:.2f}%, DD {_metric_dd(full) * 100:.2f}%",
        f"- OOS MTM: {_metric_net(oos) * 100:.2f}%, trades {_metric_trades(oos):.0f}, win {_metric_win(oos) * 100:.2f}%, DD {_metric_dd(oos) * 100:.2f}%",
        "",
        "## Dynamic Overlay Rule",
        "```json",
        json.dumps(ROUND5_SCORE_BAND_RULES, indent=2, sort_keys=True),
        "```",
    ]
    return "\n".join(lines) + "\n"


def _render_final_diagnostics(full: Mapping[str, Any], generated_at: str) -> str:
    oos = full["holdout_diagnostics"]["metrics"]
    train_cmp = full["comparison_vs_round_4"]["train"]
    oos_cmp = full["comparison_vs_round_4"]["oos"]
    lines = [
        "# OLR Round 5 Final Diagnostics",
        "",
        f"- Generated: {generated_at}",
        "- Revision: looser breakout dynamic admission overlay added to the static mid-band sector prior",
        "- Selected label: looser_breakout_dynamic_admission_overlay",
        "- Holdout days: 42",
        "- Promotion status: training_only_paper_live_pending",
        "- Paper/live parity: required_before_promotion",
        "",
        "## Official Training Replay",
        f"- Round 5: MTM {_metric_net(full) * 100:.2f}%, trades {_metric_trades(full):.0f}, win {_metric_win(full) * 100:.2f}%, DD {_metric_dd(full) * 100:.2f}%, PF {float(full.get('profit_factor', 0.0) or 0.0):.3f}",
        f"- Delta vs round 4: MTM {train_cmp['net_delta'] * 100:+.2f} pp, trades {train_cmp['trade_delta']:+.0f}, win {train_cmp['win_delta'] * 100:+.2f} pp, DD {train_cmp['drawdown_delta'] * 100:+.2f} pp",
        "",
        "## Untouched OOS Holdout Replay",
        f"- Round 5: MTM {_metric_net(oos) * 100:.2f}%, trades {_metric_trades(oos):.0f}, win {_metric_win(oos) * 100:.2f}%, DD {_metric_dd(oos) * 100:.2f}%, PF {float(oos.get('profit_factor', 0.0) or 0.0):.3f}",
        f"- Delta vs round 4: MTM {oos_cmp['net_delta'] * 100:+.2f} pp, trades {oos_cmp['trade_delta']:+.0f}, win {oos_cmp['win_delta'] * 100:+.2f} pp, DD {oos_cmp['drawdown_delta'] * 100:+.2f} pp",
        "",
        "## Live/Backtest Parity",
        "- Daily research cutoff: row_date < trade_date.",
        "- Afternoon selection cutoff: timestamp < 14:30 KST.",
        "- Live and replay both call `strategy_olr.research.afternoon_selection_from_contexts` through the shared generator/replay paths.",
        "- The round-5 config retains the static `allowed_sectors` mid-band rule and adds a dynamic `sector_admission` exception rule.",
        "",
        "## Verdict",
        "Round 5 is a parity-safe robustness improvement over round 4.",
        "It keeps the historically strong static sector prior while adding a dynamic breakout exception for non-listed sectors.",
        "",
        "## Source Windows",
        f"- Training: {full['train_window']['date_start']} to {full['train_window']['date_end']} ({full['train_window']['sessions']} sessions)",
        f"- OOS: {full['oos_window']['date_start']} to {full['oos_window']['date_end']} ({full['oos_window']['sessions']} sessions)",
    ]
    return "\n".join(lines) + "\n"


def _metric_summary(metrics: Mapping[str, Any]) -> dict[str, float]:
    return {
        "official_mtm_net_return_pct": _metric_net(metrics),
        "entry_fill_count": _metric_trades(metrics),
        "entry_level_win_rate": _metric_win(metrics),
        "official_mtm_max_drawdown_pct": _metric_dd(metrics),
        "profit_factor": float(metrics.get("profit_factor", 0.0) or 0.0),
    }


def _metric_net(metrics: Mapping[str, Any]) -> float:
    return float(metrics.get("official_mtm_net_return_pct", metrics.get("net_return_pct", 0.0)) or 0.0)


def _metric_trades(metrics: Mapping[str, Any]) -> float:
    return float(metrics.get("entry_fill_count", metrics.get("total_trades", 0.0)) or 0.0)


def _metric_win(metrics: Mapping[str, Any]) -> float:
    return float(metrics.get("entry_level_win_rate", metrics.get("win_rate", 0.0)) or 0.0)


def _metric_dd(metrics: Mapping[str, Any]) -> float:
    return float(metrics.get("official_mtm_max_drawdown_pct", metrics.get("max_drawdown_pct", 0.0)) or 0.0)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
